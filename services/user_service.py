"""
User service for LayoutCraft database operations
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
from supabase import Client
from models.user import UserProfile, UserCreate, UserUpdate, SubscriptionTier

logger = logging.getLogger(__name__)

class UserService:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
    
    async def create_user_profile(self, user_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a new user profile
        """
        try:
            # Set default values for new users
            profile_data = {
                "id": user_data["id"],
                "email": user_data["email"],
                "full_name": user_data.get("full_name"),
                "subscription_tier": "free",
                "usage_count": 0,
                "usage_reset_date": self._get_next_reset_date()
            }
            
            response = self.supabase.table("user_profiles").insert(profile_data).execute()
            
            if response.data:
                logger.info(f"Created user profile: {user_data['email']}")
                return response.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Error creating user profile: {str(e)}")
            return None
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user profile by ID
        """
        try:
            response = self.supabase.table("user_profiles").select("*").eq("id", user_id).single().execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting user profile {user_id}: {str(e)}")
            return None
    
    async def update_user_profile(self, user_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update user profile
        """
        try:
            # Filter out None values and add updated_at
            filtered_data = {k: v for k, v in update_data.items() if v is not None}
            filtered_data["updated_at"] = datetime.now().isoformat()
            
            response = self.supabase.table("user_profiles").update(filtered_data).eq("id", user_id).execute()
            
            if response.data:
                logger.info(f"Updated user profile: {user_id}")
                return response.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Error updating user profile {user_id}: {str(e)}")
            return None
    
    async def increment_usage(self, user_id: str, increment: int = 1) -> bool:
        """
        Increment user usage count with automatic reset handling
        """
        try:
            # Get current user data
            user_data = await self.get_user_profile(user_id)
            if not user_data:
                return False
            
            current_usage = user_data["usage_count"]
            reset_date = datetime.fromisoformat(user_data["usage_reset_date"].replace("Z", "+00:00"))
            
            # Check if we need to reset usage
            now = datetime.now().replace(tzinfo=reset_date.tzinfo)
            if now >= reset_date:
                # Reset usage for new period
                new_usage = increment
                new_reset_date = self._get_next_reset_date()
                logger.info(f"Resetting usage for user {user_id}")
            else:
                # Increment existing usage
                new_usage = current_usage + increment
                new_reset_date = reset_date
            
            # Update the database
            update_data = {
                "usage_count": new_usage,
                "usage_reset_date": new_reset_date.isoformat()
            }
            
            response = self.supabase.table("user_profiles").update(update_data).eq("id", user_id).execute()
            
            if response.data:
                logger.info(f"Updated usage for user {user_id}: {new_usage}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error updating usage for user {user_id}: {str(e)}")
            return False
    
    async def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive user statistics
        """
        try:
            # Get user profile
            user_profile = await self.get_user_profile(user_id)
            if not user_profile:
                return {}
            
            # Get usage limits based on tier
            tier_limits = {
                "free": 10,
                "pro": 500,
                "enterprise": float('inf')
            }
            
            tier = user_profile["subscription_tier"]
            limit = tier_limits.get(tier, 10)
            usage = user_profile["usage_count"]
            
            # Calculate remaining usage
            remaining = max(0, limit - usage) if limit != float('inf') else float('inf')
            
            # Get generation history count
            history_response = self.supabase.table("generation_history").select("id").eq("user_id", user_id).execute()
            total_generations = len(history_response.data) if history_response.data else 0
            
            # Get this month's generations
            reset_date = datetime.fromisoformat(user_profile["usage_reset_date"].replace("Z", "+00:00"))
            monthly_response = self.supabase.table("generation_history").select("id").eq("user_id", user_id).gte("created_at", reset_date.isoformat()).execute()
            monthly_generations = len(monthly_response.data) if monthly_response.data else 0
            
            return {
                "user_id": user_id,
                "subscription_tier": tier,
                "usage_count": usage,
                "usage_limit": limit if limit != float('inf') else None,
                "usage_remaining": remaining if remaining != float('inf') else None,
                "usage_reset_date": user_profile["usage_reset_date"],
                "total_generations": total_generations,
                "monthly_generations": monthly_generations,
                "account_created": user_profile["created_at"]
            }
            
        except Exception as e:
            logger.error(f"Error getting user statistics {user_id}: {str(e)}")
            return {}
    
    async def check_usage_limits(self, user_id: str) -> Dict[str, Any]:
        """
        Check if user has exceeded usage limits
        """
        try:
            user_profile = await self.get_user_profile(user_id)
            if not user_profile:
                return {"allowed": False, "reason": "User not found"}
            
            tier = user_profile["subscription_tier"]
            usage = user_profile["usage_count"]
            
            # Define limits
            tier_limits = {
                "free": 10,
                "pro": 500,
                "enterprise": float('inf')
            }
            
            limit = tier_limits.get(tier, 10)
            
            if usage >= limit:
                return {
                    "allowed": False,
                    "reason": f"Usage limit exceeded ({usage}/{limit})",
                    "tier": tier,
                    "usage": usage,
                    "limit": limit
                }
            
            return {
                "allowed": True,
                "tier": tier,
                "usage": usage,
                "limit": limit,
                "remaining": limit - usage if limit != float('inf') else float('inf')
            }
            
        except Exception as e:
            logger.error(f"Error checking usage limits for user {user_id}: {str(e)}")
            return {"allowed": False, "reason": "Internal error"}
    
    async def get_users_by_tier(self, tier: SubscriptionTier) -> List[Dict[str, Any]]:
        """
        Get all users by subscription tier
        """
        try:
            response = self.supabase.table("user_profiles").select("*").eq("subscription_tier", tier.value).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting users by tier {tier}: {str(e)}")
            return []
    
    async def update_subscription_tier(self, user_id: str, new_tier: SubscriptionTier) -> bool:
        """
        Update user subscription tier
        """
        try:
            update_data = {
                "subscription_tier": new_tier.value,
                "updated_at": datetime.now().isoformat()
            }
            
            response = self.supabase.table("user_profiles").update(update_data).eq("id", user_id).execute()
            
            if response.data:
                logger.info(f"Updated subscription tier for user {user_id}: {new_tier.value}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error updating subscription tier for user {user_id}: {str(e)}")
            return False
    
    def _get_next_reset_date(self) -> datetime:
        """
        Get the next monthly reset date (first day of next month)
        """
        now = datetime.now()
        if now.month == 12:
            return datetime(now.year + 1, 1, 1)
        else:
            return datetime(now.year, now.month + 1, 1)
    
    async def delete_user_profile(self, user_id: str) -> bool:
        """
        Delete user profile and all associated data
        """
        try:
            # This will cascade delete all related records due to foreign key constraints
            response = self.supabase.table("user_profiles").delete().eq("id", user_id).execute()
            
            if response.data:
                logger.info(f"Deleted user profile: {user_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting user profile {user_id}: {str(e)}")
            return False
