"""
Premium service for LayoutCraft tier-based features
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from supabase import Client
from models.user import SubscriptionTier

logger = logging.getLogger(__name__)

class PremiumService:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        
        # Define tier capabilities
        self.tier_features = {
            "free": {
                "max_width": 1200,
                "max_height": 630,
                "available_models": ["gemini-1.5-flash"],
                "export_formats": ["png"],
                "priority_queue": False,
                "batch_generation": False,
                "custom_templates": False,
                "generation_history_days": 7,
                "concurrent_generations": 1
            },
            "pro": {
                "max_width": 3000,
                "max_height": 3000,
                "available_models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
                "export_formats": ["png", "jpg", "webp"],
                "priority_queue": True,
                "batch_generation": True,
                "custom_templates": True,
                "generation_history_days": 30,
                "concurrent_generations": 3
            },
            "enterprise": {
                "max_width": 5000,
                "max_height": 5000,
                "available_models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
                "export_formats": ["png", "jpg", "webp", "svg"],
                "priority_queue": True,
                "batch_generation": True,
                "custom_templates": True,
                "generation_history_days": 365,
                "concurrent_generations": 10,
                "api_access": True,
                "webhook_support": True
            }
        }
    
    def get_tier_features(self, tier: str) -> Dict[str, Any]:
        """Get features available for a subscription tier"""
        return self.tier_features.get(tier, self.tier_features["free"])
    
    def can_use_dimensions(self, tier: str, width: int, height: int) -> bool:
        """Check if user can use specified dimensions"""
        features = self.get_tier_features(tier)
        return width <= features["max_width"] and height <= features["max_height"]
    
    def can_use_model(self, tier: str, model: str) -> bool:
        """Check if user can use specified model"""
        features = self.get_tier_features(tier)
        return model in features["available_models"]
    
    def can_export_format(self, tier: str, format: str) -> bool:
        """Check if user can export in specified format"""
        features = self.get_tier_features(tier)
        return format.lower() in features["export_formats"]
    
    async def get_generation_queue_priority(self, user_id: str) -> int:
        """Get queue priority for user (higher = more priority)"""
        try:
            user_response = self.supabase.table("user_profiles").select("subscription_tier").eq("id", user_id).single().execute()
            
            if not user_response.data:
                return 0
            
            tier = user_response.data["subscription_tier"]
            priority_map = {
                "free": 0,
                "pro": 50,
                "enterprise": 100
            }
            
            return priority_map.get(tier, 0)
            
        except Exception as e:
            logger.error(f"Error getting queue priority: {str(e)}")
            return 0
    
    async def can_generate_batch(self, user_id: str, batch_size: int) -> Dict[str, Any]:
        """Check if user can perform batch generation"""
        try:
            user_response = self.supabase.table("user_profiles").select("subscription_tier").eq("id", user_id).single().execute()
            
            if not user_response.data:
                return {"allowed": False, "reason": "User not found"}
            
            tier = user_response.data["subscription_tier"]
            features = self.get_tier_features(tier)
            
            if not features["batch_generation"]:
                return {"allowed": False, "reason": "Batch generation not available in your plan"}
            
            max_batch_sizes = {
                "pro": 10,
                "enterprise": 100
            }
            
            max_batch = max_batch_sizes.get(tier, 1)
            if batch_size > max_batch:
                return {"allowed": False, "reason": f"Batch size exceeds limit ({max_batch})"}
            
            return {"allowed": True, "max_batch_size": max_batch}
            
        except Exception as e:
            logger.error(f"Error checking batch generation: {str(e)}")
            return {"allowed": False, "reason": "Internal error"}
    
    async def create_custom_template(self, user_id: str, template_data: Dict[str, Any]) -> Optional[str]:
        """Create a custom template for premium users"""
        try:
            user_response = self.supabase.table("user_profiles").select("subscription_tier").eq("id", user_id).single().execute()
            
            if not user_response.data:
                return None
            
            tier = user_response.data["subscription_tier"]
            features = self.get_tier_features(tier)
            
            if not features["custom_templates"]:
                raise Exception("Custom templates not available in your plan")
            
            # Create template record
            template_record = {
                "user_id": user_id,
                "name": template_data["name"],
                "description": template_data.get("description", ""),
                "prompt_template": template_data["prompt_template"],
                "default_width": template_data.get("default_width", 1200),
                "default_height": template_data.get("default_height", 630),
                "is_public": template_data.get("is_public", False),
                "created_at": datetime.now().isoformat()
            }
            
            response = self.supabase.table("custom_templates").insert(template_record).execute()
            
            if response.data:
                logger.info(f"Created custom template for user {user_id}")
                return response.data[0]["id"]
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating custom template: {str(e)}")
            return None
    
    async def get_user_templates(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's custom templates"""
        try:
            response = self.supabase.table("custom_templates").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting user templates: {str(e)}")
            return []
    
    async def get_brand_kit(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user's brand kit (premium feature)"""
        try:
            user_response = self.supabase.table("user_profiles").select("subscription_tier").eq("id", user_id).single().execute()
            
            if not user_response.data:
                return None
            
            tier = user_response.data["subscription_tier"]
            if tier == "free":
                return None
            
            response = self.supabase.table("brand_kits").select("*").eq("user_id", user_id).single().execute()
            return response.data
            
        except Exception as e:
            logger.error(f"Error getting brand kit: {str(e)}")
            return None
    
    async def create_brand_kit(self, user_id: str, brand_data: Dict[str, Any]) -> Optional[str]:
        """Create brand kit for premium users"""
        try:
            user_response = self.supabase.table("user_profiles").select("subscription_tier").eq("id", user_id).single().execute()
            
            if not user_response.data:
                return None
            
            tier = user_response.data["subscription_tier"]
            if tier == "free":
                raise Exception("Brand kits not available in free plan")
            
            brand_kit = {
                "user_id": user_id,
                "brand_name": brand_data["brand_name"],
                "primary_colors": brand_data.get("primary_colors", []),
                "secondary_colors": brand_data.get("secondary_colors", []),
                "fonts": brand_data.get("fonts", []),
                "logo_url": brand_data.get("logo_url"),
                "style_guidelines": brand_data.get("style_guidelines", ""),
                "created_at": datetime.now().isoformat()
            }
            
            # Check if brand kit exists
            existing = self.supabase.table("brand_kits").select("id").eq("user_id", user_id).execute()
            
            if existing.data:
                # Update existing
                response = self.supabase.table("brand_kits").update(brand_kit).eq("user_id", user_id).execute()
            else:
                # Create new
                response = self.supabase.table("brand_kits").insert(brand_kit).execute()
            
            if response.data:
                logger.info(f"Created/updated brand kit for user {user_id}")
                return response.data[0]["id"]
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating brand kit: {str(e)}")
            return None
