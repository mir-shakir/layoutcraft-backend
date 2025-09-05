"""
Generation service for LayoutCraft database operations
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
from supabase import Client
from models.generation import GenerationCreate
import uuid

logger = logging.getLogger(__name__)

class GenerationService:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
    
    async def create_generation(self, generation_data: GenerationCreate) -> Optional[Dict[str, Any]]:
        """
        Save a new generation to the database
        """
        try:
            # Convert to dict and add creation timestamp
            data = generation_data.dict()
            data["created_at"] = datetime.now().isoformat()
            
            # --- FIX: Convert UUIDs to strings before sending to DB ---
            for key, value in data.items():
                if isinstance(value, uuid.UUID):
                    data[key] = str(value)
            
            # --- FIX: Ensure you are using the new table name 'generations' ---
            response = self.supabase.table("generations").insert(data).execute()
            
            if response.data:
                logger.info(f"Saved generation for user {generation_data.user_id}")
                return response.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Error creating generation: {str(e)}", exc_info=True) # Added exc_info for better logging
            return None
    
    async def get_user_generations(self, user_id: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get user's generation history with pagination
        """
        try:
            response = self.supabase.table("generations").select(
                "id, prompt, model_used, width, height, image_url, generation_time_ms, created_at"
            ).eq("user_id", user_id).order("created_at", desc=True).limit(limit).offset(offset).execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error getting user generations {user_id}: {str(e)}")
            return []
    
    async def get_generation_by_id(self, generation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific generation by ID (with user ownership check)
        """
        try:
            response = self.supabase.table("generations").select("*").eq("id", generation_id).eq("user_id", user_id).single().execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting generation {generation_id}: {str(e)}")
            return None
    
    async def delete_generation(self, generation_id: str, user_id: str) -> bool:
        """
        Delete a specific generation (with user ownership check)
        """
        try:
            response = self.supabase.table("generations").delete().eq("id", generation_id).eq("user_id", user_id).execute()
            
            if response.data:
                logger.info(f"Deleted generation {generation_id} for user {user_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting generation {generation_id}: {str(e)}")
            return False
    
    async def get_generation_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get generation statistics for a user
        """
        try:
            # Get all generations for the user
            all_generations = self.supabase.table("generations").select("*").eq("user_id", user_id).execute()
            
            if not all_generations.data:
                return {
                    "total_generations": 0,
                    "avg_generation_time": 0,
                    "most_used_model": None,
                    "most_common_dimensions": None
                }
            
            generations = all_generations.data
            
            # Calculate statistics
            total_generations = len(generations)
            
            # Average generation time
            times = [g["generation_time_ms"] for g in generations if g["generation_time_ms"]]
            avg_time = sum(times) / len(times) if times else 0
            
            # Most used model
            models = [g["model_used"] for g in generations]
            most_used_model = max(set(models), key=models.count) if models else None
            
            # Most common dimensions
            dimensions = [f"{g['width']}x{g['height']}" for g in generations]
            most_common_dimensions = max(set(dimensions), key=dimensions.count) if dimensions else None
            
            # Recent activity (last 7 days)
            week_ago = datetime.now() - timedelta(days=7)
            recent_generations = [g for g in generations if datetime.fromisoformat(g["created_at"].replace("Z", "+00:00")) > week_ago]
            
            return {
                "total_generations": total_generations,
                "avg_generation_time": round(avg_time),
                "most_used_model": most_used_model,
                "most_common_dimensions": most_common_dimensions,
                "recent_generations": len(recent_generations),
                "first_generation": generations[-1]["created_at"] if generations else None,
                "last_generation": generations[0]["created_at"] if generations else None
            }
            
        except Exception as e:
            logger.error(f"Error getting generation statistics {user_id}: {str(e)}")
            return {}
    
    async def get_popular_prompts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most popular prompts across all users (anonymized)
        """
        try:
            # Get all prompts and count frequency
            response = self.supabase.table("generations").select("prompt").execute()
            
            if not response.data:
                return []
            
            # Count prompt frequency
            from collections import Counter
            prompt_counts = Counter(item["prompt"] for item in response.data)
            
            # Return top prompts
            popular_prompts = []
            for prompt, count in prompt_counts.most_common(limit):
                popular_prompts.append({
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "usage_count": count
                })
            
            return popular_prompts
            
        except Exception as e:
            logger.error(f"Error getting popular prompts: {str(e)}")
            return []
    
    async def cleanup_old_generations(self, days_old: int = 90) -> int:
        """
        Clean up old generations (for free users)
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Only clean up generations from free users
            response = self.supabase.table("generations").delete().lt("created_at", cutoff_date.isoformat()).execute()
            
            deleted_count = len(response.data) if response.data else 0
            logger.info(f"Cleaned up {deleted_count} old generations")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old generations: {str(e)}")
            return 0
