"""
User management routes for LayoutCraft
"""
from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import List, Optional
import logging
from datetime import datetime, timedelta

from auth.dependencies import get_current_user, RequireProTier
from auth.middleware import auth_middleware
from models.user import UserResponse
from models.generation import GenerationResponse

router = APIRouter(prefix="/users", tags=["Users"])
logger = logging.getLogger(__name__)

@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    """
    Get current user profile (alias for /auth/profile)
    """
    try:
        user_profile = await auth_middleware.get_user_profile(current_user["id"])
        
        if not user_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )
        
        return UserResponse(**user_profile)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get current user error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user profile"
        )

@router.get("/usage")
async def get_usage_stats(current_user: dict = Depends(get_current_user)):
    """
    Get user usage statistics
    """
    try:
        # Get current period usage
        user_profile = await auth_middleware.get_user_profile(current_user["id"])
        
        if not user_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )
        
        # Get tier limits
        tier_limits = {
            "free": 10,
            "pro": 500,
            "enterprise": float('inf')
        }
        
        tier = user_profile["subscription_tier"]
        limit = tier_limits.get(tier, 10)
        
        # Get generation history for this period
        reset_date = datetime.fromisoformat(user_profile["usage_reset_date"].replace("Z", "+00:00"))
        
        history_response = auth_middleware.supabase.table("generation_history").select(
            "created_at, generation_time_ms"
        ).eq("user_id", current_user["id"]).gte(
            "created_at", reset_date.isoformat()
        ).execute()
        
        generations_this_period = len(history_response.data) if history_response.data else 0
        
        # Calculate average generation time
        total_time = sum(
            item["generation_time_ms"] for item in history_response.data 
            if item["generation_time_ms"] is not None
        ) if history_response.data else 0
        
        avg_time = total_time / generations_this_period if generations_this_period > 0 else 0
        
        return {
            "subscription_tier": tier,
            "usage_count": user_profile["usage_count"],
            "usage_limit": limit if limit != float('inf') else None,
            "usage_reset_date": user_profile["usage_reset_date"],
            "generations_this_period": generations_this_period,
            "average_generation_time_ms": round(avg_time),
            "remaining_generations": max(0, limit - user_profile["usage_count"]) if limit != float('inf') else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get usage stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get usage statistics"
        )

@router.get("/history", response_model=List[GenerationResponse])
async def get_generation_history(
    current_user: dict = Depends(get_current_user),
    limit: int = Query(default=10, le=100),
    offset: int = Query(default=0, ge=0)
):
    """
    Get user generation history
    """
    try:
        response = auth_middleware.supabase.table("generation_history").select(
            "id, prompt, model_used, width, height, image_url, generation_time_ms, created_at"
        ).eq("user_id", current_user["id"]).order(
            "created_at", desc=True
        ).limit(limit).offset(offset).execute()
        
        if not response.data:
            return []
        
        return [GenerationResponse(**item) for item in response.data]
        
    except Exception as e:
        logger.error(f"Get generation history error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get generation history"
        )

@router.delete("/history/{generation_id}")
async def delete_generation(
    generation_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a specific generation from history
    """
    try:
        # Check if generation belongs to current user
        check_response = auth_middleware.supabase.table("generation_history").select(
            "id"
        ).eq("id", generation_id).eq("user_id", current_user["id"]).execute()
        
        if not check_response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Generation not found"
            )
        
        # Delete the generation
        auth_middleware.supabase.table("generation_history").delete().eq(
            "id", generation_id
        ).eq("user_id", current_user["id"]).execute()
        
        logger.info(f"Generation deleted: {generation_id} by user {current_user['id']}")
        
        return {"message": "Generation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete generation"
        )

@router.get("/preferences")
async def get_user_preferences(current_user: dict = Depends(get_current_user)):
    """
    Get user preferences
    """
    try:
        response = auth_middleware.supabase.table("user_preferences").select(
            "*"
        ).eq("user_id", current_user["id"]).single().execute()
        
        if not response.data:
            # Create default preferences if they don't exist
            default_prefs = {
                "user_id": current_user["id"],
                "default_model": "gemini-1.5-flash",
                "default_width": 1200,
                "default_height": 630,
                "auto_save_drafts": True,
                "email_notifications": True
            }
            
            create_response = auth_middleware.supabase.table("user_preferences").insert(
                default_prefs
            ).execute()
            
            return create_response.data[0]
        
        return response.data
        
    except Exception as e:
        logger.error(f"Get user preferences error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user preferences"
        )

@router.put("/preferences")
async def update_user_preferences(
    preferences: dict,
    current_user: dict = Depends(get_current_user)
):
    """
    Update user preferences
    """
    try:
        # Validate preferences
        allowed_fields = {
            "default_model", "default_width", "default_height",
            "auto_save_drafts", "email_notifications"
        }
        
        update_data = {k: v for k, v in preferences.items() if k in allowed_fields}
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid preferences to update"
            )
        
        # Update preferences
        response = auth_middleware.supabase.table("user_preferences").update(
            update_data
        ).eq("user_id", current_user["id"]).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User preferences not found"
            )
        
        logger.info(f"User preferences updated: {current_user['id']}")
        
        return response.data[0]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update user preferences error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user preferences"
        )
