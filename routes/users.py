"""
User management routes for LayoutCraft
"""
from fastapi import APIRouter, HTTPException, Path, status, Depends, Query
from typing import List, Optional
import logging
from datetime import datetime, timedelta

from auth.dependencies import get_current_user, require_pro_plan
from auth.middleware import auth_middleware
from models.user import UserResponse, BrandKit
from models.generation import GenerationResponse , HistoryParent, HistoryParentsResponse ,EditGroupsResponse ,EditGroup
from auth.middleware import get_auth_middleware, AuthMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from config.decorators import retry_on_ssLError


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

@router.get("/brand-kit", response_model=BrandKit)
async def get_brand_kit(current_user: dict = Depends(get_current_user)):
    """
    Fetch the user's brand kit. Returns 404 if not set (frontend handles empty state).
    """
    try:
        auth_middleware = get_auth_middleware()
        response = auth_middleware.supabase.table("brand_kits").select("*").eq("user_id", current_user["id"]).single().execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Brand kit not found")
        return response.data
    except Exception as e:
        # Supabase .single() raises an error if no rows found, usually caught here
        raise HTTPException(status_code=404, detail="Brand kit not found")

@router.post("/brand-kit", response_model=BrandKit)
async def save_brand_kit(
    kit_data: BrandKit, 
    current_user: dict = Depends(get_current_user)
):
    """
    Upsert (Create or Update) the user's brand kit.
    """
    try:
        # Check if user is on pro plan
        if current_user.get("subscription_tier") not in ["pro", "pro-trial"]:
             raise HTTPException(status_code=403, detail="Brand Kit is a Pro feature. Please upgrade to use it.")

        data = kit_data.dict()
        data["user_id"] = current_user["id"]
        data["updated_at"] = datetime.now().isoformat()
        
        auth_middleware = get_auth_middleware()

        # upsert=True is default behavior for single row inserts with PK conflict in some libs, 
        # but for Supabase-py we use .upsert() explicitly.
        response = auth_middleware.supabase.table("brand_kits").upsert(data).execute()
        
        if not response.data:
             raise HTTPException(status_code=500, detail="Failed to save brand kit")
             
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving brand kit: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/history/parents" ,response_model=HistoryParentsResponse)
async def get_generation_parents(
    current_user: dict = Depends(get_current_user),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware), # <-- ADD THIS DEPENDENCY
    limit: int = Query(default=10, le=20),
    offset: int = Query(default=0, ge=0)
):

    try:
        response = get_user_parent_generations(current_user, auth_middleware, limit, offset)

        if not response.data:
            return HistoryParentsResponse(parents=[], has_next=False)
        
        parents = []
        for item in response.data:
            parent = HistoryParent.model_validate({
                "design_thread_id": item["design_thread_id"],
                "original_prompt": item["prompt"],
                "created_at": item["created_at"],
                "thumbnail_url": item["image_url"],
                "used_brand_kit": item.get("used_brand_kit", False)
            })
            parents.append(parent)

        design_thread_ids = [parent.design_thread_id for parent in parents]
        variations_count = get_variations_count_aggregated(auth_middleware, design_thread_ids)
        # Update parents with their total variations count
        for parent in parents:
            parent.edit_groups_count = variations_count.get(str(parent.design_thread_id), 0)

        has_next = len(parents) == limit
        next_offset = offset + limit if has_next else None

        return HistoryParentsResponse(parents=parents, has_next=has_next, next_offset=next_offset)

    except Exception as e:
        logger.error(f"Get generation parents error: {str(e)}", exc_info=True) 
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get generations parents"
        )
    
@router.get("/history/edit-groups", response_model=EditGroupsResponse)
async def get_edit_groups(
    thread_id: str = Query(..., description="Design thread ID to fetch edit groups for"),
    current_user: dict = Depends(get_current_user),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware), # <-- ADD THIS DEPENDENCY
):
    """
    Get edit groups for a specific design thread
    """
    try:
        response = get_edit_groups_for_thread(auth_middleware, thread_id, current_user["id"])

        if not response.data:
            return EditGroupsResponse(edit_groups=[])

        edit_groups = [EditGroup.model_validate({
            "generation_id": item["id"],
            "prompt": item["prompt"],
            "prompt_type": item["prompt_type"],
            "created_at": item["created_at"],
            "images_json": item["images_json"]
        }) for item in response.data]

        return EditGroupsResponse(edit_groups=edit_groups)

    except Exception as e:
        logger.error(f"Get edit groups error: {str(e)}", exc_info=True) 
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get Variations"
        )
# In routes/users.py
@router.get("/history", response_model=List[GenerationResponse])
async def get_generation_history(
    current_user: dict = Depends(get_current_user),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware), # <-- ADD THIS DEPENDENCY
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0)
):
    """
    Get user generation history
    """
    try:
        # Now auth_middleware is guaranteed to be available
        response = auth_middleware.supabase.table("generations").select(
            "id, prompt, prompt_type, image_url, model_used, theme, size_preset, created_at, user_id, design_thread_id, parent_id"
        ).eq("user_id", current_user["id"]).order(
            "created_at", desc=True
        ).limit(limit).offset(offset).execute()

        if not response.data:
            return []

        # Use from_orm as it's the correct Pydantic V2 syntax
        return [GenerationResponse.from_orm(item) for item in response.data]

    except Exception as e:
        logger.error(f"Get generation history error: {str(e)}", exc_info=True) # Add exc_info for better logging
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


@router.get("/history/design", response_model=GenerationResponse)
async def get_single_generation(
    generation_id: str,
    current_user: dict = Depends(require_pro_plan),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware)
):
    """
    Get a single generation by its ID for the current user to be used for editing.
    """
    try:

        response = get_single_generation_for_user(auth_middleware, generation_id, current_user["id"])

        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Generation not found or you do not have permission to view it."
            )

        pydantic_record = GenerationResponse.from_orm(response.data)
        response_content = jsonable_encoder(pydantic_record)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_content
        )

    except Exception as e:
        logger.error(f"Error fetching single generation {generation_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch generation details."
        )
    

@router.get("/history/{generation_id}", response_model=GenerationResponse)
async def get_single_generation(
    generation_id: str,
    current_user: dict = Depends(get_current_user),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware)
):
    """
    Get a single generation by its ID for the current user.
    """
    try:
        response = auth_middleware.supabase.table("generations").select(
            "*" # Select all columns
        ).eq("id", generation_id).eq("user_id", current_user["id"]).single().execute()

        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Generation not found or you do not have permission to view it."
            )

        return GenerationResponse.from_orm(response.data)

    except Exception as e:
        logger.error(f"Error fetching single generation {generation_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch generation details."
        )
    

@retry_on_ssLError
def get_variations_count_aggregated(auth_middleware, design_thread_ids):
    string_thread_ids = [str(uuid_obj) for uuid_obj in design_thread_ids]
    counts_response = auth_middleware.supabase.rpc(
        "get_generation_counts_for_threads", 
        {"thread_ids": string_thread_ids}
    ).execute()
    if not counts_response.data:
        logger.warning(f"No counts found for thread IDs: {string_thread_ids}")
        return {}

        # Map design thread ids to their total designs count
    variations_count = {item["design_thread_id"]: item["variation_count"] for item in counts_response.data}
    return variations_count

@retry_on_ssLError
def get_user_parent_generations(current_user, auth_middleware, limit, offset):
    response = auth_middleware.supabase.table("generations").select(
            "id, prompt, image_url, created_at, design_thread_id, used_brand_kit"
        ).eq("user_id", current_user["id"]).is_("parent_id", None).order(
            "created_at", desc=True
        ).limit(limit).offset(offset).execute()

    return response
@retry_on_ssLError
def get_edit_groups_for_thread(auth_middleware, thread_id, user_id):
    response = auth_middleware.supabase.table("generations").select(
            "id, prompt, prompt_type, created_at, images_json"
        ).eq("design_thread_id", thread_id).eq("user_id", user_id).order(
            "created_at"
        ).execute()

    return response

@retry_on_ssLError
def get_single_generation_for_user(auth_middleware, generation_id, user_id):
    response = auth_middleware.supabase.table("generations").select(
            "*"
        ).eq("id", generation_id).eq("user_id", user_id).single().execute()

    return response