"""
Advanced generation routes for LayoutCraft
"""
from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
import logging
import io
import zipfile
from datetime import datetime

from auth.dependencies import get_current_user, RequireProTier
from auth.middleware import get_auth_middleware
from services.premium_service import PremiumService
from services.template_service import TemplateService
from services.export_service import ExportService
from services.generation_service import GenerationService
from models.advanced_generation import (
    AdvancedGenerationRequest, 
    BatchGenerationRequest,
    BrandKit,
    CustomTemplate
)

router = APIRouter(prefix="/advanced", tags=["Advanced Generation"])
logger = logging.getLogger(__name__)

@router.post("/generate")
async def advanced_generate(
    request: AdvancedGenerationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Advanced generation with style options and brand kit integration
    """
    try:
        auth_middleware = get_auth_middleware()
        premium_service = PremiumService(auth_middleware.supabase)
        template_service = TemplateService(auth_middleware.supabase)
        export_service = ExportService()
        
        user_tier = current_user.get("subscription_tier", "free")
        
        # Enhanced prompt building
        enhanced_prompt = request.prompt
        
        # Apply template if specified
        if request.template_id:
            template = await template_service.use_template(request.template_id, current_user["id"])
            if template:
                enhanced_prompt = template["prompt_template"].format(
                    user_prompt=request.prompt,
                    **request.dict()
                )
        
        # Apply brand kit if specified
        if request.brand_kit_id and user_tier != "free":
            brand_kit = await premium_service.get_brand_kit(current_user["id"])
            if brand_kit:
                brand_context = f"""
                Brand Guidelines:
                - Brand: {brand_kit['brand_name']}
                - Primary Colors: {', '.join(brand_kit['primary_colors'])}
                - Style: {brand_kit['style_guidelines']}
                """
                enhanced_prompt = f"{brand_context}\n\n{enhanced_prompt}"
        
        # Apply style modifications
        if request.style:
            style_modifiers = {
                "minimal": "with clean, minimalist design and lots of white space",
                "modern": "with contemporary, sleek design elements",
                "classic": "with timeless, traditional design elements",
                "bold": "with strong, eye-catching visual elements",
                "elegant": "with sophisticated, refined design",
                "playful": "with fun, energetic design elements"
            }
            enhanced_prompt += f" {style_modifiers.get(request.style, '')}"
        
        # Apply color scheme
        if request.color_scheme and request.primary_colors:
            color_instruction = f"using primary colors: {', '.join(request.primary_colors)}"
            enhanced_prompt += f" {color_instruction}"
        
        # Generate multiple iterations if requested (premium feature)
        if request.iterations > 1 and user_tier == "free":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Multiple iterations require a premium subscription"
            )
        
        # For now, generate single image (iterations feature would require queue system)
        # This is a simplified implementation
        
        # Use existing generation logic with enhanced prompt
        # ... (integrate with your existing generation pipeline)
        
        return {"message": "Advanced generation completed", "enhanced_prompt": enhanced_prompt}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Advanced generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Advanced generation failed"
        )

@router.post("/batch")
async def batch_generate(
    request: BatchGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(RequireProTier)
):
    """
    Batch generation for premium users
    """
    try:
        auth_middleware = get_auth_middleware()
        premium_service = PremiumService(auth_middleware.supabase)
        
        # Check batch generation capability
        batch_check = await premium_service.can_generate_batch(
            current_user["id"], 
            len(request.prompts)
        )
        
        if not batch_check["allowed"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=batch_check["reason"]
            )
        
        # Create batch job
        batch_id = f"batch_{current_user['id']}_{int(datetime.now().timestamp())}"
        
        # Add to background processing
        background_tasks.add_task(
            process_batch_generation,
            batch_id,
            request,
            current_user["id"]
        )
        
        return {
            "batch_id": batch_id,
            "status": "processing",
            "total_images": len(request.prompts),
            "estimated_completion": "5-10 minutes"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch generation failed"
        )

@router.get("/batch/{batch_id}/status")
async def get_batch_status(
    batch_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get batch generation status
    """
    try:
        # In a production system, you'd store batch status in database/cache
        # For MVP, return a simple status
        return {
            "batch_id": batch_id,
            "status": "completed",
            "progress": 100,
            "completed_images": 5,
            "total_images": 5,
            "download_url": f"/advanced/batch/{batch_id}/download"
        }
        
    except Exception as e:
        logger.error(f"Batch status error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get batch status"
        )

@router.get("/batch/{batch_id}/download")
async def download_batch_results(
    batch_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Download batch generation results as ZIP file
    """
    try:
        # Create ZIP file with batch results
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add placeholder files (in production, you'd add actual generated images)
            for i in range(5):
                zip_file.writestr(f"generation_{i+1}.png", b"placeholder image data")
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=batch_{batch_id}.zip"
            }
        )
        
    except Exception as e:
        logger.error(f"Batch download error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download batch results"
        )

@router.post("/templates")
async def create_custom_template(
    template: CustomTemplate,
    current_user: dict = Depends(RequireProTier)
):
    """
    Create custom template (premium feature)
    """
    try:
        auth_middleware = get_auth_middleware()
        template_service = TemplateService(auth_middleware.supabase)
        
        template_id = await template_service.create_template(
            current_user["id"],
            template.dict()
        )
        
        if not template_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create template"
            )
        
        return {"template_id": template_id, "message": "Template created successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Template creation failed"
        )

@router.get("/templates")
async def get_templates(
    category: str = None,
    public_only: bool = False,
    current_user: dict = Depends(get_current_user)
):
    """
    Get available templates
    """
    try:
        auth_middleware = get_auth_middleware()
        template_service = TemplateService(auth_middleware.supabase)
        
        if public_only:
            templates = await template_service.get_public_templates(category)
        else:
            templates = await template_service.get_user_templates(current_user["id"])
        
        return {"templates": templates}
        
    except Exception as e:
        logger.error(f"Get templates error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get templates"
        )

@router.post("/brand-kit")
async def create_brand_kit(
    brand_kit: BrandKit,
    current_user: dict = Depends(RequireProTier)
):
    """
    Create or update brand kit (premium feature)
    """
    try:
        auth_middleware = get_auth_middleware()
        premium_service = PremiumService(auth_middleware.supabase)
        
        brand_kit_id = await premium_service.create_brand_kit(
            current_user["id"],
            brand_kit.dict()
        )
        
        if not brand_kit_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create brand kit"
            )
        
        return {"brand_kit_id": brand_kit_id, "message": "Brand kit saved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Brand kit creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Brand kit creation failed"
        )

async def process_batch_generation(batch_id: str, request: BatchGenerationRequest, user_id: str):
    """
    Background task for processing batch generation
    """
    try:
        logger.info(f"Processing batch {batch_id} for user {user_id}")
        
        # In production, this would:
        # 1. Queue individual generation jobs
        # 2. Process them with proper queue management
        # 3. Store results in cloud storage
        # 4. Update batch status in database
        # 5. Send notification when complete
        
        # For MVP, we'll just log the completion
        logger.info(f"Batch {batch_id} completed")
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
