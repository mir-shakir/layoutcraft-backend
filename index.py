import os
import asyncio
import time
import logging
import random
from xmlrpc import client
from prompts.design_prompts import DESIGN_PROMPTS,HTML_EDIT_PROMPT, SIZE_PRESET_CONTEXT
from config.dimension_presets import (
    get_preset_dimensions, 
    get_default_preset, 
    create_dynamic_prompt_context,
    validate_preset_name,
    get_multiple_presets_with_context,
    DEFAULT_PRESET
)
from typing import Dict, List, Optional, Set,Any
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Request, Depends,status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field , EmailStr
from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions
from playwright.async_api import async_playwright
import io

from routes.auth import router as auth_router
from routes.users import router as users_router
from routes.paddle import router as paddle_router
from routes.dodo import router as dodo_router
# from routes.billing import router as billing_router
# from routes.advanced_generation import router as advanced_router

from auth.dependencies import require_pro_plan, get_current_user, check_usage_limits
from auth.middleware import get_auth_middleware
from models.generation import GenerationCreate, GenerationResponse

from services.user_service import UserService
from services.generation_service import GenerationService
from auth.middleware import get_auth_middleware
from services.premium_service import PremiumService
from services.export_service import ExportService
import httpx
from fastapi.encoders import jsonable_encoder
# Add these at the top of index.py
import uuid
from fastapi.responses import JSONResponse 
from collections import defaultdict
from config.tier_config import get_tier_features # Add this import
from config.decorators import retry_on_ssLError


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('layoutcraft.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LayoutCraft Backend",
    description="AI-powered visual asset generator using LLM -> HTML -> Image workflow",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-generations-remaining", "content-disposition"]
)

# Include routers
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(paddle_router)
app.include_router(dodo_router)
# app.include_router(billing_router)
# app.include_router(advanced_router)


# Configuration constants
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # Default to 2.5-flash
PRO_GEMINI_MODEL = os.getenv("PRO_GEMINI_MODEL", "gemini-2.5-pro")  # Default to 2.5-pro
GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT", "180"))  # 2 minutes default
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))  # requests per minute
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
ANONYMOUS_USER_ID = os.getenv("ANONYMOUS_USER_ID","0e699219-d5ae-4179-966f-f30b24d82e8a")  # Fallback anonymous user ID

# MVP Rate limiting configuration
MVP_RATE_LIMIT_REQUESTS = int(os.getenv("MVP_RATE_LIMIT_REQUESTS", "5"))  # 5 requests per minute for MVP
MVP_RATE_LIMIT_WINDOW = int(os.getenv("MVP_RATE_LIMIT_WINDOW", "60"))  # seconds

anonymous_usage_tracker: Dict[str, int] = defaultdict(int)


# Available models for easy switching
AVAILABLE_MODELS = {
    "gemini-2.5-flash": "gemini-2.5-flash",  # Add when available
    "gemini-2.5-pro": "gemini-2.5-pro"      # Add when available
}

# Prompt selection configuration
PROMPT_SELECTION_MODE = os.getenv("PROMPT_SELECTION_MODE", "random")  # "random" or "sequential"
prompt_index_counter = 0  # For sequential selection

# Simple in-memory rate limiter storage
rate_limiter_storage: Dict[str, list] = defaultdict(list)
mvp_rate_limiter_storage: Dict[str, list] = defaultdict(list)  # Separate rate limiter for MVP
# A set is used for O(1) lookups, which is extremely fast.
pro_usage_tracker: Set[str] = set()

# Request model
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="User's prompt for image generation")
    width: int = Field(default=1200, ge=100, le=3000, description="Image width in pixels")
    height: int = Field(default=630, ge=100, le=3000, description="Image height in pixels")
    model: Optional[str] = Field(default=None, description="Override default model for this request")

# MVP Request model (simplified)
class MVPGenerationRequest(BaseModel):
    prompt: str = Field(..., description="User's prompt for image generation")
    model:str = Field(default=None, description="Override default model for this request (MVP)")
    theme:str = Field(default="auto", description="Theme for the image generation (MVP)")
    size_preset: Optional[str] = Field(default=None, description="Predefined size preset (e.g., 'blog_header', 'social_square')")

class GenerateMultipleRequest(BaseModel):
    prompt: str = Field(..., description="User's prompt for image generation")
    theme:str = Field(default="auto", description="Theme for the image generation (MVP)")
    size_presets : List[str] = Field(default=["blog_header"], description="List of predefined size presets")

class RefineDesignRequest(BaseModel):
    generation_id: str = Field(..., description="ID of the generation to refine")
    edit_prompt: str = Field(..., description="User's instruction for what to change")
    size_presets: List[str] = Field(default=["blog_header"], description="List of predefined size presets")

class FeedbackData(BaseModel):
    email: EmailStr
    message: str
    source: str
    user_agent: str
    timestamp: str

class EditRequest(BaseModel):
    edit_prompt: str = Field(..., description="The user's instruction for what to change.")

# Initialize Gemini AI
def initialize_gemini(model_name: str = None):
    """Initialize Gemini AI with API key and specified model."""
    logger.debug("model_name parameter: " + str(model_name))
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    # Use provided model or fall back to configured default
    selected_model = model_name or GEMINI_MODEL
    
    # Validate model exists in our available models
    if selected_model not in AVAILABLE_MODELS.values():
        logger.warning(f"Model {selected_model} not in predefined list, attempting to use anyway")
    client = genai.Client()
    return client, selected_model
    
# Rate limiting dependency
async def check_rate_limit(request: Request):
    """Simple IP-based rate limiting."""
    client_ip = request.client.host
    current_time = datetime.now()
    
    logger.debug(f"Rate limit check for IP: {client_ip}")
    
    # Clean old requests outside the window
    rate_limiter_storage[client_ip] = [
        req_time for req_time in rate_limiter_storage[client_ip]
        if current_time - req_time < timedelta(seconds=RATE_LIMIT_WINDOW)
    ]
    
    # Check if rate limit exceeded
    if len(rate_limiter_storage[client_ip]) >= RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per minute.",
            headers={"Retry-After": "60"}
        )
    
    # Add current request
    rate_limiter_storage[client_ip].append(current_time)
    logger.debug(f"Current request count for {client_ip}: {len(rate_limiter_storage[client_ip])}")

# MVP Rate limiting dependency
async def check_mvp_rate_limit(request: Request):
    """Simple IP-based rate limiting for MVP endpoints."""
    client_ip = request.client.host
    current_time = datetime.now()
    
    logger.debug(f"MVP Rate limit check for IP: {client_ip}")
    
    # Clean old requests outside the window
    mvp_rate_limiter_storage[client_ip] = [
        req_time for req_time in mvp_rate_limiter_storage[client_ip]
        if current_time - req_time < timedelta(seconds=MVP_RATE_LIMIT_WINDOW)
    ]
    
    # Check if rate limit exceeded
    if len(mvp_rate_limiter_storage[client_ip]) >= MVP_RATE_LIMIT_REQUESTS:
        logger.warning(f"MVP Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {MVP_RATE_LIMIT_REQUESTS} requests per minute.",
            headers={"Retry-After": "60"}
        )
    
    # Add current request
    mvp_rate_limiter_storage[client_ip].append(current_time)
    logger.debug(f"Current MVP request count for {client_ip}: {len(mvp_rate_limiter_storage[client_ip])}")

def get_next_prompt_template() -> dict:
    """Get the next prompt template based on selection mode."""
    global prompt_index_counter
    
    if PROMPT_SELECTION_MODE == "sequential":
        template = DESIGN_PROMPTS[prompt_index_counter % len(DESIGN_PROMPTS)]
        prompt_index_counter += 1
        logger.debug(f"Selected sequential prompt: {template['name']} (index: {prompt_index_counter-1})")
    else:  # random
        template = random.choice(DESIGN_PROMPTS)
        logger.debug(f"Selected random prompt: {template['name']}")
    
    return template
def get_design_prompt_template(theme: str) -> dict:
    """Get a design prompt template based on the specified theme."""
    if theme == "auto":
        return random.choice(DESIGN_PROMPTS)
    # Get theme using name
    for template in DESIGN_PROMPTS:
        if template["name"].lower() == theme.lower():
            logger.debug(f"Selected design prompt for theme '{theme}': {template['name']}")
            return template
    # If no match found, return a default or raise an error
    logger.warning(f"No design prompt found for theme '{theme}', using default")
    return get_next_prompt_template()  # Fallback to random template if not found

def create_refine_prompt(original_html: str, edit_prompt: str, size_preset_context: str) -> str:
    """Create the full edit prompt for refining existing HTML."""
    full_prompt = HTML_EDIT_PROMPT.format(
        original_html=original_html,
        edit_prompt=edit_prompt,
        size_preset_context=size_preset_context
    )
    return full_prompt
def create_full_generation_prompt(user_prompt: str, size_presets: List[str], theme: str) -> str:
    """
        This function is an adaptation of create_generation_prompt_with_template to do all the promt building in one place.
        include multiple size presets in the prmpt with dimet
    """
    full_preset_prompt = SIZE_PRESET_CONTEXT + get_multiple_presets_with_context(size_presets)
    prompt_template = get_design_prompt_template(theme)
    full_prompt = prompt_template["prompt"].format(user_prompt=user_prompt,preset_context=full_preset_prompt)
    return full_prompt
        

def create_generation_prompt_with_template(user_prompt: str, template: dict, preset_context: str = "", width: int = 1200, height: int = 630) -> str:
    """Create the system prompt using selected template and optional preset context."""
    base_prompt = template["prompt"].format(user_prompt=user_prompt)
    
    # Inject dimension information into the prompt
    dimension_context = f"**CRITICAL DIMENSION REQUIREMENT:**\n- The container must be exactly {width}x{height} pixels\n- Use width: {width}px; height: {height}px; in your CSS container\n\n"
    
    # Find insertion point for dimension context
    context_insertion_point = base_prompt.find("**User's Core Idea:**")
    if context_insertion_point != -1:
        enhanced_prompt = base_prompt[:context_insertion_point]
        
        # Add dimension context
        enhanced_prompt += dimension_context
        
        # Add preset context if available
        if preset_context:
            enhanced_prompt += f"**Design Context:**\n{preset_context}\n\n"
        
        # Add the user's core idea section
        enhanced_prompt += base_prompt[context_insertion_point:]
        
        return enhanced_prompt
    
    # Fallback: prepend dimension context if no insertion point found
    return dimension_context + base_prompt

async def generate_html_with_gemini(model_name:str, prompt: str, client_ip: str) -> str:
    """Generate HTML using Gemini AI."""
    logger.info(f"Starting HTML generation for IP: {client_ip}")
    
    try:
        vertex_client = genai.Client(vertexai=True)
        logger.info(f"Test:Using Gemini model: {model_name} for IP: {client_ip}")
        response = await asyncio.wait_for(
            asyncio.to_thread(
                vertex_client.models.generate_content,
                model=PRO_GEMINI_MODEL,
                contents=prompt
            ),
            timeout=GENERATION_TIMEOUT
        )
        if response and response.candidates:
            candidate = response.candidates[0]
            if candidate.content and hasattr(candidate.content, 'parts'):
                parts = candidate.content.parts
                if parts:
                    text_output = "".join(p.text for p in parts if hasattr(p, 'text'))
                    return text_output

        logger.error("No text parts found in Gemini response")
        raise HTTPException(status_code=500, detail="Empty response from model")


    except Exception as e:
        logger.error(f"Error in HTML generation for IP {client_ip}: {str(e)}: Trying to reinitialize model.")
    
    try:
        client, _ = initialize_gemini(PRO_GEMINI_MODEL)
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model=PRO_GEMINI_MODEL,
                contents=prompt
            ),
            timeout=GENERATION_TIMEOUT
        )
        logger.info(f"Retrying with Pro model: {PRO_GEMINI_MODEL} for IP: {client_ip}")
        logger.info(f"Raw Gemini response: {response}")
        if response and response.candidates:
            candidate = response.candidates[0]
            if candidate.content and hasattr(candidate.content, 'parts'):
                parts = candidate.content.parts
                if parts:
                    text_output = "".join(p.text for p in parts if hasattr(p, 'text'))
                    logger.info(f"Generated text: {text_output}")
                    return text_output

        logger.error("No text parts found in Gemini response")
        raise HTTPException(status_code=500, detail="Empty response from model")
    except Exception as e:
        logger.error(f"Attempt 2: Error in HTML generation for IP {client_ip}: {str(e)} trying again with flash model")
    try:
        # vertex_client = genai.Client(vertexai=True)
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model=GEMINI_MODEL,
                contents=prompt
            ),
            timeout=GENERATION_TIMEOUT
        )
        logger.info(f"Raw Gemini response: {response}")
        if response and response.candidates:
            candidate = response.candidates[0]
            if candidate.content and hasattr(candidate.content, 'parts'):
                parts = candidate.content.parts
                if parts:
                    text_output = "".join(p.text for p in parts if hasattr(p, 'text'))
                    logger.info(f"Generated text: {text_output}")
                    return text_output

        logger.error("No text parts found in Gemini response")
        raise HTTPException(status_code=500, detail="Empty response from model")

    except Exception as e:
        logger.error(f"Attempt 3: Error in HTML generation for IP {client_ip}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate the design. Please try again later."
        )
    

def clean_html_response(html_content: str) -> str:
    """Clean the HTML response by removing markdown fences and extra whitespace."""
    logger.debug("Cleaning HTML response")
    
    original_length = len(html_content)
    
    # Remove markdown code fences
    html_content = html_content.replace("``````", "")
    
    # Remove leading/trailing whitespace
    html_content = html_content.strip()
    
    # Ensure it starts with DOCTYPE
    if not html_content.startswith("<!DOCTYPE"):
        # Try to find the start of HTML
        start_idx = html_content.find("<!DOCTYPE")
        if start_idx != -1:
            html_content = html_content[start_idx:]
            logger.debug(f"Trimmed HTML content from index {start_idx}")
    
    cleaned_length = len(html_content)
    logger.debug(f"HTML cleaned: {original_length} -> {cleaned_length} characters")
    
    return html_content


    
async def get_multiple_images_from_html(
    html_content: str, 
    size_presets: List[str]
) -> List[Dict[str, Any]]:
    output_images = []
    
    try:
        async with async_playwright() as p:
            # Launch one browser instance for all operations
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Set a large enough viewport to ensure all elements are rendered
            await page.set_viewport_size({"width": 1920, "height": 1080}) 
            
            # Load the HTML content once
            await page.set_content(html_content, wait_until="networkidle")

            # Iterate through the requested presets and take a screenshot of each
            for preset_name in size_presets:
                try:
                    # Use the preset name as the class selector
                    container_selector = f".{preset_name}"
                    container = page.locator(container_selector).first
                    
                    # Wait for the element to be visible to ensure it has rendered
                    await container.wait_for(timeout=5000) 
                    
                    logger.info(f"Capturing screenshot for preset: '{preset_name}'")
                    
                    screenshot_bytes = await container.screenshot(type="png")
                    
                    output_images.append({
                        "size_preset": preset_name,
                        "image_bytes": screenshot_bytes
                    })
                except Exception as e:
                    logger.error(f"Failed to capture screenshot for preset '{preset_name}': {e}")
                    # Continue to the next preset even if one fails
                    continue

            await browser.close()
            # if no images were captured, log a warning and throw an error
            if not output_images:
                logger.error("No images were captured from the HTML content.")
                raise HTTPException(status_code=500, detail="Image Generation Failed")
            
            logger.info(f"Successfully captured {len(output_images)} images from HTML.")
            return output_images
            
    except Exception as e:
        logger.error(f"An error occurred during multi-image rendering: {e}", exc_info=True)
        # Re-raise the exception to be handled by the main API endpoint
        raise

async def render_html_to_image(html_content: str, width: int, height: int, client_ip: str) -> bytes:
    """Render HTML to PNG image using Playwright with a fallback mechanism."""
    start_time = time.time()
    logger.info(f"Starting HTML rendering to image ({width}x{height}) for IP: {client_ip}")
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            await page.set_viewport_size({"width": width, "height": height})
            await page.set_content(html_content, wait_until="networkidle")

            # --- START OF NEW ROBUST FIX ---
            screenshot_bytes = None
            
            # Try to locate the preferred element first, with a short timeout.
            try:
                container = page.locator('.container').first
                # Use a short timeout (e.g., 2 seconds) to avoid long waits
                await container.wait_for(timeout=2000) 
                
                logger.info(f"Found '.container' element. Taking element screenshot.")
                screenshot_bytes = await container.screenshot(type="png")

            except Exception as e:
                # This will catch timeout errors or if the element is not found.
                logger.warning(f"Could not find '.container' element, falling back to full-page screenshot. Reason: {e}")
                
                # Fallback to the original full-page screenshot method.
                # This might have the scrollbar issue, but it's better than failing.
                screenshot_bytes = await page.screenshot(
                    type="png",
                    clip={"x": 0, "y": 0, "width": width, "height": height}
                )
            # --- END OF NEW ROBUST FIX ---
            
            await browser.close()
            
            render_time = time.time() - start_time
            image_size = len(screenshot_bytes)
            logger.info(f"Image rendered in {render_time:.2f}s, size: {image_size} bytes for IP: {client_ip}")
            
            return screenshot_bytes
            
    except Exception as e:
        logger.error(f"Error in HTML rendering for IP {client_ip}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate design. Please try again"
        )

def resolve_dimensions_from_preset(request: MVPGenerationRequest) -> tuple[int, int, str]:
    """
    Resolve dimensions from size_preset if provided, otherwise use default preset.
    
    Args:
        request: The MVP generation request
        
    Returns:
        Tuple of (width, height, preset_context)
    """
    if request.size_preset and validate_preset_name(request.size_preset):
        # Use preset dimensions
        preset_data = get_preset_dimensions(request.size_preset)
        width = preset_data["width"]
        height = preset_data["height"]
        preset_context = create_dynamic_prompt_context(request.size_preset)
        logger.info(f"Using preset '{request.size_preset}': {width}x{height}")
        return width, height, preset_context
    elif request.size_preset:
        # Invalid preset name provided, log warning and fall back to default
        logger.warning(f"Invalid preset name '{request.size_preset}', using default preset '{DEFAULT_PRESET}'")
        preset_data = get_default_preset()
        width = preset_data["width"]
        height = preset_data["height"]
        preset_context = create_dynamic_prompt_context(DEFAULT_PRESET)
        return width, height, preset_context
    else:
        # No preset specified, use default preset
        preset_data = get_default_preset()
        width = preset_data["width"]
        height = preset_data["height"]
        preset_context = create_dynamic_prompt_context(DEFAULT_PRESET)
        logger.info(f"Using default preset '{DEFAULT_PRESET}': {width}x{height}")
        return width, height, preset_context

def get_model_for_request(request: GenerationRequest):
    """Get the appropriate model for the request."""
    if request.model:
        # User specified a model override
        if request.model in AVAILABLE_MODELS:
            model_name = AVAILABLE_MODELS[request.model]
            logger.info(f"Using user-specified model: {model_name}")
        else:
            # Try to use the model name directly
            model_name = request.model
            logger.info(f"Using custom model name: {model_name}")
        
        return initialize_gemini(model_name)
    else:
        # Use default model
        return gemini_model
    
async def upload_image(image_data : Dict[str, Any], current_user_id: str, storage_client) -> Optional[Dict[str, str]]:
    """Helper function to upload a single image and return its data."""
    image_bytes = image_data.get("image_bytes")
    preset_name = image_data.get("size_preset")

    if not image_bytes or not preset_name:
        return None



    # Upload the image
    try:
        # Upload the image
        image_url = upload_image_and_get_url(current_user_id, storage_client, image_bytes, preset_name)

        logger.info(f"Successfully uploaded image for preset: {preset_name}")
        return {
            "size_preset": preset_name,
            "image_url": image_url
        }
    except Exception as e:
        logger.error(f"Failed to upload image for preset '{preset_name}'. Reason: {e}", exc_info=True)
        return None
    
@retry_on_ssLError
def upload_image_and_get_url(current_user_id, storage_client, image_bytes, preset_name):
    file_path = f"{current_user_id}/{uuid.uuid4()}_{preset_name}.png"
    storage_client.upload(
            file=image_bytes,
            path=file_path,
            file_options={"content-type": "image/png"}
        )

        # Get the public URL
    image_url = storage_client.get_public_url(file_path)
    return image_url


# Initialize Gemini model at startup
gemini_model = None
pro_gemini_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global gemini_model
    global pro_gemini_model
    try:
        gemini_client = genai.Client(http_options=HttpOptions(api_version="v1"))
        gemini_model_name = GEMINI_MODEL
        pro_gemini_model_name = PRO_GEMINI_MODEL
        logger.info("✅ LayoutCraft Backend started successfully")
        logger.info("✅ LayoutCraft Backend started successfully")
        logger.info(f"✅ Default model: {gemini_model_name}")
        logger.info(f"✅ Pro model: {pro_gemini_model_name}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LayoutCraft Backend: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check requested")
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "model": GEMINI_MODEL,
        "available_models": list(AVAILABLE_MODELS.keys())
    }

@app.get("/api/presets")
async def get_dimension_presets():
    """Get available dimension presets for the frontend."""
    from config.dimension_presets import get_preset_info_for_frontend, DEFAULT_PRESET
    
    preset_info = get_preset_info_for_frontend()
    
    return {
        "presets": preset_info,
        "default_preset": DEFAULT_PRESET,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/generate", response_model=GenerationResponse)
async def generate_image_anonymous(
    request: GenerateMultipleRequest,
    http_request: Request
):
    """
    Generates a visual asset for an anonymous user with a strict 3-generation limit.
    """
    start_time = time.time()
    client_ip = http_request.client.host
    request_id = f"anon_{client_ip}_{int(time.time())}"
    logger.info(f"[Anonymous Generation][{request_id}] Request from IP: {client_ip}")

    # --- 1. Enforce Anonymous User Limits ---
    anon_features = get_tier_features("anonymous")
    current_usage = anonymous_usage_tracker[client_ip]

    if current_usage >= anon_features["total_generations_limit"]:
        logger.warning(f"Anonymous limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Please sign up to unlock higher quotas and more features.",
        )

    try:
        model_to_use = PRO_GEMINI_MODEL
        full_prompt = create_full_generation_prompt(request.prompt,request.size_presets,request.theme)
        html_content = await generate_html_with_gemini(model_to_use, full_prompt, client_ip)
        cleaned_html = clean_html_response(html_content)
        output_images = await get_multiple_images_from_html(cleaned_html,request.size_presets)

        auth_middleware = get_auth_middleware()
        storage_client = auth_middleware.supabase.storage.from_("generations")
        upload_tasks = [upload_image(img, ANONYMOUS_USER_ID, storage_client) for img in output_images]
        results = await asyncio.gather(*upload_tasks)
        outputs_for_db = [res for res in results if res is not None]
        if not outputs_for_db:
            logger.error(f"[{request_id}] Failed to upload images for anonymous user.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="We are unable to process your request at this time. Please try again later."
            )

        generation_time = int((time.time() - start_time) * 1000)
        #  set first image from json as image_url to be used as thumbnail in frontend
        image_url = outputs_for_db[0]["image_url"] if outputs_for_db else None
        generation_data = GenerationCreate(
            user_id=ANONYMOUS_USER_ID,
            design_thread_id=uuid.uuid4(),  # New thread for a new generation
            parent_id=None,                 # No parent for a new generation
            prompt=request.prompt,
            prompt_type='creation',
            generated_html=cleaned_html,
            model_used=model_to_use,
            theme=request.theme,
            generation_time_ms=generation_time,
            images_json=outputs_for_db,
            image_url=image_url
        )
        generation_service = GenerationService(auth_middleware.supabase)
        new_generation_record = await generation_service.create_generation(generation_data)

        # --- 3. Increment Usage ---
        anonymous_usage_tracker[client_ip] += 1
        remaining = anon_features["total_generations_limit"] - anonymous_usage_tracker[client_ip]
        logger.info(f"Anonymous usage for {client_ip} is now {anonymous_usage_tracker[client_ip]}. {remaining} remaining.")

        pydantic_record = GenerationResponse.from_orm(new_generation_record)
        response_content = jsonable_encoder(pydantic_record)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=response_content,
            headers={"x-generations-remaining": str(remaining)}
        )

    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error for anonymous user: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during image generation."
        )


#  =======================New Authenticated Endpoints ============================


@app.post("/api/v1/generate", response_model=GenerationResponse)
async def generate_image_authenticated( # Renamed for clarity
    request: GenerateMultipleRequest,
    http_request: Request,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(check_mvp_rate_limit)
):
    """
    Generates a visual asset for an authenticated user, saves it to storage and the database,
    and returns the full generation record.
    """
   
    client_ip = http_request.client.host
    request_id = f"{current_user['id']}_{int(time.time())}"
    logger.info(f"[Authenticated Generation][{request_id}] Request from user: {current_user['email']}")
    start_time = time.time()

    try:
        user_profile = current_user
        if not user_profile:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found.")
        
        if user_profile.get("subscription_tier") not in ["pro", "pro-trial"]:
            if len(request.size_presets) > 1:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Multi-dimension generation is a Pro feature. Please upgrade to continue."
                )
        model_to_use = PRO_GEMINI_MODEL 

        full_prompt = create_full_generation_prompt(request.prompt,request.size_presets,request.theme)
        logger.info(f"full prommpt : {full_prompt}")
        html_content = await generate_html_with_gemini(model_to_use, full_prompt, client_ip)
        cleaned_html = clean_html_response(html_content)
        output_images = await get_multiple_images_from_html(cleaned_html,request.size_presets)


        auth_middleware = get_auth_middleware()
        storage_client = auth_middleware.supabase.storage.from_("generations")
        upload_tasks = [upload_image(img, current_user['id'], storage_client) for img in output_images]
        results = await asyncio.gather(*upload_tasks)
        outputs_for_db = [res for res in results if res is not None]

        # Save Generation Record to Database ---
        generation_service = GenerationService(auth_middleware.supabase)
        generation_time = int((time.time() - start_time) * 1000)
        image_url = outputs_for_db[0]["image_url"] if outputs_for_db else None
        generation_data = GenerationCreate(
            user_id=current_user["id"],
            design_thread_id=uuid.uuid4(),  # New thread for a new generation
            parent_id=None,                 # No parent for a new generation
            prompt=request.prompt,
            prompt_type='creation',
            generated_html=cleaned_html,
            model_used=model_to_use,
            theme=request.theme,
            generation_time_ms=generation_time,
            images_json=outputs_for_db,
            image_url=image_url
        )

        new_generation_record = await generation_service.create_generation(generation_data)
        if not new_generation_record:
            raise HTTPException(status_code=500, detail="Failed to save generation record.")

        logger.info(f"[{request_id}] Generation saved with ID: {new_generation_record['id']}")

        pydantic_record = GenerationResponse.from_orm(new_generation_record)
        response_content = jsonable_encoder(pydantic_record)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=response_content
        )

    except HTTPException as e:
        logger.error(f"[{request_id}] HTTP error: {e.status_code} - {e.detail}")
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during image generation. Try again later."
        )


@app.post("/api/feedback")
async def submit_feedback(data: FeedbackData, _: None = Depends(check_mvp_rate_limit)):
    """
    Receives feedback from the frontend and securely forwards it to a Google Sheet
    via a Google Apps Script Web App.
    """
    google_sheet_url = os.getenv("GOOGLE_SHEET_WEB_APP_URL")

    if not google_sheet_url:
        logger.error("GOOGLE_SHEET_WEB_APP_URL environment variable is not set.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server is not configured to handle feedback."
        )

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            # Forward the validated data to your Google Apps Script
            response = await client.post(google_sheet_url, json=data.model_dump())

            # Check if the request to Google Sheets was successful
            response.raise_for_status()

        logger.info(f"Feedback successfully submitted to Google Sheets from {data.email}")
        return {"status": "success", "message": "Feedback submitted successfully."}

    except httpx.RequestError as e:
        logger.error(f"Could not connect to Google Sheets webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to submit feedback due to a server-side connection issue."
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during feedback submission: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred."
        )
 


@app.post("/api/refine", response_model=GenerationResponse)
async def refine_design(
    request: RefineDesignRequest,
    http_request: Request,
    current_user: dict = Depends(require_pro_plan),
    _: None = Depends(check_mvp_rate_limit)
):
    # Implementation for refining the design goes here
    client_ip = http_request.client.host
    start_time = time.time()
    try:
        if not current_user:
            raise HTTPException(status_code=404, detail="User profile not found.")
        auth_middleware = get_auth_middleware()
        generation_service = GenerationService(auth_middleware.supabase)
        user_service = UserService(auth_middleware.supabase)
        generation_id = request.generation_id
        parent_generation = await generation_service.get_generation_by_id(generation_id, current_user["id"])
        if not parent_generation:
            raise HTTPException(status_code=404, detail="Generation not found. Start a new design instead.")
        size_presets = request.size_presets
        if not size_presets or len(size_presets) == 0:
            raise HTTPException(status_code=400, detail="At least one size preset must be specified for the edit.")
        parent_generation_html = parent_generation['generated_html']
        edit_prompt = request.edit_prompt
        # check if the size presets are present in parent generation images_json
        parent_images_json = parent_generation.get('images_json', [])
        parent_presets = {img['size_preset'] for img in parent_images_json if 'size_preset' in img}
        for preset in size_presets:
            if preset not in parent_presets:
                raise HTTPException(status_code=400, detail=f"Size preset '{preset}' not found in the original generation.")
        full_edit_prompt = create_refine_prompt(parent_generation_html, edit_prompt, size_presets)
        model_to_use = PRO_GEMINI_MODEL

        modified_html_content = await generate_html_with_gemini(model_to_use, full_edit_prompt, client_ip)
        cleaned_modified_html = clean_html_response(modified_html_content)

        output_images = await get_multiple_images_from_html(cleaned_modified_html, size_presets)
        
        storage_client = auth_middleware.supabase.storage.from_("generations")
        upload_tasks = [upload_image(img, current_user['id'], storage_client) for img in output_images]
        results = await asyncio.gather(*upload_tasks)
        outputs_for_db = [res for res in results if res is not None]
        if not outputs_for_db:
            logger.error(f"Failed to upload refined images for user {current_user['id']}.")
            raise HTTPException(
                status_code=500,
                detail="We are unable to process your request at this time. Please try again later."
            )
        # add existing images from parent generation that are not in the current edit
        for img in parent_images_json:
            if img['size_preset'] not in size_presets:
                outputs_for_db.append(img)
        generation_time = int((time.time() - start_time) * 1000)
        image_url = outputs_for_db[0]["image_url"] if outputs_for_db else None
        generation_data = GenerationCreate(
            user_id=current_user["id"],
            design_thread_id=parent_generation['design_thread_id'],
            parent_id=parent_generation['id'],
            prompt=edit_prompt,
            prompt_type='refine',
            generated_html=cleaned_modified_html,
            model_used=model_to_use,
            theme=parent_generation['theme'],
            size_preset=None,  # Not applicable for refine
            generation_time_ms=generation_time,
            images_json=outputs_for_db,
            image_url=image_url
        )
        new_generation_record = await generation_service.create_generation(generation_data)
        if not new_generation_record:
            raise HTTPException(status_code=500, detail="Failed to save refined generation record.")
        logger.info(f"Refinement saved with ID: {new_generation_record['id']}")
        pydantic_record = GenerationResponse.from_orm(new_generation_record)
        response_content = jsonable_encoder(pydantic_record)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=response_content
        )

    except Exception as e:
        logger.error(f"Error in refine_design: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during design refinement.")

@app.post("/users/history/{generation_id}/edit", response_model=GenerationResponse)
async def edit_generation(
    generation_id: str,
    request: EditRequest,
    http_request: Request,
    current_user: dict = Depends(get_current_user)
):
    """
    Edits an existing generation by modifying its HTML blueprint and creating a new version.
    """
    client_ip = http_request.client.host
    request_id = f"{current_user['id']}_{int(time.time())}_edit"
    logger.info(f"[{request_id}] Edit request for generation {generation_id}")
    start_time = time.time()

    auth_middleware = get_auth_middleware()
    generation_service = GenerationService(auth_middleware.supabase)

    try:

        if not current_user:
            raise HTTPException(status_code=404, detail="User profile not found.")
        
        # user_tier = current_user.get("subscription_tier", "free")
        # pro_usage_count = current_user.get("pro_usage_count", 0)
        # edit_usage_count = current_user.get("edit_usage_count", 0)
        # tier_features = get_tier_features(user_tier)

        # if user_tier == "free" and tier_features.get("monthly_edit_limit", 0) <= edit_usage_count:
        #     raise HTTPException(
        #         status_code=403, # Forbidden
        #         detail="Upgrade to Pro to unlock unlimited editing."
        #     )
        # if user_tier == "pro" and pro_usage_count >= tier_features.get("monthly_pro_generations_limit", 0):
        #     raise HTTPException(
        #         status_code=403, # Forbidden
        #         detail="You have reached your monthly Pro usage limit."
        #     )
        # 1. Fetch the original/parent generation record
        parent_generation = await generation_service.get_generation_by_id(generation_id, current_user["id"])
        if not parent_generation:
            raise HTTPException(status_code=404, detail="Generation not found. Start a new design instead.")
        
        preset_data = get_preset_dimensions(parent_generation['size_preset'])
        parent_width = preset_data['width']
        parent_height = preset_data['height']

        # 2. Create the specialized "edit" prompt
        edit_prompt_for_llm = HTML_EDIT_PROMPT.format(
            original_html=parent_generation['generated_html'],
            edit_prompt=request.edit_prompt
        )

        # 3. Call the LLM to get the MODIFIED HTML
        model_to_use = PRO_GEMINI_MODEL if 'pro' in parent_generation['model_used'] else GEMINI_MODEL
        modified_html_content = await generate_html_with_gemini(model_to_use, edit_prompt_for_llm, client_ip)
        cleaned_modified_html = clean_html_response(modified_html_content)

        # 4. Render the new HTML to a new image
        modified_image_bytes = await render_html_to_image(
            cleaned_modified_html, parent_width, parent_height, client_ip
        )

        # 5. Upload the new image to Storage
        new_image_url = upload_and_get_url(current_user, auth_middleware, modified_image_bytes)

        # 6. Save a NEW entry to the generations table
        generation_time = int((time.time() - start_time) * 1000)
        new_generation_data = GenerationCreate(
            user_id=current_user["id"],
            design_thread_id=parent_generation['design_thread_id'],
            parent_id=parent_generation['id'],
            prompt=request.edit_prompt,
            prompt_type='edit',
            generated_html=cleaned_modified_html,
            image_url=new_image_url,
            model_used=parent_generation['model_used'],
            theme=parent_generation['theme'],
            size_preset=parent_generation['size_preset'],
            generation_time_ms=generation_time
        )

        new_record = await generation_service.create_generation(new_generation_data)
        if not new_record:
            logger.error(f"[{request_id}] Failed to save edited generation record.")
        else:
            logger.info(f"[{request_id}] Edit successful. New generation ID: {new_record['id']}")
        # 6. Update user's usage counts
        # await user_service.update_usage(
        #     current_user,
        #     usage_increment=1,  # Increment normal usage count
        #     pro_increment=1 if 'pro' in parent_generation['model_used'] else 0,  # Increment Pro usage if applicable
        #     edit_increment=1  # Increment edit usage count
        # )


        # 7. Return the new generation record
        pydantic_record = GenerationResponse.from_orm(new_record)
        return jsonable_encoder(pydantic_record)

    except HTTPException as e:
        logger.error(f"[{request_id}] HTTP error during edit: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error during edit: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during the edit.")

@retry_on_ssLError
def upload_and_get_url(current_user, auth_middleware, modified_image_bytes):
    file_path = f"{current_user['id']}/{uuid.uuid4()}.png"
    auth_middleware.supabase.storage.from_("generations").upload(
            file=modified_image_bytes,
            path=file_path,
            file_options={"content-type": "image/png"}
        )
    new_image_url = auth_middleware.supabase.storage.from_("generations").get_public_url(file_path)
    return new_image_url


# ==================== EXISTING PREMIUM ENDPOINTS (UNCHANGED) ====================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "LayoutCraft Backend API",
        "version": "2.0.0",
        "model": GEMINI_MODEL,
        "available_models": list(AVAILABLE_MODELS.keys()),
        "endpoints": {
            "generate": "/api/generate (MVP - No Auth)",
            "generate_premium": "/api/generate-premium (Full Features)",
            "health": "/health", 
            "auth": "/auth",
            "users": "/users",
            "billing": "/billing",
            "premium": "/premium",
            "advanced": "/advanced"
        },
        "mvp_info": {
            "description": "MVP endpoint available at /api/generate without authentication",
            "rate_limit": f"{MVP_RATE_LIMIT_REQUESTS} requests per minute",
            "dimension_system": "preset-based",
            "default_preset": DEFAULT_PRESET,
            "supported_formats": ["png"],
            "design_templates": [template["name"] for template in DESIGN_PROMPTS],
            "prompt_selection_mode": PROMPT_SELECTION_MODE,
            "presets_endpoint": "/api/presets"
        }
    }

@app.get("/users/analytics")
async def get_user_analytics(current_user: dict = Depends(get_current_user)):
    """
    Get comprehensive user analytics
    """
    try:
        auth_middleware = get_auth_middleware()
        user_service = UserService(auth_middleware.supabase)
        generation_service = GenerationService(auth_middleware.supabase)
        
        # Get user statistics
        user_stats = await user_service.get_user_statistics(current_user["id"])
        generation_stats = await generation_service.get_generation_statistics(current_user["id"])
        
        return {
            "user_stats": user_stats,
            "generation_stats": generation_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting user analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get user analytics"
        )

@app.post("/premium/templates")
async def create_custom_template(
    template_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Create custom template (premium feature)"""
    try:
        auth_middleware = get_auth_middleware()
        premium_service = PremiumService(auth_middleware.supabase)
        
        template_id = await premium_service.create_custom_template(current_user["id"], template_data)
        
        if not template_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Custom templates not available in your plan"
            )
        
        return {"template_id": template_id, "message": "Template created successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating custom template: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create custom template"
        )

@app.get("/premium/templates")
async def get_user_templates(current_user: dict = Depends(get_current_user)):
    """Get user's custom templates"""
    try:
        auth_middleware = get_auth_middleware()
        premium_service = PremiumService(auth_middleware.supabase)
        
        templates = await premium_service.get_user_templates(current_user["id"])
        
        return {"templates": templates}
        
    except Exception as e:
        logger.error(f"Error getting user templates: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get user templates"
        )

@app.post("/premium/brand-kit")
async def create_brand_kit(
    brand_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Create brand kit (premium feature)"""
    try:
        auth_middleware = get_auth_middleware()
        premium_service = PremiumService(auth_middleware.supabase)
        
        brand_kit_id = await premium_service.create_brand_kit(current_user["id"], brand_data)
        
        if not brand_kit_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Brand kits not available in your plan"
            )
        
        return {"brand_kit_id": brand_kit_id, "message": "Brand kit created successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating brand kit: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create brand kit"
        )

@app.get("/premium/brand-kit")
async def get_brand_kit(current_user: dict = Depends(get_current_user)):
    """Get user's brand kit"""
    try:
        auth_middleware = get_auth_middleware()
        premium_service = PremiumService(auth_middleware.supabase)
        
        brand_kit = await premium_service.get_brand_kit(current_user["id"])
        
        return {"brand_kit": brand_kit}
        
    except Exception as e:
        logger.error(f"Error getting brand kit: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get brand kit"
        )

# Add this entire function to index.py


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")