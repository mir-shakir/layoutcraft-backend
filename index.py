import os
import asyncio
import time
import logging
import random
from prompts.design_prompts import DESIGN_PROMPTS,HTML_EDIT_PROMPT
from config.dimension_presets import (
    get_preset_dimensions, 
    get_default_preset, 
    create_dynamic_prompt_context,
    validate_preset_name,
    DEFAULT_PRESET
)
from typing import Dict, List, Optional, Set
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Request, Depends,status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field , EmailStr
from dotenv import load_dotenv
import google.generativeai as genai
from playwright.async_api import async_playwright
import io

from routes.auth import router as auth_router
from routes.users import router as users_router
from routes.paddle import router as paddle_router
# from routes.billing import router as billing_router
# from routes.advanced_generation import router as advanced_router

from auth.dependencies import get_current_user, check_usage_limits
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
# app.include_router(billing_router)
# app.include_router(advanced_router)


# Configuration constants
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # Default to 2.5-flash
PRO_GEMINI_MODEL = os.getenv("PRO_GEMINI_MODEL", "gemini-2.5-pro")  # Default to 2.5-pro
GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT", "180"))  # 2 minutes default
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))  # requests per minute
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds

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
    
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info(f"Initializing Gemini with model: {selected_model}")
    return genai.GenerativeModel(selected_model)

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

def create_generation_prompt(user_prompt: str) -> str:
    """Create the system prompt for HTML generation."""
    return f"""You are a world-class digital designer and frontend developer. Your task is to create a single, visually stunning, self-contained HTML file that looks like a modern tech blog header, based on the user's core idea. Your entire response must be ONLY the raw HTML code, starting with `<!DOCTYPE html>` and ending with `</html>`.

**Your Design System & Creative Instructions:**
1.  **Aesthetic Goal:** The final image must feel modern, airy, and professional. The key design language is **"Glassmorphism"** with soft, layered backgrounds.
2.  **Background Construction:**
    * Do not use a solid color. Create a **subtle, multi-color gradient background**. Use multiple `radial-gradient` or `linear-gradient` with soft, transparent edges to create a gentle, out-of-focus feel.
    * On top of the gradient, add **faint, abstract geometric shapes or lines** with a very low opacity and a low `z-index` to create a sense of depth.
3.  **Foreground Element (The "Glass Card"):**
    * Place the main content (title, text) inside a central "card" element.
    * This card **must** have a semi-transparent white background (e.g., `background: rgba(255, 255, 255, 0.1);`).
    * It **must** use the `backdrop-filter: blur(20px);` property to create the frosted glass effect.
    * Give it a subtle, 1px white border with low opacity (e.g., `border: 1px solid rgba(255, 255, 255, 0.2);`).
4.  **Typography & Color:**
    * Use a modern, clean sans-serif font.
    * Create a strong visual hierarchy. The main title should be large and bold. Subtitles and tags should be smaller and lighter.
    * Choose a professional and harmonious color palette that complements the soft background.

**Strict Technical Compliance:**
- ALL styling must be in a single `<style>` tag.
- The container must be 1200x630 pixels unless specified otherwise by the user.
- No external files, no JavaScript, and no explanations or markdown in your response.

---

**User's Core Idea:**
"{user_prompt}"
---
Based on all the above, generate the complete HTML code."""

async def generate_html_with_gemini(model, prompt: str, client_ip: str) -> str:
    """Generate HTML using Gemini AI."""
    start_time = time.time()
    logger.info(f"Starting HTML generation for IP: {client_ip}")
    
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, prompt),
            timeout=GENERATION_TIMEOUT
        )
        
        generation_time = time.time() - start_time
        logger.info(f"HTML generation completed in {generation_time:.2f}s for IP: {client_ip}")
        response_length = len(response.text) if response.text else 0
        logger.debug(f"Generated HTML length: {response_length} characters")
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error in HTML generation for IP {client_ip}: {str(e)}: Trying to reinitialize model.")
    
    try:
        model = initialize_gemini(model.model_name)
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, prompt),
            timeout=GENERATION_TIMEOUT
        )

        generation_time = time.time() - start_time
        logger.info(f"HTML generation completed in {generation_time:.2f}s for IP: {client_ip}")
        response_length = len(response.text) if response.text else 0
        logger.debug(f"Generated HTML length: {response_length} characters")

        return response.text

    except Exception as e:
        logger.error(f"Attempt 2: Error in HTML generation for IP {client_ip}: {str(e)} trying again with flash model")
    try:
        model = initialize_gemini()  # Default to flash model
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, prompt),
            timeout=GENERATION_TIMEOUT
        )

        generation_time = time.time() - start_time
        logger.info(f"HTML generation completed in {generation_time:.2f}s for IP: {client_ip}")
        response_length = len(response.text) if response.text else 0
        logger.debug(f"Generated HTML length: {response_length} characters")

        return response.text
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

# async def render_html_to_image(html_content: str, width: int, height: int, client_ip: str) -> bytes:
#     """Render HTML to PNG image using Playwright."""
#     start_time = time.time()
#     logger.info(f"Starting HTML rendering to image ({width}x{height}) for IP: {client_ip}")
    
#     try:
#         async with async_playwright() as p:
#             browser = await p.chromium.launch(headless=True)
#             page = await browser.new_page()
            
#             # Set viewport size to match requested dimensions
#             await page.set_viewport_size({"width": width, "height": height})
            
#             # Set HTML content
#             await page.set_content(html_content, wait_until="networkidle")
            
#             # Take screenshot
#             screenshot_bytes = await page.screenshot(
#                 type="png",
#                 full_page=False,
#                 clip={"x": 0, "y": 0, "width": width, "height": height}
#             )
            
#             await browser.close()
            
#             render_time = time.time() - start_time
#             image_size = len(screenshot_bytes)
#             logger.info(f"Image rendered in {render_time:.2f}s, size: {image_size} bytes for IP: {client_ip}")
            
#             return screenshot_bytes
            
#     except Exception as e:
#         logger.error(f"Error in HTML rendering for IP {client_ip}: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error rendering image: {str(e)}"
#         )
# This is inside your ImageCreationService class in index.py
# This is inside your ImageCreationService class in index.py

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
            detail=f"Error rendering image: {str(e)}"
        )
async def generate_visual_asset(
    request: GenerationRequest,
    model,
    client_ip: str
) -> bytes:
    """Main logic for generating visual assets."""
    logger.info(f"Starting visual asset generation for IP: {client_ip}, prompt: '{request.prompt}...'")
    
    # Get random/sequential prompt template
    prompt_template = get_next_prompt_template()
    full_prompt = create_generation_prompt_with_template(request.prompt, prompt_template, "", request.width, request.height)
    logger.info(f"Using prompt template: {prompt_template['name']}")
    
    # Generate HTML with Gemini
    html_content = await generate_html_with_gemini(model, full_prompt, client_ip)
    
    # Clean HTML response
    cleaned_html = clean_html_response(html_content)
    
    # Render HTML to image
    image_bytes = await render_html_to_image(cleaned_html, request.width, request.height, client_ip)
    
    logger.info(f"Visual asset generation completed successfully for IP: {client_ip}")
    return image_bytes

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


# MVP version of generate_visual_asset (simplified)
async def generate_visual_asset_mvp(
    request: MVPGenerationRequest,
    model,
    client_ip: str,
) -> bytes:
    """Simplified logic for generating visual assets in MVP mode."""
    
    # Resolve dimensions from preset or use provided dimensions
    width, height, preset_context = resolve_dimensions_from_preset(request)
    logger.info(f"[MVP] Resolved dimensions: {width}x{height}, preset context: '{preset_context}'")
    
    # Get design prompt template based on theme
    prompt_template = get_design_prompt_template(request.theme)
    
    # Create full prompt with preset context and dimensions
    full_prompt = create_generation_prompt_with_template(request.prompt, prompt_template, preset_context, width, height)
    logger.info(f"[MVP] Using prompt template: {prompt_template['name']}, dimensions: {width}x{height}, prompt length: {len(full_prompt)} characters")  # Log first 100 chars for brevity

    # Generate HTML with Gemini
    html_content = await generate_html_with_gemini(model, full_prompt, client_ip)
    logger.debug(f"[MVP] Generated HTML content : {html_content} ")
    
    logger.info(f"[MVP] HTML generation completed for IP: {client_ip}, length: {len(html_content)} characters")
    # Clean HTML response
    cleaned_html = clean_html_response(html_content)
    
    # Render HTML to image using resolved dimensions
    image_bytes = await render_html_to_image(cleaned_html, width, height, client_ip)
    
    logger.info(f"[MVP] Visual asset generation completed successfully for IP: {client_ip}")
    return image_bytes

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

# Initialize Gemini model at startup
gemini_model = None
pro_gemini_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global gemini_model
    global pro_gemini_model
    try:
        gemini_model = initialize_gemini()
        pro_gemini_model = initialize_gemini(PRO_GEMINI_MODEL)
        logger.info("✅ LayoutCraft Backend started successfully")
        logger.info(f"✅ Default model: {GEMINI_MODEL}")
        logger.info(f"✅ Pro model: {PRO_GEMINI_MODEL}")
        logger.info(f"✅ Available models: {list(AVAILABLE_MODELS.keys())}")
        logger.info(f"✅ Rate limit: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s")
        logger.info(f"✅ MVP Rate limit: {MVP_RATE_LIMIT_REQUESTS} requests per {MVP_RATE_LIMIT_WINDOW}s")
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

# ==================== MVP ENDPOINTS ====================
# No authorization. 1 pro request per ip rest fast model requests

# Replace the existing /api/generate endpoint in index.py


@app.post("/api/generate")
async def generate_image_anonymous(
    request: MVPGenerationRequest,
    http_request: Request
):
    """
    Generates a visual asset for an anonymous user with a strict 3-generation limit.
    """
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
            detail="Please sign up to unlock more generations and Pro quality for free."
        )

    # if 'pro' in (request.model or '').lower():
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="Sign up to use Pro quality generations."
    #     )

    try:
        # --- 2. Generate Image and Save to Storage (consistent with v1 endpoint) ---
        if request.model and 'pro' in request.model.lower():
            model_to_use = pro_gemini_model
        else:
            model_to_use = gemini_model 
        # ... (The logic for generating html and rendering the image is the same) ...
        # ... (You will need to import the necessary functions like resolve_dimensions_from_preset, etc.)
        width, height, preset_context = resolve_dimensions_from_preset(request)
        prompt_template = get_design_prompt_template(request.theme)
        full_prompt = create_generation_prompt_with_template(
            request.prompt, prompt_template, preset_context, width, height
        )
        html_content = await generate_html_with_gemini(model_to_use, full_prompt, client_ip)
        cleaned_html = clean_html_response(html_content)
        image_bytes = await render_html_to_image(cleaned_html, width, height, client_ip)

        # --- 3. Increment Usage ---
        anonymous_usage_tracker[client_ip] += 1
        remaining = anon_features["total_generations_limit"] - anonymous_usage_tracker[client_ip]
        logger.info(f"Anonymous usage for {client_ip} is now {anonymous_usage_tracker[client_ip]}. {remaining} remaining.")

        # --- 4. Return Image Stream ---
        # NOTE: We don't save to the DB for anonymous users, we just stream the image.
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=generated-image.png",
                "x-generations-remaining": str(remaining) # Send remaining quota to UI
            }
        )

    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error for anonymous user: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during image generation."
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
 


#  =======================New Authenticated Endpoints ============================


@app.post("/api/v1/generate", response_model=GenerationResponse)
async def generate_image_authenticated( # Renamed for clarity
    request: MVPGenerationRequest,
    http_request: Request,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(check_mvp_rate_limit)
    # You can add back rate limiting or other dependencies here if needed
):
    """
    Generates a visual asset for an authenticated user, saves it to storage and the database,
    and returns the full generation record.
    """
   
    client_ip = http_request.client.host
    user_service = UserService(get_auth_middleware().supabase)
    request_id = f"{current_user['id']}_{int(time.time())}"
    logger.info(f"[Authenticated Generation][{request_id}] Request from user: {current_user['email']}")
    start_time = time.time()

    try:
        # --- 1. NEW: Robust Permissions Check ---
        user_profile = current_user
        if not user_profile:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found.")

        # tier = user_profile.get("subscription_tier", "free")
        # tier_features = get_tier_features(tier)
        # usage_count = user_profile.get("usage_count", 0)
        # pro_usage_count = user_profile.get("pro_usage_count", 0)

        # if usage_count >= tier_features["monthly_generations_limit"]:
        #     raise HTTPException(
        #         status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        #         detail=f"You have used all {tier_features['monthly_generations_limit']} of your generations for this month. Please upgrade for more."
        #     )

        # Check Pro generation request
        # is_pro_request = 'pro' in (request.model or '').lower()
        # if is_pro_request and not tier_features["allow_pro_generations"]:
        #     raise HTTPException(
        #         status_code=status.HTTP_403_FORBIDDEN,
        #         detail="Upgrade to Pro to use Pro quality generations."
        #     )
        # if is_pro_request and pro_usage_count >= tier_features["monthly_pro_generations_limit"]:
        #     raise HTTPException(
        #         status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        #         detail=f"You have used {tier_features['monthly_pro_generations_limit']}  Pro generations for this month. Please upgrade for more."
        #     )
        # --- 1. Determine Model ---
        model_to_use = pro_gemini_model if 'pro' in (request.model or '').lower() else gemini_model

        # --- 2. Generate HTML (Existing Logic) ---
        width, height, preset_context = resolve_dimensions_from_preset(request)
        prompt_template = get_design_prompt_template(request.theme)
        full_prompt = create_generation_prompt_with_template(
            request.prompt, prompt_template, preset_context, width, height
        )
        html_content = await generate_html_with_gemini(model_to_use, full_prompt, client_ip)
        cleaned_html = clean_html_response(html_content)

        # --- 3. Render Image (Existing Logic) ---
        image_bytes = await render_html_to_image(cleaned_html, width, height, client_ip)

        # --- 4. NEW: Upload Image to Supabase Storage ---
        auth_middleware = get_auth_middleware()
        file_path = f"{current_user['id']}/{uuid.uuid4()}.png"

        # Use the Supabase client to upload
        auth_middleware.supabase.storage.from_("generations").upload(
            file=image_bytes,
            path=file_path,
            file_options={"content-type": "image/png"}
        )

        # Get the public URL for the uploaded image
        image_url_response = auth_middleware.supabase.storage.from_("generations").get_public_url(file_path)
        image_url = image_url_response

        # --- 5. NEW: Save Generation Record to Database ---
        generation_service = GenerationService(auth_middleware.supabase)
        generation_time = int((time.time() - start_time) * 1000)

        generation_data = GenerationCreate(
            user_id=current_user["id"],
            design_thread_id=uuid.uuid4(),  # New thread for a new generation
            parent_id=None,                 # No parent for a new generation
            prompt=request.prompt,
            prompt_type='creation',
            generated_html=cleaned_html,
            image_url=image_url,
            model_used=model_to_use.model_name.split('/')[-1],
            theme=request.theme,
            size_preset=request.size_preset or DEFAULT_PRESET,
            generation_time_ms=generation_time
        )

        new_generation_record = await generation_service.create_generation(generation_data)
        if not new_generation_record:
            raise HTTPException(status_code=500, detail="Failed to save generation record.")

        logger.info(f"[{request_id}] Generation saved with ID: {new_generation_record['id']}")
        # await user_service.increment_usage(current_user["id"])
        # pro_usage_increment = 1 if is_pro_request else 0
        # await user_service.update_usage(user_profile, usage_increment=1, pro_increment=pro_usage_increment)

        # --- 6. NEW: Return the Full Generation Object as JSON ---
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
    user_service = UserService(get_auth_middleware().supabase)

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
        model_to_use = pro_gemini_model if 'pro' in parent_generation['model_used'] else gemini_model
        modified_html_content = await generate_html_with_gemini(model_to_use, edit_prompt_for_llm, client_ip)
        cleaned_modified_html = clean_html_response(modified_html_content)

        # 4. Render the new HTML to a new image
        modified_image_bytes = await render_html_to_image(
            cleaned_modified_html, parent_width, parent_height, client_ip
        )

        # 5. Upload the new image to Storage
        file_path = f"{current_user['id']}/{uuid.uuid4()}.png"
        auth_middleware.supabase.storage.from_("generations").upload(
            file=modified_image_bytes,
            path=file_path,
            file_options={"content-type": "image/png"}
        )
        new_image_url = auth_middleware.supabase.storage.from_("generations").get_public_url(file_path)

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


# ==================== EXISTING PREMIUM ENDPOINTS (UNCHANGED) ====================

@app.post("/api/generate-premium")
async def generate_image(
    request: GenerationRequest,
    http_request: Request,
    current_user: dict = Depends(get_current_user),
    _: dict = Depends(check_usage_limits),
    __: None = Depends(check_rate_limit),
    format: str = "png",
    quality: int = 95
):
    """
    Generate a visual asset with premium features support
    """
    client_ip = http_request.client.host
    request_id = f"{current_user['id']}_{int(time.time())}"
    
    logger.info(f"[{request_id}] Generation request from user: {current_user['email']}")
    
    start_time = time.time()
    
    try:
        # Initialize services
        auth_middleware = get_auth_middleware()
        user_service = UserService(auth_middleware.supabase)
        generation_service = GenerationService(auth_middleware.supabase)
        premium_service = PremiumService(auth_middleware.supabase)
        export_service = ExportService()
        
        # Get user tier
        user_tier = current_user.get("subscription_tier", "free")
        
        # Check premium feature restrictions
        if not premium_service.can_use_dimensions(user_tier, request.width, request.height):
            tier_features = premium_service.get_tier_features(user_tier)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Dimensions {request.width}x{request.height} exceed your plan limit ({tier_features['max_width']}x{tier_features['max_height']})"
            )
        
        if not premium_service.can_use_model(user_tier, request.model or GEMINI_MODEL):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Model {request.model} not available in your plan"
            )
        
        if not premium_service.can_export_format(user_tier, format):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Export format {format} not available in your plan"
            )
        
        # Check usage limits
        usage_check = await user_service.check_usage_limits(current_user["id"])
        if not usage_check["allowed"]:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=usage_check["reason"]
            )
        
        # Get queue priority for premium users
        priority = await premium_service.get_generation_queue_priority(current_user["id"])
        
        # Get appropriate model for this request
        model = get_model_for_request(request)
        
        # Generate the visual asset
        image_bytes = await generate_visual_asset(request, model, client_ip)
        
        # Convert to requested format if not PNG
        if format.lower() != "png":
            image_bytes = await export_service.convert_image(image_bytes, format, quality)
        
        # Calculate generation time
        generation_time = int((time.time() - start_time) * 1000)
        
        # Create generation record
        generation_data = GenerationCreate(
            user_id=current_user["id"],
            prompt=request.prompt,
            model_used=request.model or GEMINI_MODEL,
            width=request.width,
            height=request.height,
            generation_time_ms=generation_time
        )
        
        # Save to database
        generation_record = await generation_service.create_generation(generation_data)
        
        # Update user usage count
        await user_service.increment_usage(current_user["id"])
        
        logger.info(f"[{request_id}] Generation completed successfully with format: {format}")
        
        # Return the image with enhanced headers
        content_type = export_service.get_content_type(format)
        file_extension = export_service.get_file_extension(format)
        
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type=content_type,
            headers={
                "Content-Disposition": f"inline; filename=generated-image{file_extension}",
                "Cache-Control": "no-cache",
                "X-Request-ID": request_id,
                "X-Generation-Time": str(generation_time),
                "X-Generation-ID": generation_record["id"] if generation_record else "unknown",
                "X-User-Tier": user_tier,
                "X-Queue-Priority": str(priority),
                # "X-Design-Template": prompt_template['name'] // todo fix this. commenting out for now
            }
        )
        
    except HTTPException as e:
        logger.error(f"[{request_id}] HTTP error: {e.status_code} - {e.detail}")
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during image generation: {str(e)}"
        )
  
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

# Add new premium features endpoints
@app.post("/api/batch-generate")
async def batch_generate(
    requests: List[GenerationRequest],
    current_user: dict = Depends(get_current_user)
):
    """
    Batch generate multiple images (premium feature)
    """
    try:
        auth_middleware = get_auth_middleware()
        premium_service = PremiumService(auth_middleware.supabase)
        
        # Check if user can perform batch generation
        batch_check = await premium_service.can_generate_batch(current_user["id"], len(requests))
        if not batch_check["allowed"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=batch_check["reason"]
            )
        
        # Process batch requests
        results = []
        for i, request in enumerate(requests):
            try:
                # Generate individual image
                result = await generate_image(request, current_user)
                results.append({"index": i, "status": "success", "result": result})
            except Exception as e:
                results.append({"index": i, "status": "error", "error": str(e)})
        
        return {"batch_results": results}
        
    except Exception as e:
        logger.error(f"Batch generation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Batch generation failed"
        )

@app.get("/premium/features")
async def get_premium_features(current_user: dict = Depends(get_current_user)):
    """
    Get available premium features for current user
    """
    try:
        auth_middleware = get_auth_middleware()
        premium_service = PremiumService(auth_middleware.supabase)
        
        user_tier = current_user.get("subscription_tier", "free")
        features = premium_service.get_tier_features(user_tier)
        
        return {
            "tier": user_tier,
            "features": features,
            "limits": {
                "max_dimensions": f"{features['max_width']}x{features['max_height']}",
                "available_models": features["available_models"],
                "export_formats": features["export_formats"],
                "generation_history_days": features["generation_history_days"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting premium features: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get premium features"
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