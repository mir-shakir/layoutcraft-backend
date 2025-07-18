import os
import asyncio
import time
import logging
from typing import Dict, Optional
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import google.generativeai as genai
from playwright.async_api import async_playwright
import io

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
)

# Configuration constants
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # Default to 1.5-flash
GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT", "120"))  # 2 minutes default
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))  # requests per minute
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds

# Available models for easy switching
AVAILABLE_MODELS = {
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-1.5-pro": "gemini-1.5-pro", 
    "gemini-2.0-flash": "gemini-2.0-flash-exp",
    "gemini-2.5-flash": "gemini-2.5-flash",  # Add when available
    "gemini-2.5-pro": "gemini-2.5-pro"      # Add when available
}

# Simple in-memory rate limiter storage
rate_limiter_storage: Dict[str, list] = defaultdict(list)

# Request model
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="User's prompt for image generation")
    width: int = Field(default=1200, ge=100, le=3000, description="Image width in pixels")
    height: int = Field(default=630, ge=100, le=3000, description="Image height in pixels")
    model: Optional[str] = Field(default=None, description="Override default model for this request")

# Initialize Gemini AI
def initialize_gemini(model_name: str = None):
    """Initialize Gemini AI with API key and specified model."""
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
        
        # Log response length for debugging
        response_length = len(response.text) if response.text else 0
        logger.debug(f"Generated HTML length: {response_length} characters")
        
        return response.text
        
    except asyncio.TimeoutError:
        logger.error(f"HTML generation timeout after {GENERATION_TIMEOUT}s for IP: {client_ip}")
        raise HTTPException(
            status_code=408,
            detail="HTML generation timed out. Please try again with a simpler prompt."
        )
    except Exception as e:
        logger.error(f"Error in HTML generation for IP {client_ip}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating HTML: {str(e)}"
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

async def render_html_to_image(html_content: str, width: int, height: int, client_ip: str) -> bytes:
    """Render HTML to PNG image using Playwright."""
    start_time = time.time()
    logger.info(f"Starting HTML rendering to image ({width}x{height}) for IP: {client_ip}")
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Set viewport size to match requested dimensions
            await page.set_viewport_size({"width": width, "height": height})
            
            # Set HTML content
            await page.set_content(html_content, wait_until="networkidle")
            
            # Take screenshot
            screenshot_bytes = await page.screenshot(
                type="png",
                full_page=False,
                clip={"x": 0, "y": 0, "width": width, "height": height}
            )
            
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
    logger.info(f"Starting visual asset generation for IP: {client_ip}, prompt: '{request.prompt[:50]}...'")
    
    # Create generation prompt
    full_prompt = create_generation_prompt(request.prompt)
    logger.debug(f"Generated system prompt length: {len(full_prompt)} characters")
    
    # Generate HTML with Gemini
    html_content = await generate_html_with_gemini(model, full_prompt, client_ip)
    
    # Clean HTML response
    cleaned_html = clean_html_response(html_content)
    
    # Render HTML to image
    image_bytes = await render_html_to_image(cleaned_html, request.width, request.height, client_ip)
    
    logger.info(f"Visual asset generation completed successfully for IP: {client_ip}")
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

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global gemini_model
    try:
        gemini_model = initialize_gemini()
        logger.info("✅ LayoutCraft Backend started successfully")
        logger.info(f"✅ Default model: {GEMINI_MODEL}")
        logger.info(f"✅ Available models: {list(AVAILABLE_MODELS.keys())}")
        logger.info(f"✅ Rate limit: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s")
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

@app.post("/api/generate")
async def generate_image(
    request: GenerationRequest,
    http_request: Request,
    _: None = Depends(check_rate_limit)
):
    """
    Generate a visual asset based on the user's prompt.
    
    This endpoint is structured to easily accommodate future authentication
    by adding a dependency injection parameter like:
    current_user: User = Depends(get_current_user)
    """
    client_ip = http_request.client.host
    request_id = f"{client_ip}_{int(time.time())}"
    
    logger.info(f"[{request_id}] Generation request received")
    logger.debug(f"[{request_id}] Request details: {request.dict()}")
    
    try:
        # Get appropriate model for this request
        model = get_model_for_request(request)
        
        # Generate the visual asset
        image_bytes = await generate_visual_asset(request, model, client_ip)
        
        logger.info(f"[{request_id}] Generation completed successfully")
        
        # Return the image as a streaming response
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=generated-image.png",
                "Cache-Control": "no-cache",
                "X-Request-ID": request_id
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
        "version": "1.0.0",
        "model": GEMINI_MODEL,
        "available_models": list(AVAILABLE_MODELS.keys()),
        "endpoints": {
            "generate": "/api/generate",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
