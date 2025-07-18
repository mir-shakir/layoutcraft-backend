import os
import re
import logging
import tempfile
from typing import Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict
from io import BytesIO

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import google.generativeai as genai
from playwright.async_api import async_playwright
import asyncio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LayoutCraft API",
    description="AI-Powered Visual Asset Generator",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting storage (in-memory for MVP)
rate_limit_storage: Dict[str, list] = defaultdict(list)

# Request/Response Models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="User's prompt for image generation")
    width: int = Field(default=1200, ge=400, le=2000, description="Image width in pixels")
    height: int = Field(default=630, ge=300, le=1500, description="Image height in pixels")

class ErrorResponse(BaseModel):
    error: str
    message: str

# Configuration
class Config:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Rate limiting config
        self.rate_limit_requests = 10  # requests per window
        self.rate_limit_window = 60  # seconds

config = Config()

# Helper Functions
def create_generation_prompt(user_prompt: str, width: int = 1200, height: int = 630) -> str:
    """Create the system prompt for HTML generation"""
    template = f"""
You are a world-class digital designer and frontend developer. Your task is to create a single, visually stunning, self-contained HTML file that looks like a modern tech blog header, based on the user's core idea. Your entire response must be ONLY the raw HTML code, starting with `<!DOCTYPE html>` and ending with `</html>`.

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
- The container must be {width}x{height} pixels.
- No external files, no JavaScript, and no explanations or markdown in your response.
- Ensure the HTML is valid and well-formed.

**User's Core Idea:**
"{user_prompt}"

Based on all the above, generate the complete HTML code.
"""
    return template.strip()

def clean_html_response(raw_response: str) -> str:
    """Clean the HTML response from Gemini API"""
    try:
        # Remove markdown code fences
        cleaned = re.sub(r'```html\s*', '', raw_response, flags=re.IGNORECASE)
        cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
        
        # Remove extra whitespace but preserve structure
        cleaned = cleaned.strip()
        
        # Basic validation - ensure it starts with DOCTYPE and ends with </html>
        if not cleaned.lower().startswith('<!doctype html'):
            raise ValueError("Response does not start with valid HTML DOCTYPE")
        
        if not cleaned.lower().rstrip().endswith('</html>'):
            raise ValueError("Response does not end with valid HTML closing tag")
        
        return cleaned
    
    except Exception as e:
        logger.error(f"Error cleaning HTML response: {e}")
        raise ValueError(f"Failed to clean HTML response: {str(e)}")

async def generate_html_with_gemini(prompt: str, width: int, height: int) -> str:
    """Generate HTML using Gemini API"""
    try:
        system_prompt = create_generation_prompt(prompt, width, height)
        
        logger.info(f"Generating HTML for prompt: {prompt[:100]}...")
        
        response = await asyncio.to_thread(
            config.model.generate_content,
            system_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=40,
                max_output_tokens=8192,
            )
        )
        
        if not response.text:
            raise ValueError("Empty response from Gemini API")
        
        cleaned_html = clean_html_response(response.text)
        logger.info("HTML generated successfully")
        
        return cleaned_html
    
    except Exception as e:
        logger.error(f"Error generating HTML with Gemini: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate HTML: {str(e)}")

async def render_html_to_image(html_content: str, width: int, height: int) -> bytes:
    """Render HTML to PNG image using Playwright"""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={'width': width, 'height': height},
                device_scale_factor=1
            )
            
            page = await context.new_page()
            
            # Set the HTML content
            await page.set_content(html_content, wait_until='networkidle')
            
            # Wait a bit for any CSS animations to settle
            await asyncio.sleep(1)
            
            # Take screenshot
            screenshot_bytes = await page.screenshot(
                type='png',
                full_page=False,
                clip={'x': 0, 'y': 0, 'width': width, 'height': height}
            )
            
            await browser.close()
            
            logger.info(f"Image rendered successfully: {width}x{height}")
            return screenshot_bytes
    
    except Exception as e:
        logger.error(f"Error rendering HTML to image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to render image: {str(e)}")

def check_rate_limit(client_ip: str) -> bool:
    """Simple IP-based rate limiting"""
    now = datetime.now()
    cutoff = now - timedelta(seconds=config.rate_limit_window)
    
    # Clean old entries
    rate_limit_storage[client_ip] = [
        timestamp for timestamp in rate_limit_storage[client_ip]
        if timestamp > cutoff
    ]
    
    # Check if under limit
    if len(rate_limit_storage[client_ip]) >= config.rate_limit_requests:
        return False
    
    # Add current request
    rate_limit_storage[client_ip].append(now)
    return True

async def get_client_ip(request: Request) -> str:
    """Get client IP address"""
    # Check for forwarded headers first (for reverse proxies)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    return request.client.host if request.client else "unknown"

# API Endpoints
@app.post("/api/generate", response_class=StreamingResponse)
async def generate_image(
    request: GenerateRequest,
    http_request: Request,
    client_ip: str = Depends(get_client_ip)
):
    """Generate image from text prompt"""
    try:
        # Rate limiting
        if not check_rate_limit(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        logger.info(f"Processing request from IP: {client_ip}, prompt: {request.prompt[:100]}...")
        
        # Generate HTML
        html_content = await generate_html_with_gemini(
            request.prompt,
            request.width,
            request.height
        )
        
        # Render to image
        image_bytes = await render_html_to_image(
            html_content,
            request.width,
            request.height
        )
        
        # Create response
        image_stream = BytesIO(image_bytes)
        
        return StreamingResponse(
            image_stream,
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=generated_image.png",
                "Cache-Control": "no-cache"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "LayoutCraft API is running", "version": "1.0.0"}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return {"error": "HTTP Error", "message": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unexpected error: {str(exc)}")
    return {"error": "Internal Server Error", "message": "An unexpected error occurred"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)