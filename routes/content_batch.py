"""
ContentBatch router -- Bulk image generation from CSV + template.
"""
import os
import io
import re
import asyncio
import zipfile
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from google import genai
from playwright.async_api import async_playwright

from auth.dependencies import get_current_user
from auth.middleware import get_auth_middleware
from services.content_batch_service import ContentBatchService
from prompts.content_batch_prompts import build_content_batch_prompt, build_text_to_csv_prompt

router = APIRouter(prefix="/api/cb", tags=["ContentBatch"])
logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT", "180"))


# ── Request / Response models ─────────────────────────────────

class GenerateBatchRequest(BaseModel):
    template_id: str = Field(..., description="UUID of the cb_template to use")
    csv_data: List[Dict[str, Any]] = Field(
        ..., description="Parsed CSV rows as list of objects"
    )
    column_mapping: Dict[str, str] = Field(
        ...,
        description="Map of template placeholders to CSV column names, e.g. {'quote': 'Quote', 'author': 'Author'}",
    )


class BatchStatusResponse(BaseModel):
    batch_id: str
    status: str
    result_zip_url: Optional[str] = None
    error_log: Optional[str] = None
    created_at: Optional[str] = None


class TextToCSVRequest(BaseModel):
    raw_text: str = Field(..., description="Unstructured text to convert into CSV")
    columns: List[str] = Field(..., description="Target CSV column names")
    intent: Optional[str] = Field(None, description="Optional description of what the user wants")


# ── Helpers ────────────────────────────────────────────────────

def _get_service() -> ContentBatchService:
    auth = get_auth_middleware()
    return ContentBatchService(auth.supabase)


def _clean_html(html: str) -> str:
    """Remove markdown fences the LLM may wrap around the HTML."""
    html = html.replace("```html", "").replace("```", "").strip()
    start = html.find("<!DOCTYPE")
    if start == -1:
        start = html.find("<html")
    if start > 0:
        html = html[start:]
    return html


def _clean_csv(text: str) -> str:
    """Remove markdown fences the LLM may wrap around CSV output."""
    text = text.replace("```csv", "").replace("```", "").strip()
    return text


async def _call_gemini(prompt: str) -> str:
    """Call Gemini with the given prompt and return raw text."""
    try:
        client = genai.Client()
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model=GEMINI_MODEL,
                contents=prompt,
            ),
            timeout=GENERATION_TIMEOUT,
        )
        if response and response.candidates:
            candidate = response.candidates[0]
            if candidate.content and hasattr(candidate.content, "parts"):
                parts = candidate.content.parts
                if parts:
                    return "".join(p.text for p in parts if hasattr(p, "text"))
        raise HTTPException(status_code=500, detail="Empty response from AI model")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI generation timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LLM generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate content")


async def _render_and_zip(html: str, count: int) -> bytes:
    """
    Open the master HTML in Playwright, screenshot each variation container,
    and return an in-memory ZIP archive as bytes.
    """
    buf = io.BytesIO()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_viewport_size({"width": 1920, "height": 1080})
        await page.set_content(html, wait_until="networkidle")

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(count):
                selector = f"#variation-{i}"
                try:
                    el = page.locator(selector).first
                    await el.wait_for(timeout=5000)
                    screenshot = await el.screenshot(type="png")
                    zf.writestr(f"image-{i + 1}.png", screenshot)
                except Exception as e:
                    logger.warning(f"Failed to capture {selector}: {e}")
                    continue

        await browser.close()

    buf.seek(0)
    return buf.read()


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')[:50]


async def _upload_zip(supabase_client, user_id: str, batch_id: str, zip_bytes: bytes, template_name: str = "", row_count: int = 0) -> str:
    """Upload the ZIP to the content-batch Supabase bucket and return public URL."""
    slug = _slugify(template_name) if template_name else batch_id
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    file_path = f"{user_id}/{slug}_{row_count}rows_{timestamp}.zip"
    supabase_client.storage.from_("content-batch").upload(
        path=file_path,
        file=zip_bytes,
        file_options={"content-type": "application/zip"},
    )
    return supabase_client.storage.from_("content-batch").get_public_url(file_path)


# ── Background job ─────────────────────────────────────────────

async def _process_batch(batch_id: str, user_id: str, template: dict, csv_data: list, column_mapping: dict):
    """
    Full pipeline executed as a background task:
    1. Build prompt  2. LLM -> master HTML  3. Playwright screenshots  4. ZIP & upload
    """
    service = _get_service()
    auth = get_auth_middleware()
    try:
        # 1. Build prompt
        prompt = build_content_batch_prompt(
            template_html=template["html_structure"],
            css_styles=template.get("css_styles"),
            ai_rules=template.get("ai_rules"),
            column_mapping=column_mapping,
            data_rows=csv_data,
        )

        # 2. Generate master HTML
        raw_html = await _call_gemini(prompt)
        master_html = _clean_html(raw_html)

        # 3. Render screenshots & zip
        zip_bytes = await _render_and_zip(master_html, len(csv_data))

        # 4. Upload to Supabase
        zip_url = await _upload_zip(auth.supabase, user_id, batch_id, zip_bytes, template_name=template.get("name", ""), row_count=len(csv_data))

        # 5. Mark completed
        await service.update_batch_completed(batch_id, zip_url)
        logger.info(f"Batch {batch_id} completed successfully")

    except Exception as e:
        logger.error(f"Batch {batch_id} failed: {e}", exc_info=True)
        await service.update_batch_failed(batch_id, str(e))


# ── Endpoints ──────────────────────────────────────────────────

@router.post("/generate", response_model=BatchStatusResponse)
async def generate_batch(
    req: GenerateBatchRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    """
    Start a new ContentBatch generation job.
    Returns immediately with a batch_id; processing happens in background.
    """
    service = _get_service()
    user_id = current_user["id"]

    # Validate template exists
    template = await service.get_template(req.template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    # Validate CSV data is non-empty
    if not req.csv_data or len(req.csv_data) == 0:
        raise HTTPException(status_code=400, detail="CSV data is empty")

    # Cap at 100 rows per batch
    if len(req.csv_data) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 rows per batch")

    # Create batch record
    batch = await service.create_batch(
        user_id=user_id,
        template_id=req.template_id,
        input_data=req.csv_data,
        column_mapping=req.column_mapping,
    )
    if not batch:
        raise HTTPException(status_code=500, detail="Failed to create batch")

    # Kick off background processing
    background_tasks.add_task(
        _process_batch,
        batch_id=batch["id"],
        user_id=user_id,
        template=template,
        csv_data=req.csv_data,
        column_mapping=req.column_mapping,
    )

    return BatchStatusResponse(
        batch_id=batch["id"],
        status="processing",
        created_at=batch.get("created_at"),
    )


@router.get("/batch/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(
    batch_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Poll the status of a batch job."""
    service = _get_service()
    batch = await service.get_batch(batch_id)

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Ownership check
    if batch["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    return BatchStatusResponse(
        batch_id=batch["id"],
        status=batch["status"],
        result_zip_url=batch.get("result_zip_url"),
        error_log=batch.get("error_log"),
        created_at=batch.get("created_at"),
    )


@router.get("/templates")
async def list_templates(current_user: dict = Depends(get_current_user)):
    """Return all active ContentBatch templates."""
    service = _get_service()
    templates = await service.list_templates()
    return {"templates": templates}


@router.get("/batches")
async def list_batches(current_user: dict = Depends(get_current_user)):
    """Return all batches for the current user."""
    service = _get_service()
    batches = await service.list_user_batches(current_user["id"])
    return {"batches": batches}


@router.post("/text-to-csv")
async def text_to_csv(
    req: TextToCSVRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Convert unstructured text into CSV using AI.
    Returns the CSV text for user verification -- does NOT auto-process it.
    """
    if not req.raw_text or not req.raw_text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if not req.columns or len(req.columns) == 0:
        raise HTTPException(status_code=400, detail="At least one target column is required")

    prompt = build_text_to_csv_prompt(
        raw_text=req.raw_text,
        columns=req.columns,
        intent=req.intent or "",
    )

    raw_csv = await _call_gemini(prompt)
    csv_text = _clean_csv(raw_csv)

    return {"csv_text": csv_text}
