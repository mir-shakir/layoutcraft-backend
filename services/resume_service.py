import uuid
import logging
from typing import Dict, Any
from playwright.async_api import async_playwright
from auth.middleware import get_auth_middleware

logger = logging.getLogger(__name__)

RESUME_TYPE = "RESUME_HTML"
STATUS_NEW = "new"
STATUS_DONE = "done"


def _get_supabase():
    return get_auth_middleware().supabase


def fetch_latest_resume_html() -> Dict[str, Any]:
    """Fetch the latest n8n-data row where type=RESUME_HTML and status=new."""
    supabase = _get_supabase()
    response = (
        supabase.table("n8n-data")
        .select("*")
        .eq("type", RESUME_TYPE)
        .eq("status", STATUS_NEW)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not response.data:
        return None
    return response.data[0]


def mark_record_done(record_id: int) -> None:
    """Update the n8n-data row status to done."""
    supabase = _get_supabase()
    supabase.table("n8n-data").update({"status": STATUS_DONE}).eq("id", record_id).execute()


async def render_html_to_pdf(html_content: str) -> bytes:
    """Render HTML to PDF using Playwright headless Chromium."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.set_content(html_content, wait_until="networkidle")
            pdf_bytes = await page.pdf(
                format="A4",
                print_background=True,
                margin={"top": "0.8in", "right": "0.4in", "bottom": "0.4in", "left": "0.4in"},
            )
            return pdf_bytes
        finally:
            await browser.close()


def upload_pdf_to_supabase(pdf_bytes: bytes, file_name: str) -> str:
    """Upload PDF bytes to Supabase storage and return the public URL."""
    storage_client = _get_supabase().storage.from_("resumes")

    file_path = f"{uuid.uuid4()}_{file_name}.pdf"
    storage_client.upload(
        file=pdf_bytes,
        path=file_path,
        file_options={"content-type": "application/pdf"},
    )
    return storage_client.get_public_url(file_path)
