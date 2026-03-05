import logging
from fastapi import APIRouter, HTTPException, status
from models.resume import ResumePDFResponse
from services.resume_service import (
    fetch_latest_resume_html,
    mark_record_done,
    render_html_to_pdf,
    upload_pdf_to_supabase,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resume", tags=["Resume"])


@router.get("/generate-pdf", response_model=ResumePDFResponse)
async def generate_resume_pdf():
    """Fetch latest RESUME_HTML from n8n-data, render PDF, upload, mark done."""
    record = fetch_latest_resume_html()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No pending resume HTML found",
        )

    html_content = record.get("text_data")
    if not html_content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Record has no HTML content in text_data",
        )

    record_id = record["id"]

    try:
        pdf_bytes = await render_html_to_pdf(html_content)
    except Exception as e:
        logger.error(f"PDF rendering failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to render PDF from provided HTML",
        )

    try:
        pdf_url = upload_pdf_to_supabase(pdf_bytes, "resume")
    except Exception as e:
        logger.error(f"PDF upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload PDF to storage",
        )

    mark_record_done(record_id)

    return ResumePDFResponse(pdf_url=pdf_url, record_id=record_id)
