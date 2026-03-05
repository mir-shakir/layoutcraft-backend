from pydantic import BaseModel


class ResumePDFResponse(BaseModel):
    pdf_url: str
    record_id: int
