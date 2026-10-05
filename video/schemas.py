from pydantic import BaseModel, Field, field_validator

class VideoRenderRequest(BaseModel):
    html: str = Field(..., description="Self-contained HTML animation document.")
    width: int = Field(1920, description="Width of the video", le=1920, ge=1)
    height: int = Field(1080, description="Height of the video", le=1080, ge=1)
    fps: int = Field(30, description="Frames per second", le=30, ge=1)
    duration_seconds: float = Field(..., description="Duration of the video in seconds", le=30.0, ge=0.1)

    @field_validator("html")
    @classmethod
    def check_html_size(cls, v: str) -> str:
        if len(v.encode('utf-8')) > 2 * 1024 * 1024:
            raise ValueError("HTML size must be less than or equal to 2 MB")
        return v

class VideoRenderResponse(BaseModel):
    success: bool
    video_url: str | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    format: str | None = None
    size_bytes: int | None = None
    error: str | None = None
