"""
Generation model for LayoutCraft
"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class GenerationHistory(BaseModel):
    id: str
    user_id: str
    prompt: str
    model_used: str
    width: int
    height: int
    image_url: Optional[str] = None
    generation_time_ms: Optional[int] = None
    created_at: datetime

class GenerationCreate(BaseModel):
    user_id: str
    prompt: str
    model_used: str
    width: int
    height: int
    image_url: Optional[str] = None
    generation_time_ms: Optional[int] = None

class GenerationResponse(BaseModel):
    id: str
    prompt: str
    model_used: str
    width: int
    height: int
    image_url: Optional[str] = None
    generation_time_ms: Optional[int] = None
    created_at: datetime
