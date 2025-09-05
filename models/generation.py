"""
Generation models for LayoutCraft, updated for history and editing features.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid

class GenerationCreate(BaseModel):
    """
    Model for creating a new generation record in the database.
    """
    user_id: str
    design_thread_id: uuid.UUID
    parent_id: Optional[uuid.UUID] = None
    prompt: str
    prompt_type: str = 'creation'
    generated_html: str
    image_url: str
    model_used: str
    theme: str
    size_preset: str
    generation_time_ms: Optional[int] = None # This field is from your old model, keeping it.

class GenerationResponse(BaseModel):
    """
    Model for returning a generation record to the frontend.
    """
    id: uuid.UUID
    user_id: str
    design_thread_id: uuid.UUID
    parent_id: Optional[uuid.UUID] = None
    prompt: str
    prompt_type: str
    image_url: str
    model_used: str
    theme: str
    size_preset: str
    created_at: datetime

    class Config:
        from_attributes = True # Use this to allow the model to be created from ORM objects