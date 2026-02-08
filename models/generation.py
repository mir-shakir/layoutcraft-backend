"""
Generation models for LayoutCraft, updated for history and editing features.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid

class GenerationOutput(BaseModel):
    size_preset: str
    image_url: str

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
    image_url: Optional[str] = None
    model_used: str
    theme: str
    size_preset: Optional[str] = None
    generation_time_ms: Optional[int] = None 
    images_json: Optional[List[GenerationOutput]] = None
    used_brand_kit: bool = False
    

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
    image_url: Optional[str] = None
    model_used: str
    theme: str
    size_preset: Optional[str] = None
    created_at: datetime
    images_json: Optional[List[GenerationOutput]] = None # Updated to list of GenerationOutput
    used_brand_kit: bool = False

    class Config:
        from_attributes = True # Use this to allow the model to be created from ORM objects

class HistoryParent(BaseModel):
    """
    Model for returning a list of design threads (parents) to the frontend.
    """
    design_thread_id: uuid.UUID
    original_prompt: Optional[str] = None
    created_at: Optional[datetime] = None
    thumbnail_url: Optional[str] = None
    edit_groups_count: Optional[int] = None
    total_designs_count: Optional[int] = None
    used_brand_kit: bool = False

    class Config:
        from_attributes = True
class HistoryParentsResponse(BaseModel):
    """
    Model for returning a paginated list of design threads (parents) to the frontend.
    """
    parents: List[HistoryParent]
    has_next: bool
    next_offset: Optional[int] = None

    class Config:
        from_attributes = True

class EditGroup(BaseModel):
    """
    Model for returning a list of edit groups (generations) within a design thread to the frontend.
    """
    generation_id: uuid.UUID
    prompt: str
    prompt_type: str
    created_at: datetime
    images_json: Optional[List[GenerationOutput]] = None 

    class Config:
        from_attributes = True

class EditGroupsResponse(BaseModel):
    """
    Model for returning a list of edit groups (generations) within a design thread to the frontend.
    """
    edit_groups: List[EditGroup]

    class Config:
        from_attributes = True