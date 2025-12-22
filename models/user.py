"""
User model for LayoutCraft
"""
from pydantic import BaseModel, EmailStr,Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

class SubscriptionTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    PRO_TRIAL = "pro-trial" # <-- ADD THIS LINE
    ENTERPRISE = "enterprise"

class BrandKit(BaseModel):
    colors: Dict[str, str] = {} # e.g. {"primary": "#000000", "secondary": "#ffffff"}
    fonts: Dict[str, str] = {}  # e.g. {"heading": "Inter", "body": "Roboto"}
    guidelines: str = ""


class UserProfile(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    usage_count: int = 0
    usage_reset_date: datetime
    created_at: datetime
    updated_at: datetime
    trial_ends_at: Optional[datetime] = None

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    subscription_tier: SubscriptionTier
    trial_ends_at: Optional[datetime] = None
    usage_count: int
    usage_reset_date: datetime
    created_at: datetime

class EditRequest(BaseModel):
    edit_prompt: str = Field(..., description="The user's instruction for what to change.")