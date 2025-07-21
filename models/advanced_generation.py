"""
Advanced generation models for LayoutCraft
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class GenerationStyle(str, Enum):
    MINIMAL = "minimal"
    MODERN = "modern"
    CLASSIC = "classic"
    BOLD = "bold"
    ELEGANT = "elegant"
    PLAYFUL = "playful"

class ColorScheme(str, Enum):
    MONOCHROME = "monochrome"
    COMPLEMENTARY = "complementary"
    ANALOGOUS = "analogous"
    TRIADIC = "triadic"
    CUSTOM = "custom"

class AdvancedGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Main prompt for generation")
    width: int = Field(default=1200, ge=100, le=5000)
    height: int = Field(default=630, ge=100, le=5000)
    model: Optional[str] = Field(default=None)
    
    # Advanced options
    style: Optional[GenerationStyle] = Field(default=None)
    color_scheme: Optional[ColorScheme] = Field(default=None)
    primary_colors: Optional[List[str]] = Field(default=None)
    secondary_colors: Optional[List[str]] = Field(default=None)
    fonts: Optional[List[str]] = Field(default=None)
    
    # Template and brand kit
    template_id: Optional[str] = Field(default=None)
    brand_kit_id: Optional[str] = Field(default=None)
    
    # Export options
    export_format: str = Field(default="png")
    quality: int = Field(default=95, ge=1, le=100)
    
    # Advanced features
    iterations: int = Field(default=1, ge=1, le=10)
    seed: Optional[int] = Field(default=None)

class BatchGenerationRequest(BaseModel):
    prompts: List[str] = Field(..., max_items=100)
    width: int = Field(default=1200, ge=100, le=5000)
    height: int = Field(default=630, ge=100, le=5000)
    model: Optional[str] = Field(default=None)
    template_id: Optional[str] = Field(default=None)
    export_format: str = Field(default="png")
    quality: int = Field(default=95, ge=1, le=100)

class BrandKit(BaseModel):
    brand_name: str
    primary_colors: List[str] = Field(default=[])
    secondary_colors: List[str] = Field(default=[])
    fonts: List[str] = Field(default=[])
    logo_url: Optional[str] = None
    style_guidelines: str = ""

class CustomTemplate(BaseModel):
    name: str
    description: str = ""
    prompt_template: str
    default_width: int = 1200
    default_height: int = 630
    category: str = "general"
    tags: List[str] = Field(default=[])
    is_public: bool = False
