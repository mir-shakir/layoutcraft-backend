"""
Dimension Presets Configuration for LayoutCraft

This module contains the centralized configuration for predefined dimension presets
that users can select from instead of specifying manual dimensions.
"""

from typing import Dict, List, TypedDict


class DimensionPreset(TypedDict):
    """Type definition for a dimension preset."""
    width: int
    height: int
    purpose: str


# Centralized dimension presets configuration
DIMENSION_PRESETS: Dict[str, DimensionPreset] = {
    "blog_header": {
        "width": 1200, 
        "height": 630, 
        "purpose": "A wide blog header graphic (16:9 ratio)"
    },
    "social_square": {
        "width": 1080, 
        "height": 1080, 
        "purpose": "A square social media post (1:1 ratio)"
    },
    "social_portrait": {
        "width": 1080, 
        "height": 1350, 
        "purpose": "A vertical social media post (4:5 ratio)"
    },
    "story": {
        "width": 1080, 
        "height": 1920, 
        "purpose": "A tall story for Instagram or TikTok (9:16 ratio)"
    },
    "twitter_post": {
        "width": 1600, 
        "height": 900, 
        "purpose": "A wide image for a Twitter (X) post (16:9 ratio)"
    },
    "presentation_slide": {
        "width": 1920, 
        "height": 1080, 
        "purpose": "A standard presentation slide (16:9 ratio)"
    },
    "youtube_thumbnail": {
        "width": 1280, 
        "height": 720, 
        "purpose": "A thumbnail for a YouTube video (16:9 ratio)"
    },
}

# Default preset to use when none is specified
DEFAULT_PRESET = "blog_header"


def get_preset_dimensions(preset_name: str) -> DimensionPreset:
    """
    Get dimensions and purpose for a given preset name.
    
    Args:
        preset_name: The name of the preset to look up
        
    Returns:
        DimensionPreset containing width, height, and purpose
        
    Raises:
        KeyError: If the preset name is not found
    """
    if preset_name not in DIMENSION_PRESETS:
        raise KeyError(f"Preset '{preset_name}' not found. Available presets: {list(DIMENSION_PRESETS.keys())}")
    
    return DIMENSION_PRESETS[preset_name]


def get_default_preset() -> DimensionPreset:
    """Get the default preset dimensions."""
    return DIMENSION_PRESETS[DEFAULT_PRESET]


def list_available_presets() -> List[str]:
    """Get a list of all available preset names."""
    return list(DIMENSION_PRESETS.keys())


def get_preset_info_for_frontend() -> Dict[str, Dict[str, any]]:
    """
    Get preset information formatted for frontend consumption.
    
    Returns:
        Dictionary with preset names as keys and preset info including display name
    """
    preset_info = {}
    
    # Create display names from preset keys
    display_names = {
        "blog_header": "Blog Header",
        "social_square": "Social Post - Square", 
        "social_portrait": "Social Post - Portrait",
        "story": "Instagram/TikTok Story",
        "twitter_post": "Twitter Post",
        "presentation_slide": "Presentation Slide",
        "youtube_thumbnail": "YouTube Thumbnail"
    }
    
    for preset_name, preset_data in DIMENSION_PRESETS.items():
        preset_info[preset_name] = {
            "width": preset_data["width"],
            "height": preset_data["height"], 
            "purpose": preset_data["purpose"],
            "display_name": display_names.get(preset_name, preset_name.replace("_", " ").title()),
            "aspect_ratio": f"{preset_data['width']}x{preset_data['height']}"
        }
    
    return preset_info


def validate_preset_name(preset_name: str) -> bool:
    """
    Validate if a preset name exists.
    
    Args:
        preset_name: The preset name to validate
        
    Returns:
        True if preset exists, False otherwise
    """
    return preset_name in DIMENSION_PRESETS


def create_dynamic_prompt_context(preset_name: str) -> str:
    """
    Create prompt context based on the selected preset dimensions and purpose.
    
    Args:
        preset_name: The name of the preset
        
    Returns:
        String containing context about dimensions and purpose for the AI prompt
    """
    if not validate_preset_name(preset_name):
        preset_name = DEFAULT_PRESET
    
    preset = DIMENSION_PRESETS[preset_name]
    
    # Create context based on aspect ratio and purpose
    width, height = preset["width"], preset["height"]
    aspect_ratio = width / height
    
    if aspect_ratio > 1.5:  # Wide format
        layout_context = "wide, horizontal layout"
    elif aspect_ratio < 0.8:  # Tall format  
        layout_context = "tall, vertical layout"
    else:  # Square-ish format
        layout_context = "square or balanced layout"
        
    context = f"""
    
    Target dimensions: {width}x{height} pixels.
    """
    
    return context.strip()
    
def get_multiple_presets_with_context(preset_names: List[str]) -> str:
    """
    Get a combined prompt context for multiple presets.
    
    Args:
        preset_names: List of preset names to include in the context
        
    Returns:
        Combined string context for all valid presets in the format:
        {
        - preset_name: width x height pixels
        }
    """
    context_lines = []
    
    for preset_name in preset_names:
        if validate_preset_name(preset_name):
            preset = DIMENSION_PRESETS[preset_name]
            context_lines.append(f"- {preset_name}: {preset['width']}x{preset['height']} pixels")
    context = "\n".join(context_lines)

    if not context:
        # If no valid presets, return default
        default_preset = DIMENSION_PRESETS[DEFAULT_PRESET]
        context_lines.append(f"- {DEFAULT_PRESET}: {default_preset['width']}x{default_preset['height']} pixels")

    return context
    