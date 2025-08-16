# config/tier_config.py

from typing import Dict, Any, List

TIER_CONFIG: Dict[str, Dict[str, Any]] = {
    "anonymous": {
        "name": "Anonymous",
        "total_generations_limit": 3,
        "allow_pro_generations": False,
        "allow_editing": False,
        "monthly_generations_limit": 0,
        "monthly_edit_limit": 0,
        "monthly_pro_generations_limit": 0,
    },
    "free": {
        "name": "Free",
        "monthly_generations_limit": 10,
        "allow_pro_generations": True, # For the one-time credit
        "allow_editing": True, # For the one-time credit
        "monthly_pro_generations_limit": 3, # No Pro generations allowed
        "monthly_edit_limit": 1, # Free users can edit up to 5 times
    },
    "pro": {
        "name": "Pro",
        "monthly_generations_limit": 500,
        "allow_pro_generations": True,
        "allow_editing": True,
        "monthly_pro_generations_limit": 50, # Pro generations allowed
        "monthly_edit_limit": 50, # Pro users can edit up to 10 times
    },
    "enterprise": {
        "name": "Enterprise",
        "monthly_generations_limit": float('inf'), # Represents unlimited
        "allow_pro_generations": True,
        "allow_editing": True,
        "monthly_pro_generations_limit": float('inf'), # Represents unlimited
        "monthly_edit_limit": float('inf'), # Represents unlimited
    }
}

def get_tier_features(tier: str) -> Dict[str, Any]:
    """Safely get the feature configuration for a given tier."""
    return TIER_CONFIG.get(tier, TIER_CONFIG["free"])