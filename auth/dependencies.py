"""
Authentication dependencies for LayoutCraft
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from .middleware import get_auth_middleware
from models.user import UserProfile, SubscriptionTier

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Get current authenticated user
    """
    auth_middleware = get_auth_middleware()
    return await auth_middleware.verify_token(credentials)

async def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[dict]:
    """
    Get current user if authenticated, otherwise return None
    """
    if not credentials:
        return None
    
    try:
        auth_middleware = get_auth_middleware()
        return await auth_middleware.verify_token(credentials)
    except HTTPException:
        return None

def require_subscription_tier(required_tier: SubscriptionTier):
    async def _require_tier(user: dict = Depends(get_current_user)):
        tier_hierarchy = {"free": 0, "pro": 1, "enterprise": 2}
        user_tier = user.get("subscription_tier", "free")
        if tier_hierarchy[user_tier] < tier_hierarchy[required_tier.value]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This feature requires {required_tier.value} subscription",
            )
        return user
    return _require_tier

async def check_usage_limits(user: dict = Depends(get_current_user)) -> dict:
    """
    Check if user has exceeded usage limits
    """
    tier = user.get("subscription_tier", "free")
    usage_count = user.get("usage_count", 0)
    
    # Define usage limits per tier
    usage_limits = {
        "free": 10,
        "pro": 500,
        "enterprise": float('inf')
    }
    
    limit = usage_limits.get(tier, 10)
    
    if usage_count >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Usage limit exceeded. Upgrade to increase your limit."
        )
    
    return user
def require_pro_plan(current_user: dict = Depends(get_current_user)):
    """
    Dependency that raises an exception if the user does not have an active
    pro plan or trial.
    """
    tier = current_user.get("subscription_tier")
    if tier not in ["pro", "pro-trial"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This feature requires a Pro plan. Please upgrade to continue."
        )
    return current_user

# Convenience dependencies
RequireProTier = require_subscription_tier(SubscriptionTier.PRO)
RequireEnterpriseTier = require_subscription_tier(SubscriptionTier.ENTERPRISE)
