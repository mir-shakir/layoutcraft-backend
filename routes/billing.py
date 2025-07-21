"""
Billing routes for LayoutCraft subscription management
"""
from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel
from typing import Optional, List
import logging
from datetime import datetime

from auth.dependencies import get_current_user
from auth.middleware import get_auth_middleware
from services.stripe_service import StripeService

router = APIRouter(prefix="/billing", tags=["Billing"])
logger = logging.getLogger(__name__)

class CheckoutRequest(BaseModel):
    plan_type: str  # "pro_monthly", "pro_yearly", "enterprise_monthly", "enterprise_yearly"

class SubscriptionResponse(BaseModel):
    tier: str
    status: str
    current_period_start: Optional[str] = None
    current_period_end: Optional[str] = None
    cancel_at_period_end: bool = False

@router.post("/create-checkout")
async def create_checkout_session(
    request: CheckoutRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Create Stripe checkout session for subscription
    """
    try:
        auth_middleware = get_auth_middleware()
        stripe_service = StripeService(auth_middleware.supabase)
        
        # Validate plan type
        valid_plans = ["pro_monthly", "pro_yearly", "enterprise_monthly", "enterprise_yearly"]
        if request.plan_type not in valid_plans:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid plan type"
            )
        
        # Get price ID for the plan
        price_id = stripe_service.price_ids.get(request.plan_type)
        if not price_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Plan not configured"
            )
        
        # Create checkout session
        checkout_url = await stripe_service.create_checkout_session(
            current_user["id"], 
            price_id, 
            request.plan_type
        )
        
        if not checkout_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create checkout session"
            )
        
        logger.info(f"Created checkout session for user {current_user['id']}, plan: {request.plan_type}")
        
        return {"checkout_url": checkout_url}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Checkout creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create checkout session"
        )

@router.post("/create-portal")
async def create_portal_session(current_user: dict = Depends(get_current_user)):
    """
    Create Stripe customer portal session
    """
    try:
        auth_middleware = get_auth_middleware()
        stripe_service = StripeService(auth_middleware.supabase)
        
        portal_url = await stripe_service.create_portal_session(current_user["id"])
        
        if not portal_url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No subscription found"
            )
        
        logger.info(f"Created portal session for user {current_user['id']}")
        
        return {"portal_url": portal_url}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portal creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create portal session"
        )

@router.get("/subscription", response_model=SubscriptionResponse)
async def get_subscription_details(current_user: dict = Depends(get_current_user)):
    """
    Get current user's subscription details
    """
    try:
        auth_middleware = get_auth_middleware()
        stripe_service = StripeService(auth_middleware.supabase)
        
        subscription_details = await stripe_service.get_subscription_details(current_user["id"])
        
        return SubscriptionResponse(**subscription_details)
        
    except Exception as e:
        logger.error(f"Error getting subscription details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get subscription details"
        )

@router.get("/plans")
async def get_available_plans():
    """
    Get available subscription plans with pricing
    """
    plans = [
        {
            "id": "pro_monthly",
            "name": "Pro Monthly",
            "price": 19.99,
            "currency": "USD",
            "interval": "month",
            "features": [
                "500 generations per month",
                "All AI models",
                "Priority generation",
                "Advanced export formats",
                "Email support"
            ]
        },
        {
            "id": "pro_yearly",
            "name": "Pro Yearly",
            "price": 199.99,
            "currency": "USD",
            "interval": "year",
            "discount": "Save 17%",
            "features": [
                "500 generations per month",
                "All AI models",
                "Priority generation",
                "Advanced export formats",
                "Email support"
            ]
        },
        {
            "id": "enterprise_monthly",
            "name": "Enterprise Monthly",
            "price": 99.99,
            "currency": "USD",
            "interval": "month",
            "features": [
                "Unlimited generations",
                "All AI models",
                "Highest priority",
                "All export formats",
                "Priority support",
                "Custom integrations",
                "Team collaboration"
            ]
        },
        {
            "id": "enterprise_yearly",
            "name": "Enterprise Yearly",
            "price": 999.99,
            "currency": "USD",
            "interval": "year",
            "discount": "Save 17%",
            "features": [
                "Unlimited generations",
                "All AI models",
                "Highest priority",
                "All export formats",
                "Priority support",
                "Custom integrations",
                "Team collaboration"
            ]
        }
    ]
    
    return {"plans": plans}

@router.post("/webhook")
async def stripe_webhook(request: Request):
    """
    Handle Stripe webhook events
    """
    try:
        payload = await request.body()
        signature = request.headers.get("stripe-signature")
        
        if not signature:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing Stripe signature"
            )
        
        auth_middleware = get_auth_middleware()
        stripe_service = StripeService(auth_middleware.supabase)
        
        result = await stripe_service.handle_webhook(payload, signature)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook processing failed"
        )
