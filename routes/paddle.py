from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel
import logging
from typing import Dict, Any

from auth.dependencies import get_current_user
from services.paddle_service import PaddleService
from services.user_service import UserService
from auth.middleware import get_auth_middleware

router = APIRouter(prefix="/paddle", tags=["Paddle Payments"])
logger = logging.getLogger(__name__)

class CheckoutRequest(BaseModel):
    plan_id: str

class CheckoutResponse(BaseModel):
    transaction_id: str
    status: str

class WebhookResponse(BaseModel):
    status: str
    message: str = "Webhook processed successfully"

@router.post("/create-checkout", response_model=CheckoutResponse)
async def create_checkout(
    request: CheckoutRequest,
    current_user: dict = Depends(get_current_user)
) -> CheckoutResponse:
    """
    Create a Paddle checkout session for a subscription plan
    """
    try:
        # Validate plan_id
        if not request.plan_id or request.plan_id not in ["pro", "max"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid plan_id. Must be 'pro' or 'max'."
            )

        # Initialize services
        paddle_service = PaddleService()
        user_service = UserService(get_auth_middleware().supabase)
        
        # Get user profile and email
        user_profile = await user_service.get_user_profile(current_user["id"])
        if not user_profile or not user_profile.get("email"):
            raise HTTPException(
                status_code=404, 
                detail="User email not found. Please complete your profile."
            )

        # Create checkout session
        transaction_id = paddle_service.create_transaction_for_checkout(
            plan_id=request.plan_id,
            user_id=current_user["id"],
            user_email=user_profile["email"]
        )

        logger.info(f"Created checkout for user {current_user['id']}, plan: {request.plan_id}")
        
        return CheckoutResponse(
            transaction_id=transaction_id,
            status="success"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        # Handle validation errors from PaddleService
        logger.error(f"Validation error creating checkout for user {current_user['id']}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create checkout for user {current_user['id']}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Could not create payment session. Please try again later."
        )

@router.post("/webhook", response_model=WebhookResponse)
async def paddle_webhook(request: Request) -> WebhookResponse:
    """
    Handle Paddle webhook events for subscription updates
    """
    try:
        # Get request body and signature
        request_body = await request.body()
        signature_header = request.headers.get("paddle-signature")
        
        if not signature_header:
            logger.warning("Webhook request missing paddle-signature header")
            raise HTTPException(status_code=400, detail="Missing signature header.")
            
        if not request_body:
            logger.warning("Webhook request has empty body")
            raise HTTPException(status_code=400, detail="Empty request body.")

        # Initialize Paddle service
        paddle_service = PaddleService()
        
        # Verify and parse the webhook
        try:
            event_data = paddle_service.verify_and_parse_webhook(request_body, signature_header)
        except ValueError as e:
            logger.warning(f"Webhook verification failed: {e}")
            raise HTTPException(status_code=400, detail=f"Webhook verification failed: {str(e)}")
        
        # Process the verified webhook data
        try:
            supabase_client = get_auth_middleware().supabase
            await paddle_service.process_webhook_data(event_data, supabase_client)
        except Exception as e:
            logger.error(f"Webhook data processing failed: {e}", exc_info=True)
            # Don't raise here - we've verified the webhook is legitimate
            # Return success to prevent Paddle from retrying
            return WebhookResponse(
                status="error", 
                message="Webhook received but processing failed"
            )

        logger.info(f"Successfully processed webhook event: {event_data.get('event_type', 'unknown')}")
        return WebhookResponse(status="success")
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in webhook handler: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Webhook processing failed due to internal error."
        )

@router.get("/plans", response_model=Dict[str, Any])
async def get_available_plans() -> Dict[str, Any]:
    """
    Get available subscription plans and their Paddle price IDs
    """
    try:
        # Get price IDs from environment (without exposing the actual IDs)
        plans = {
            "pro": {
                "name": "Pro",
                "available": bool(paddle_service.PADDLE_PRICE_IDS.get("pro"))
            },
            "max": {
                "name": "Max", 
                "available": bool(paddle_service.PADDLE_PRICE_IDS.get("max"))
            }
        }
        
        return {
            "plans": plans,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error fetching available plans: {e}")
        raise HTTPException(
            status_code=500,
            detail="Could not fetch available plans."
        )

@router.get("/subscription-status")
async def get_subscription_status(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current user's subscription status
    """
    try:
        user_service = UserService(get_auth_middleware().supabase)
        supabase_client = get_auth_middleware().supabase
        
        # Get user profile
        user_profile = await user_service.get_user_profile(current_user["id"])
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found.")
        
        # Get subscription details if exists
        subscription_data = None
        if user_profile.get("subscription_tier") != "free":
            try:
                subscription_response = await supabase_client.table("subscriptions").select("*").eq("user_id", current_user["id"]).order("updated_at", desc=True).limit(1).single().execute()
                subscription_data = subscription_response.data if subscription_response.data else None
            except Exception as e:
                logger.warning(f"Could not fetch subscription details for user {current_user['id']}: {e}")
        
        return {
            "user_id": current_user["id"],
            "subscription_tier": user_profile.get("subscription_tier", "free"),
            "paddle_customer_id": user_profile.get("paddle_customer_id"),
            "subscription": subscription_data,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching subscription status for user {current_user['id']}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Could not fetch subscription status."
        )