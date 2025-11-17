"""
Billing routes for Dodo Payments subscription management.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from auth.dependencies import get_current_user
from auth.middleware import get_auth_middleware
from services.dodo_service import DodoService

router = APIRouter(prefix="/dodo", tags=["Dodo Billing"])
logger = logging.getLogger(__name__)

class CheckoutResponse(BaseModel):
    checkout_url: str

class PortalResponse(BaseModel):
    portal_url: str

@router.post("/create-checkout", response_model=CheckoutResponse)
async def create_dodo_checkout_session(current_user: dict = Depends(get_current_user)):
    """
    Creates a Dodo Payments checkout session for the current user to subscribe to the pro plan.
    """
    try:
        auth_middleware = get_auth_middleware()
        dodo_service = DodoService(auth_middleware.supabase)
        trial_period_days = await dodo_service.get_no_of_trial_days_for_user(current_user)

        checkout_url = await dodo_service.create_checkout_session(
            user_id=current_user["id"],
            user_email=current_user["email"],
            full_name=current_user.get("full_name", ""),
            dodo_customer_id=current_user.get("dodo_customer_id", ""),
            trial_period_days=trial_period_days
        )

        if not checkout_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create checkout session."
            )

        logger.info(f"Created Dodo checkout session for user {current_user['id']}")
        return CheckoutResponse(checkout_url=checkout_url)

    except Exception as e:
        logger.error(f"Dodo checkout creation error for user {current_user.get('id', 'unknown')}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/webhook")
async def dodo_webhook(request: Request):
    """
    Handles incoming webhook events from Dodo Payments.
    """
    payload = await request.body()
    headers = dict(request.headers)

    
    try:
        auth_middleware = get_auth_middleware()
        dodo_service = DodoService(auth_middleware.supabase)
        result = await dodo_service.process_webhook(payload, headers)
        return result

    except ValueError as e: # Catches signature verification errors
        logger.warning(f"Dodo webhook verification failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error processing Dodo webhook: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Webhook processing failed due to an internal error."
        )
@router.post("/portal", response_model=PortalResponse)
async def create_dodo_portal_session(current_user: dict = Depends(get_current_user)):
    """
    Creates a Dodo Payments customer portal session for the current user.
    """
    logger.info(f"Creating Dodo portal session for user {current_user.get('id', 'unknown')}")
    dodo_customer_id = current_user.get("dodo_customer_id")
    if not dodo_customer_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No subscription found for this user."
        )

    try:
        auth_middleware = get_auth_middleware()
        dodo_service = DodoService(auth_middleware.supabase)

        portal_url = await dodo_service.create_portal_session(dodo_customer_id)

        if not portal_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create customer portal session."
            )

        return PortalResponse(portal_url=portal_url)

    except Exception as e:
        logger.error(f"Dodo portal creation error for user {current_user.get('id', 'unknown')}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )