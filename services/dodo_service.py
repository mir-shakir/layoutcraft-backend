"""
Dodo Payments service for LayoutCraft subscription management.
FINAL - Verified against the user-provided webhook payload.
"""
import os
import logging
from dodopayments import AsyncDodoPayments
from standardwebhooks import Webhook
from supabase import Client
from typing import Dict, Any
from datetime import datetime, timezone
import math
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DodoService:
    def __init__(self, supabase_client: Client):
        """Initializes the Dodo Service and the Dodo Payments SDK client."""
        self.supabase = supabase_client
        self.dodo_api_key = os.getenv("DODO_API_KEY")
        self.webhook_secret = os.getenv("DODO_WEBHOOK_SECRET")
        self.pro_plan_id = os.getenv("DODO_PRO_PLAN_ID")
        self.frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")

        if not self.dodo_api_key or not self.webhook_secret or not self.pro_plan_id:
            logger.error("Dodo Payments environment variables are not fully configured.")
            raise ValueError("DODO_API_KEY, DODO_WEBHOOK_SECRET, and DODO_PRO_PLAN_ID must be set.")

        self.async_client = AsyncDodoPayments(
            bearer_token=self.dodo_api_key,
            environment="live_mode"
        )

    async def create_checkout_session(self, user_id: str, user_email: str, full_name: str, dodo_customer_id: str, trial_period_days: int) -> str:
        """
        Creates a Dodo Payments checkout session for the pro plan.
        """
        try:
            logger.info(f"Creating Dodo checkout session for user_id: {user_id}")
            if dodo_customer_id:
                customer_payload = {"customer_id": dodo_customer_id}
            else:
                customer_payload = {"email": user_email, "name": full_name}
                
            if trial_period_days > 0:
                subscription_data = {"trial_period_days": trial_period_days}
            else:
                subscription_data = {}

            checkout_session = await self.async_client.checkout_sessions.create(
                customer=customer_payload,
                product_cart=[{"product_id": self.pro_plan_id, "quantity": 1}],
                return_url=f"{self.frontend_url}/account?payment=success",
                # return_url="http://0.0.0.0:8080/account?payment=success",
                metadata={"user_id": user_id},
                subscription_data=subscription_data
            )
            return checkout_session.checkout_url
        except Exception as e:
            logger.error(f"Failed to create Dodo checkout session for user {user_id}: {e}", exc_info=True)
            raise Exception("Could not create payment session.")

    async def get_no_of_trial_days_for_user(self, user: Dict[str, Any]) -> int:
        """
        Returns the number of remaining trial days for a user.
        Ensures timezone awareness and avoids truncating fractional days.
        """
        subscription_tier = user.get("subscription_tier", "free")
        trial_ends_at = user.get("trial_ends_at")

        if subscription_tier != "pro-trial" or not trial_ends_at:
            return 0

        try:
            # Parse ISO string safely with timezone
            trial_end_date = datetime.fromisoformat(trial_ends_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)

            # Compute fractional days accurately
            seconds_left = (trial_end_date - now).total_seconds()
            if seconds_left <= 0:
                return 0

            # Round up to ensure user keeps the full remaining day
            remaining_days = math.ceil(seconds_left / 86400)
            return remaining_days

        except Exception as e:
            # Log and fail gracefully
            logger.warning(f"Failed to parse trial_ends_at for user: {e}")
            return 0

    async def process_webhook(self, payload: bytes, headers: Dict[str, Any]):
        """
        Verifies and processes an incoming webhook from Dodo Payments.
        """
        try:
            wh = Webhook(self.webhook_secret)
            event = wh.verify(payload, headers)
        except Exception as e:
            logger.error(f"Webhook signature verification failed: {e}")
            raise ValueError("Invalid webhook signature.")

        await self._log_webhook_event(event)

        event_type = event.get("type")
        event_data = event.get("data", {})
        
        event_handlers = {
            "subscription.active": self._handle_subscription_activated,
            "subscription.renewed": self._handle_subscription_renewed,
            "subscription.on_hold": self._handle_subscription_on_hold,
            "subscription.cancelled": self._handle_subscription_canceled,
            "subscription.expired": self._handle_subscription_expired,
        }

        handler = event_handlers.get(event_type)
        if handler:
            await handler(event_data)
        else:
            logger.info(f"Ignoring unhandled webhook event type: {event_type}")

        return {"status": "success", "event_type": event_type}

    async def _handle_subscription_activated(self, data: Dict[str, Any]):
        """
        Handles 'subscription.active' by creating a new subscription record.
        """
        user_id = data.get("metadata", {}).get("user_id")
        if not user_id:
            return logger.error("Webhook error: user_id not found in metadata for active subscription.")

        subscription_record = {
            "user_id": user_id,
            "dodo_subscription_id": data.get("subscription_id"), # CORRECTED
            "plan_id": "pro",
            "status": "active",
            "current_period_start": data.get("previous_billing_date"), # CORRECTED
            "current_period_end": data.get("next_billing_date"), # CORRECTED
        }
        self.supabase.table("subscriptions").upsert(subscription_record, on_conflict="dodo_subscription_id").execute()

        (self.supabase.table("user_profiles")
            .update({
                "subscription_tier": "pro",
                "dodo_customer_id": data.get("customer", {}).get("customer_id") # CORRECTED
            })
            .eq("id", user_id)
            .execute())
        logger.info(f"New subscription activated for user {user_id}.")

    async def _handle_subscription_renewed(self, data: Dict[str, Any]):
        """
        Handles 'subscription.renewed' by updating the billing cycle and ensuring status is active.
        """
        dodo_sub_id = data.get("subscription_id") # CORRECTED
        if not dodo_sub_id:
            return logger.error("Webhook error: subscription_id not found in renewed subscription data.")

        update_data = {
            "status": "active",
            "current_period_start": data.get("previous_billing_date"), # CORRECTED
            "current_period_end": data.get("next_billing_date"), # CORRECTED
            "cancel_at_period_end": False,
        }
        (self.supabase.table("subscriptions")
            .update(update_data)
            .eq("dodo_subscription_id", dodo_sub_id)
            .execute())
        logger.info(f"Subscription {dodo_sub_id} was successfully renewed.")

    async def _handle_subscription_on_hold(self, data: Dict[str, Any]):
        """
        Handles 'subscription.on_hold' by marking the subscription as 'past_due'.
        """
        dodo_sub_id = data.get("subscription_id") # CORRECTED
        if not dodo_sub_id:
             return logger.error("Webhook error: subscription_id not found for on_hold event.")

        (self.supabase.table("subscriptions")
            .update({"status": "past_due"})
            .eq("dodo_subscription_id", dodo_sub_id)
            .execute())
        logger.warning(f"Subscription {dodo_sub_id} marked as 'past_due' and is on hold.")

    async def _handle_subscription_canceled(self, data: Dict[str, Any]):
        """
        Handles 'subscription.cancelled'. The user remains 'pro' until the period ends.
        """
        dodo_sub_id = data.get("subscription_id") # CORRECTED
        if not dodo_sub_id:
            return logger.error("Webhook error: Missing subscription_id in canceled event.")

        update_data = { "cancel_at_period_end": True }
        (self.supabase.table("subscriptions")
            .update(update_data)
            .eq("dodo_subscription_id", dodo_sub_id)
            .execute())
        logger.info(f"Subscription {dodo_sub_id} is set to be canceled at the end of the period.")

    async def _handle_subscription_expired(self, data: Dict[str, Any]):
        """
        Handles 'subscription.expired'. This is the definitive end of the subscription.
        """
        dodo_sub_id = data.get("subscription_id") # CORRECTED
        user_id = data.get("metadata", {}).get("user_id")
        if not dodo_sub_id or not user_id:
            return logger.error(f"Webhook error: Missing IDs in expired subscription data.")

        (self.supabase.table("subscriptions")
            .update({"status": "expired"})
            .eq("dodo_subscription_id", dodo_sub_id)
            .execute())

        (self.supabase.table("user_profiles")
            .update({"subscription_tier": "free"})
            .eq("id", user_id)
            .execute())
        logger.info(f"Subscription {dodo_sub_id} for user {user_id} has expired.")

    async def _log_webhook_event(self, event: Dict[str, Any]):
        """Logs the raw webhook event to the database for auditing."""
        try:
            (self.supabase.table("dodo_webhook_events")
                .insert({"event_id": event.get("id"), "event_type": event.get("type"), "payload": event, "status": "processed"})
                .execute())
        except Exception as e:
            logger.error(f"Failed to log webhook event {event.get('id')}: {e}")



    async def create_portal_session(self, dodo_customer_id: str) -> str:
        """
        Creates a Dodo Payments customer portal session.
        """
        if not dodo_customer_id:
            logger.error("Attempted to create portal session without a dodo_customer_id.")
            raise ValueError("Customer does not have a payment history.")

        try:
            logger.info(f"Creating Dodo portal session for customer_id: {dodo_customer_id}")
            
            # This corresponds to the `POST /customers/{customer_id}/portal` endpoint
            portal_session = await self.async_client.customers.customer_portal.create(
                customer_id=dodo_customer_id,
            )
            return portal_session.link
        except Exception as e:
            logger.error(f"Failed to create Dodo portal session for {dodo_customer_id}: {e}", exc_info=True)
            raise Exception("Could not create customer portal session.")