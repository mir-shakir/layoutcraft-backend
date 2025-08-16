import os
import logging
import httpx
import hmac
import hashlib
import json
from typing import Optional, Dict, Any
from supabase import Client

logger = logging.getLogger(__name__)

PADDLE_API_KEY = os.getenv("PADDLE_API_KEY")
PADDLE_WEBHOOK_SECRET = os.getenv("PADDLE_WEBHOOK_SECRET")
PADDLE_API_URL = "https://sandbox-api.paddle.com/transactions"

PADDLE_PRICE_IDS = {
    "pro": os.getenv("PADDLE_PRICE_ID_PRO"),
    "max": os.getenv("PADDLE_PRICE_ID_MAX"),
}

class PaddleService:
    def __init__(self):
        if not PADDLE_API_KEY:
            raise ValueError("PADDLE_API_KEY is required.")
        logger.info("✅ Paddle Service initialized for direct API calls in SANDBOX mode.")

    def create_transaction_for_checkout(self, plan_id: str, user_id: str, user_email: str) -> str:
        """
        Create a Paddle transaction and return the checkout URL (not just transaction ID)
        """
        price_id = PADDLE_PRICE_IDS.get(plan_id)
        if not price_id:
            raise ValueError(f"Invalid plan ID: {plan_id}")

        headers = {
            "Authorization": f"Bearer {PADDLE_API_KEY}",
            "Content-Type": "application/json",
        }

        # Create transaction with custom_data containing user_id
        payload = {
            "items": [{"price_id": price_id, "quantity": 1}],
            "customer": {"email": user_email},
            "custom_data": {
                "user_id": user_id 
            }
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(PADDLE_API_URL, headers=headers, json=payload)
                response.raise_for_status()

            data = response.json()
            
            # Extract checkout URL from response
            transaction_data = data.get('data', {})
            checkout_data = transaction_data.get('checkout', {})
            checkout_url = checkout_data.get('url')
            
            if not checkout_url:
                # Fallback: try to get transaction ID and construct checkout URL
                transaction_id = transaction_data.get('id')
                if transaction_id:
                    # Paddle checkout URLs typically follow this pattern
                    checkout_url = f"https://checkout.paddle.com/transaction/{transaction_id}"
                else:
                    raise Exception("Paddle did not return a checkout URL or transaction ID.")

            transaction_id = transaction_data.get('id', 'unknown')
            logger.info(f"Created Paddle transaction {transaction_id} for user {user_id}")
            return transaction_id
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error creating Paddle transaction: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Paddle API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error creating Paddle transaction: {e}", exc_info=True)
            raise

    def verify_and_parse_webhook(self, request_body: bytes, signature_header: str) -> Dict[str, Any]:
        """
        Verify Paddle webhook signature and parse the payload
        """
        if not PADDLE_WEBHOOK_SECRET:
            raise ValueError("PADDLE_WEBHOOK_SECRET is not configured.")
        
        try:
            # Parse the signature header: "ts=timestamp;h1=signature"
            signature_parts = signature_header.split(';')
            if len(signature_parts) != 2:
                raise ValueError("Invalid signature header format")
                
            ts_part, h1_part = signature_parts
            
            # Extract timestamp and signature
            if not ts_part.startswith('ts=') or not h1_part.startswith('h1='):
                raise ValueError("Invalid signature header components")
                
            ts = ts_part[3:]  # Remove 'ts=' prefix
            h1 = h1_part[3:]  # Remove 'h1=' prefix

            # Create the signed payload
            signed_payload = f"{ts}:{request_body.decode('utf-8')}"
            
            # Compute the expected signature
            computed_signature = hmac.new(
                key=PADDLE_WEBHOOK_SECRET.encode('utf-8'),
                msg=signed_payload.encode('utf-8'),
                digestmod=hashlib.sha256
            ).hexdigest()

            # Compare signatures securely
            if not hmac.compare_digest(h1, computed_signature):
                raise ValueError("Webhook signature is invalid.")
            
            logger.info("✅ Paddle webhook signature verified successfully.")
            return json.loads(request_body)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in webhook payload: {e}")
            raise ValueError("Invalid JSON in webhook payload.")
        except Exception as e:
            logger.error(f"Webhook verification failed: {e}")
            raise ValueError(f"Webhook verification failed: {str(e)}")

    async def process_webhook_data(self, event_data: Dict[str, Any], supabase_client: Client) -> None:
        """
        Process verified webhook data and update database
        """
        try:
            event_type = event_data.get('event_type')
            if event_type not in ["subscription.created", "subscription.updated", "subscription.canceled"]:
                logger.info(f"Ignoring webhook event type: {event_type}")
                return

            logger.info(f"Processing Paddle webhook event: {event_type}")
            
            subscription_payload = event_data.get('data', {})
            
            # Validate essential data exists
            if not subscription_payload:
                logger.error("Webhook error: No subscription data found.")
                return
            
            # Get user_id from custom_data
            custom_data = subscription_payload.get("custom_data", {})
            user_id = custom_data.get("user_id") if isinstance(custom_data, dict) else None
            
            if not user_id:
                logger.error("Webhook error: user_id not found in custom_data.")
                return

            # Safely extract plan name
            plan_name = self._extract_plan_name(subscription_payload)
            if not plan_name:
                logger.error("Webhook error: Could not extract plan name.")
                return

            # Look up plan in database
            plan_id = self._get_plan_id(supabase_client, plan_name)
            if not plan_id:
                logger.error(f"Webhook error: Plan '{plan_name}' not found in database.")
                return

            # Prepare subscription record with safe extraction
            subscription_record = {
                "user_id": user_id,
                "plan_id": plan_id,
                "paddle_subscription_id": subscription_payload.get("id"),
                "paddle_customer_id": subscription_payload.get("customer_id"),
                "status": subscription_payload.get("status"),
                "cancel_at_period_end": self._get_cancel_at_period_end(subscription_payload),
                "renews_at": subscription_payload.get("next_billed_at"),
                "ends_at": subscription_payload.get("canceled_at"),
                "updated_at": subscription_payload.get("updated_at")
            }

            # Validate required fields
            if not subscription_record["paddle_subscription_id"]:
                logger.error("Webhook error: paddle_subscription_id is missing.")
                return

            # Update subscription in database
            try:
                supabase_client.table("subscriptions").upsert(
                    subscription_record, 
                    on_conflict="paddle_subscription_id"
                ).execute()
            except Exception as e:
                logger.error(f"Webhook error: Failed to upsert subscription - {e}")
                return
            
            # Determine new tier
            subscription_status = subscription_payload.get("status", "").lower()
            new_tier = plan_name if subscription_status in ["trialing", "active"] else "free"
            
            # Update user profile
            try:
                supabase_client.table("user_profiles").update({
                    "subscription_tier": new_tier,
                    "paddle_customer_id": subscription_payload.get("customer_id")
                }).eq("id", user_id).execute()
            except Exception as e:
                logger.error(f"Webhook error: Failed to update user profile - {e}")
                return

            logger.info(f"Successfully processed webhook for user {user_id}. New tier: {new_tier}")
            
        except Exception as e:
            logger.error(f"Unexpected error processing webhook: {e}", exc_info=True)
            raise

    def _extract_plan_name(self, subscription_payload: Dict[str, Any]) -> Optional[str]:
        """
        Safely extract plan name from subscription payload
        """
        try:
            items = subscription_payload.get("items", [])
            if not items or len(items) == 0:
                return None
                
            first_item = items[0]
            # price_info = first_item.get("price", {})
            product_info = first_item.get("product", {})
            product_name = product_info.get("name")
            
            if not product_name:
                return None
                
            return product_name.lower().strip()
            
        except (KeyError, IndexError, AttributeError) as e:
            logger.error(f"Error extracting plan name: {e}")
            return None

    def _get_plan_id(self, supabase_client: Client, plan_name: str) -> Optional[str]:
        """
        Get plan ID from database by plan name
        """
        try:
            plan_response = (
                supabase_client.table("plans")
                .select("id")
                .eq("plan_name", plan_name)
                .single()
                .execute()
            )

            if plan_response.data:
                return plan_response.data['id']
            return None

        except Exception as e:
            logger.error(f"Database error while fetching plan: {e}")
            return None

    def _get_cancel_at_period_end(self, subscription_payload: Dict[str, Any]) -> bool:
        """
        Safely extract cancel_at_period_end status
        """
        try:
            scheduled_change = subscription_payload.get("scheduled_change", {})
            if isinstance(scheduled_change, dict):
                return scheduled_change.get("action") == "cancel"
            return False
        except Exception:
            return False
            if plan_response.data:
                return plan_response.data['id']
            return None

        except Exception as e:
            logger.error(f"Database error while fetching plan: {e}")
            return None

    def _get_cancel_at_period_end(self, subscription_payload: Dict[str, Any]) -> bool:
        """
        Safely extract cancel_at_period_end status
        """
        try:
            scheduled_change = subscription_payload.get("scheduled_change", {})
            if isinstance(scheduled_change, dict):
                return scheduled_change.get("action") == "cancel"
            return False
        except Exception:
            return False