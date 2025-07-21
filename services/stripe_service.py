"""
Stripe service for LayoutCraft subscription management
"""
import stripe
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
from fastapi import HTTPException
from supabase import Client

logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

class StripeService:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
        self.frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
        
        # Price IDs for different plans
        self.price_ids = {
            "pro_monthly": os.getenv("STRIPE_PRICE_PRO_MONTHLY"),
            "pro_yearly": os.getenv("STRIPE_PRICE_PRO_YEARLY"),
            "enterprise_monthly": os.getenv("STRIPE_PRICE_ENTERPRISE_MONTHLY"),
            "enterprise_yearly": os.getenv("STRIPE_PRICE_ENTERPRISE_YEARLY")
        }
    
    async def create_customer(self, user_id: str, email: str, name: str = None) -> Optional[str]:
        """
        Create a Stripe customer for a user
        """
        try:
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata={"user_id": user_id}
            )
            
            # Update user profile with Stripe customer ID
            self.supabase.table("user_profiles").update({
                "stripe_customer_id": customer.id
            }).eq("id", user_id).execute()
            
            logger.info(f"Created Stripe customer {customer.id} for user {user_id}")
            return customer.id
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating customer: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error creating customer: {str(e)}")
            return None
    
    async def get_or_create_customer(self, user_id: str, email: str, name: str = None) -> Optional[str]:
        """
        Get existing customer or create new one
        """
        try:
            # Check if user already has a Stripe customer ID
            user_response = self.supabase.table("user_profiles").select("stripe_customer_id").eq("id", user_id).single().execute()
            
            if user_response.data and user_response.data.get("stripe_customer_id"):
                return user_response.data["stripe_customer_id"]
            
            # Create new customer
            return await self.create_customer(user_id, email, name)
            
        except Exception as e:
            logger.error(f"Error getting or creating customer: {str(e)}")
            return None
    
    async def create_checkout_session(self, user_id: str, price_id: str, plan_type: str) -> Optional[str]:
        """
        Create Stripe checkout session for subscription
        """
        try:
            # Get user data
            user_response = self.supabase.table("user_profiles").select("*").eq("id", user_id).single().execute()
            if not user_response.data:
                raise HTTPException(status_code=404, detail="User not found")
            
            user_data = user_response.data
            
            # Get or create customer
            customer_id = await self.get_or_create_customer(
                user_id, 
                user_data["email"], 
                user_data.get("full_name")
            )
            
            if not customer_id:
                raise HTTPException(status_code=500, detail="Failed to create customer")
            
            # Create checkout session
            session = stripe.checkout.Session.create(
                customer=customer_id,
                payment_method_types=['card'],
                mode='subscription',
                line_items=[{
                    'price': price_id,
                    'quantity': 1,
                }],
                metadata={
                    "user_id": user_id,
                    "plan_type": plan_type
                },
                success_url=f"{self.frontend_url}/dashboard?session_id={{CHECKOUT_SESSION_ID}}&success=true",
                cancel_url=f"{self.frontend_url}/pricing?canceled=true",
                automatic_tax={'enabled': True},
                tax_id_collection={'enabled': True},
                customer_update={
                    'address': 'auto',
                    'name': 'auto'
                }
            )
            
            logger.info(f"Created checkout session {session.id} for user {user_id}")
            return session.url
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating checkout session: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Payment error: {str(e)}")
        except Exception as e:
            logger.error(f"Error creating checkout session: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create checkout session")
    
    async def create_portal_session(self, user_id: str) -> Optional[str]:
        """
        Create Stripe customer portal session
        """
        try:
            # Get customer ID
            user_response = self.supabase.table("user_profiles").select("stripe_customer_id").eq("id", user_id).single().execute()
            
            if not user_response.data or not user_response.data.get("stripe_customer_id"):
                raise HTTPException(status_code=404, detail="No subscription found")
            
            customer_id = user_response.data["stripe_customer_id"]
            
            # Create portal session
            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=f"{self.frontend_url}/dashboard"
            )
            
            logger.info(f"Created portal session for user {user_id}")
            return session.url
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating portal session: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Portal error: {str(e)}")
        except Exception as e:
            logger.error(f"Error creating portal session: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create portal session")
    
    async def handle_webhook(self, payload: bytes, signature: str) -> Dict[str, Any]:
        """
        Handle Stripe webhook events
        """
        try:
            event = stripe.Webhook.construct_event(
                payload, signature, self.webhook_secret
            )
            
            logger.info(f"Processing webhook event: {event['type']}")
            
            # Handle different event types
            if event['type'] == 'checkout.session.completed':
                await self._handle_checkout_completed(event['data']['object'])
            
            elif event['type'] == 'invoice.payment_succeeded':
                await self._handle_payment_succeeded(event['data']['object'])
            
            elif event['type'] == 'invoice.payment_failed':
                await self._handle_payment_failed(event['data']['object'])
            
            elif event['type'] == 'customer.subscription.updated':
                await self._handle_subscription_updated(event['data']['object'])
            
            elif event['type'] == 'customer.subscription.deleted':
                await self._handle_subscription_canceled(event['data']['object'])
            
            return {"status": "success", "event_type": event['type']}
            
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Webhook signature verification failed: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid signature")
        except Exception as e:
            logger.error(f"Webhook processing error: {str(e)}")
            raise HTTPException(status_code=500, detail="Webhook processing failed")
    
    async def _handle_checkout_completed(self, session):
        """Handle successful checkout completion"""
        try:
            user_id = session['metadata']['user_id']
            plan_type = session['metadata']['plan_type']
            
            # Get subscription details
            subscription = stripe.Subscription.retrieve(session['subscription'])
            
            # Update user subscription in database
            await self._update_user_subscription(user_id, subscription, plan_type)
            
            logger.info(f"Checkout completed for user {user_id}, plan: {plan_type}")
            
        except Exception as e:
            logger.error(f"Error handling checkout completion: {str(e)}")
    
    async def _handle_payment_succeeded(self, invoice):
        """Handle successful payment"""
        try:
            subscription = stripe.Subscription.retrieve(invoice['subscription'])
            customer = stripe.Customer.retrieve(subscription['customer'])
            user_id = customer['metadata']['user_id']
            
            # Reset usage count for new billing period
            self.supabase.table("user_profiles").update({
                "usage_count": 0,
                "usage_reset_date": datetime.now().isoformat()
            }).eq("id", user_id).execute()
            
            logger.info(f"Payment succeeded for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error handling payment success: {str(e)}")
    
    async def _handle_payment_failed(self, invoice):
        """Handle failed payment"""
        try:
            subscription = stripe.Subscription.retrieve(invoice['subscription'])
            customer = stripe.Customer.retrieve(subscription['customer'])
            user_id = customer['metadata']['user_id']
            
            # Update subscription status
            self.supabase.table("subscriptions").update({
                "status": "past_due"
            }).eq("user_id", user_id).execute()
            
            logger.info(f"Payment failed for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error handling payment failure: {str(e)}")
    
    async def _handle_subscription_updated(self, subscription):
        """Handle subscription updates"""
        try:
            customer = stripe.Customer.retrieve(subscription['customer'])
            user_id = customer['metadata']['user_id']
            
            # Determine plan type from price ID
            price_id = subscription['items']['data'][0]['price']['id']
            plan_type = self._get_plan_type_from_price(price_id)
            
            # Update user subscription
            await self._update_user_subscription(user_id, subscription, plan_type)
            
            logger.info(f"Subscription updated for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error handling subscription update: {str(e)}")
    
    async def _handle_subscription_canceled(self, subscription):
        """Handle subscription cancellation"""
        try:
            customer = stripe.Customer.retrieve(subscription['customer'])
            user_id = customer['metadata']['user_id']
            
            # Downgrade to free tier
            self.supabase.table("user_profiles").update({
                "subscription_tier": "free"
            }).eq("id", user_id).execute()
            
            # Update subscription status
            self.supabase.table("subscriptions").update({
                "status": "canceled"
            }).eq("user_id", user_id).execute()
            
            logger.info(f"Subscription canceled for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error handling subscription cancellation: {str(e)}")
    
    async def _update_user_subscription(self, user_id: str, subscription, plan_type: str):
        """Update user subscription in database"""
        try:
            # Update user profile
            self.supabase.table("user_profiles").update({
                "subscription_tier": plan_type.split('_')[0]  # Extract 'pro' or 'enterprise'
            }).eq("id", user_id).execute()
            
            # Upsert subscription record
            subscription_data = {
                "user_id": user_id,
                "stripe_subscription_id": subscription['id'],
                "stripe_customer_id": subscription['customer'],
                "plan_type": plan_type,
                "status": subscription['status'],
                "current_period_start": datetime.fromtimestamp(subscription['current_period_start']).isoformat(),
                "current_period_end": datetime.fromtimestamp(subscription['current_period_end']).isoformat()
            }
            
            # Check if subscription exists
            existing = self.supabase.table("subscriptions").select("id").eq("user_id", user_id).execute()
            
            if existing.data:
                # Update existing
                self.supabase.table("subscriptions").update(subscription_data).eq("user_id", user_id).execute()
            else:
                # Insert new
                self.supabase.table("subscriptions").insert(subscription_data).execute()
            
        except Exception as e:
            logger.error(f"Error updating user subscription: {str(e)}")
    
    def _get_plan_type_from_price(self, price_id: str) -> str:
        """Get plan type from Stripe price ID"""
        for plan, pid in self.price_ids.items():
            if pid == price_id:
                return plan
        return "unknown"
    
    async def get_subscription_details(self, user_id: str) -> Dict[str, Any]:
        """Get detailed subscription information for a user"""
        try:
            # Get subscription from database
            subscription_response = self.supabase.table("subscriptions").select("*").eq("user_id", user_id).single().execute()
            
            if not subscription_response.data:
                return {"tier": "free", "status": "inactive"}
            
            subscription_data = subscription_response.data
            
            # Get Stripe subscription details
            stripe_subscription = stripe.Subscription.retrieve(subscription_data["stripe_subscription_id"])
            
            return {
                "tier": subscription_data["plan_type"].split('_')[0],
                "status": subscription_data["status"],
                "current_period_start": subscription_data["current_period_start"],
                "current_period_end": subscription_data["current_period_end"],
                "cancel_at_period_end": stripe_subscription.get("cancel_at_period_end", False),
                "stripe_subscription_id": subscription_data["stripe_subscription_id"]
            }
            
        except Exception as e:
            logger.error(f"Error getting subscription details: {str(e)}")
            return {"tier": "free", "status": "error"}
