"""
Subscription model for LayoutCraft
"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum

class PlanType(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    CANCELED = "canceled"
    PAST_DUE = "past_due"
    UNPAID = "unpaid"

class Subscription(BaseModel):
    id: str
    user_id: str
    stripe_subscription_id: Optional[str] = None
    stripe_customer_id: Optional[str] = None
    plan_type: PlanType
    status: SubscriptionStatus
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

class SubscriptionCreate(BaseModel):
    user_id: str
    plan_type: PlanType
    stripe_subscription_id: Optional[str] = None
    stripe_customer_id: Optional[str] = None

class SubscriptionUpdate(BaseModel):
    plan_type: Optional[PlanType] = None
    status: Optional[SubscriptionStatus] = None
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
