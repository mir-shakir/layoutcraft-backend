"""
Authentication routes for LayoutCraft
"""
import os
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, Union
import logging
from datetime import datetime, timedelta

from auth.middleware import get_auth_middleware
from auth.dependencies import get_current_user, security
from auth.utils import validate_email, validate_password
from models.user import UserCreate, UserResponse, UserUpdate
import asyncio
from asyncio import TimeoutError as AsyncTimeoutError


router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = logging.getLogger(__name__)

class RegistrationPendingResponse(BaseModel):
    message: str
    user_id: str
    email_confirmation_required: bool = True

class RegistrationSuccessResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

RegistrationResponse = Union[RegistrationSuccessResponse, RegistrationPendingResponse]

@router.post("/register", response_model=RegistrationResponse)
async def register(request: RegisterRequest):
    """
    Register a new user with timeout handling
    """
    auth_middleware = get_auth_middleware()
    
    try:
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
        # Validate inputs
        if not validate_email(request.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email format"
            )
        
        is_valid, message = validate_password(request.password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        # Register user with Supabase with timeout
        try:
            auth_response = await asyncio.wait_for(
            asyncio.to_thread(
                auth_middleware.supabase.auth.sign_up,
                {
                    "email": request.email,
                    "password": request.password,
                    "options": {
                        "data": {
                            "full_name": request.full_name
                        },
                        "email_redirect_to": f"{frontend_url}/auth/callback"
                    }
                }
            ),
            timeout=120
        )
        except AsyncTimeoutError:
            logger.error(f"Registration timeout for email: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="Registration request timed out. Please try again."
            )
        
        if auth_response.user is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Registration failed. Email might already be registered."
            )
        
        # Handle email confirmation
        if not auth_response.user.email_confirmed_at:
            logger.info(f"User registered but needs email confirmation: {request.email}")
            return RegistrationPendingResponse(
                message="Registration successful. Please check your email to confirm your account.",
                user_id=auth_response.user.id,
                email_confirmation_required=True
            )
        
        # Create access token for confirmed users
        access_token = auth_middleware.create_access_token(
            auth_response.user.id,
            auth_response.user.email
        )
        
        # Get user profile
        user_profile = await auth_middleware.get_user_profile(auth_response.user.id)
        
        if not user_profile:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user profile"
            )
        
        logger.info(f"User registered and confirmed successfully: {request.email}")
        
        return RegistrationSuccessResponse(
            access_token=access_token,
            user=UserResponse(**user_profile)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )
@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Login user and return access token
    """
    auth_middleware = get_auth_middleware()
    
    try:
        # Authenticate with Supabase
        auth_response = auth_middleware.supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
        
        if auth_response.user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Create access token
        access_token = auth_middleware.create_access_token(
            auth_response.user.id,
            auth_response.user.email
        )
        
        # Get user profile
        user_profile = await auth_middleware.get_user_profile(auth_response.user.id)
        
        if not user_profile:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="User profile not found"
            )
        
        logger.info(f"User logged in successfully: {request.email}")
        
        return LoginResponse(
            access_token=access_token,
            user=UserResponse(**user_profile)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )


@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Logout user (invalidate token)
    """
    try:
        auth_middleware = get_auth_middleware()
        # Sign out from Supabase
        auth_middleware.supabase.auth.sign_out()
        
        logger.info("User logged out successfully")
        
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        # Don't raise error for logout, just log it
        return {"message": "Logged out successfully"}

@router.get("/profile", response_model=UserResponse)
async def get_profile(current_user: dict = Depends(get_current_user)):
    """
    Get current user profile
    """
    try:
        auth_middleware = get_auth_middleware()
        user_profile = await auth_middleware.get_user_profile(current_user["id"])
        
        if not user_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )
        
        return UserResponse(**user_profile)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get profile error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user profile"
        )

@router.put("/profile", response_model=UserResponse)
async def update_profile(
    update_data: UserUpdate,
    current_user: dict = Depends(get_current_user)
):
    """
    Update current user profile
    """
    try:
        # Prepare update data
        update_fields = {}
        if update_data.full_name is not None:
            update_fields["full_name"] = update_data.full_name
        if update_data.avatar_url is not None:
            update_fields["avatar_url"] = update_data.avatar_url
        
        if not update_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )
        
        auth_middleware = get_auth_middleware()
        # Update user profile
        response = auth_middleware.supabase.table("user_profiles").update(
            update_fields
        ).eq("id", current_user["id"]).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )
        
        logger.info(f"User profile updated: {current_user['id']}")
        
        return UserResponse(**response.data[0])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update profile error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile"
        )

@router.post("/password-reset")
async def request_password_reset(request: PasswordResetRequest):
    """
    Request password reset
    """
    try:
        auth_middleware = get_auth_middleware()
        # Send password reset email via Supabase
        auth_middleware.supabase.auth.reset_password_email(request.email)
        
        logger.info(f"Password reset requested for: {request.email}")
        
        return {"message": "Password reset email sent if account exists"}
        
    except Exception as e:
        logger.error(f"Password reset error: {str(e)}")
        # Don't reveal if email exists or not
        return {"message": "Password reset email sent if account exists"}

@router.delete("/account")
async def delete_account(current_user: dict = Depends(get_current_user)):
    """
    Delete current user account
    """
    try:
        auth_middleware = get_auth_middleware()
        # Delete user profile (cascades to related tables)
        auth_middleware.supabase.table("user_profiles").delete().eq("id", current_user["id"]).execute()
        
        # Delete from Supabase auth
        auth_middleware.supabase.auth.admin.delete_user(current_user["id"])
        
        logger.info(f"User account deleted: {current_user['id']}")
        
        return {"message": "Account deleted successfully"}
        
    except Exception as e:
        logger.error(f"Delete account error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account"
        )
