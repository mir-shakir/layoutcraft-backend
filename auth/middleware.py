"""
Authentication middleware for LayoutCraft with local JWT validation
"""
import jwt
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import os
from supabase import create_client, Client
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# JWT settings
JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_AUDIENCE = "authenticated"

security = HTTPBearer()

class AuthMiddleware:
    def __init__(self):
        # Initialize Supabase client with environment variables
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not supabase_url:
            raise ValueError("SUPABASE_URL environment variable is required")
        if not supabase_key:
            raise ValueError("SUPABASE_SERVICE_KEY environment variable is required")
        if not JWT_SECRET:
            raise ValueError("SUPABASE_JWT_SECRET environment variable is required")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        print(f"✅ Supabase client initialized with local JWT validation")
    
    async def verify_token(self, credentials: HTTPAuthorizationCredentials) -> dict:
        """
        Verify JWT token locally without round-trip to Supabase
        """
        try:
            # Decode and verify JWT token locally
            payload = jwt.decode(
                credentials.credentials,
                JWT_SECRET,
                algorithms=[JWT_ALGORITHM],
                audience=JWT_AUDIENCE
            )
            
            # Extract user information from token
            user_id = payload.get("sub")
            email = payload.get("email")
            
            if not user_id or not email:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: missing user information"
                )
            
            # Get user profile from database
            user_response = self.supabase.table("user_profiles").select("*").eq("id", user_id).single().execute()
            
            if not user_response.data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User profile not found"
                )
            
            return {
                "id": user_response.data["id"],
                "email": user_response.data["email"],
                "subscription_tier": user_response.data["subscription_tier"],
                "usage_count": user_response.data["usage_count"],
                "usage_reset_date": user_response.data["usage_reset_date"],
                "pro_usage_count": user_response.data.get("pro_usage_count", 0),
                "edit_usage_count": user_response.data.get("edit_usage_count", 0)
            }
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidAudienceError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token audience"
            )
        except jwt.InvalidSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token signature"
            )
        except jwt.JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication failed: {str(e)}"
            )
    
    def create_access_token(self, user_id: str, email: str) -> str:
        """
        Create JWT access token (for custom auth flows)
        """
        expire = datetime.utcnow() + timedelta(hours=24)
        payload = {
            "sub": user_id,
            "email": email,
            "aud": JWT_AUDIENCE,
            "exp": expire,
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    async def get_user_profile(self, user_id: str) -> Optional[dict]:
        """
        Get user profile from database
        """
        try:
            response = self.supabase.table("user_profiles").select("*").eq("id", user_id).single().execute()
            return response.data
        except Exception as e:
            print(f"Error getting user profile: {e}")
            return None
    
    async def update_user_usage(self, user_id: str, increment: int = 1) -> bool:
        """
        Update user usage count
        """
        try:
            # Get current usage
            current_response = self.supabase.table("user_profiles").select("usage_count, usage_reset_date").eq("id", user_id).single().execute()
            
            if not current_response.data:
                return False
            
            current_usage = current_response.data["usage_count"]
            reset_date = datetime.fromisoformat(current_response.data["usage_reset_date"].replace("Z", "+00:00"))
            
            # Check if we need to reset usage (monthly reset)
            now = datetime.now().replace(tzinfo=reset_date.tzinfo)
            if now >= reset_date:
                # Reset usage and set new reset date
                new_reset_date = now.replace(day=1) + timedelta(days=32)
                new_reset_date = new_reset_date.replace(day=1)  # First day of next month
                
                self.supabase.table("user_profiles").update({
                    "usage_count": increment,
                    "usage_reset_date": new_reset_date.isoformat()
                }).eq("id", user_id).execute()
            else:
                # Just increment usage
                self.supabase.table("user_profiles").update({
                    "usage_count": current_usage + increment
                }).eq("id", user_id).execute()
            
            return True
            
        except Exception as e:
            print(f"Error updating user usage: {e}")
            return False

# Global auth middleware instance - will be initialized when imported
auth_middleware = None

def get_auth_middleware():
    """Get or create auth middleware instance"""
    global auth_middleware
    if auth_middleware is None:
        auth_middleware = AuthMiddleware()
    return auth_middleware
