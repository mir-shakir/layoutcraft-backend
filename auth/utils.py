"""
Authentication utilities for LayoutCraft
"""
import bcrypt
from typing import Optional
import secrets
import string

def hash_password(password: str) -> str:
    """
    Hash password using bcrypt
    """
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    """
    Verify password against hash
    """
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def generate_random_string(length: int = 32) -> str:
    """
    Generate random string for tokens
    """
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def validate_email(email: str) -> bool:
    """
    Basic email validation
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password: str) -> tuple[bool, str]:
    """
    Validate password strength
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    
    return True, "Password is valid"
