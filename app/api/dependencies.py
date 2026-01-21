"""
API Dependencies - shared dependencies for route handlers.
"""
from fastapi import Header, HTTPException

from app.config import get_settings


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """
    Verify the API key from request header.
    Used to authenticate requests from the backend service.
    """
    settings = get_settings()
    
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return x_api_key
