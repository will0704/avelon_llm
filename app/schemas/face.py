"""
Face recognition schemas.
"""
from pydantic import BaseModel
from typing import Optional


class FaceVerifyResponse(BaseModel):
    """Response from face matching endpoint."""
    passed: bool            # True if selfie matches government ID
    score: float            # Similarity score 0-1 (higher = more similar)
    confidence: float       # Model confidence in the result 0-1
    message: Optional[str] = None  # Human-readable explanation on failure
