"""
Verification request/response schemas.
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from app.schemas.document import DocumentType


class DocumentVerifyRequest(BaseModel):
    """Request for single document verification."""
    document_type: DocumentType
    image_base64: str


class DocumentVerifyResponse(BaseModel):
    """Response from single document verification."""
    valid: bool
    document_type: DocumentType
    confidence: float
    extracted_data: Dict[str, Any]
    fraud_indicators: List[str]
    message: Optional[str] = None


class WalletData(BaseModel):
    """Wallet information for analysis."""
    address: str
    age_days: Optional[int] = None
    transaction_count: Optional[int] = None
    balance_eth: Optional[float] = None


class CompleteVerificationRequest(BaseModel):
    """Request for complete KYC verification."""
    user_id: str
    
    # Document images (base64 encoded)
    government_id_base64: str
    proof_of_income_base64: str
    proof_of_address_base64: str
    
    # Wallet data
    wallet_data: WalletData


class CompleteVerificationResponse(BaseModel):
    """Response from complete KYC verification."""
    status: str  # "approved", "rejected", "pending"
    credit_score: int
    tier: Optional[str]  # "basic", "standard", "premium", "vip", None if rejected
    
    extracted_data: Dict[str, Any]
    document_scores: Dict[str, float]
    
    fraud_flags: List[str]
    rejection_reasons: List[str]
