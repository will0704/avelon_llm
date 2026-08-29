"""Schemas package - Pydantic models for request/response validation."""

from app.schemas.document import (
    DocumentType,
    DocumentClassification,
    ExtractedDocumentData,
)
from app.schemas.score import ScoreBreakdown, LoanHistory
from app.schemas.verification import (
    DocumentVerifyRequest,
    DocumentVerifyResponse,
    WalletData,
    CompleteVerificationRequest,
    CompleteVerificationResponse,
)
from app.schemas.fraud import FraudFlag, FraudFlagType, FraudResult
from app.schemas.volatility import PriceBand, LiquidationRisk, VolatilityResponse

__all__ = [
    # Document
    "DocumentType",
    "DocumentClassification",
    "ExtractedDocumentData",
    # Score
    "ScoreBreakdown",
    "LoanHistory",
    # Verification
    "DocumentVerifyRequest",
    "DocumentVerifyResponse",
    "WalletData",
    "CompleteVerificationRequest",
    "CompleteVerificationResponse",
    # Fraud
    "FraudFlag",
    "FraudFlagType",
    "FraudResult",
]
