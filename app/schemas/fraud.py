"""
Fraud detection schemas.
"""
from enum import Enum
from typing import Literal, List, Optional
from pydantic import BaseModel


class FraudFlagType(str, Enum):
    """Types of fraud indicators."""
    METADATA_MANIPULATION = "metadata_manipulation"
    QUALITY_INCONSISTENCY = "quality_inconsistency"
    TEXT_MISMATCH = "text_mismatch"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    DATE_INCONSISTENCY = "date_inconsistency"
    FORMAT_ANOMALY = "format_anomaly"


class FraudFlag(BaseModel):
    """Individual fraud indicator."""
    flag_type: FraudFlagType
    description: str
    severity: Literal["low", "medium", "high"]
    confidence: float  # 0.0 to 1.0


class FraudResult(BaseModel):
    """Complete fraud analysis result."""
    is_suspicious: bool
    fraud_probability: float  # 0.0 to 1.0
    flags: List[FraudFlag] = []
    recommendation: str  # "approve", "review", "reject"
    details: Optional[str] = None
