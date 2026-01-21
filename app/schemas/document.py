"""
Document-related schemas.
"""
from enum import Enum
from pydantic import BaseModel
from typing import Optional, Dict, Any


class DocumentType(str, Enum):
    """Types of documents accepted for KYC verification."""
    GOVERNMENT_ID = "government_id"
    PROOF_OF_INCOME = "proof_of_income"
    PROOF_OF_ADDRESS = "proof_of_address"


class DocumentClassification(BaseModel):
    """Result of document classification."""
    document_type: DocumentType
    confidence: float
    is_valid: bool


class ExtractedDocumentData(BaseModel):
    """Data extracted from a document via OCR and NER."""
    raw_text: Optional[str] = None
    
    # Common fields
    name: Optional[str] = None
    address: Optional[str] = None
    date_of_birth: Optional[str] = None
    
    # Government ID specific
    id_number: Optional[str] = None
    id_type: Optional[str] = None  # e.g., "Philippine National ID", "Driver's License"
    expiry_date: Optional[str] = None
    
    # Proof of Income specific
    employer_name: Optional[str] = None
    monthly_income: Optional[float] = None
    employment_type: Optional[str] = None  # "permanent", "contract", "self-employed"
    
    # Proof of Address specific
    utility_type: Optional[str] = None  # "electric", "water", "internet"
    billing_date: Optional[str] = None
    
    # Additional extracted fields
    extra: Dict[str, Any] = {}
