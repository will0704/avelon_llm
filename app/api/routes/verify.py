"""
Document verification endpoints.
Handles document upload, classification, OCR, and fraud detection.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import Optional

from app.schemas.document import DocumentType
from app.schemas.verification import (
    DocumentVerifyRequest,
    DocumentVerifyResponse,
    CompleteVerificationRequest,
    CompleteVerificationResponse,
)
from app.api.dependencies import verify_api_key

router = APIRouter()


@router.post("/verify/document", response_model=DocumentVerifyResponse)
async def verify_document(
    document_type: DocumentType,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
):
    """
    Verify a single document.
    
    - Classifies document type
    - Extracts text via OCR
    - Extracts structured data via NER
    - Checks for fraud indicators
    
    Args:
        document_type: Expected document type (government_id, proof_of_income, proof_of_address)
        file: Uploaded document image
        
    Returns:
        Verification result with extracted data and confidence scores
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image (JPEG, PNG)."
        )
    
    # TODO: Implement actual verification logic
    # For now, return a stub response
    return DocumentVerifyResponse(
        valid=True,
        document_type=document_type,
        confidence=0.0,  # Will be populated by AI model
        extracted_data={},
        fraud_indicators=[],
        message="Document verification not yet implemented. This is a stub response.",
    )


@router.post("/verify/complete", response_model=CompleteVerificationResponse)
async def complete_verification(
    request: CompleteVerificationRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Complete KYC verification with all documents and wallet data.
    
    - Verifies all three required documents
    - Analyzes wallet history
    - Calculates credit score
    - Assigns tier
    
    Args:
        request: Contains all document data and wallet information
        
    Returns:
        Complete verification result with credit score and tier
    """
    # TODO: Implement complete verification logic
    # For now, return a stub response
    return CompleteVerificationResponse(
        status="pending",
        credit_score=0,
        tier=None,
        extracted_data={
            "name": None,
            "address": None,
            "income": None,
        },
        document_scores={
            "government_id": 0.0,
            "proof_of_income": 0.0,
            "proof_of_address": 0.0,
        },
        fraud_flags=[],
        rejection_reasons=["Verification not yet implemented. This is a stub response."],
    )
