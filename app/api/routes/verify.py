"""
Document verification endpoints.
Handles document upload, classification, OCR, and fraud detection.
"""
import base64
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

from app.schemas.document import DocumentType
from app.schemas.verification import (
    DocumentVerifyResponse,
    CompleteVerificationRequest,
    CompleteVerificationResponse,
)
from app.api.dependencies import verify_api_key
from app.config import get_settings
from app.services.preprocessing_service import get_preprocessing_service
from app.services.classifier_service import get_classifier_service
from app.services.entity_extractor_service import get_entity_extractor_service
from app.services.fraud_detector_service import get_fraud_detector_service
from app.services.scorer_service import get_scorer_service

logger = logging.getLogger(__name__)
router = APIRouter()

settings = get_settings()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _verify_single_document(
    image_bytes: bytes,
    expected_type: DocumentType,
) -> dict:
    """
    Run the full single-document pipeline and return a result dict.

    Pipeline:
        1. Preprocessing  – validate image & quality check
        2. Classification – confirm document type matches expected
        3. Entity extraction – OCR + NER
        4. Fraud detection – metadata / quality / text consistency
    """
    preprocessing = get_preprocessing_service()
    classifier = get_classifier_service()
    extractor = get_entity_extractor_service()
    fraud_detector = get_fraud_detector_service()

    # 1. Validate image
    is_valid, error_msg = preprocessing.validate_image(image_bytes)
    if not is_valid:
        return {
            "valid": False,
            "document_type": expected_type,
            "confidence": 0.0,
            "extracted_data": {},
            "fraud_indicators": [],
            "message": f"Image validation failed: {error_msg}",
        }

    # 2. Classify document
    classified_type, confidence = classifier.classify(image_bytes)

    type_match = classified_type == expected_type
    effective_confidence = confidence if type_match else confidence * 0.5

    # 3. Extract entities
    extracted = extractor.extract_from_image(image_bytes)
    extracted_dict = extracted.model_dump(exclude_none=True)

    # 4. Fraud analysis
    fraud_result = fraud_detector.analyze(
        image_bytes=image_bytes,
        extracted_data=extracted,
        document_type=expected_type,
    )

    fraud_indicators = [flag.description for flag in fraud_result.flags]

    # Determine overall validity
    is_document_valid = (
        effective_confidence >= settings.confidence_threshold
        and fraud_result.recommendation != "reject"
    )

    message = None
    if not type_match:
        message = (
            f"Classified as {classified_type.value} but expected {expected_type.value}. "
            "Confidence has been penalised."
        )
    elif fraud_result.recommendation == "review":
        message = "Document flagged for manual review due to fraud indicators."
    elif fraud_result.recommendation == "reject":
        message = "Document rejected due to high fraud probability."

    # Sanitize values for JSON serialization (EasyOCR returns numpy types)
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        return obj

    return {
        "valid": is_document_valid,
        "document_type": expected_type,
        "confidence": float(round(effective_confidence, 4)),
        "extracted_data": _sanitize(extracted_dict),
        "fraud_indicators": fraud_indicators,
        "message": message,
    }


# ---------------------------------------------------------------------------
# POST /verify/document
# ---------------------------------------------------------------------------

@router.post("/verify/document", response_model=DocumentVerifyResponse)
async def verify_document(
    document_type: DocumentType,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
):
    """
    Verify a single document.

    Pipeline: Preprocess → Classify → OCR/NER → Fraud Detection.

    Args:
        document_type: Expected document type (government_id, proof_of_income, proof_of_address)
        file: Uploaded document image

    Returns:
        Verification result with extracted data and confidence scores
    """
    # Validate MIME type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image (JPEG, PNG).",
        )

    image_bytes = await file.read()

    try:
        result = await _verify_single_document(image_bytes, document_type)
    except Exception as e:
        logger.exception("Document verification failed")
        raise HTTPException(status_code=500, detail="Internal verification error")

    return DocumentVerifyResponse(**result)


# ---------------------------------------------------------------------------
# POST /verify/complete
# ---------------------------------------------------------------------------

@router.post("/verify/complete", response_model=CompleteVerificationResponse)
async def complete_verification(
    request: CompleteVerificationRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Complete KYC verification with all three documents and wallet data.

    Pipeline:
        1. Verify each document (government_id, proof_of_income, proof_of_address)
        2. Aggregate extracted data
        3. Calculate credit score
        4. Determine status & tier
    """
    scorer = get_scorer_service()

    doc_map = {
        DocumentType.GOVERNMENT_ID: request.government_id_base64,
        DocumentType.PROOF_OF_INCOME: request.proof_of_income_base64,
        DocumentType.PROOF_OF_ADDRESS: request.proof_of_address_base64,
    }

    all_fraud_flags: list[str] = []
    rejection_reasons: list[str] = []
    document_scores: dict[str, float] = {}
    merged_extracted: dict = {}
    verified_documents: dict = {}

    for doc_type, b64_data in doc_map.items():
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(b64_data)
        except Exception:
            rejection_reasons.append(f"Invalid base64 data for {doc_type.value}")
            document_scores[doc_type.value] = 0.0
            continue

        try:
            result = await _verify_single_document(image_bytes, doc_type)
        except Exception:
            logger.exception("Document verification failed for %s", doc_type.value)
            rejection_reasons.append(f"{doc_type.value} verification failed")
            document_scores[doc_type.value] = 0.0
            continue

        document_scores[doc_type.value] = result["confidence"]
        all_fraud_flags.extend(result["fraud_indicators"])

        if not result["valid"]:
            reason = result.get("message") or f"{doc_type.value} failed verification"
            rejection_reasons.append(reason)

        # Merge extracted data (later docs overwrite only if value exists)
        for key, value in result["extracted_data"].items():
            if key == "extra":
                continue
            if value is not None:
                merged_extracted[key] = value

        # Track per-document verification status for scorer
        verified_documents[doc_type.value] = {
            "is_verified": result["valid"],
            "confidence": result["confidence"],
        }

    # Build scorer input
    scorer_extracted = {
        **merged_extracted,
        "verified_documents": verified_documents,
    }

    wallet_data = {
        "address": request.wallet_data.address,
        "age_days": request.wallet_data.age_days or 0,
        "transaction_count": request.wallet_data.transaction_count or 0,
        "balance_eth": request.wallet_data.balance_eth or 0.0,
    }

    # Calculate credit score
    try:
        total_score, breakdown, tier = scorer.calculate_score(
            extracted_data=scorer_extracted,
            wallet_data=wallet_data,
            loan_history=None,  # First-time KYC has no loan history
        )
    except Exception as e:
        logger.exception("Credit scoring failed")
        raise HTTPException(status_code=500, detail="Internal scoring error")

    # Determine status
    if rejection_reasons:
        status = "rejected"
        tier = None
    elif any("review" in f.lower() for f in all_fraud_flags):
        status = "pending"
    else:
        status = "approved"

    return CompleteVerificationResponse(
        status=status,
        credit_score=total_score,
        tier=tier,
        extracted_data=merged_extracted,
        document_scores=document_scores,
        fraud_flags=all_fraud_flags,
        rejection_reasons=rejection_reasons,
    )
