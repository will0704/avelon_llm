"""
Health check endpoints for service monitoring.
"""
import logging
from fastapi import APIRouter

from app.services.classifier_service import get_classifier_service
from app.services.entity_extractor_service import get_entity_extractor_service
from app.services.fraud_detector_service import get_fraud_detector_service
from app.services.scorer_service import get_scorer_service
from app.services.preprocessing_service import get_preprocessing_service
from app.services.ocr_service import get_ocr_service

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_model_status() -> dict:
    """Collect load / availability status from every service."""
    classifier = get_classifier_service()
    extractor = get_entity_extractor_service()
    fraud = get_fraud_detector_service()
    scorer = get_scorer_service()
    preprocessing = get_preprocessing_service()
    ocr = get_ocr_service()

    return {
        "document_classifier": classifier.is_loaded,
        "ner_extractor": extractor.is_available,
        "ocr_engine": ocr.is_available,
        "fraud_detector": True,  # rule-based, always available
        "credit_scorer": True,   # rule-based, always available
        "preprocessing": preprocessing._cv2_available,
    }


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    Returns service status and loaded models info.
    """
    models = _get_model_status()

    return {
        "status": "healthy",
        "service": "avelon-llm",
        "models_loaded": models,
    }


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness check – verifies all required models are loaded.
    Used by Kubernetes / Docker for readiness probes.
    """
    models = _get_model_status()

    # Core services that must be available for the service to accept traffic
    required_keys = ["document_classifier", "ner_extractor", "fraud_detector", "credit_scorer"]
    models_ready = all(models.get(k, False) for k in required_keys)

    return {
        "ready": models_ready,
        "models": models,
        "message": "All systems ready" if models_ready else "Some required models are not loaded",
    }
