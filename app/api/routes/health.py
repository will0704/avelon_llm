"""
Health check endpoints for service monitoring.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    Returns service status and loaded models info.
    """
    return {
        "status": "healthy",
        "service": "avelon-llm",
        "models_loaded": {
            "document_classifier": False,  # TODO: Check actual model status
            "ner_extractor": False,
            "fraud_detector": False,
            "credit_scorer": False,
        }
    }


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness check - verifies all required models are loaded.
    Used by Kubernetes/Docker for readiness probes.
    """
    # TODO: Actually verify models are loaded
    models_ready = False  # Will be True when models are implemented
    
    return {
        "ready": models_ready,
        "message": "Models not yet loaded" if not models_ready else "All systems ready"
    }
