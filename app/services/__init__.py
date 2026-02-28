"""Services package - AI model wrappers and business logic."""

from app.services.preprocessing_service import (
    PreprocessingService,
    get_preprocessing_service,
    ImageValidationError,
)
from app.services.ocr_service import OCRService, get_ocr_service
from app.services.classifier_service import ClassifierService, get_classifier_service
from app.services.entity_extractor_service import (
    EntityExtractorService,
    get_entity_extractor_service,
)
from app.services.fraud_detector_service import (
    FraudDetectorService,
    get_fraud_detector_service,
)
from app.services.scorer_service import ScorerService, get_scorer_service

__all__ = [
    # Preprocessing
    "PreprocessingService",
    "get_preprocessing_service",
    "ImageValidationError",
    # OCR (legacy Tesseract)
    "OCRService",
    "get_ocr_service",
    # Classifier (unified MobileNetV2)
    "ClassifierService",
    "get_classifier_service",
    # Entity Extractor (EasyOCR + NER)
    "EntityExtractorService",
    "get_entity_extractor_service",
    # Fraud Detector
    "FraudDetectorService",
    "get_fraud_detector_service",
    # Scorer
    "ScorerService",
    "get_scorer_service",
]
