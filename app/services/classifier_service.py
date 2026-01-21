"""
Document Classifier Service
Uses a CNN (ResNet-18) to classify document types.
"""
from typing import Tuple
from app.schemas.document import DocumentType


class ClassifierService:
    """
    Document classification using trained CNN model.
    
    TODO: Implement actual model loading and inference when models are trained.
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained classifier model."""
        if self.model_path:
            # TODO: Load PyTorch model
            # self.model = torch.load(self.model_path)
            pass
    
    def classify(self, image_bytes: bytes) -> Tuple[DocumentType, float]:
        """
        Classify a document image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Tuple of (document_type, confidence_score)
        """
        # TODO: Implement actual classification
        # 1. Preprocess image (resize to 224x224, normalize)
        # 2. Run through model
        # 3. Get prediction and confidence
        
        # Stub response
        return (DocumentType.GOVERNMENT_ID, 0.0)
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


# Singleton instance
_classifier_service = None

def get_classifier_service() -> ClassifierService:
    """Get or create classifier service instance."""
    global _classifier_service
    if _classifier_service is None:
        from app.config import get_settings
        settings = get_settings()
        _classifier_service = ClassifierService(settings.classifier_model_path)
    return _classifier_service
