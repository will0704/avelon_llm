"""
Document Classifier Service
Uses MobileNetV2 to classify document types.
Currently trained for: GOVERNMENT_ID
Ready to extend for: PROOF_OF_INCOME, PROOF_OF_ADDRESS
"""
from typing import Tuple, Optional, Dict
from io import BytesIO
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms, models
    from torchvision.models import MobileNet_V2_Weights
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from app.config import get_settings
from app.schemas.document import DocumentType

logger = logging.getLogger(__name__)


class DocumentClassifierModel(nn.Module):
    """MobileNetV2-based document classifier."""
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class ClassifierService:
    """
    Unified document classifier using trained MobileNetV2 model.
    
    Currently supports:
    - government_id (trained with PhilID data)
    
    Architecture ready for:
    - proof_of_income (add training data)
    - proof_of_address (add training data)
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.settings = get_settings()
        self.model_path = model_path or self.settings.philid_model_path
        self.model = None
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.classes = None
        self.class_mapping = None
        
        # Image preprocessing
        if TORCH_AVAILABLE:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = None
        
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load the trained PyTorch model."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, classifier disabled")
            return False
        
        if not self.model_path:
            logger.info("No model path configured, classifier will use heuristic fallback")
            return False
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            num_classes = checkpoint.get('config', {}).get('num_classes', 2)
            
            self.model = DocumentClassifierModel(num_classes=num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.class_mapping = checkpoint.get('class_mapping', {})
            self.classes = checkpoint.get('classes', ['id', 'not_id'])
            
            logger.info(f"Classifier loaded from {self.model_path}")
            logger.info(f"Classes: {self.classes}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            return False
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None
    
    @property
    def is_trained(self) -> bool:
        """Check if model is trained (has a valid model path)."""
        return self.model is not None and self.model_path is not None
    
    def _preprocess(self, image_bytes: bytes) -> 'torch.Tensor':
        """Preprocess image for model input."""
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)
    
    def _map_to_document_type(self, class_name: str, confidence: float) -> Tuple[DocumentType, float]:
        """Map model class to DocumentType enum."""
        class_lower = class_name.lower()
        
        # ID-related classes → GOVERNMENT_ID
        if class_lower in ['id', 'valid', 'government_id', 'philid']:
            return (DocumentType.GOVERNMENT_ID, confidence)
        
        # Not an ID → return with low confidence to indicate uncertainty
        elif class_lower in ['not_id', 'invalid']:
            # Still might be another document type, but we're not sure
            return (DocumentType.GOVERNMENT_ID, 0.0)
        
        # Income-related
        elif class_lower in ['income', 'proof_of_income', 'payslip']:
            return (DocumentType.PROOF_OF_INCOME, confidence)
        
        # Address-related
        elif class_lower in ['address', 'proof_of_address', 'utility_bill']:
            return (DocumentType.PROOF_OF_ADDRESS, confidence)
        
        # Default
        return (DocumentType.GOVERNMENT_ID, 0.0)
    
    def classify(self, image_bytes: bytes) -> Tuple[DocumentType, float]:
        """
        Classify a document image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Tuple of (document_type, confidence_score)
        """
        if not self.is_loaded:
            logger.warning("Model not loaded, returning fallback classification")
            return (DocumentType.GOVERNMENT_ID, 0.0)
        
        try:
            input_tensor = self._preprocess(image_bytes)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()
            
            # Map to DocumentType
            if self.classes and predicted_idx < len(self.classes):
                class_name = self.classes[predicted_idx]
                return self._map_to_document_type(class_name, confidence)
            
            return (DocumentType.GOVERNMENT_ID, 0.0)
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return (DocumentType.GOVERNMENT_ID, 0.0)
    
    def classify_with_all_scores(self, image_bytes: bytes) -> Dict:
        """
        Classify and return detailed scores.
        
        Returns:
            Dict with prediction, confidence, scores for each class
        """
        result = {
            "predicted": DocumentType.GOVERNMENT_ID,
            "confidence": 0.0,
            "scores": {},
            "model": "mobilenetv2"
        }
        
        if not self.is_loaded:
            return result
        
        try:
            input_tensor = self._preprocess(image_bytes)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
            
            # Build scores dict
            for idx, class_name in enumerate(self.classes):
                result["scores"][class_name] = probabilities[idx].item()
            
            confidence, predicted_idx = torch.max(probabilities, 0)
            class_name = self.classes[predicted_idx.item()]
            
            doc_type, conf = self._map_to_document_type(class_name, confidence.item())
            result["predicted"] = doc_type
            result["confidence"] = conf
            
            return result
            
        except Exception as e:
            logger.error(f"Classification with scores failed: {e}")
            return result


# Singleton instance
_classifier_service: Optional[ClassifierService] = None


def get_classifier_service() -> ClassifierService:
    """Get or create the classifier service singleton."""
    global _classifier_service
    if _classifier_service is None:
        _classifier_service = ClassifierService()
    return _classifier_service
