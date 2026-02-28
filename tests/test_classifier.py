"""
Unit tests for the Document Classifier Service.
"""
import pytest
from io import BytesIO
from PIL import Image

from app.services.classifier_service import ClassifierService, get_classifier_service
from app.schemas.document import DocumentType


class TestClassifierService:
    """Tests for ClassifierService."""
    
    @pytest.fixture
    def service(self):
        """Create a classifier service instance (without model)."""
        return ClassifierService(model_path=None)
    
    @pytest.fixture
    def test_image_bytes(self):
        """Create test image bytes."""
        img = Image.new('RGB', (224, 224), color='blue')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    def test_classify_returns_tuple(self, service, test_image_bytes):
        """Test classify returns (DocumentType, confidence) tuple."""
        result = service.classify(test_image_bytes)
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        doc_type, confidence = result
        assert isinstance(doc_type, DocumentType)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_classify_fallback_without_model(self, service, test_image_bytes):
        """Test classify returns fallback when model not loaded."""
        doc_type, confidence = service.classify(test_image_bytes)
        # Without a model, should return fallback
        assert doc_type == DocumentType.GOVERNMENT_ID
        assert confidence == 0.0
    
    def test_classify_with_all_scores(self, service, test_image_bytes):
        """Test classify_with_all_scores returns expected structure."""
        result = service.classify_with_all_scores(test_image_bytes)
        assert "predicted" in result
        assert "confidence" in result
        assert "scores" in result
        assert isinstance(result["scores"], dict)
    
    def test_is_loaded_property(self, service):
        """Test is_loaded property."""
        # Without PyTorch, model won't be loaded
        assert isinstance(service.is_loaded, bool)
    
    def test_is_trained_property(self, service):
        """Test is_trained property."""
        # Without model path, should be False
        assert service.is_trained is False
    
    def test_singleton_get_service(self):
        """Test singleton pattern returns same instance."""
        service1 = get_classifier_service()
        service2 = get_classifier_service()
        assert service1 is service2
    
    def test_class_mapping_covers_document_types(self, service):
        """Test class mapping includes all expected document types."""
        expected_types = {
            DocumentType.GOVERNMENT_ID,
            DocumentType.PROOF_OF_INCOME,
            DocumentType.PROOF_OF_ADDRESS
        }
        mapped_types = set(service.CLASS_MAPPING.values())
        assert mapped_types == expected_types
