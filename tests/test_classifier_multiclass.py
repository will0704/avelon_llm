"""
Unit tests for multi-class document classifier.
Tests 4-class classification (government_id, not_id, proof_of_income, proof_of_address).
"""
import os
import pytest
from io import BytesIO
from PIL import Image

from app.services.classifier_service import ClassifierService
from app.schemas.document import DocumentType


class TestMultiClassClassifier:
    """Tests for multi-class document classification."""

    @pytest.fixture
    def service_no_model(self):
        """Create classifier service without model."""
        return ClassifierService(model_path="/nonexistent/model.pt")

    @pytest.fixture
    def test_image_bytes(self):
        """Create test image bytes."""
        img = Image.new('RGB', (224, 224), color='blue')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    def test_map_government_id_classes(self, service_no_model):
        """Mapping should handle ID-related class names."""
        id_names = ['id', 'valid', 'validid', 'valid_id', 'government_id', 'philid']
        for name in id_names:
            doc_type, conf = service_no_model._map_to_document_type(name, 0.9)
            assert doc_type == DocumentType.GOVERNMENT_ID

    def test_map_proof_of_income_classes(self, service_no_model):
        """Mapping should handle income-related class names."""
        income_names = ['income', 'proof_of_income', 'payslip']
        for name in income_names:
            doc_type, conf = service_no_model._map_to_document_type(name, 0.9)
            assert doc_type == DocumentType.PROOF_OF_INCOME
            assert conf == 0.9

    def test_map_proof_of_address_classes(self, service_no_model):
        """Mapping should handle address-related class names."""
        address_names = ['address', 'proof_of_address', 'utility_bill']
        for name in address_names:
            doc_type, conf = service_no_model._map_to_document_type(name, 0.85)
            assert doc_type == DocumentType.PROOF_OF_ADDRESS
            assert conf == 0.85

    def test_map_not_id_classes(self, service_no_model):
        """Mapping should handle negative/not-id class names."""
        not_id_names = ['not_id', 'invalid', 'nonvalid', 'non_valid', 'not_valid']
        for name in not_id_names:
            doc_type, conf = service_no_model._map_to_document_type(name, 0.95)
            assert conf == 0.0  # Low confidence for negative class

    def test_map_unknown_class_fallback(self, service_no_model):
        """Unknown classes should fall back gracefully."""
        doc_type, conf = service_no_model._map_to_document_type("unknown_class", 0.5)
        assert conf == 0.0  # Unknown should return low confidence

    def test_classify_fallback_without_model(self, service_no_model, test_image_bytes):
        """Without model, classify should return fallback."""
        doc_type, confidence = service_no_model.classify(test_image_bytes)
        assert doc_type == DocumentType.GOVERNMENT_ID
        assert confidence == 0.0

    def test_classify_with_all_scores_structure(self, service_no_model, test_image_bytes):
        """classify_with_all_scores should return proper structure."""
        result = service_no_model.classify_with_all_scores(test_image_bytes)
        assert "predicted" in result
        assert "confidence" in result
        assert "scores" in result
        assert "model" in result

    def test_class_mapping_constant(self, service_no_model):
        """CLASS_MAPPING should map class names to DocumentType."""
        assert hasattr(service_no_model, 'CLASS_MAPPING')
        mapping = service_no_model.CLASS_MAPPING
        assert isinstance(mapping, dict)
        # Must cover all 3 document types
        mapped_types = set(mapping.values())
        expected = {DocumentType.GOVERNMENT_ID, DocumentType.PROOF_OF_INCOME, DocumentType.PROOF_OF_ADDRESS}
        assert mapped_types == expected

    def test_four_class_model_loads(self):
        """If 4-class model exists, it should load correctly."""
        model_path = os.path.join(
            os.path.dirname(__file__), '..', 'app', 'models', 'philid_classifier.pt'
        )
        if os.path.exists(model_path):
            service = ClassifierService(model_path=model_path)
            assert service.is_loaded is True
