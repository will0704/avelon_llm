"""
Unit tests for the Fraud Detector Service.
"""
import pytest
from io import BytesIO
from PIL import Image

from app.services.fraud_detector_service import FraudDetectorService, get_fraud_detector_service
from app.schemas.fraud import FraudResult, FraudFlagType
from app.schemas.document import DocumentType, ExtractedDocumentData


class TestFraudDetectorService:
    """Tests for FraudDetectorService."""
    
    @pytest.fixture
    def service(self):
        """Create a fraud detector service instance."""
        return FraudDetectorService()
    
    @pytest.fixture
    def test_image_bytes(self):
        """Create test image bytes."""
        img = Image.new('RGB', (300, 300), color='white')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    @pytest.fixture
    def extracted_data_complete(self):
        """Create complete extracted data."""
        return ExtractedDocumentData(
            raw_text="Sample document",
            name="Juan Dela Cruz",
            id_number="1234-5678-9012",
            address="Manila, Philippines"
        )
    
    @pytest.fixture
    def extracted_data_missing_fields(self):
        """Create extracted data with missing required fields."""
        return ExtractedDocumentData(
            raw_text="Incomplete document"
            # Missing name, id_number, etc.
        )
    
    def test_analyze_returns_fraud_result(self, service, test_image_bytes):
        """Test analyze returns FraudResult."""
        result = service.analyze(test_image_bytes)
        assert isinstance(result, FraudResult)
    
    def test_analyze_has_required_fields(self, service, test_image_bytes):
        """Test FraudResult has all required fields."""
        result = service.analyze(test_image_bytes)
        assert hasattr(result, 'is_suspicious')
        assert hasattr(result, 'fraud_probability')
        assert hasattr(result, 'flags')
        assert hasattr(result, 'recommendation')
    
    def test_recommendation_values(self, service, test_image_bytes):
        """Test recommendation is one of expected values."""
        result = service.analyze(test_image_bytes)
        assert result.recommendation in ["approve", "review", "reject"]
    
    def test_fraud_probability_range(self, service, test_image_bytes):
        """Test fraud probability is in valid range."""
        result = service.analyze(test_image_bytes)
        assert 0.0 <= result.fraud_probability <= 1.0
    
    def test_detect_manipulation_returns_list(self, service, test_image_bytes):
        """Test detect_manipulation returns list of flags."""
        flags = service.detect_manipulation(test_image_bytes)
        assert isinstance(flags, list)
    
    def test_check_consistency_missing_fields(self, service, extracted_data_missing_fields):
        """Test consistency check flags missing required fields."""
        flags = service.check_text_consistency(
            extracted_data_missing_fields,
            DocumentType.GOVERNMENT_ID
        )
        # Should flag missing name and id_number for government ID
        assert len(flags) > 0
        missing_field_flags = [f for f in flags if f.flag_type == FraudFlagType.MISSING_REQUIRED_FIELD]
        assert len(missing_field_flags) >= 1
    
    def test_check_consistency_complete_data(self, service, extracted_data_complete):
        """Test consistency check passes for complete data."""
        flags = service.check_text_consistency(
            extracted_data_complete,
            DocumentType.GOVERNMENT_ID
        )
        # Should not flag missing required fields
        missing_field_flags = [f for f in flags if f.flag_type == FraudFlagType.MISSING_REQUIRED_FIELD]
        assert len(missing_field_flags) == 0
    
    def test_singleton_get_service(self):
        """Test singleton pattern returns same instance."""
        service1 = get_fraud_detector_service()
        service2 = get_fraud_detector_service()
        assert service1 is service2
