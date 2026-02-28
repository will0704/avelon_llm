"""
Unit tests for the OCR Service.
"""
import pytest

from app.services.ocr_service import OCRService, get_ocr_service


class TestOCRService:
    """Tests for OCRService."""
    
    @pytest.fixture
    def service(self):
        """Create an OCR service instance."""
        return OCRService()
    
    def test_service_available_property(self, service):
        """Test is_available property exists."""
        # This will be False if pytesseract is not installed
        assert hasattr(service, 'is_available')
    
    def test_extract_text_returns_string(self, service):
        """Test extract_text always returns a string."""
        result = service.extract_text(b"fake image data")
        assert isinstance(result, str)
    
    def test_extract_structured_returns_dict(self, service):
        """Test extract_structured returns expected structure."""
        result = service.extract_structured(b"fake image data")
        assert isinstance(result, dict)
        assert "text" in result
        assert "words" in result
        assert "boxes" in result
        assert "confidences" in result
        assert "average_confidence" in result
    
    def test_extract_lines_returns_list(self, service):
        """Test extract_lines returns a list."""
        result = service.extract_lines(b"fake image data")
        assert isinstance(result, list)
    
    def test_get_orientation_returns_dict(self, service):
        """Test get_orientation returns a dict."""
        result = service.get_orientation(b"fake image data")
        assert isinstance(result, dict)
    
    def test_singleton_get_service(self):
        """Test singleton pattern returns same instance."""
        service1 = get_ocr_service()
        service2 = get_ocr_service()
        assert service1 is service2
    
    def test_languages_default(self, service):
        """Test default languages are set."""
        assert service.languages == "eng+fil"
