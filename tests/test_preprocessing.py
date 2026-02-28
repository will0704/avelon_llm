"""
Unit tests for the Preprocessing Service.
"""
import pytest
from io import BytesIO
from PIL import Image

from app.services.preprocessing_service import PreprocessingService, get_preprocessing_service


class TestPreprocessingService:
    """Tests for PreprocessingService."""
    
    @pytest.fixture
    def service(self):
        """Create a preprocessing service instance."""
        return PreprocessingService()
    
    @pytest.fixture
    def valid_image_bytes(self):
        """Create valid test image bytes."""
        img = Image.new('RGB', (500, 500), color='white')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    @pytest.fixture
    def small_image_bytes(self):
        """Create small test image bytes (below minimum resolution)."""
        img = Image.new('RGB', (100, 100), color='white')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    def test_validate_image_valid(self, service, valid_image_bytes):
        """Test validation passes for valid image."""
        is_valid, error = service.validate_image(valid_image_bytes)
        assert is_valid is True
        assert error is None
    
    def test_validate_image_too_small_resolution(self, service, small_image_bytes):
        """Test validation fails for small resolution."""
        is_valid, error = service.validate_image(small_image_bytes)
        assert is_valid is False
        assert "Resolution too low" in error
    
    def test_validate_image_too_small_bytes(self, service):
        """Test validation fails for tiny file."""
        is_valid, error = service.validate_image(b"tiny")
        assert is_valid is False
        assert "too small" in error.lower()
    
    def test_validate_image_invalid_data(self, service):
        """Test validation fails for invalid image data."""
        is_valid, error = service.validate_image(b"not an image at all")
        assert is_valid is False
        assert "Invalid" in error
    
    def test_get_image_info(self, service, valid_image_bytes):
        """Test getting image info."""
        info = service.get_image_info(valid_image_bytes)
        assert info["format"] == "PNG"
        assert info["width"] == 500
        assert info["height"] == 500
    
    def test_normalize_for_cnn(self, service, valid_image_bytes):
        """Test CNN normalization produces correct size."""
        normalized = service.normalize_for_cnn(valid_image_bytes)
        assert normalized is not None
        
        # Verify the normalized image is 224x224
        img = Image.open(BytesIO(normalized))
        assert img.size == (224, 224)
    
    def test_check_quality(self, service, valid_image_bytes):
        """Test quality check returns expected structure."""
        result = service.check_quality(valid_image_bytes)
        assert "blur_score" in result
        assert "brightness" in result
        assert "is_acceptable" in result
        assert "issues" in result
    
    def test_singleton_get_service(self):
        """Test singleton pattern returns same instance."""
        service1 = get_preprocessing_service()
        service2 = get_preprocessing_service()
        assert service1 is service2
