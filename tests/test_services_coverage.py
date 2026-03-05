"""
Additional coverage tests for preprocessing and OCR services.
"""
import pytest
from io import BytesIO

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def _make_image(fmt="JPEG", width=300, height=300) -> bytes:
    img = Image.new("RGB", (width, height), color="green")
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestPreprocessingServiceAdditional:
    """Additional coverage for preprocessing_service.py."""

    @pytest.fixture
    def service(self):
        from app.services.preprocessing_service import PreprocessingService
        return PreprocessingService()

    def test_validate_too_large(self, service):
        """Reject files exceeding MAX_FILE_SIZE."""
        huge = b"x" * (11 * 1024 * 1024)
        valid, msg = service.validate_image(huge)
        assert valid is False
        assert "too large" in msg.lower()

    def test_validate_too_small(self, service):
        """Reject files smaller than 100 bytes."""
        valid, msg = service.validate_image(b"tiny")
        assert valid is False
        assert "too small" in msg.lower()

    def test_validate_unsupported_format(self, service):
        """Reject unsupported image formats."""
        bmp_img = Image.new("RGB", (300, 300))
        buf = BytesIO()
        bmp_img.save(buf, format="BMP")
        valid, msg = service.validate_image(buf.getvalue())
        assert valid is False
        assert "Unsupported" in msg

    def test_validate_low_resolution(self, service):
        """Reject images below MIN_RESOLUTION."""
        small_img = _make_image(width=50, height=50)
        valid, msg = service.validate_image(small_img)
        assert valid is False
        assert "Resolution" in msg

    def test_validate_valid_png(self, service):
        """Accept valid PNG images."""
        png = _make_image(fmt="PNG")
        valid, msg = service.validate_image(png)
        assert valid is True
        assert msg is None

    def test_normalize_for_cnn_output_size(self, service):
        """CNN normalization should produce 224x224."""
        img_bytes = _make_image()
        result = service.normalize_for_cnn(img_bytes)
        assert result is not None
        result_img = Image.open(BytesIO(result))
        assert result_img.size == (224, 224)

    def test_normalize_for_cnn_converts_rgba(self, service):
        """CNN normalization should handle RGBA images."""
        rgba = Image.new("RGBA", (300, 300), color=(255, 0, 0, 128))
        buf = BytesIO()
        rgba.save(buf, format="PNG")
        result = service.normalize_for_cnn(buf.getvalue())
        assert result is not None

    def test_preprocess_for_ocr_returns_bytes(self, service):
        """OCR preprocessing should return bytes."""
        img_bytes = _make_image()
        result = service.preprocess_for_ocr(img_bytes)
        assert result is not None
        assert isinstance(result, bytes)

    def test_get_image_info(self, service):
        """get_image_info returns correct metadata."""
        img_bytes = _make_image(fmt="JPEG", width=400, height=300)
        info = service.get_image_info(img_bytes)
        assert info["format"] == "JPEG"
        assert info["width"] == 400
        assert info["height"] == 300
        assert info["mode"] == "RGB"

    def test_get_image_info_invalid(self, service):
        """get_image_info handles invalid input."""
        info = service.get_image_info(b"not an image")
        assert "error" in info

    def test_check_quality_bright_image(self, service):
        """Quality check should detect overexposed images."""
        bright = Image.new("RGB", (300, 300), color=(255, 255, 255))
        buf = BytesIO()
        bright.save(buf, format="JPEG")
        result = service.check_quality(buf.getvalue())
        assert result["brightness"] > 200

    def test_check_quality_dark_image(self, service):
        """Quality check should detect dark images."""
        dark = Image.new("RGB", (300, 300), color=(10, 10, 10))
        buf = BytesIO()
        dark.save(buf, format="JPEG")
        result = service.check_quality(buf.getvalue())
        assert result["brightness"] < 50


class TestOCRServiceBasic:
    """Basic coverage for ocr_service.py."""

    @pytest.fixture
    def service(self):
        from app.services.ocr_service import OCRService
        return OCRService()

    def test_is_available_property(self, service):
        """is_available should return a boolean."""
        assert isinstance(service.is_available, bool)

    def test_extract_text_returns_string(self, service):
        """extract_text should always return a string."""
        result = service.extract_text(b"invalid image data" * 10)
        assert isinstance(result, str)

    def test_extract_structured_returns_dict(self, service):
        """extract_structured should return expected dict keys."""
        result = service.extract_structured(b"invalid data" * 10)
        assert "text" in result
        assert "words" in result
        assert "average_confidence" in result

    def test_extract_lines_returns_list(self, service):
        """extract_lines should return a list."""
        result = service.extract_lines(b"invalid data" * 10)
        assert isinstance(result, list)

    def test_get_orientation_returns_dict(self, service):
        """get_orientation should return a dict."""
        result = service.get_orientation(b"invalid data" * 10)
        assert isinstance(result, dict)

    def test_preprocess_image_fallback(self, service):
        """_preprocess_image should work with valid image bytes."""
        img_bytes = _make_image()
        result = service._preprocess_image(img_bytes)
        assert result is not None
