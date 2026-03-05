"""
Unit tests for the Fraud Detector ML model training and integration.
Tests both the ML model path and rule-based fallback.
"""
import os
import pytest
import tempfile
from io import BytesIO
from PIL import Image

from app.services.fraud_detector_service import FraudDetectorService
from app.schemas.fraud import FraudResult


class TestFraudDetectorMLModel:
    """Tests for ML-enhanced fraud detection."""

    @pytest.fixture
    def test_image_bytes(self):
        """Create test image bytes."""
        img = Image.new('RGB', (300, 300), color='white')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    @pytest.fixture
    def service_no_model(self):
        """Create service without ML model (rule-based only)."""
        return FraudDetectorService(model_path=None)

    def test_service_without_model_uses_rules(self, service_no_model, test_image_bytes):
        """Service should work with rules only when no model loaded."""
        result = service_no_model.analyze(test_image_bytes)
        assert isinstance(result, FraudResult)
        assert 0.0 <= result.fraud_probability <= 1.0

    def test_ml_model_is_loaded_property(self, service_no_model):
        """Service should report ml_model_loaded correctly."""
        assert service_no_model.ml_model_loaded is False

    def test_service_with_model_path_loads(self):
        """Service should load model from valid path."""
        model_path = os.path.join(
            os.path.dirname(__file__), '..', 'app', 'models', 'fraud_detector.pkl'
        )
        if os.path.exists(model_path):
            service = FraudDetectorService(model_path=model_path)
            assert service.ml_model_loaded is True

    def test_service_with_invalid_path_falls_back(self):
        """Service should fall back gracefully with invalid model path."""
        service = FraudDetectorService(model_path="/nonexistent/model.pkl")
        assert service.ml_model_loaded is False
        # Should still work with rules
        img = Image.new('RGB', (100, 100), color='red')
        buf = BytesIO()
        img.save(buf, format='PNG')
        result = service.analyze(buf.getvalue())
        assert isinstance(result, FraudResult)

    def test_extract_image_features_returns_dict(self, service_no_model, test_image_bytes):
        """extract_image_features should return a feature dict."""
        features = service_no_model.extract_image_features(test_image_bytes)
        assert isinstance(features, dict)
        expected_keys = [
            'compression_ratio', 'laplacian_variance', 'has_exif',
            'edge_density', 'mean_brightness', 'std_brightness'
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"

    def test_extract_image_features_values_reasonable(self, service_no_model, test_image_bytes):
        """Feature values should be in reasonable ranges."""
        features = service_no_model.extract_image_features(test_image_bytes)
        assert 0.0 <= features['compression_ratio'] <= 1.0
        assert features['laplacian_variance'] >= 0.0
        assert isinstance(features['has_exif'], (int, float))
        assert 0.0 <= features['edge_density'] <= 1.0
        assert 0.0 <= features['mean_brightness'] <= 255.0
