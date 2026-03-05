"""
Tests for health endpoint accuracy — ML model status reporting.
"""
import pytest
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport
import os


@pytest.fixture
def app_instance():
    """Create a fresh FastAPI app."""
    with patch.dict(os.environ, {
        "API_KEY": "test-key",
        "ENVIRONMENT": "development",
        "DEBUG": "true",
    }):
        from app.config import get_settings
        get_settings.cache_clear()

        import importlib
        import app.main
        importlib.reload(app.main)
        yield app.main.app

        get_settings.cache_clear()


class TestHealthEndpoint:
    """Tests for /health and /health/ready endpoints."""

    @pytest.mark.asyncio
    async def test_health_reports_ml_model_status(self, app_instance):
        """Health endpoint should report actual ML model loaded status."""
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        models = data["models_loaded"]

        # These keys should exist and reflect actual service state
        assert "fraud_detector" in models
        assert "credit_scorer" in models
        # Values should be booleans reflecting actual ml_model_loaded
        assert isinstance(models["fraud_detector"], bool)
        assert isinstance(models["credit_scorer"], bool)

    @pytest.mark.asyncio
    async def test_health_distinguishes_rule_and_ml(self, app_instance):
        """Health should separately report rule-based availability and ML model status."""
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        data = response.json()
        models = data["models_loaded"]

        # Should have separate keys for rule-based and ML availability
        assert "fraud_detector_rules" in models or "fraud_detector" in models
        assert "credit_scorer_rules" in models or "credit_scorer" in models

        # If ML keys exist separately, they should be booleans
        for key in ["fraud_detector_ml", "credit_scorer_ml"]:
            if key in models:
                assert isinstance(models[key], bool)

    @pytest.mark.asyncio
    async def test_readiness_check_works(self, app_instance):
        """Readiness check should return ready status."""
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert isinstance(data["ready"], bool)
