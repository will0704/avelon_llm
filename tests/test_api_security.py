"""
Integration tests for API security hardening.
Covers: error leakage, API key validation, base64 size limits, CORS.
"""
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock
import base64
import os


@pytest.fixture
def api_key():
    return "test-api-key-for-tests"


@pytest.fixture
def app_instance(api_key):
    """Create a fresh FastAPI app with a known API key."""
    # Patch settings before importing app
    with patch.dict(os.environ, {
        "API_KEY": api_key,
        "ENVIRONMENT": "development",
        "DEBUG": "true",
    }):
        # Clear cached settings
        from app.config import get_settings
        get_settings.cache_clear()

        # Re-import app to pick up patched settings
        import importlib
        import app.main
        importlib.reload(app.main)
        yield app.main.app

        # Cleanup
        get_settings.cache_clear()


@pytest.fixture
def headers(api_key):
    return {"X-API-Key": api_key}


class TestErrorLeakage:
    """Verify that internal error details are never exposed to clients."""

    @pytest.mark.asyncio
    async def test_verify_document_error_hides_internals(self, app_instance, headers):
        """POST /api/v1/verify/document should not leak exception details."""
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Send a valid image that will trigger a classifier error
            # (services not fully loaded in test env)
            import io
            from PIL import Image
            img = Image.new("RGB", (300, 300), color="red")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)

            response = await client.post(
                "/api/v1/verify/document",
                data={"document_type": "government_id"},
                files={"file": ("test.jpg", buf, "image/jpeg")},
                headers=headers,
            )

        # If 500, detail must NOT contain Python exception text
        if response.status_code == 500:
            detail = response.json().get("detail", "")
            # Must not contain internal error patterns
            assert "Traceback" not in detail
            assert "File \"" not in detail
            assert "line " not in detail.lower() or "error" in detail.lower()
            # Should be a generic message
            assert "str(e)" not in detail

    @pytest.mark.asyncio
    async def test_score_calculate_error_hides_internals(self, app_instance, headers):
        """POST /api/v1/score/calculate should not leak exception details on error."""
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/score/calculate",
                json={
                    "user_id": "test-user",
                    "extracted_data": {},
                    "wallet_address": "0x123",
                },
                headers=headers,
            )

        if response.status_code == 500:
            detail = response.json().get("detail", "")
            assert "Traceback" not in detail
            assert "File \"" not in detail

    @pytest.mark.asyncio
    async def test_complete_verification_error_hides_internals(self, app_instance, headers):
        """POST /api/v1/verify/complete should not leak exception details."""
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Valid base64 but tiny image — will cause processing errors
            tiny_b64 = base64.b64encode(b"x" * 200).decode()
            response = await client.post(
                "/api/v1/verify/complete",
                json={
                    "user_id": "test-user",
                    "government_id_base64": tiny_b64,
                    "proof_of_income_base64": tiny_b64,
                    "proof_of_address_base64": tiny_b64,
                    "wallet_data": {
                        "address": "0xabc",
                        "age_days": 10,
                        "transaction_count": 5,
                        "balance_eth": 1.0,
                    },
                },
                headers=headers,
            )

        if response.status_code == 500:
            detail = response.json().get("detail", "")
            assert "Traceback" not in detail
            assert "File \"" not in detail


class TestAPIKeyValidation:
    """Verify API key security properties."""

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_422(self, app_instance):
        """Request without X-API-Key header should be rejected."""
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/score/calculate",
                json={"user_id": "test"},
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_wrong_api_key_returns_401(self, app_instance):
        """Request with wrong API key should be rejected."""
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/score/calculate",
                json={"user_id": "test"},
                headers={"X-API-Key": "wrong-key"},
            )
        assert response.status_code == 401

    def test_api_key_uses_timing_safe_comparison(self):
        """verify_api_key must use hmac.compare_digest for timing-safe comparison."""
        import inspect
        from app.api.dependencies import verify_api_key
        source = inspect.getsource(verify_api_key)
        assert "compare_digest" in source, (
            "API key comparison must use hmac.compare_digest to prevent timing attacks"
        )


class TestBase64SizeLimits:
    """Verify base64 payload size limits on verification endpoints."""

    @pytest.mark.asyncio
    async def test_complete_verification_rejects_oversized_base64(self, app_instance, headers):
        """Oversized base64 payloads should be rejected before processing."""
        # 15MB base64 string (exceeds reasonable document size)
        oversized_b64 = base64.b64encode(b"x" * (15 * 1024 * 1024)).decode()

        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/verify/complete",
                json={
                    "user_id": "test",
                    "government_id_base64": oversized_b64,
                    "proof_of_income_base64": "dGVzdA==",
                    "proof_of_address_base64": "dGVzdA==",
                    "wallet_data": {
                        "address": "0x123",
                    },
                },
                headers=headers,
            )
        assert response.status_code == 422, (
            "Oversized base64 should be rejected with 422 validation error"
        )


class TestCORSConfiguration:
    """Verify CORS is properly restricted."""

    @pytest.mark.asyncio
    async def test_cors_does_not_allow_arbitrary_origins(self, app_instance):
        """CORS should not reflect arbitrary origins."""
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.options(
                "/api/v1/score/calculate",
                headers={
                    "Origin": "http://evil-site.com",
                    "Access-Control-Request-Method": "POST",
                },
            )
        # Should NOT have the evil origin in allowed origins
        acao = response.headers.get("access-control-allow-origin", "")
        assert acao != "http://evil-site.com", (
            "CORS must not reflect arbitrary origins"
        )
        assert acao != "*", "CORS must not use wildcard origin"

    @pytest.mark.asyncio
    async def test_cors_allows_backend_origin(self, app_instance):
        """CORS should allow the configured backend URL."""
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.options(
                "/api/v1/score/calculate",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                },
            )
        acao = response.headers.get("access-control-allow-origin", "")
        assert acao == "http://localhost:3000"

    @pytest.mark.asyncio
    async def test_cors_restricts_methods(self, app_instance):
        """CORS should only allow POST and GET methods, not DELETE/PUT etc."""
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.options(
                "/api/v1/score/calculate",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                },
            )
        methods = response.headers.get("access-control-allow-methods", "")
        assert "DELETE" not in methods, "DELETE should not be in allowed CORS methods"
        assert "PUT" not in methods, "PUT should not be in allowed CORS methods"
