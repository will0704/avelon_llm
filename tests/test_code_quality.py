"""
Tests for singleton service initialization safety.
"""
import pytest


class TestFraudDetectorPublicAPI:
    """FraudDetectorService should use public Pillow API."""

    def test_uses_public_getexif(self):
        """FraudDetectorService should use image.getexif() not image._getexif()."""
        import inspect
        from app.services.fraud_detector_service import FraudDetectorService
        source = inspect.getsource(FraudDetectorService)
        # _getexif is private API — should use getexif (public)
        assert "._getexif()" not in source, (
            "Use image.getexif() (public API) instead of image._getexif() (private)"
        )


class TestCompleteVerificationPerDocumentErrors:
    """complete_verification should handle per-document errors gracefully."""

    @pytest.mark.asyncio
    async def test_single_doc_failure_doesnt_crash_entire_request(self):
        """If one document fails, others should still be processed."""
        import os
        from unittest.mock import patch
        from httpx import AsyncClient, ASGITransport
        import base64

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
            the_app = app.main.app

            # Use invalid base64 for one doc but valid-ish for others
            transport = ASGITransport(app=the_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/verify/complete",
                    json={
                        "user_id": "test-user",
                        "government_id_base64": "!!!INVALID_BASE64!!!",
                        "proof_of_income_base64": base64.b64encode(b"x" * 100).decode(),
                        "proof_of_address_base64": base64.b64encode(b"x" * 100).decode(),
                        "wallet_data": {
                            "address": "0xabc",
                            "age_days": 10,
                            "transaction_count": 5,
                            "balance_eth": 1.0,
                        },
                    },
                    headers={"X-API-Key": "test-key"},
                )

            # Should NOT be a 500 — should handle gracefully
            assert response.status_code != 500, (
                "Per-document failure should not crash the entire request"
            )

            get_settings.cache_clear()
