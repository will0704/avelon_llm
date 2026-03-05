"""
Integration tests for API endpoints — covers route-level logic.
"""
import pytest
import base64
import io
import os
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@pytest.fixture
def api_key():
    return "test-integration-key"


@pytest.fixture
def app_instance(api_key):
    with patch.dict(os.environ, {
        "API_KEY": api_key,
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


@pytest.fixture
def headers(api_key):
    return {"X-API-Key": api_key}


def _make_test_jpeg(width=300, height=300) -> bytes:
    """Create a valid JPEG image in memory."""
    img = Image.new("RGB", (width, height), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestVerifyDocumentEndpoint:
    """Tests for POST /api/v1/verify/document."""

    @pytest.mark.asyncio
    async def test_rejects_non_image_file(self, app_instance, headers):
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/verify/document?document_type=government_id",
                files={"file": ("test.txt", b"plain text", "text/plain")},
                headers=headers,
            )
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_accepts_valid_jpeg(self, app_instance, headers):
        img_bytes = _make_test_jpeg()
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/verify/document?document_type=government_id",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")},
                headers=headers,
            )
        assert response.status_code == 200
        data = response.json()
        assert "valid" in data
        assert "confidence" in data
        assert "extracted_data" in data

    @pytest.mark.asyncio
    async def test_returns_fraud_indicators_list(self, app_instance, headers):
        img_bytes = _make_test_jpeg()
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/verify/document?document_type=government_id",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")},
                headers=headers,
            )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["fraud_indicators"], list)

    @pytest.mark.asyncio
    async def test_requires_auth(self, app_instance):
        img_bytes = _make_test_jpeg()
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/verify/document?document_type=government_id",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            )
        assert response.status_code == 422  # Missing header


class TestScoreCalculateEndpoint:
    """Tests for POST /api/v1/score/calculate."""

    @pytest.mark.asyncio
    async def test_returns_credit_score(self, app_instance, headers):
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/score/calculate",
                json={
                    "user_id": "user-1",
                    "extracted_data": {
                        "verified_documents": {
                            "government_id": {"is_verified": True, "confidence": 0.95},
                            "proof_of_income": {"is_verified": True, "confidence": 0.9},
                            "proof_of_address": {"is_verified": True, "confidence": 0.85},
                        },
                        "monthly_income": 50000,
                        "employment_type": "permanent",
                        "years_employed": 3,
                    },
                    "wallet_address": "0xabc123",
                    "wallet_age_days": 200,
                    "wallet_transaction_count": 60,
                    "wallet_balance_eth": 1.5,
                },
                headers=headers,
            )
        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert "tier" in data
        assert "breakdown" in data
        assert "recommendations" in data
        assert 0 <= data["score"] <= 100

    @pytest.mark.asyncio
    async def test_returns_breakdown_fields(self, app_instance, headers):
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/score/calculate",
                json={
                    "user_id": "user-2",
                    "extracted_data": {},
                    "wallet_address": "0x000",
                },
                headers=headers,
            )
        data = response.json()
        breakdown = data["breakdown"]
        assert "document_score" in breakdown
        assert "financial_score" in breakdown
        assert "history_score" in breakdown
        assert "wallet_score" in breakdown

    @pytest.mark.asyncio
    async def test_recommendations_for_low_score(self, app_instance, headers):
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/score/calculate",
                json={
                    "user_id": "user-low",
                    "extracted_data": {},
                    "wallet_address": "0x000",
                },
                headers=headers,
            )
        data = response.json()
        assert len(data["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_validates_missing_fields(self, app_instance, headers):
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/score/calculate",
                json={},
                headers=headers,
            )
        assert response.status_code == 422


class TestScoreRecalculateEndpoint:
    """Tests for POST /api/v1/score/recalculate."""

    @pytest.mark.asyncio
    async def test_recalculate_with_loan_history(self, app_instance, headers):
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/score/recalculate",
                json={
                    "user_id": "user-3",
                    "extracted_data": {
                        "verified_documents": {
                            "government_id": {"is_verified": True, "confidence": 0.9},
                        },
                    },
                    "wallet_address": "0xdef",
                    "wallet_age_days": 100,
                    "wallet_transaction_count": 30,
                    "wallet_balance_eth": 0.5,
                    "loan_history": {
                        "total_loans": 3,
                        "repaid_loans": 2,
                        "defaulted_loans": 0,
                        "late_payments": 1,
                    },
                },
                headers=headers,
            )
        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert data["breakdown"]["history_score"] >= 0


class TestCompleteVerificationEndpoint:
    """Tests for POST /api/v1/verify/complete."""

    @pytest.mark.asyncio
    async def test_returns_complete_response(self, app_instance, headers):
        img_bytes = _make_test_jpeg()
        b64 = base64.b64encode(img_bytes).decode()

        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/verify/complete",
                json={
                    "user_id": "user-kyc",
                    "government_id_base64": b64,
                    "proof_of_income_base64": b64,
                    "proof_of_address_base64": b64,
                    "wallet_data": {
                        "address": "0xwallet",
                        "age_days": 90,
                        "transaction_count": 25,
                        "balance_eth": 0.8,
                    },
                },
                headers=headers,
            )
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ("approved", "rejected", "pending")
        assert "credit_score" in data
        assert "document_scores" in data
        assert "fraud_flags" in data
        assert "rejection_reasons" in data

    @pytest.mark.asyncio
    async def test_handles_invalid_base64_gracefully(self, app_instance, headers):
        img_bytes = _make_test_jpeg()
        b64 = base64.b64encode(img_bytes).decode()

        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/verify/complete",
                json={
                    "user_id": "user-bad",
                    "government_id_base64": "!!!NOT_BASE64!!!",
                    "proof_of_income_base64": b64,
                    "proof_of_address_base64": b64,
                    "wallet_data": {
                        "address": "0xwallet",
                    },
                },
                headers=headers,
            )
        assert response.status_code == 200
        data = response.json()
        assert len(data["rejection_reasons"]) > 0

    @pytest.mark.asyncio
    async def test_requires_auth(self, app_instance):
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/verify/complete",
                json={"user_id": "test"},
            )
        assert response.status_code == 422


class TestRootEndpoint:
    """Tests for GET /."""

    @pytest.mark.asyncio
    async def test_root_returns_service_info(self, app_instance):
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Avelon LLM"
        assert data["status"] == "running"
