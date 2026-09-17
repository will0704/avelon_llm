"""
Uploads a phone can realistically send that are not usable images. Each must
come back as a 4xx the backend can turn into "retake the photo", never a 500
the backend has to treat as an outage.
"""
import io
import os
from unittest.mock import patch, MagicMock

import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image

API_KEY = "test-malformed-key"


@pytest.fixture
def app_instance():
    with patch.dict(os.environ, {"API_KEY": API_KEY, "ENVIRONMENT": "development", "DEBUG": "true"}):
        from app.config import get_settings
        get_settings.cache_clear()
        import importlib
        import app.main
        importlib.reload(app.main)
        yield app.main.app
        get_settings.cache_clear()


def _jpeg(width=300, height=300) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color="white").save(buf, format="JPEG")
    return buf.getvalue()


async def _post_document(app, payload: bytes, content_type="image/jpeg"):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(
            "/api/v1/verify/document?document_type=government_id",
            files={"file": ("id.jpg", payload, content_type)},
            headers={"X-API-Key": API_KEY},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b"", b"\xff\xd8\xff" + b"\x00" * 50, _jpeg()[:400]])
async def test_unreadable_document_is_a_client_error(app_instance, payload):
    response = await _post_document(app_instance, payload)
    assert response.status_code == 422
    assert "could not be read" in response.json()["detail"]


@pytest.mark.asyncio
async def test_oversized_document_is_refused_before_processing(app_instance):
    payload = _jpeg() + b"\x00" * (16 * 1024 * 1024)
    with patch("app.api.routes.verify._verify_single_document") as pipeline:
        response = await _post_document(app_instance, payload)
    assert response.status_code == 413
    pipeline.assert_not_called()


@pytest.mark.asyncio
async def test_unreadable_selfie_is_a_client_error(app_instance):
    face = MagicMock()
    face.is_available = True
    face.verify.side_effect = RuntimeError("cv2 could not decode")
    with patch("app.api.routes.verify.get_face_recognition_service", return_value=face):
        async with AsyncClient(transport=ASGITransport(app=app_instance), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/verify/face",
                files={
                    "selfie_file": ("selfie.jpg", b"not really a jpeg", "image/jpeg"),
                    "government_id_file": ("id.jpg", _jpeg(), "image/jpeg"),
                },
                headers={"X-API-Key": API_KEY},
            )
    assert response.status_code == 422
    assert "could not be read" in response.json()["detail"]
    face.verify.assert_not_called()
