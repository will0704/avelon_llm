"""
Tests for the face recognition service and /verify/face endpoint.

All heavy dependencies (insightface, onnxruntime) are mocked at module level
so this test file is self-contained and does not require those packages.
"""
import sys
import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# ── Mock heavy deps at module level ─────────────────────────────────────────
sys.modules.setdefault("insightface", MagicMock())
sys.modules.setdefault("insightface.app", MagicMock())

# ── Import service directly to avoid app.services.__init__ (pulls in torch) ─
import importlib.util as _ilu
import pathlib as _pl

_svc_path = _pl.Path(__file__).parent.parent / "app" / "services" / "face_recognition_service.py"
_spec = _ilu.spec_from_file_location("face_recognition_service", _svc_path)
_mod = _ilu.module_from_spec(_spec)   # type: ignore[arg-type]
_spec.loader.exec_module(_mod)         # type: ignore[union-attr]

FaceRecognitionService = _mod.FaceRecognitionService
FaceRecognitionError = _mod.FaceRecognitionError
MATCH_THRESHOLD = _mod.MATCH_THRESHOLD
get_face_recognition_service = _mod.get_face_recognition_service


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_svc(available: bool = True) -> FaceRecognitionService:
    """Create a service instance without running __init__."""
    _mod._face_service_instance = None
    svc = FaceRecognitionService.__new__(FaceRecognitionService)
    svc._insightface_available = available
    svc._app = MagicMock() if available else None
    return svc


def _make_face(embedding: list, det_score: float = 0.9):
    """Create a mock InsightFace face object."""
    face = MagicMock()
    face.normed_embedding = np.array(embedding, dtype=np.float32)
    face.det_score = det_score
    return face


def _fake_decode(*args, **kwargs):
    """Return a dummy image array (valid shape for cv2.imdecode)."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


# ── FaceRecognitionService unit tests ─────────────────────────────────────────

class TestFaceRecognitionService:

    def test_is_available_true_when_insightface_loaded(self):
        svc = _make_svc(available=True)
        assert svc.is_available is True

    def test_is_available_false_when_insightface_missing(self):
        svc = _make_svc(available=False)
        assert svc.is_available is False

    def test_verify_raises_when_service_unavailable(self):
        svc = _make_svc(available=False)
        with pytest.raises(FaceRecognitionError) as exc_info:
            svc.verify(b"selfie", b"gov_id")
        assert exc_info.value.error_code == "MODEL_UNAVAILABLE"

    def test_verify_calls_app_get_for_both_images(self):
        svc = _make_svc(available=True)
        emb = [1.0, 0.0, 0.0]
        face = _make_face(emb)
        svc._app.get.return_value = [face]

        with patch("cv2.imdecode", side_effect=_fake_decode):
            svc.verify(b"selfie", b"gov_id")

        assert svc._app.get.call_count == 2

    def test_verify_returns_passed_true_when_similarity_above_threshold(self):
        svc = _make_svc(available=True)
        # dot([1,0,0], [1,0,0]) = 1.0 — well above threshold
        face = _make_face([1.0, 0.0, 0.0])
        svc._app.get.return_value = [face]

        with patch("cv2.imdecode", side_effect=_fake_decode):
            passed, score, confidence, message = svc.verify(b"selfie", b"gov_id")

        assert passed is True
        assert score == pytest.approx(1.0, abs=0.01)
        assert message is None

    def test_verify_returns_passed_false_when_similarity_below_threshold(self):
        svc = _make_svc(available=True)
        # dot([1,0,0], [0,1,0]) = 0.0 — below threshold (0.35)
        selfie_face = _make_face([1.0, 0.0, 0.0])
        id_face = _make_face([0.0, 1.0, 0.0])
        svc._app.get.side_effect = [[selfie_face], [id_face]]

        with patch("cv2.imdecode", side_effect=_fake_decode):
            passed, score, confidence, message = svc.verify(b"selfie", b"gov_id")

        assert passed is False
        assert score == pytest.approx(0.0, abs=0.01)
        assert message is not None

    def test_verify_raises_no_face_in_selfie(self):
        svc = _make_svc(available=True)
        svc._app.get.side_effect = [[], [_make_face([1.0, 0.0, 0.0])]]

        with patch("cv2.imdecode", side_effect=_fake_decode):
            with pytest.raises(FaceRecognitionError) as exc_info:
                svc.verify(b"selfie", b"gov_id")

        assert exc_info.value.error_code == "NO_FACE_IN_SELFIE"

    def test_verify_raises_no_face_in_id(self):
        svc = _make_svc(available=True)
        svc._app.get.side_effect = [[_make_face([1.0, 0.0, 0.0])], []]

        with patch("cv2.imdecode", side_effect=_fake_decode):
            with pytest.raises(FaceRecognitionError) as exc_info:
                svc.verify(b"selfie", b"gov_id")

        assert exc_info.value.error_code == "NO_FACE_IN_ID"

    def test_verify_raises_decode_error_for_invalid_selfie(self):
        svc = _make_svc(available=True)

        with patch("cv2.imdecode", return_value=None):
            with pytest.raises(FaceRecognitionError) as exc_info:
                svc.verify(b"bad", b"gov_id")

        assert exc_info.value.error_code == "NO_FACE_IN_SELFIE"

    def test_score_clamped_to_zero_for_negative_similarity(self):
        svc = _make_svc(available=True)
        # dot([1,0,0], [-1,0,0]) = -1.0
        selfie_face = _make_face([1.0, 0.0, 0.0])
        id_face = _make_face([-1.0, 0.0, 0.0])
        svc._app.get.side_effect = [[selfie_face], [id_face]]

        with patch("cv2.imdecode", side_effect=_fake_decode):
            passed, score, confidence, message = svc.verify(b"selfie", b"gov_id")

        assert score == 0.0
        assert passed is False

    def test_similarity_exactly_at_threshold_passes(self):
        """Similarity >= MATCH_THRESHOLD should pass. Use float64 to avoid float32 rounding."""
        svc = _make_svc(available=True)
        # Use float64 to avoid float32 rounding below threshold
        target = MATCH_THRESHOLD + 1e-6
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        sin_val = float(np.sqrt(max(0.0, 1.0 - target ** 2)))
        emb2 = np.array([target, sin_val, 0.0], dtype=np.float64)

        svc._app.get.side_effect = [[_make_face(emb1.tolist())], [_make_face(emb2.tolist())]]

        with patch("cv2.imdecode", side_effect=_fake_decode):
            passed, _, _, _ = svc.verify(b"selfie", b"gov_id")

        assert passed is True

    def test_similarity_just_below_threshold_fails(self):
        svc = _make_svc(available=True)
        below = MATCH_THRESHOLD - 0.01
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sin_val = float(np.sqrt(max(0.0, 1.0 - below ** 2)))
        emb2 = np.array([below, sin_val, 0.0], dtype=np.float32)

        svc._app.get.side_effect = [[_make_face(emb1.tolist())], [_make_face(emb2.tolist())]]

        with patch("cv2.imdecode", side_effect=_fake_decode):
            passed, _, _, _ = svc.verify(b"selfie", b"gov_id")

        assert passed is False

    def test_rejects_multiple_faces_in_selfie(self):
        """A group photo must not be silently reduced to one selected face."""
        svc = _make_svc(available=True)
        low_conf = _make_face([0.0, 1.0, 0.0], det_score=0.3)   # orthogonal — would fail
        high_conf = _make_face([1.0, 0.0, 0.0], det_score=0.95)  # identical — would pass
        id_face = _make_face([1.0, 0.0, 0.0])
        svc._app.get.side_effect = [[low_conf, high_conf], [id_face]]

        with patch("cv2.imdecode", side_effect=_fake_decode), pytest.raises(FaceRecognitionError) as exc:
            svc.verify(b"selfie", b"gov_id")

        assert exc.value.error_code == "MULTIPLE_FACES_IN_SELFIE"

    def test_get_face_recognition_service_returns_singleton(self):
        _mod._face_service_instance = None
        svc1 = get_face_recognition_service()
        svc2 = get_face_recognition_service()
        assert svc1 is svc2

    def test_error_includes_code(self):
        err = FaceRecognitionError("test message", "TEST_CODE")
        assert err.error_code == "TEST_CODE"
        assert str(err) == "test message"


# ── /verify/face endpoint tests ───────────────────────────────────────────────

@pytest.fixture
def app_instance():
    """Create FastAPI app, mocking heavy deps to avoid torch/insightface install."""
    with patch.dict(os.environ, {
        "API_KEY": "test-key",
        "ENVIRONMENT": "development",
        "DEBUG": "true",
    }):
        _TorchModule = type("Module", (object,), {
            "__init_subclass__": classmethod(lambda cls, **kw: None),
        })
        _nn_mock = MagicMock()
        _nn_mock.Module = _TorchModule
        torch_mock = MagicMock()
        torch_mock.nn = _nn_mock

        heavy_mocks = {
            "torch": torch_mock,
            "torch.nn": _nn_mock,
            "torch.nn.functional": MagicMock(),
            "torchvision": MagicMock(),
            "torchvision.models": MagicMock(),
            "torchvision.transforms": MagicMock(),
            "cv2": MagicMock(),
            "easyocr": MagicMock(),
            "xgboost": MagicMock(),
            "pytesseract": MagicMock(),
            "joblib": MagicMock(),
            "PIL": MagicMock(),
            "PIL.Image": MagicMock(),
            "numpy": MagicMock(),
            "sklearn": MagicMock(),
            "sklearn.ensemble": MagicMock(),
            "transformers": MagicMock(),
            "insightface": MagicMock(),
            "insightface.app": MagicMock(),
            "onnxruntime": MagicMock(),
        }

        app_keys = [k for k in sys.modules if k.startswith("app.") or k == "app"]
        saved = {k: sys.modules.pop(k) for k in app_keys}

        with patch.dict(sys.modules, heavy_mocks):
            from app.config import get_settings
            get_settings.cache_clear()

            import importlib
            import app.main
            importlib.reload(app.main)
            yield app.main.app

            get_settings.cache_clear()

        sys.modules.update(saved)


class TestFaceVerifyEndpoint:
    HEADERS = {"X-API-Key": "test-key"}

    @pytest.mark.asyncio
    async def test_requires_api_key(self, app_instance):
        from httpx import AsyncClient, ASGITransport

        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/verify/face",
                headers={"X-API-Key": "wrong-key"},
                files={
                    "selfie_file": ("selfie.jpg", b"fake", "image/jpeg"),
                    "government_id_file": ("id.jpg", b"fake", "image/jpeg"),
                },
            )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_rejects_non_image_selfie(self, app_instance):
        from httpx import AsyncClient, ASGITransport

        mock_svc = MagicMock()
        mock_svc.is_available = True

        with patch("app.api.routes.verify.get_face_recognition_service", return_value=mock_svc):
            transport = ASGITransport(app=app_instance)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/verify/face",
                    headers=self.HEADERS,
                    files={
                        "selfie_file": ("selfie.pdf", b"fake-pdf", "application/pdf"),
                        "government_id_file": ("id.jpg", b"fake-id", "image/jpeg"),
                    },
                )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_returns_503_when_service_unavailable(self, app_instance):
        from httpx import AsyncClient, ASGITransport

        mock_svc = MagicMock()
        mock_svc.is_available = False

        with patch("app.api.routes.verify.get_face_recognition_service", return_value=mock_svc):
            transport = ASGITransport(app=app_instance)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/verify/face",
                    headers=self.HEADERS,
                    files={
                        "selfie_file": ("selfie.jpg", b"fake", "image/jpeg"),
                        "government_id_file": ("id.jpg", b"fake", "image/jpeg"),
                    },
                )
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_returns_200_on_match(self, app_instance):
        from httpx import AsyncClient, ASGITransport

        mock_svc = MagicMock()
        mock_svc.is_available = True
        mock_svc.verify.return_value = (True, 0.88, 0.95, None)

        with patch("app.api.routes.verify.get_face_recognition_service", return_value=mock_svc):
            transport = ASGITransport(app=app_instance)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/verify/face",
                    headers=self.HEADERS,
                    files={
                        "selfie_file": ("selfie.jpg", b"fake-selfie", "image/jpeg"),
                        "government_id_file": ("id.jpg", b"fake-id", "image/jpeg"),
                    },
                )

        assert response.status_code == 200
        data = response.json()
        assert data["passed"] is True
        assert abs(data["score"] - 0.88) < 0.01
        assert abs(data["confidence"] - 0.95) < 0.01
        assert data["message"] is None

    @pytest.mark.asyncio
    async def test_returns_200_on_mismatch(self, app_instance):
        from httpx import AsyncClient, ASGITransport

        mock_svc = MagicMock()
        mock_svc.is_available = True
        mock_svc.verify.return_value = (False, 0.25, 0.3, "Face does not match the government ID photo.")

        with patch("app.api.routes.verify.get_face_recognition_service", return_value=mock_svc):
            transport = ASGITransport(app=app_instance)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/verify/face",
                    headers=self.HEADERS,
                    files={
                        "selfie_file": ("selfie.jpg", b"fake", "image/jpeg"),
                        "government_id_file": ("id.jpg", b"fake", "image/jpeg"),
                    },
                )

        assert response.status_code == 200
        data = response.json()
        assert data["passed"] is False
        assert data["message"] is not None

    @pytest.mark.asyncio
    async def test_returns_422_on_no_face_detected(self, app_instance):
        from httpx import AsyncClient, ASGITransport
        from app.services.face_recognition_service import FaceRecognitionError as AppFaceRecognitionError

        mock_svc = MagicMock()
        mock_svc.is_available = True
        mock_svc.verify.side_effect = AppFaceRecognitionError("No face", "NO_FACE_IN_SELFIE")

        with patch("app.api.routes.verify.get_face_recognition_service", return_value=mock_svc):
            transport = ASGITransport(app=app_instance)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/verify/face",
                    headers=self.HEADERS,
                    files={
                        "selfie_file": ("selfie.jpg", b"fake", "image/jpeg"),
                        "government_id_file": ("id.jpg", b"fake", "image/jpeg"),
                    },
                )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_health_includes_face_recognition_status(self, app_instance):
        from httpx import AsyncClient, ASGITransport

        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        models = response.json()["models_loaded"]
        assert "face_recognition" in models
        assert isinstance(models["face_recognition"], bool)
