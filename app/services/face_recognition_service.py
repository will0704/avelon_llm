"""
Face recognition service using InsightFace (ArcFace ResNet50 via ONNX Runtime).

Compares a selfie image against a government ID photo to verify identity.
Uses cosine similarity — similarity >= 0.35 = same person.

InsightFace's buffalo_l model outputs L2-normalised embeddings;
cosine similarity is therefore the dot product, ranging from -1 to 1.
Typical same-person similarity: 0.4–0.9 (higher with better photo quality).
Typical different-person similarity: < 0.2.
ID-photo-vs-selfie introduces ~0.1–0.2 degradation due to domain shift.
"""
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Similarity threshold — insightface buffalo_l ArcFace:
# >= 0.35 = same person (conservative, accounts for ID-photo quality loss)
MATCH_THRESHOLD = 0.35

_face_service_instance: Optional["FaceRecognitionService"] = None


class FaceRecognitionError(Exception):
    """Raised when face detection or comparison fails."""
    def __init__(self, message: str, error_code: str):
        super().__init__(message)
        self.error_code = error_code


class FaceRecognitionService:
    """
    Wraps InsightFace (ArcFace ResNet50, ONNX Runtime) for face verification.

    InsightFace is lazy-imported so the service starts up even if not installed.
    Unlike DeepFace, InsightFace has no TensorFlow dependency and works on
    Python 3.14+ via onnxruntime.
    """

    def __init__(self):
        self._insightface_available = False
        self._app = None
        self._load_insightface()

    def _load_insightface(self) -> None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
            import numpy  # noqa: F401 — needed by insightface internals
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._app = app
            self._insightface_available = True
            logger.info("[FaceRecognition] InsightFace (buffalo_l ArcFace) loaded successfully")
        except ImportError:
            logger.warning(
                "[FaceRecognition] InsightFace not installed. "
                "Run: pip install insightface onnxruntime. Face verification will be unavailable."
            )
        except Exception as e:
            logger.warning("[FaceRecognition] Failed to load InsightFace model: %s", e)

    @property
    def is_available(self) -> bool:
        return self._insightface_available and self._app is not None

    def verify(
        self,
        selfie_bytes: bytes,
        gov_id_bytes: bytes,
    ) -> Tuple[bool, float, float, Optional[str]]:
        """
        Compare selfie against government ID photo.

        Returns:
            (passed, score, confidence, message)
            - passed:     True if the same person is in both images
            - score:      cosine similarity 0-1 (higher = more similar)
            - confidence: how far the score is from the threshold (0-1)
            - message:    failure explanation or None on success
        """
        if not self.is_available:
            raise FaceRecognitionError(
                "Face recognition model is not available. Install insightface and onnxruntime.",
                "MODEL_UNAVAILABLE",
            )

        import numpy as np
        import cv2

        def _decode(img_bytes: bytes, label: str):
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise FaceRecognitionError(
                    f"Could not decode {label} image.",
                    "NO_FACE_IN_SELFIE" if label == "selfie" else "NO_FACE_IN_ID",
                )
            return img

        selfie_img = _decode(selfie_bytes, "selfie")
        id_img = _decode(gov_id_bytes, "government ID")

        # Detect faces and extract embeddings
        selfie_faces = self._app.get(selfie_img)
        id_faces = self._app.get(id_img)

        if not selfie_faces:
            raise FaceRecognitionError(
                "No face detected in the selfie. Please take a clear, well-lit photo facing the camera.",
                "NO_FACE_IN_SELFIE",
            )
        if not id_faces:
            raise FaceRecognitionError(
                "No face detected in the government ID. Please upload a clear photo ID.",
                "NO_FACE_IN_ID",
            )

        # Use the highest-confidence face from each image
        selfie_face = max(selfie_faces, key=lambda f: f.det_score)
        id_face = max(id_faces, key=lambda f: f.det_score)

        # Cosine similarity (embeddings are already L2-normalised)
        similarity: float = float(np.dot(selfie_face.normed_embedding, id_face.normed_embedding))

        # Clamp to [0, 1] for the score (negative similarity = clearly different people)
        score = round(max(0.0, min(1.0, similarity)), 4)

        passed = similarity >= MATCH_THRESHOLD

        # Confidence: distance from threshold normalised to [0, 1]
        # Well above threshold → high confidence; near threshold → low confidence
        margin = similarity - MATCH_THRESHOLD
        confidence = round(min(1.0, max(0.0, 0.5 + margin / MATCH_THRESHOLD)), 4)

        message = None
        if not passed:
            if similarity >= 0.2:
                message = "Face similarity is low. The selfie may not match the ID photo."
            else:
                message = "Face does not match the government ID photo."

        return passed, score, confidence, message


def get_face_recognition_service() -> FaceRecognitionService:
    """Return the singleton FaceRecognitionService instance (lazy-loaded)."""
    global _face_service_instance
    if _face_service_instance is None:
        _face_service_instance = FaceRecognitionService()
    return _face_service_instance
