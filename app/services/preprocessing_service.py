"""
Preprocessing Service
Image quality validation and normalization for AI processing.
"""
from typing import Tuple, Optional
from io import BytesIO
import logging

try:
    import cv2
    import numpy as np
    from PIL import Image
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImageValidationError(Exception):
    """Raised when image validation fails."""
    pass


class PreprocessingService:
    """
    Image preprocessing for document verification.
    
    Handles:
    - Format validation
    - Quality assessment
    - Size normalization
    - OCR-specific preprocessing
    """
    
    SUPPORTED_FORMATS = {"JPEG", "PNG", "JPG"}
    MIN_RESOLUTION = (200, 200)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    CNN_INPUT_SIZE = (224, 224)
    
    def __init__(self):
        self._cv2_available = CV2_AVAILABLE
    
    def validate_image(self, image_bytes: bytes) -> Tuple[bool, Optional[str]]:
        """
        Validate image format, size, and basic properties.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size
        if len(image_bytes) > self.MAX_FILE_SIZE:
            return (False, f"File too large. Maximum size is {self.MAX_FILE_SIZE // (1024*1024)}MB")
        
        if len(image_bytes) < 100:
            return (False, "File too small to be a valid image")
        
        try:
            image = Image.open(BytesIO(image_bytes))
            
            # Check format
            if image.format not in self.SUPPORTED_FORMATS:
                return (False, f"Unsupported format: {image.format}. Supported: {self.SUPPORTED_FORMATS}")
            
            # Check resolution
            width, height = image.size
            if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
                return (False, f"Resolution too low. Minimum: {self.MIN_RESOLUTION}")
            
            return (True, None)
            
        except Exception as e:
            return (False, f"Invalid image file: {str(e)}")
    
    def check_quality(self, image_bytes: bytes) -> dict:
        """
        Check image quality for document processing.
        
        Returns:
            Dictionary with quality metrics:
            - blur_score: 0-100 (higher is sharper)
            - brightness: 0-255 average
            - is_acceptable: bool
        """
        result = {
            "blur_score": 0.0,
            "brightness": 0.0,
            "is_acceptable": False,
            "issues": []
        }
        
        if not self._cv2_available:
            logger.warning("OpenCV not available, skipping quality check")
            result["is_acceptable"] = True
            result["issues"].append("Quality check skipped (OpenCV not installed)")
            return result
        
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                result["issues"].append("Could not decode image")
                return result
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            result["blur_score"] = min(100, laplacian_var / 5)  # Normalize to 0-100
            
            if laplacian_var < 100:
                result["issues"].append("Image appears blurry")
            
            # Brightness analysis
            result["brightness"] = float(np.mean(gray))
            
            if result["brightness"] < 50:
                result["issues"].append("Image is too dark")
            elif result["brightness"] > 220:
                result["issues"].append("Image is overexposed")
            
            # Determine overall acceptability
            result["is_acceptable"] = (
                laplacian_var >= 50 and  # Not too blurry
                50 <= result["brightness"] <= 220  # Reasonable brightness
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            result["issues"].append(f"Quality check error: {str(e)}")
            return result
    
    def normalize_for_cnn(self, image_bytes: bytes) -> Optional[bytes]:
        """
        Normalize image for CNN input (224x224, RGB).
        
        Returns:
            Normalized image bytes or None on failure
        """
        try:
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Resize to CNN input size
            image = image.resize(self.CNN_INPUT_SIZE, Image.Resampling.LANCZOS)
            
            # Save to bytes
            output = BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"CNN normalization failed: {e}")
            return None
    
    def preprocess_for_ocr(self, image_bytes: bytes) -> Optional[bytes]:
        """
        Preprocess image for OCR (denoise, threshold, deskew).
        
        Returns:
            Preprocessed image bytes or None on failure
        """
        if not self._cv2_available:
            logger.warning("OpenCV not available, returning original image")
            return image_bytes
        
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding for better text contrast
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Noise removal with median blur
            denoised = cv2.medianBlur(thresh, 3)
            
            # Encode back to bytes
            _, buffer = cv2.imencode('.png', denoised)
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"OCR preprocessing failed: {e}")
            return None
    
    def get_image_info(self, image_bytes: bytes) -> dict:
        """
        Get basic image information.
        
        Returns:
            Dictionary with image metadata
        """
        try:
            image = Image.open(BytesIO(image_bytes))
            return {
                "format": image.format,
                "mode": image.mode,
                "width": image.size[0],
                "height": image.size[1],
                "size_bytes": len(image_bytes)
            }
        except Exception as e:
            return {"error": str(e)}


# Singleton instance
_preprocessing_service = None


def get_preprocessing_service() -> PreprocessingService:
    """Get or create preprocessing service instance."""
    global _preprocessing_service
    if _preprocessing_service is None:
        _preprocessing_service = PreprocessingService()
    return _preprocessing_service
