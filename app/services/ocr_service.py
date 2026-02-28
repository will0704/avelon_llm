"""
OCR Service
Uses Tesseract for text extraction from document images.
"""
from typing import Optional, Dict, Any, List
from io import BytesIO
import logging

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class OCRService:
    """
    OCR text extraction using Tesseract.
    
    Supports English + Filipino language extraction with
    confidence scores and bounding box information.
    """
    
    def __init__(self, tesseract_path: Optional[str] = None):
        self.languages = "eng+fil"  # English + Filipino
        self._available = TESSERACT_AVAILABLE
        
        # Set Tesseract path if provided (Windows often needs this)
        if tesseract_path and TESSERACT_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    @property
    def is_available(self) -> bool:
        """Check if Tesseract is available."""
        return self._available
    
    def extract_text(self, image_bytes: bytes) -> str:
        """
        Extract text from an image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Extracted text string
        """
        if not self._available:
            logger.warning("Tesseract not available, returning empty string")
            return ""
        
        try:
            # Preprocess image if OpenCV is available
            processed_image = self._preprocess_image(image_bytes)
            
            # Run OCR
            text = pytesseract.image_to_string(
                processed_image,
                lang=self.languages
            )
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def extract_structured(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract text with bounding boxes and confidence.
        
        Returns:
            Dictionary with text, boxes, and confidence scores
        """
        if not self._available:
            logger.warning("Tesseract not available, returning empty result")
            return {
                "text": "",
                "words": [],
                "boxes": [],
                "confidences": [],
                "average_confidence": 0.0
            }
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image_bytes)
            
            # Get detailed data
            data = pytesseract.image_to_data(
                processed_image,
                lang=self.languages,
                output_type=pytesseract.Output.DICT
            )
            
            # Process results
            words = []
            boxes = []
            confidences = []
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                # Filter out empty entries and low confidence
                conf = int(data['conf'][i])
                text = data['text'][i].strip()
                
                if conf > 0 and text:
                    words.append(text)
                    boxes.append({
                        "x": data['left'][i],
                        "y": data['top'][i],
                        "width": data['width'][i],
                        "height": data['height'][i]
                    })
                    confidences.append(conf)
            
            # Calculate average confidence
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                "text": " ".join(words),
                "words": words,
                "boxes": boxes,
                "confidences": confidences,
                "average_confidence": avg_conf
            }
            
        except Exception as e:
            logger.error(f"Structured OCR extraction failed: {e}")
            return {
                "text": "",
                "words": [],
                "boxes": [],
                "confidences": [],
                "average_confidence": 0.0,
                "error": str(e)
            }
    
    def extract_lines(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Extract text line by line with positions.
        
        Returns:
            List of line dictionaries with text and bounding box
        """
        if not self._available:
            return []
        
        try:
            processed_image = self._preprocess_image(image_bytes)
            
            data = pytesseract.image_to_data(
                processed_image,
                lang=self.languages,
                output_type=pytesseract.Output.DICT
            )
            
            lines = []
            current_line = {"text": [], "line_num": -1}
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                line_num = data['line_num'][i]
                conf = int(data['conf'][i])
                
                if conf > 0 and text:
                    if line_num != current_line["line_num"]:
                        if current_line["text"]:
                            lines.append({
                                "text": " ".join(current_line["text"]),
                                "line_number": current_line["line_num"]
                            })
                        current_line = {"text": [text], "line_num": line_num}
                    else:
                        current_line["text"].append(text)
            
            # Add last line
            if current_line["text"]:
                lines.append({
                    "text": " ".join(current_line["text"]),
                    "line_number": current_line["line_num"]
                })
            
            return lines
            
        except Exception as e:
            logger.error(f"Line extraction failed: {e}")
            return []
    
    def _preprocess_image(self, image_bytes: bytes) -> Image.Image:
        """
        Preprocess image for better OCR accuracy.
        """
        if CV2_AVAILABLE:
            try:
                # Convert to numpy array
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Convert to grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Apply thresholding
                    thresh = cv2.threshold(
                        gray, 0, 255, 
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )[1]
                    
                    # Convert back to PIL Image
                    return Image.fromarray(thresh)
            except Exception as e:
                logger.warning(f"OpenCV preprocessing failed, using raw image: {e}")
        
        # Fallback: use raw image
        return Image.open(BytesIO(image_bytes))
    
    def get_orientation(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Detect image orientation and script.
        
        Returns:
            Dictionary with orientation info and rotation angle
        """
        if not self._available:
            return {"error": "Tesseract not available"}
        
        try:
            image = Image.open(BytesIO(image_bytes))
            osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
            return {
                "orientation": osd.get('orientation', 0),
                "rotate": osd.get('rotate', 0),
                "script": osd.get('script', 'Unknown'),
                "confidence": osd.get('orientation_conf', 0)
            }
        except Exception as e:
            logger.error(f"Orientation detection failed: {e}")
            return {"error": str(e)}


# Singleton instance
_ocr_service = None


def get_ocr_service() -> OCRService:
    """Get or create OCR service instance."""
    global _ocr_service
    if _ocr_service is None:
        from app.config import get_settings
        settings = get_settings()
        _ocr_service = OCRService(
            tesseract_path=getattr(settings, 'tesseract_path', None)
        )
    return _ocr_service
