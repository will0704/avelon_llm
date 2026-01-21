"""
OCR Service
Uses Tesseract for text extraction from document images.
"""
from typing import Optional


class OCRService:
    """
    OCR text extraction using Tesseract.
    
    TODO: Implement actual OCR when pytesseract is installed.
    """
    
    def __init__(self):
        self.languages = "eng+fil"  # English + Filipino
    
    def extract_text(self, image_bytes: bytes) -> str:
        """
        Extract text from an image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Extracted text string
        """
        # TODO: Implement actual OCR
        # 1. Convert bytes to PIL Image
        # 2. Preprocess (denoise, deskew)
        # 3. Run pytesseract
        
        # Stub response
        return ""
    
    def extract_structured(self, image_bytes: bytes) -> dict:
        """
        Extract text with bounding boxes and confidence.
        
        Returns:
            Dictionary with text, boxes, and confidence scores
        """
        # TODO: Implement structured extraction
        return {
            "text": "",
            "boxes": [],
            "confidences": []
        }


# Singleton instance
_ocr_service = None

def get_ocr_service() -> OCRService:
    """Get or create OCR service instance."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service
