"""
Entity Extractor Service
Uses EasyOCR for text extraction + regex/NER for entity extraction.
Supports English and Filipino (Tagalog) languages.
"""
from typing import Dict, Any, List, Optional
from io import BytesIO
import logging
import re
import gc

# EasyOCR
try:
    import easyocr
    import numpy as np
    from PIL import Image
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Optional: BERT NER for enhanced entity extraction
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from app.schemas.document import ExtractedDocumentData

logger = logging.getLogger(__name__)


class EntityExtractorService:
    """
    Entity extraction using EasyOCR + regex patterns.
    
    Supports:
    - English and Filipino (Tagalog) text extraction
    - PhilID-specific field extraction
    - General document entity extraction
    """
    
    # Entity mapping from generic NER labels to our domain
    ENTITY_MAPPING = {
        "PER": "name", "PERSON": "name", "B-PER": "name", "I-PER": "name",
        "LOC": "address", "LOCATION": "address", "GPE": "address", "B-LOC": "address",
        "ORG": "employer_name", "ORGANIZATION": "employer_name", "B-ORG": "employer_name",
        "DATE": "date", "MONEY": "amount", "CARDINAL": "amount",
    }
    
    # Regex patterns for entity extraction
    PATTERNS = {
        "id_number": [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # PhilID (16 digits)
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # 12-digit ID
            r'\b[A-Z]\d{2}[-\s]?\d{2}[-\s]?\d{6}\b',  # Driver's license
        ],
        "date": [
            r'\b(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b',
            r'\b\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}\b',
        ],
        "amount": [
            r'(?:PHP|₱|P)\s*[\d,]+(?:\.\d{2})?',  # Philippine Peso
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b',  # Comma-separated
        ],
        "phone": [
            r'(?:\+63|0)[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{4}\b',  # Philippine phone
        ],
    }
    
    def __init__(self, use_gpu: bool = False):
        self._reader = None
        self._ner_pipeline = None
        self._use_gpu = use_gpu
        self._available = EASYOCR_AVAILABLE
    
    @property
    def reader(self):
        """Lazy-load EasyOCR reader to save memory."""
        if self._reader is None and EASYOCR_AVAILABLE:
            logger.info("Initializing EasyOCR reader (English + Tagalog)...")
            self._reader = easyocr.Reader(
                ['en', 'tl'],
                gpu=self._use_gpu,
                verbose=False
            )
            logger.info("EasyOCR reader initialized")
        return self._reader
    
    @property
    def ner_pipeline(self):
        """Lazy-load NER pipeline for enhanced extraction."""
        if self._ner_pipeline is None and TRANSFORMERS_AVAILABLE:
            try:
                self._ner_pipeline = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple"
                )
                logger.info("NER pipeline loaded")
            except Exception as e:
                logger.warning(f"Failed to load NER pipeline: {e}")
        return self._ner_pipeline
    
    @property
    def is_available(self) -> bool:
        """Check if EasyOCR is available."""
        return EASYOCR_AVAILABLE
    
    def extract_text(self, image_bytes: bytes) -> str:
        """
        Extract text from image using EasyOCR.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Extracted text as string
        """
        if not self.is_available:
            logger.warning("EasyOCR not available")
            return ""
        
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            
            # Resize if too large
            max_size = 1280
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            image_np = np.array(image)
            results = self.reader.readtext(image_np, detail=0)
            
            del image_np
            gc.collect()
            
            return " ".join(results)
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract named entities from text using regex + optional NER.
        """
        entities = {
            "names": [], "addresses": [], "dates": [],
            "amounts": [], "id_numbers": [], "employers": [],
            "phones": [],
        }
        
        # Run NER if available
        if self.ner_pipeline:
            try:
                for entity in self.ner_pipeline(text):
                    entity_type = entity.get("entity_group", "")
                    word = entity.get("word", "")
                    score = entity.get("score", 0.0)
                    
                    mapped = self.ENTITY_MAPPING.get(entity_type)
                    if mapped == "name":
                        entities["names"].append({"value": word, "confidence": score})
                    elif mapped == "address":
                        entities["addresses"].append({"value": word, "confidence": score})
                    elif mapped == "employer_name":
                        entities["employers"].append({"value": word, "confidence": score})
            except Exception as e:
                logger.warning(f"NER failed: {e}")
        
        # Run regex patterns
        self._extract_with_regex(text, entities)
        
        return entities
    
    def _extract_with_regex(self, text: str, entities: Dict[str, List]) -> None:
        """Extract entities using regex patterns."""
        for pattern_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = " ".join(m for m in match if m)
                    
                    entry = {"value": match.strip(), "confidence": 0.8, "source": "regex"}
                    key = pattern_type + "s" if pattern_type != "phone" else "phones"
                    
                    # Avoid duplicates
                    if not any(e["value"] == entry["value"] for e in entities.get(key, [])):
                        if key in entities:
                            entities[key].append(entry)
    
    def extract_from_image(self, image_bytes: bytes) -> ExtractedDocumentData:
        """
        Full pipeline: OCR + entity extraction from image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            ExtractedDocumentData with populated fields
        """
        # Extract text
        text = self.extract_text(image_bytes)
        
        if not text:
            return ExtractedDocumentData(raw_text="")
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Get best match helper
        def get_best(lst: List[Dict]) -> Optional[str]:
            if not lst:
                return None
            return sorted(lst, key=lambda x: x["confidence"], reverse=True)[0]["value"]
        
        # Parse amount
        def parse_amount(s: Optional[str]) -> Optional[float]:
            if not s:
                return None
            try:
                return float(re.sub(r'[₱PHP,\s]', '', s))
            except ValueError:
                return None
        
        return ExtractedDocumentData(
            raw_text=text,
            name=get_best(entities["names"]),
            address=get_best(entities["addresses"]),
            id_number=get_best(entities["id_numbers"]),
            date_of_birth=get_best(entities["dates"]),
            employer_name=get_best(entities["employers"]),
            monthly_income=parse_amount(get_best(entities["amounts"])),
            extra={
                "all_entities": entities,
                "extraction_method": "easyocr"
            }
        )
    
    def extract_from_back_image(self, image_bytes: bytes) -> ExtractedDocumentData:
        """
        Extraction pipeline for the back of a government ID.

        Detects QR code, signature presence, and emergency contact info.
        Does NOT attempt to extract front-specific fields (name, DOB, id_number).
        """
        # Extract any text present on the back
        text = self.extract_text(image_bytes)

        # Detect QR code
        has_qr, qr_data = self._detect_qr_code(image_bytes)

        # Detect signature region
        has_sig = self._detect_signature_region(image_bytes)

        # Extract emergency contact phone number from OCR text
        emergency_contact: Optional[str] = None
        if text:
            for pattern in self.PATTERNS.get("phone", []):
                match = re.search(pattern, text)
                if match:
                    emergency_contact = match.group(0)
                    break

        extra: Dict[str, Any] = {"extraction_method": "easyocr_back"}
        if qr_data:
            extra["qr_data"] = qr_data

        return ExtractedDocumentData(
            raw_text=text or "",
            has_qr_code=has_qr,
            has_signature=has_sig,
            emergency_contact=emergency_contact,
            extra=extra,
        )

    def _detect_qr_code(self, image_bytes: bytes) -> tuple:
        """Detect QR code presence and optionally decode its data."""
        try:
            import cv2
            import numpy as np
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return False, None
            detector = cv2.QRCodeDetector()
            data, points, _ = detector.detectAndDecode(img)
            if points is not None:
                return True, data if data else None
            return False, None
        except Exception:
            return False, None

    def _detect_signature_region(self, image_bytes: bytes) -> bool:
        """Heuristic: check for ink-like marks in the signature strip area."""
        try:
            import cv2
            import numpy as np
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False
            h, w = img.shape
            # Signature strip is typically in the bottom third of the card
            strip = img[int(h * 0.55):int(h * 0.85), int(w * 0.1):int(w * 0.9)]
            _, thresh = cv2.threshold(strip, 100, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Filter for contours that look like handwriting strokes
            sig_contours = [c for c in contours if cv2.contourArea(c) > 50]
            return len(sig_contours) >= 3
        except Exception:
            return False

    def extract_to_document_data(self, text: str) -> ExtractedDocumentData:
        """
        Extract entities from pre-extracted text.
        (For backwards compatibility)
        """
        entities = self.extract_entities(text)
        
        def get_best(lst: List[Dict]) -> Optional[str]:
            if not lst:
                return None
            return sorted(lst, key=lambda x: x["confidence"], reverse=True)[0]["value"]
        
        return ExtractedDocumentData(
            raw_text=text,
            name=get_best(entities["names"]),
            address=get_best(entities["addresses"]),
            id_number=get_best(entities["id_numbers"]),
            date_of_birth=get_best(entities["dates"]),
            employer_name=get_best(entities["employers"]),
        )


# Singleton instance
_entity_extractor_service: Optional[EntityExtractorService] = None


def get_entity_extractor_service() -> EntityExtractorService:
    """Get or create entity extractor service instance."""
    global _entity_extractor_service
    if _entity_extractor_service is None:
        _entity_extractor_service = EntityExtractorService()
    return _entity_extractor_service
