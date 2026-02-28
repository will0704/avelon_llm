"""
Fraud Detector Service
Detects document manipulation and inconsistencies.
"""
from typing import Dict, Any, List, Optional
from io import BytesIO
import logging

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from app.schemas.fraud import FraudFlag, FraudFlagType, FraudResult
from app.schemas.document import DocumentType, ExtractedDocumentData

logger = logging.getLogger(__name__)


class FraudDetectorService:
    """
    Document fraud detection service.
    
    Analyzes images and extracted data for signs of manipulation
    or inconsistencies that might indicate fraudulent documents.
    """
    
    # Expected fields for document types
    REQUIRED_FIELDS = {
        DocumentType.GOVERNMENT_ID: ["name", "id_number"],
        DocumentType.PROOF_OF_INCOME: ["name", "monthly_income"],
        DocumentType.PROOF_OF_ADDRESS: ["name", "address"],
    }
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._pillow_available = PILLOW_AVAILABLE
        self._cv2_available = CV2_AVAILABLE
    
    def analyze(
        self,
        image_bytes: bytes,
        extracted_data: Optional[ExtractedDocumentData] = None,
        document_type: Optional[DocumentType] = None
    ) -> FraudResult:
        """
        Perform comprehensive fraud analysis.
        
        Args:
            image_bytes: Document image bytes
            extracted_data: Extracted document data from OCR/NER
            document_type: Classified document type
            
        Returns:
            FraudResult with fraud probability and flags
        """
        flags: List[FraudFlag] = []
        
        # 1. Image manipulation analysis
        manipulation_flags = self.detect_manipulation(image_bytes)
        flags.extend(manipulation_flags)
        
        # 2. Text consistency checks
        if extracted_data and document_type:
            consistency_flags = self.check_text_consistency(extracted_data, document_type)
            flags.extend(consistency_flags)
        
        # 3. Calculate overall fraud probability
        fraud_probability = self._calculate_fraud_probability(flags)
        
        # 4. Determine recommendation
        if fraud_probability >= 0.7:
            recommendation = "reject"
        elif fraud_probability >= 0.4:
            recommendation = "review"
        else:
            recommendation = "approve"
        
        return FraudResult(
            is_suspicious=fraud_probability >= 0.4,
            fraud_probability=fraud_probability,
            flags=flags,
            recommendation=recommendation,
            details=self._generate_details(flags)
        )
    
    def detect_manipulation(self, image_bytes: bytes) -> List[FraudFlag]:
        """
        Detect signs of image manipulation.
        
        Checks:
        - EXIF metadata inconsistencies
        - Compression artifacts
        - Image quality anomalies
        """
        flags = []
        
        if not self._pillow_available:
            return flags
        
        try:
            image = Image.open(BytesIO(image_bytes))
            
            # Check EXIF metadata
            exif_flags = self._check_exif(image)
            flags.extend(exif_flags)
            
            # Check for editing software signatures
            software_flags = self._check_software_metadata(image)
            flags.extend(software_flags)
            
        except Exception as e:
            logger.error(f"Manipulation detection failed: {e}")
        
        # OpenCV-based analysis
        if self._cv2_available:
            try:
                quality_flags = self._analyze_image_quality(image_bytes)
                flags.extend(quality_flags)
            except Exception as e:
                logger.error(f"Image quality analysis failed: {e}")
        
        return flags
    
    def _check_exif(self, image: Image.Image) -> List[FraudFlag]:
        """Check EXIF metadata for inconsistencies."""
        flags = []
        
        try:
            exif_data = image._getexif()
            if exif_data is None:
                # No EXIF data - could be legitimate or could be stripped
                flags.append(FraudFlag(
                    flag_type=FraudFlagType.METADATA_MANIPULATION,
                    description="No EXIF metadata found - may have been stripped",
                    severity="low",
                    confidence=0.3
                ))
                return flags
            
            # Check for suspicious software
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                
                if tag == "Software":
                    suspicious_software = ["photoshop", "gimp", "paint.net", "lightroom"]
                    if any(sw in str(value).lower() for sw in suspicious_software):
                        flags.append(FraudFlag(
                            flag_type=FraudFlagType.METADATA_MANIPULATION,
                            description=f"Image edited with software: {value}",
                            severity="high",
                            confidence=0.9
                        ))
                
                if tag == "DateTime" or tag == "DateTimeOriginal":
                    # Could check for date inconsistencies
                    pass
                    
        except Exception as e:
            logger.debug(f"EXIF check failed: {e}")
        
        return flags
    
    def _check_software_metadata(self, image: Image.Image) -> List[FraudFlag]:
        """Check for image editing software signatures in metadata."""
        flags = []
        
        # Check image info for suspicious entries
        info = image.info
        if info:
            for key, value in info.items():
                if isinstance(value, str):
                    value_lower = value.lower()
                    if "adobe" in value_lower or "photoshop" in value_lower:
                        flags.append(FraudFlag(
                            flag_type=FraudFlagType.METADATA_MANIPULATION,
                            description=f"Adobe software signature detected in {key}",
                            severity="medium",
                            confidence=0.7
                        ))
        
        return flags
    
    def _analyze_image_quality(self, image_bytes: bytes) -> List[FraudFlag]:
        """Analyze image quality for anomalies."""
        flags = []
        
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return flags
            
            # Check for extremely low quality (possibly many re-saves)
            # JPEG quality estimation through file size ratio
            height, width = img.shape[:2]
            expected_size = width * height * 3  # Uncompressed estimate
            actual_size = len(image_bytes)
            compression_ratio = actual_size / expected_size
            
            if compression_ratio < 0.01:  # Very high compression
                flags.append(FraudFlag(
                    flag_type=FraudFlagType.QUALITY_INCONSISTENCY,
                    description="Extremely high compression detected - possible multiple re-saves",
                    severity="medium",
                    confidence=0.6
                ))
            
            # Check for uniform regions (possible tampering/editing)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var < 10:  # Very smooth image
                flags.append(FraudFlag(
                    flag_type=FraudFlagType.QUALITY_INCONSISTENCY,
                    description="Unusually smooth image - possible airbrushing or tampering",
                    severity="medium",
                    confidence=0.5
                ))
            
        except Exception as e:
            logger.debug(f"Quality analysis failed: {e}")
        
        return flags
    
    def check_text_consistency(
        self,
        extracted_data: ExtractedDocumentData,
        document_type: DocumentType
    ) -> List[FraudFlag]:
        """
        Check extracted text for consistency and required fields.
        """
        flags = []
        
        # Check for required fields
        required = self.REQUIRED_FIELDS.get(document_type, [])
        for field in required:
            value = getattr(extracted_data, field, None)
            if not value:
                flags.append(FraudFlag(
                    flag_type=FraudFlagType.MISSING_REQUIRED_FIELD,
                    description=f"Required field '{field}' not found for {document_type.value}",
                    severity="high",
                    confidence=0.9
                ))
        
        # Check name format
        if extracted_data.name:
            name = extracted_data.name
            # Check for suspicious name patterns
            if len(name) < 3:
                flags.append(FraudFlag(
                    flag_type=FraudFlagType.SUSPICIOUS_PATTERN,
                    description="Name appears too short",
                    severity="medium",
                    confidence=0.6
                ))
            
            # Check for numbers in name (unusual)
            if any(char.isdigit() for char in name):
                flags.append(FraudFlag(
                    flag_type=FraudFlagType.SUSPICIOUS_PATTERN,
                    description="Name contains numeric characters",
                    severity="low",
                    confidence=0.5
                ))
        
        # Check ID number format
        if extracted_data.id_number:
            id_num = extracted_data.id_number
            # Philippine IDs typically have specific formats
            if len(id_num) < 8:
                flags.append(FraudFlag(
                    flag_type=FraudFlagType.SUSPICIOUS_PATTERN,
                    description="ID number appears too short",
                    severity="medium",
                    confidence=0.7
                ))
        
        return flags
    
    def _calculate_fraud_probability(self, flags: List[FraudFlag]) -> float:
        """Calculate overall fraud probability from flags."""
        if not flags:
            return 0.0
        
        # Weight by severity
        severity_weights = {
            "low": 0.1,
            "medium": 0.25,
            "high": 0.5
        }
        
        total_weight = 0.0
        for flag in flags:
            weight = severity_weights.get(flag.severity, 0.1)
            total_weight += weight * flag.confidence
        
        # Normalize to 0-1 range (cap at 1.0)
        return min(1.0, total_weight)
    
    def _generate_details(self, flags: List[FraudFlag]) -> str:
        """Generate human-readable details from flags."""
        if not flags:
            return "No fraud indicators detected."
        
        high_severity = [f for f in flags if f.severity == "high"]
        medium_severity = [f for f in flags if f.severity == "medium"]
        
        details = []
        if high_severity:
            details.append(f"High severity issues: {len(high_severity)}")
        if medium_severity:
            details.append(f"Medium severity issues: {len(medium_severity)}")
        
        details.append(f"Total flags: {len(flags)}")
        
        return " | ".join(details)


# Singleton instance
_fraud_detector_service = None


def get_fraud_detector_service() -> FraudDetectorService:
    """Get or create fraud detector service instance."""
    global _fraud_detector_service
    if _fraud_detector_service is None:
        from app.config import get_settings
        settings = get_settings()
        _fraud_detector_service = FraudDetectorService(
            model_path=getattr(settings, 'fraud_model_path', None)
        )
    return _fraud_detector_service
