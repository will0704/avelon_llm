"""
Unit tests for the Entity Extractor Service.
"""
import pytest

from app.services.entity_extractor_service import EntityExtractorService, get_entity_extractor_service
from app.schemas.document import ExtractedDocumentData


class TestEntityExtractorService:
    """Tests for EntityExtractorService."""
    
    @pytest.fixture
    def service(self):
        """Create an entity extractor service instance."""
        return EntityExtractorService()
    
    def test_extract_entities_returns_dict(self, service):
        """Test extract_entities returns expected structure."""
        result = service.extract_entities("John Doe lives in Manila.")
        assert isinstance(result, dict)
        assert "names" in result
        assert "addresses" in result
        assert "dates" in result
        assert "amounts" in result
        assert "id_numbers" in result
    
    def test_extract_id_number_regex(self, service):
        """Test regex extraction of Philippine ID numbers."""
        text = "ID Number: 1234-5678-9012"
        result = service.extract_entities(text)
        assert len(result["id_numbers"]) > 0
    
    def test_extract_amount_regex(self, service):
        """Test regex extraction of monetary amounts."""
        text = "Monthly salary: PHP 50,000.00"
        result = service.extract_entities(text)
        assert len(result["amounts"]) > 0
    
    def test_extract_date_regex(self, service):
        """Test regex extraction of dates."""
        text = "Date of birth: 01/15/1990"
        result = service.extract_entities(text)
        assert len(result["dates"]) > 0
    
    def test_extract_phone_regex(self, service):
        """Test regex extraction of phone numbers."""
        text = "Contact: +63 912 345 6789"
        result = service.extract_entities(text)
        assert len(result["phones"]) > 0
    
    def test_extract_to_document_data(self, service):
        """Test conversion to ExtractedDocumentData."""
        text = "Name: Juan Dela Cruz. ID: 1234-5678-9012. Income: PHP 45,000"
        result = service.extract_to_document_data(text)
        assert isinstance(result, ExtractedDocumentData)
        assert result.raw_text == text
    
    def test_entity_list_structure(self, service):
        """Test entity lists contain expected structure."""
        result = service.extract_entities("Sample text with Amount: PHP 1,000")
        for amount in result["amounts"]:
            assert "value" in amount
            assert "confidence" in amount
    
    def test_singleton_get_service(self):
        """Test singleton pattern returns same instance."""
        service1 = get_entity_extractor_service()
        service2 = get_entity_extractor_service()
        assert service1 is service2
