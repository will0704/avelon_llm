"""
Configuration settings for Avelon LLM service.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import model_validator
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Environment
    environment: str = "development"
    debug: bool = True
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS
    backend_url: str = "http://localhost:3000"
    
    # API Security
    api_key: str = "dev-api-key-change-in-production"
    
    # Model Paths (optional until models are trained)
    classifier_model_path: Optional[str] = None
    ner_model_path: Optional[str] = None
    fraud_model_path: Optional[str] = None
    scorer_model_path: Optional[str] = None
    philid_model_path: Optional[str] = None  # PyTorch MobileNetV2 for PhilID
    
    # OCR Configuration
    tesseract_path: Optional[str] = None  # Path to tesseract executable (Windows)
    
    # NER Configuration  
    ner_model_name: str = "dslim/bert-base-NER"  # Default HuggingFace model
    
    # Processing Thresholds
    confidence_threshold: float = 0.7  # Minimum confidence for entity extraction
    fraud_threshold: float = 0.4  # Threshold for flagging suspicious documents

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @model_validator(mode="after")
    def _reject_default_key_in_production(self) -> "Settings":
        if (
            self.environment == "production"
            and self.api_key == "dev-api-key-change-in-production"
        ):
            raise ValueError(
                "API_KEY must be changed from the default value in production"
            )
        return self


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
