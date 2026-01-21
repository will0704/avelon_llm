"""
Configuration settings for Avelon LLM service.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings
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
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
