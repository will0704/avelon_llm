"""
Tests for schema validation hardening.
Covers: FraudFlag severity validation, Config deprecation fix.
"""
import pytest


class TestFraudFlagSeverity:
    """FraudFlag.severity must only accept valid severity levels."""

    def test_valid_severity_low(self):
        from app.schemas.fraud import FraudFlag, FraudFlagType
        flag = FraudFlag(
            flag_type=FraudFlagType.SUSPICIOUS_PATTERN,
            description="test",
            severity="low",
            confidence=0.5,
        )
        assert flag.severity == "low"

    def test_valid_severity_medium(self):
        from app.schemas.fraud import FraudFlag, FraudFlagType
        flag = FraudFlag(
            flag_type=FraudFlagType.SUSPICIOUS_PATTERN,
            description="test",
            severity="medium",
            confidence=0.5,
        )
        assert flag.severity == "medium"

    def test_valid_severity_high(self):
        from app.schemas.fraud import FraudFlag, FraudFlagType
        flag = FraudFlag(
            flag_type=FraudFlagType.SUSPICIOUS_PATTERN,
            description="test",
            severity="high",
            confidence=0.5,
        )
        assert flag.severity == "high"

    def test_invalid_severity_rejected(self):
        """Invalid severity values must be rejected by Pydantic validation."""
        from pydantic import ValidationError
        from app.schemas.fraud import FraudFlag, FraudFlagType

        with pytest.raises(ValidationError):
            FraudFlag(
                flag_type=FraudFlagType.SUSPICIOUS_PATTERN,
                description="test",
                severity="critical",  # Not a valid severity
                confidence=0.5,
            )

    def test_empty_severity_rejected(self):
        """Empty string severity must be rejected."""
        from pydantic import ValidationError
        from app.schemas.fraud import FraudFlag, FraudFlagType

        with pytest.raises(ValidationError):
            FraudFlag(
                flag_type=FraudFlagType.SUSPICIOUS_PATTERN,
                description="test",
                severity="",
                confidence=0.5,
            )


class TestConfigModelConfig:
    """Settings class should use modern Pydantic v2 ConfigDict, not deprecated class Config."""

    def test_settings_uses_model_config(self):
        """Settings should use model_config = SettingsConfigDict(...) not class Config."""
        import inspect
        from app.config import Settings
        source = inspect.getsource(Settings)
        assert "class Config" not in source, (
            "Settings must use model_config = SettingsConfigDict(...) "
            "instead of deprecated class Config"
        )

    def test_settings_loads_from_env(self):
        """Settings should still load from .env file."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"API_KEY": "test-key-123"}):
            from app.config import get_settings
            get_settings.cache_clear()
            settings = get_settings()
            assert settings.api_key == "test-key-123"
            get_settings.cache_clear()


class TestConfigFailFast:
    """Production config must fail-fast if API key is not set."""

    def test_production_rejects_default_api_key(self):
        """In production, the default API key must not be accepted."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "API_KEY": "dev-api-key-change-in-production",
        }, clear=False):
            from app.config import get_settings
            get_settings.cache_clear()

            with pytest.raises((ValueError, SystemExit, Exception)):
                get_settings()

            get_settings.cache_clear()

    def test_development_allows_default_api_key(self):
        """In development, the default API key should be allowed."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {
            "ENVIRONMENT": "development",
            "API_KEY": "dev-api-key-change-in-production",
        }, clear=False):
            from app.config import get_settings
            get_settings.cache_clear()
            settings = get_settings()
            assert settings.api_key == "dev-api-key-change-in-production"
            get_settings.cache_clear()


class TestScorerServiceTypeAnnotation:
    """ScorerService.model_path should have Optional[str] type."""

    def test_model_path_accepts_none(self):
        """ScorerService should accept None as model_path."""
        from app.services.scorer_service import ScorerService
        service = ScorerService(model_path=None)
        assert service.model_path is None

    def test_model_path_accepts_string(self):
        """ScorerService should accept a string model_path."""
        from app.services.scorer_service import ScorerService
        service = ScorerService(model_path="/nonexistent/path.json")
        assert service.model_path == "/nonexistent/path.json"
