"""
Unit tests for the XGBoost Credit Scorer model training and integration.
Tests both the XGBoost model path and rule-based fallback.
"""
import os
import pytest

from app.services.scorer_service import ScorerService
from app.schemas.score import LoanHistory, ScoreBreakdown


class TestScorerMLModel:
    """Tests for ML-enhanced credit scoring."""

    @pytest.fixture
    def service_no_model(self):
        """Create scorer service without XGBoost model."""
        return ScorerService(model_path=None)

    @pytest.fixture
    def full_extracted_data(self):
        """Complete extracted data for scoring."""
        return {
            "verified_documents": {
                "government_id": {"is_verified": True, "confidence": 0.95},
                "proof_of_income": {"is_verified": True},
                "proof_of_address": {"is_verified": True}
            },
            "monthly_income": 50000,
            "employment_type": "permanent",
            "years_employed": 3,
            "debt_to_income_ratio": 0.2,
        }

    @pytest.fixture
    def full_wallet_data(self):
        """Complete wallet data for scoring."""
        return {
            "age_days": 200,
            "transaction_count": 60,
            "balance_eth": 1.5,
        }

    def test_ml_model_loaded_property_false(self, service_no_model):
        """Service should report ml_model_loaded=False when no model."""
        assert service_no_model.ml_model_loaded is False

    def test_service_without_model_uses_rules(self, service_no_model, full_extracted_data, full_wallet_data):
        """Score calculation should work with rules only."""
        score, breakdown, tier = service_no_model.calculate_score(
            full_extracted_data, full_wallet_data
        )
        assert isinstance(score, int)
        assert 0 <= score <= 100
        assert isinstance(breakdown, ScoreBreakdown)
        assert tier is not None

    def test_service_with_invalid_path_falls_back(self, full_extracted_data, full_wallet_data):
        """Service should fall back gracefully with invalid model path."""
        service = ScorerService(model_path="/nonexistent/model.json")
        assert service.ml_model_loaded is False

        score, breakdown, tier = service.calculate_score(
            full_extracted_data, full_wallet_data
        )
        assert isinstance(score, int)
        assert 0 <= score <= 100

    def test_service_with_model_loads(self):
        """Service should load XGBoost model from valid path."""
        model_path = os.path.join(
            os.path.dirname(__file__), '..', 'app', 'models', 'credit_scorer.json'
        )
        if os.path.exists(model_path):
            service = ScorerService(model_path=model_path)
            assert service.ml_model_loaded is True

    def test_feature_engineering(self, service_no_model, full_extracted_data, full_wallet_data):
        """Feature engineering should return correct feature vector."""
        features = service_no_model.engineer_features(
            full_extracted_data, full_wallet_data
        )
        assert isinstance(features, list)
        assert len(features) > 0
        # All values should be numeric
        for val in features:
            assert isinstance(val, (int, float))

    def test_blended_score_within_range(self):
        """When model is loaded, blended score should be within valid range."""
        model_path = os.path.join(
            os.path.dirname(__file__), '..', 'app', 'models', 'credit_scorer.json'
        )
        if os.path.exists(model_path):
            service = ScorerService(model_path=model_path)
            score, breakdown, tier = service.calculate_score(
                {
                    "verified_documents": {"government_id": {"is_verified": True}},
                    "monthly_income": 30000,
                    "employment_type": "contract",
                },
                {"age_days": 90, "transaction_count": 10, "balance_eth": 0.1}
            )
            assert 0 <= score <= 100
