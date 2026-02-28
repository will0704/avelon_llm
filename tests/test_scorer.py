"""
Unit tests for the Credit Scorer Service.
"""
import pytest

from app.services.scorer_service import ScorerService, get_scorer_service
from app.schemas.score import LoanHistory


class TestScorerService:
    """Tests for ScorerService."""
    
    @pytest.fixture
    def service(self):
        """Create a scorer service instance."""
        return ScorerService()
    
    def test_calculate_score_returns_tuple(self, service):
        """Test calculate_score returns (score, breakdown, tier) tuple."""
        result = service.calculate_score(
            extracted_data={},
            wallet_data={"age_days": 100, "transaction_count": 30, "balance_eth": 0.5}
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        score, breakdown, tier = result
        assert isinstance(score, int)
        assert 0 <= score <= 100
    
    def test_document_score_verified_docs(self, service):
        """Test document score calculation with verified documents."""
        data = {
            "verified_documents": {
                "government_id": {"is_verified": True, "confidence": 0.95},
                "proof_of_income": {"is_verified": True},
                "proof_of_address": {"is_verified": True}
            }
        }
        score = service._calculate_document_score(data)
        # All verified: 15 + 2 (high conf bonus) + 15 + 10 = 42, capped at 40
        assert score == 40
    
    def test_document_score_no_docs(self, service):
        """Test document score with no verified documents."""
        score = service._calculate_document_score({})
        assert score == 0
    
    def test_financial_score_high_income(self, service):
        """Test financial score for high income."""
        data = {"monthly_income": 150000, "employment_type": "permanent"}
        score = service._calculate_financial_score(data)
        # 15 (income) + 10 (permanent) = 25
        assert score >= 25
    
    def test_financial_score_low_income(self, service):
        """Test financial score for low income."""
        data = {"monthly_income": 10000, "employment_type": "contract"}
        score = service._calculate_financial_score(data)
        # 5 (income below 15k) + 7 (contract) = 12
        assert score == 12
    
    def test_financial_score_employment_duration(self, service):
        """Test financial score includes employment duration bonus."""
        data = {"monthly_income": 50000, "employment_type": "permanent", "years_employed": 6}
        score = service._calculate_financial_score(data)
        # 12 + 10 + 5 (6 years bonus) = 27
        assert score == 27
    
    def test_financial_score_debt_penalty(self, service):
        """Test debt-to-income ratio penalty."""
        data = {"monthly_income": 50000, "debt_to_income_ratio": 0.6}
        score = service._calculate_financial_score(data)
        # 12 - 5 (high DTI penalty) = 7
        assert score == 7
    
    def test_history_score_first_time_user(self, service):
        """Test history score for first-time user."""
        score = service._calculate_history_score(None)
        assert score == 5.0  # Base score
    
    def test_history_score_good_history(self, service):
        """Test history score for user with good history."""
        history = LoanHistory(
            repaid_loans=3,
            defaulted_loans=0,
            late_payments=0
        )
        score = service._calculate_history_score(history)
        # 5 (base) + 9 (3 repaid * 3) = 14
        assert score == 14
    
    def test_history_score_with_default(self, service):
        """Test history score with loan default."""
        history = LoanHistory(
            repaid_loans=2,
            defaulted_loans=1,
            late_payments=1
        )
        score = service._calculate_history_score(history)
        # 5 + 6 - 15 - 2 = -6, capped at 0
        assert score == 0
    
    def test_wallet_score(self, service):
        """Test wallet score calculation."""
        wallet_data = {
            "age_days": 200,
            "transaction_count": 60,
            "balance_eth": 1.5
        }
        score = service._calculate_wallet_score(wallet_data)
        # 4 (age > 180) + 3 (tx > 50) + 3 (balance > 1) = 10
        assert score == 10
    
    def test_tier_vip(self, service):
        """Test VIP tier assignment."""
        tier = service._get_tier(95)
        assert tier == "vip"
    
    def test_tier_premium(self, service):
        """Test premium tier assignment."""
        tier = service._get_tier(85)
        assert tier == "premium"
    
    def test_tier_standard(self, service):
        """Test standard tier assignment."""
        tier = service._get_tier(70)
        assert tier == "standard"
    
    def test_tier_basic(self, service):
        """Test basic tier assignment."""
        tier = service._get_tier(50)
        assert tier == "basic"
    
    def test_tier_rejected(self, service):
        """Test rejection for low score."""
        tier = service._get_tier(30)
        assert tier is None
    
    def test_singleton_get_service(self):
        """Test singleton pattern returns same instance."""
        service1 = get_scorer_service()
        service2 = get_scorer_service()
        assert service1 is service2
