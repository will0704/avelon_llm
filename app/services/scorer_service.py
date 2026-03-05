"""
Credit Scorer Service
Uses XGBoost for credit score calculation with rule-based fallback.
"""
from typing import Dict, Any, Optional, List
import logging
import os

try:
    import xgboost as xgb
    import numpy as np
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from app.schemas.score import ScoreBreakdown, LoanHistory

logger = logging.getLogger(__name__)


class ScorerService:
    """
    Credit scoring using XGBoost model and rule-based components.
    
    Score Components (0-100 total):
    - Document Verification: 40 points max
    - Financial Indicators: 35 points max
    - Avelon History: 15 points max
    - Wallet Analysis: 10 points max
    """
    
    # Feature names expected by the XGBoost model
    FEATURE_NAMES = [
        'gov_id_verified', 'gov_id_confidence',
        'income_verified', 'address_verified',
        'fraud_flag_count', 'fraud_high_count',
        'monthly_income', 'years_employed',
        'debt_to_income_ratio', 'employment_score',
        'total_loans', 'repaid_loans', 'defaulted_loans', 'late_payments',
        'wallet_age_days', 'wallet_tx_count', 'wallet_balance_eth',
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained XGBoost scorer model."""
        if not XGB_AVAILABLE:
            logger.warning("xgboost not available, using rule-based scoring only")
            return
        
        if not self.model_path or not os.path.exists(self.model_path):
            logger.info("No valid scorer model path, using rule-based scoring only")
            return
        
        try:
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
            logger.info(f"XGBoost scorer loaded from {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load scorer model: {e}")
            self.model = None
    
    @property
    def ml_model_loaded(self) -> bool:
        """Check if the XGBoost model is loaded."""
        return self.model is not None
    
    def engineer_features(
        self,
        extracted_data: Dict[str, Any],
        wallet_data: Dict[str, Any],
        loan_history: Optional[LoanHistory] = None,
    ) -> List[float]:
        """
        Engineer feature vector from raw scoring inputs.
        
        Returns:
            List of numeric feature values matching FEATURE_NAMES order.
        """
        documents = extracted_data.get("verified_documents", {})
        gov_id = documents.get("government_id", {})
        income_doc = documents.get("proof_of_income", {})
        address_doc = documents.get("proof_of_address", {})
        fraud_flags = extracted_data.get("fraud_flags", [])
        
        # Employment type → numeric score
        emp_type = extracted_data.get("employment_type", "").lower()
        emp_scores = {"permanent": 4, "contract": 3, "self-employed": 2}
        employment_score = emp_scores.get(emp_type, 1) if emp_type else 0
        
        # Loan history
        history = loan_history or LoanHistory()
        
        return [
            float(gov_id.get("is_verified", False)),
            float(gov_id.get("confidence", 0.0)),
            float(income_doc.get("is_verified", False)),
            float(address_doc.get("is_verified", False)),
            float(len(fraud_flags)),
            float(sum(1 for f in fraud_flags if f.get("severity") == "high")),
            float(extracted_data.get("monthly_income", 0)),
            float(extracted_data.get("years_employed", 0)),
            float(extracted_data.get("debt_to_income_ratio", 0)),
            float(employment_score),
            float(history.total_loans),
            float(history.repaid_loans),
            float(history.defaulted_loans),
            float(history.late_payments),
            float(wallet_data.get("age_days", 0)),
            float(wallet_data.get("transaction_count", 0)),
            float(wallet_data.get("balance_eth", 0)),
        ]
    
    def calculate_score(
        self,
        extracted_data: Dict[str, Any],
        wallet_data: Dict[str, Any],
        loan_history: Optional[LoanHistory] = None,
    ) -> tuple:
        """
        Calculate credit score.
        
        When ML model is loaded, blends XGBoost prediction (70%)
        with rule-based score (30%). Otherwise uses rules only.
        
        Returns:
            Tuple of (total_score, breakdown, tier)
        """
        # Calculate rule-based components
        doc_score = self._calculate_document_score(extracted_data)
        financial_score = self._calculate_financial_score(extracted_data)
        history_score = self._calculate_history_score(loan_history)
        wallet_score = self._calculate_wallet_score(wallet_data)
        
        rule_score = int(doc_score + financial_score + history_score + wallet_score)
        rule_score = max(0, min(100, rule_score))
        
        # Blend with ML model if available
        if self.ml_model_loaded and XGB_AVAILABLE:
            try:
                features = self.engineer_features(extracted_data, wallet_data, loan_history)
                dmatrix = xgb.DMatrix(
                    np.array([features]),
                    feature_names=self.FEATURE_NAMES,
                )
                ml_score = float(self.model.predict(dmatrix)[0])
                ml_score = max(0.0, min(100.0, ml_score))
                total_score = int(0.7 * ml_score + 0.3 * rule_score)
            except Exception as e:
                logger.warning(f"ML scoring failed, using rules only: {e}")
                total_score = rule_score
        else:
            total_score = rule_score
        
        total_score = max(0, min(100, total_score))
        
        breakdown = ScoreBreakdown(
            document_score=doc_score,
            financial_score=financial_score,
            history_score=history_score,
            wallet_score=wallet_score,
        )
        
        tier = self._get_tier(total_score)
        
        return (total_score, breakdown, tier)
    
    def _calculate_document_score(self, data: Dict[str, Any]) -> float:
        """Calculate document verification score (max 40 points)."""
        score = 0.0
        
        # Check for verified documents
        documents = data.get("verified_documents", {})
        
        # Valid Government ID: +15
        gov_id = documents.get("government_id", {})
        if gov_id.get("is_verified", False):
            score += 15
            # Bonus for high confidence
            if gov_id.get("confidence", 0) > 0.9:
                score += 2
        
        # Proof of Income: +15
        income_doc = documents.get("proof_of_income", {})
        if income_doc.get("is_verified", False):
            score += 15
        
        # Proof of Address: +10
        address_doc = documents.get("proof_of_address", {})
        if address_doc.get("is_verified", False):
            score += 10
        
        # Penalties for fraud flags
        fraud_flags = data.get("fraud_flags", [])
        for flag in fraud_flags:
            if flag.get("severity") == "high":
                score -= 10
            elif flag.get("severity") == "medium":
                score -= 5
        
        return max(0, min(40, score))
    
    def _calculate_financial_score(self, data: Dict[str, Any]) -> float:
        """Calculate financial indicators score (max 35 points)."""
        score = 0.0
        
        # Get monthly income (PHP)
        monthly_income = data.get("monthly_income", 0)
        
        # Income brackets (PHP)
        if monthly_income >= 100000:
            score += 15
        elif monthly_income >= 50000:
            score += 12
        elif monthly_income >= 30000:
            score += 10
        elif monthly_income >= 15000:
            score += 8
        elif monthly_income > 0:
            score += 5
        
        # Employment type score
        employment_type = data.get("employment_type", "").lower()
        if employment_type == "permanent":
            score += 10
        elif employment_type == "contract":
            score += 7
        elif employment_type == "self-employed":
            score += 5
        elif employment_type:
            score += 3  # Some employment is better than none
        
        # Employment duration bonus (years)
        years_employed = data.get("years_employed", 0)
        if years_employed >= 5:
            score += 5
        elif years_employed >= 2:
            score += 3
        elif years_employed >= 1:
            score += 1
        
        # Debt-to-income ratio penalty
        dti_ratio = data.get("debt_to_income_ratio", 0)
        if dti_ratio > 0.5:
            score -= 5
        elif dti_ratio > 0.3:
            score -= 2
        
        return max(0, min(35, score))
    
    def _calculate_history_score(self, history: Optional[LoanHistory]) -> float:
        """Calculate Avelon history score (max 15 points)."""
        if history is None:
            return 5.0  # First-time user base score
        
        score = 5.0  # Base score
        
        # Repaid loans: +3 per loan (max +10)
        score += min(history.repaid_loans * 3, 10)
        
        # Defaults: -15 per default
        score -= history.defaulted_loans * 15
        
        # Late payments: -2 per late payment
        score -= history.late_payments * 2
        
        return max(0, score)
    
    def _calculate_wallet_score(self, wallet_data: Dict[str, Any]) -> float:
        """Calculate wallet analysis score (max 10 points)."""
        score = 0.0
        
        age_days = wallet_data.get("age_days", 0)
        tx_count = wallet_data.get("transaction_count", 0)
        balance = wallet_data.get("balance_eth", 0)
        
        # Wallet age
        if age_days > 180:
            score += 4
        elif age_days > 90:
            score += 2
        else:
            score += 1
        
        # Transaction count
        if tx_count > 50:
            score += 3
        elif tx_count > 20:
            score += 2
        
        # Balance
        if balance > 1.0:
            score += 3
        elif balance > 0.5:
            score += 2
        
        return min(score, 10)
    
    def _get_tier(self, score: int) -> Optional[str]:
        """Get tier based on score."""
        if score >= 90:
            return "vip"
        elif score >= 80:
            return "premium"
        elif score >= 60:
            return "standard"
        elif score >= 40:
            return "basic"
        else:
            return None  # Rejected


# Singleton instance
_scorer_service = None

def get_scorer_service() -> ScorerService:
    """Get or create scorer service instance."""
    global _scorer_service
    if _scorer_service is None:
        from app.config import get_settings
        settings = get_settings()
        _scorer_service = ScorerService(settings.scorer_model_path)
    return _scorer_service
