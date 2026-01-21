"""
Credit scoring schemas.
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class LoanHistory(BaseModel):
    """User's loan history on Avelon platform."""
    total_loans: int = 0
    repaid_loans: int = 0
    defaulted_loans: int = 0
    late_payments: int = 0


class ScoreBreakdown(BaseModel):
    """Breakdown of credit score components."""
    document_score: float  # Max 40 points
    financial_score: float  # Max 35 points
    history_score: float  # Max 15 points
    wallet_score: float  # Max 10 points


class CreditScoreRequest(BaseModel):
    """Request for credit score calculation."""
    user_id: str
    
    # Extracted data from documents
    extracted_data: Dict[str, Any]
    
    # Wallet analysis data
    wallet_address: str
    wallet_age_days: Optional[int] = None
    wallet_transaction_count: Optional[int] = None
    wallet_balance_eth: Optional[float] = None
    
    # Avelon history
    loan_history: Optional[LoanHistory] = None


class CreditScoreResponse(BaseModel):
    """Response with calculated credit score."""
    score: int  # 0-100
    tier: Optional[str]  # "basic", "standard", "premium", "vip", None if < 40
    breakdown: ScoreBreakdown
    recommendations: List[str]
