"""
Credit scoring endpoints.
"""
from fastapi import APIRouter, Depends

from app.schemas.score import (
    CreditScoreRequest,
    CreditScoreResponse,
    ScoreBreakdown,
)
from app.api.dependencies import verify_api_key

router = APIRouter()


@router.post("/score/calculate", response_model=CreditScoreResponse)
async def calculate_credit_score(
    request: CreditScoreRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Calculate credit score for a user.
    
    Components:
    - Document Verification: 40 points max
    - Financial Indicators: 35 points max  
    - Avelon History: 15 points max
    - Wallet Analysis: 10 points max
    
    Args:
        request: User data including extracted document info, wallet data, and history
        
    Returns:
        Credit score (0-100), breakdown, and assigned tier
    """
    # TODO: Implement actual scoring logic
    # For now, return a stub response
    return CreditScoreResponse(
        score=0,
        tier=None,
        breakdown=ScoreBreakdown(
            document_score=0,
            financial_score=0,
            history_score=0,
            wallet_score=0,
        ),
        recommendations=[
            "Credit scoring not yet implemented. This is a stub response."
        ],
    )


@router.post("/score/recalculate", response_model=CreditScoreResponse)
async def recalculate_credit_score(
    request: CreditScoreRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Recalculate credit score after loan events (repayment, default, etc).
    
    Called by backend when:
    - User successfully repays a loan (+3 to +5 points)
    - User defaults on a loan (-15 points)
    - User makes late payments (-2 points per late payment)
    """
    # TODO: Implement recalculation logic
    return CreditScoreResponse(
        score=0,
        tier=None,
        breakdown=ScoreBreakdown(
            document_score=0,
            financial_score=0,
            history_score=0,
            wallet_score=0,
        ),
        recommendations=[
            "Score recalculation not yet implemented. This is a stub response."
        ],
    )
