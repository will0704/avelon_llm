"""
Credit scoring endpoints.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException

from app.schemas.score import (
    CreditScoreRequest,
    CreditScoreResponse,
    ScoreBreakdown,
    LoanHistory,
)
from app.api.dependencies import verify_api_key
from app.services.scorer_service import get_scorer_service

logger = logging.getLogger(__name__)
router = APIRouter()


def _build_recommendations(tier: str | None, breakdown: ScoreBreakdown) -> list[str]:
    """Generate actionable recommendations based on score breakdown."""
    recommendations: list[str] = []

    if breakdown.document_score < 30:
        recommendations.append(
            "Submit all required KYC documents (government ID, proof of income, proof of address) "
            "to improve your document score."
        )
    if breakdown.financial_score < 25:
        recommendations.append(
            "Provide proof of stable income to boost your financial indicator score."
        )
    if breakdown.history_score < 10:
        recommendations.append(
            "Build a positive loan repayment history on Avelon to increase your history score."
        )
    if breakdown.wallet_score < 7:
        recommendations.append(
            "Maintain an active wallet with regular transactions and a healthy balance."
        )

    if tier is None:
        recommendations.append(
            "Your score is currently below the minimum threshold. Complete KYC and build "
            "transaction history to become eligible for loans."
        )
    elif tier == "basic":
        recommendations.append("Repay loans on time to qualify for the Standard tier.")
    elif tier == "standard":
        recommendations.append("Keep a strong repayment record to reach the Premium tier.")
    elif tier == "premium":
        recommendations.append("Maintain excellent history to reach VIP status.")

    return recommendations if recommendations else ["Your score is excellent — keep it up!"]


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
    scorer = get_scorer_service()

    wallet_data = {
        "address": request.wallet_address,
        "age_days": request.wallet_age_days or 0,
        "transaction_count": request.wallet_transaction_count or 0,
        "balance_eth": request.wallet_balance_eth or 0.0,
    }

    try:
        total_score, breakdown, tier = scorer.calculate_score(
            extracted_data=request.extracted_data,
            wallet_data=wallet_data,
            loan_history=request.loan_history,
        )
    except Exception as e:
        logger.exception("Credit score calculation failed")
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

    recommendations = _build_recommendations(tier, breakdown)

    return CreditScoreResponse(
        score=total_score,
        tier=tier,
        breakdown=breakdown,
        recommendations=recommendations,
    )


@router.post("/score/recalculate", response_model=CreditScoreResponse)
async def recalculate_credit_score(
    request: CreditScoreRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Recalculate credit score after loan events (repayment, default, etc).

    Called by the backend when:
    - User successfully repays a loan  (+3 to +5 points)
    - User defaults on a loan          (-15 points)
    - User makes late payments          (-2 per late payment)

    The recalculation uses the same scorer but emphasises the updated
    loan_history component.
    """
    scorer = get_scorer_service()

    wallet_data = {
        "address": request.wallet_address,
        "age_days": request.wallet_age_days or 0,
        "transaction_count": request.wallet_transaction_count or 0,
        "balance_eth": request.wallet_balance_eth or 0.0,
    }

    try:
        total_score, breakdown, tier = scorer.calculate_score(
            extracted_data=request.extracted_data,
            wallet_data=wallet_data,
            loan_history=request.loan_history,
        )
    except Exception as e:
        logger.exception("Credit score recalculation failed")
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

    recommendations = _build_recommendations(tier, breakdown)

    return CreditScoreResponse(
        score=total_score,
        tier=tier,
        breakdown=breakdown,
        recommendations=recommendations,
    )
