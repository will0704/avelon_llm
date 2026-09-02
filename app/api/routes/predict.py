"""
Market prediction endpoints.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import verify_api_key
from app.schemas.volatility import VolatilityResponse
from app.services.volatility_service import get_volatility_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/predict/volatility", response_model=VolatilityResponse)
async def predict_volatility(
    horizon_days: int = Query(7, ge=1, le=30),
    stake_ratio_bps: int = Query(4000, ge=1, le=20000),
    min_ratio_bps: int = Query(3500, ge=1, le=20000),
    api_key: str = Depends(verify_api_key),
):
    """
    Forecast ETH/PHP volatility for advisory research.

    The LSTM predicts forward annualized realized volatility from the last 30 days
    of price action. The legacy terminal-threshold scenario is returned only for
    research compatibility. Because collateral and debt are both ETH, it is not
    used to authorize liquidation.

    Args:
        horizon_days: Forecast window, 1-30 days
        stake_ratio_bps: Borrower's collateral as a share of debt, basis points
        min_ratio_bps: CollateralManager.minCollateralRatio, basis points

    Returns:
        Volatility forecast, projected price bands, and an advisory terminal scenario
    """
    service = get_volatility_service()

    try:
        return service.predict(
            horizon_days=horizon_days,
            stake_ratio_bps=stake_ratio_bps,
            min_ratio_bps=min_ratio_bps,
        )
    except RuntimeError as e:
        # No price data from CoinGecko or the local snapshot.
        logger.error(f"Volatility prediction unavailable: {e}")
        raise HTTPException(status_code=503, detail="Price data unavailable")
    except Exception:
        logger.exception("Volatility prediction failed")
        raise HTTPException(status_code=500, detail="Internal prediction error")
