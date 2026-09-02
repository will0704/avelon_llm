"""
ETH volatility prediction schemas.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class PriceBand(BaseModel):
    """Projected price bounds at the end of the horizon."""
    lower: float
    upper: float


class LiquidationRisk(BaseModel):
    """Legacy terminal-price scenario; advisory only, not a liquidation trigger."""
    stake_ratio_bps: int
    min_ratio_bps: int
    price_drop_to_liquidation: float  # fraction, e.g. 0.125 = a 12.5% fall
    probability: float                # 0.0 - 1.0
    interpretation: str
    advisory_only: bool = True


class VolatilityResponse(BaseModel):
    """Forward volatility forecast and its collateral implications."""
    horizon_days: int
    current_price_php: float
    price_source: str                 # "coingecko" or "snapshot"
    model: str                        # "lstm" or "ewma_fallback"
    predicted_volatility: float       # annualized
    realized_volatility_24h: float    # annualized, what actually just happened
    horizon_volatility: float         # over the horizon, not annualized
    risk_level: str                   # LOW | MODERATE | HIGH | EXTREME
    price_range_68: PriceBand
    price_range_95: PriceBand
    liquidation: LiquidationRisk
    recent_prices: List[float]       # last 7 days of closes, thinned for charting
    model_metadata: Optional[Dict[str, Any]] = None
    advisory_only: bool = True
