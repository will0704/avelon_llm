"""
ETH Volatility Prediction Service

Forecasts forward annualized realized volatility for ETH/PHP with an LSTM over
hourly bars, then converts that forecast into the probability that a borrower's
stake falls below CollateralManager's liquidation ratio before the horizon is out.

Degrades to an EWMA estimate when the trained model is absent, matching the
rule-based fallbacks in the scorer and fraud detector.
"""
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

logger = logging.getLogger(__name__)

# Hourly bars. Daily data caps out at 365 points on CoinGecko's free tier, which
# leaves ~300 training windows — too few to fit an LSTM. 90 days of hourly bars
# gives ~2000, and the model still forecasts in annualized terms either way.
SEQUENCE_LENGTH = 72        # 3 days of context
HORIZON_HOURS = 24          # target window: next 24 hours of realized volatility
# Crypto trades continuously, so a year is 8760 hours rather than 252 sessions.
PERIODS_PER_YEAR = 365 * 24
# Longest rolling window in build_features; the first WARMUP returns produce no row.
WARMUP = 168

COINGECKO_URL = (
    "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
    "?vs_currency=php&days=90"
)
PRICE_CACHE_TTL = 900  # seconds; CoinGecko's free tier is rate limited per minute
# Lives under app/models/ rather than data/ because .dockerignore excludes data/,
# and this snapshot has to reach the Cloud Run image as the offline fallback.
SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "models" / "eth_php_hourly.json"


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing sample standard deviation, aligned so index i covers values[i-window+1:i+1]."""
    out = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        out[i] = values[i - window + 1 : i + 1].std(ddof=1)
    return out


def build_features(prices: np.ndarray) -> np.ndarray:
    """
    Per-hour feature rows: the bar's log return and its magnitude, plus trailing
    realized volatility over three windows. Shared with train_volatility.py so
    training and inference cannot drift apart.
    """
    returns = np.diff(np.log(prices))
    columns = [returns, np.abs(returns)]
    for window in (24, 72, 168):
        columns.append(_rolling_std(returns, window) * np.sqrt(PERIODS_PER_YEAR))

    features = np.column_stack(columns)
    # Drop the warmup rows where the 168-hour window is still incomplete.
    return features[WARMUP - 1 :]


if TORCH_AVAILABLE:

    class VolatilityLSTM(nn.Module):
        """
        Two-layer LSTM over 72 hourly feature rows.

        Predicts log volatility, not volatility. Realized vol is roughly lognormal,
        so MSE on the raw scale is dominated by a handful of spike days and the fit
        collapses toward the mean.
        """

        def __init__(self, input_size: int = 5, hidden_size: int = 48, num_layers: int = 2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :])

else:  # pragma: no cover - torch is a hard dependency in every deployed image
    VolatilityLSTM = None


def _normal_cdf(x: float) -> float:
    """Standard normal CDF. Avoids pulling scipy in for one function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class VolatilityService:
    """ETH/PHP volatility forecasting and collateral risk translation."""

    # Annualized volatility bands. ETH typically sits in the 50-80% range, so the
    # boundaries are set against that rather than equity-market norms.
    RISK_BANDS = [(0.50, "LOW"), (0.75, "MODERATE"), (1.10, "HIGH")]

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.feature_mean = None
        self.feature_std = None
        self.metadata: Dict[str, Any] = {}
        self._price_cache: Optional[tuple] = None
        self._load_model()

    def _load_model(self):
        if not TORCH_AVAILABLE:
            logger.warning("torch not available, volatility falls back to EWMA")
            return

        if not self.model_path:
            logger.info("No volatility model path, using EWMA estimate only")
            return

        model_file = Path(self.model_path)
        if not model_file.is_absolute():
            model_file = Path(__file__).resolve().parent.parent.parent / model_file

        if not model_file.exists():
            logger.info(f"Volatility model not found at {model_file}, using EWMA estimate only")
            return

        try:
            checkpoint = torch.load(str(model_file), map_location="cpu", weights_only=False)
            model = VolatilityLSTM()
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.model = model
            self.feature_mean = np.array(checkpoint["feature_mean"], dtype=np.float32)
            self.feature_std = np.array(checkpoint["feature_std"], dtype=np.float32)
            self.metadata = {
                "log_vol_mae": checkpoint.get("log_vol_mae"),
                "baselines": checkpoint.get("baselines"),
                "trained_points": checkpoint.get("trained_points"),
                "horizon_hours": checkpoint.get("horizon_hours"),
            }
            logger.info(f"Volatility LSTM loaded from {model_file}")
        except Exception as e:
            logger.warning(f"Failed to load volatility model: {e}")
            self.model = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _fetch_prices(self) -> tuple:
        """
        Recent hourly ETH/PHP closes. Returns (prices, source).

        Cached for PRICE_CACHE_TTL because Cloud Run may hold several workers and
        CoinGecko's free tier rate limits per minute. Falls back to the snapshot
        committed for training so a demo survives an API outage.
        """
        now = time.time()
        if self._price_cache and now - self._price_cache[0] < PRICE_CACHE_TTL:
            return self._price_cache[1], self._price_cache[2]

        try:
            import httpx

            response = httpx.get(COINGECKO_URL, timeout=10.0)
            response.raise_for_status()
            prices = np.array([p[1] for p in response.json()["prices"]], dtype=np.float64)
            if len(prices) >= SEQUENCE_LENGTH + WARMUP:
                self._price_cache = (now, prices, "coingecko")
                return prices, "coingecko"
            logger.warning(f"CoinGecko returned only {len(prices)} points, using snapshot")
        except Exception as e:
            logger.warning(f"CoinGecko fetch failed ({e}), using snapshot")

        if SNAPSHOT_PATH.exists():
            import json

            payload = json.loads(SNAPSHOT_PATH.read_text())
            prices = np.array([p[1] for p in payload["prices"]], dtype=np.float64)
            self._price_cache = (now, prices, "snapshot")
            return prices, "snapshot"

        raise RuntimeError("No ETH price data available from CoinGecko or the local snapshot")

    def _predict_annualized_vol(self, prices: np.ndarray) -> tuple:
        """Returns (annualized_vol, method)."""
        returns = np.diff(np.log(prices))

        if not self.is_loaded:
            # RiskMetrics EWMA, decay rescaled from its daily 0.94 to hourly bars.
            lam = 0.94 ** (1 / 24)
            weights = lam ** np.arange(len(returns) - 1, -1, -1)
            variance = float((weights * returns**2).sum() / weights.sum())
            return math.sqrt(variance * PERIODS_PER_YEAR), "ewma_fallback"

        features = build_features(prices)[-SEQUENCE_LENGTH:]
        scaled = (features - self.feature_mean) / self.feature_std
        with torch.no_grad():
            prediction = self.model(torch.tensor(scaled, dtype=torch.float32).unsqueeze(0))
        # The head emits log volatility.
        return float(np.exp(prediction.item())), "lstm"

    def _risk_level(self, annualized_vol: float) -> str:
        for ceiling, label in self.RISK_BANDS:
            if annualized_vol < ceiling:
                return label
        return "EXTREME"

    def predict(
        self,
        horizon_days: int = 7,
        stake_ratio_bps: int = 4000,
        min_ratio_bps: int = 3500,
    ) -> Dict[str, Any]:
        """
        Forecast volatility and translate it into liquidation risk.

        stake_ratio_bps is the borrower's collateral as a share of debt today;
        min_ratio_bps is CollateralManager.minCollateralRatio. Both are basis
        points, passed in by the backend so the thresholds are not duplicated here.
        """
        prices, source = self._fetch_prices()
        annualized_vol, method = self._predict_annualized_vol(prices)

        # Square-root-of-time scaling from the annualized figure to the horizon.
        horizon_vol = annualized_vol * math.sqrt(horizon_days / 365)
        current_price = float(prices[-1])

        # Lognormal bands with zero drift — over days the drift term is noise next
        # to the volatility term, and assuming none is the conservative choice.
        def band(z):
            return {
                "lower": round(current_price * math.exp(-z * horizon_vol), 2),
                "upper": round(current_price * math.exp(z * horizon_vol), 2),
            }

        # A stake at ratio R is liquidated once the price falls far enough that
        # R * (1 - drop) < min_ratio.
        if stake_ratio_bps <= min_ratio_bps:
            drop_to_liquidation = 0.0
            liquidation_probability = 1.0
        else:
            drop_to_liquidation = 1.0 - (min_ratio_bps / stake_ratio_bps)
            log_move = math.log(1.0 - drop_to_liquidation)
            liquidation_probability = _normal_cdf(log_move / horizon_vol)

        recent = np.diff(np.log(prices[-(HORIZON_HOURS + 1):]))
        realized_24h = float(recent.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR))

        # A week of closes for the admin chart, thinned to keep the payload small.
        window = prices[-168:]
        step = max(1, len(window) // 48)
        sparkline = [round(float(v), 2) for v in window[::step]]

        return {
            "horizon_days": horizon_days,
            "current_price_php": round(current_price, 2),
            "price_source": source,
            "model": method,
            "predicted_volatility": round(annualized_vol, 4),
            "realized_volatility_24h": round(realized_24h, 4),
            "horizon_volatility": round(horizon_vol, 4),
            "risk_level": self._risk_level(annualized_vol),
            "price_range_68": band(1.0),
            "price_range_95": band(1.96),
            "liquidation": {
                "stake_ratio_bps": stake_ratio_bps,
                "min_ratio_bps": min_ratio_bps,
                "price_drop_to_liquidation": round(drop_to_liquidation, 4),
                "probability": round(liquidation_probability, 4),
                "interpretation": "Terminal-price threshold scenario only; not first-passage probability and not used for ETH/ETH liquidation.",
                "advisory_only": True,
            },
            "recent_prices": sparkline,
            "model_metadata": self.metadata,
            "advisory_only": True,
        }


# Singleton instance
_volatility_service = None


def get_volatility_service() -> VolatilityService:
    """Get or create volatility service instance."""
    global _volatility_service
    if _volatility_service is None:
        from app.config import get_settings

        settings = get_settings()
        _volatility_service = VolatilityService(settings.volatility_model_path)
    return _volatility_service
