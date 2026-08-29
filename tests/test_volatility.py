"""
Tests for the ETH volatility predictor — features, fallbacks, and liquidation maths.
"""
import math

import numpy as np
import pytest

from app.services.volatility_service import (
    HORIZON_HOURS,
    PERIODS_PER_YEAR,
    SEQUENCE_LENGTH,
    WARMUP,
    VolatilityService,
    _normal_cdf,
    build_features,
)


@pytest.fixture
def synthetic_prices():
    """A geometric random walk long enough to clear the warmup window."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0, 0.01, 600)
    return 150000 * np.exp(np.cumsum(returns))


class TestFeatures:
    def test_shape_and_warmup(self, synthetic_prices):
        features = build_features(synthetic_prices)
        assert features.shape[1] == 5
        # One row per return, less the incomplete warmup rows.
        assert len(features) == len(synthetic_prices) - 1 - (WARMUP - 1)

    def test_no_nan_after_warmup(self, synthetic_prices):
        assert not np.isnan(build_features(synthetic_prices)).any()

    def test_volatility_columns_are_annualized(self, synthetic_prices):
        # 1% hourly moves annualize to roughly 0.01 * sqrt(8760) ≈ 94%.
        features = build_features(synthetic_prices)
        assert 0.5 < features[:, 2].mean() < 1.5

    def test_higher_variance_raises_the_estimate(self):
        rng = np.random.default_rng(3)
        calm = 150000 * np.exp(np.cumsum(rng.normal(0, 0.002, 600)))
        wild = 150000 * np.exp(np.cumsum(rng.normal(0, 0.02, 600)))
        assert build_features(calm)[:, 2].mean() < build_features(wild)[:, 2].mean()


class TestFallback:
    def test_ewma_used_without_weights(self, monkeypatch, synthetic_prices):
        service = VolatilityService(None)
        monkeypatch.setattr(service, "_fetch_prices", lambda: (synthetic_prices, "test"))

        result = service.predict()
        assert service.is_loaded is False
        assert result["model"] == "ewma_fallback"
        assert result["predicted_volatility"] > 0

    def test_missing_model_file_does_not_raise(self):
        # A bad path must degrade, not crash the service at import time.
        assert VolatilityService("app/models/does_not_exist.pt").is_loaded is False

    def test_raises_when_no_price_data_at_all(self, monkeypatch):
        service = VolatilityService(None)
        monkeypatch.setattr(
            "app.services.volatility_service.COINGECKO_URL", "http://127.0.0.1:9/dead"
        )
        monkeypatch.setattr(
            "app.services.volatility_service.SNAPSHOT_PATH",
            type("P", (), {"exists": staticmethod(lambda: False)})(),
        )
        with pytest.raises(RuntimeError):
            service.predict()


class TestLiquidationMaths:
    @pytest.fixture
    def service(self, monkeypatch, synthetic_prices):
        s = VolatilityService(None)
        monkeypatch.setattr(s, "_fetch_prices", lambda: (synthetic_prices, "test"))
        return s

    def test_drop_to_liquidation(self, service):
        # A 40% stake against a 35% floor survives a 12.5% fall, no more.
        result = service.predict(stake_ratio_bps=4000, min_ratio_bps=3500)
        assert result["liquidation"]["price_drop_to_liquidation"] == pytest.approx(0.125, abs=1e-4)

    def test_stake_at_the_floor_is_certain(self, service):
        result = service.predict(stake_ratio_bps=3500, min_ratio_bps=3500)
        assert result["liquidation"]["probability"] == 1.0

    def test_thicker_stake_is_safer(self, service):
        thin = service.predict(stake_ratio_bps=4000)["liquidation"]["probability"]
        thick = service.predict(stake_ratio_bps=6000)["liquidation"]["probability"]
        assert thick < thin

    def test_longer_horizon_is_riskier(self, service):
        near = service.predict(horizon_days=1)["liquidation"]["probability"]
        far = service.predict(horizon_days=30)["liquidation"]["probability"]
        assert far > near

    def test_horizon_vol_scales_with_sqrt_time(self, service):
        one = service.predict(horizon_days=1)["horizon_volatility"]
        four = service.predict(horizon_days=4)["horizon_volatility"]
        assert four == pytest.approx(2 * one, rel=0.01)

    def test_price_bands_bracket_the_spot(self, service):
        result = service.predict()
        spot = result["current_price_php"]
        assert result["price_range_95"]["lower"] < result["price_range_68"]["lower"] < spot
        assert spot < result["price_range_68"]["upper"] < result["price_range_95"]["upper"]


class TestRiskBands:
    @pytest.mark.parametrize(
        "vol,expected",
        [(0.30, "LOW"), (0.60, "MODERATE"), (0.90, "HIGH"), (1.50, "EXTREME")],
    )
    def test_labels(self, vol, expected):
        assert VolatilityService(None)._risk_level(vol) == expected


class TestNormalCdf:
    def test_known_values(self):
        assert _normal_cdf(0.0) == pytest.approx(0.5)
        assert _normal_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)
        assert _normal_cdf(1.645) == pytest.approx(0.95, abs=1e-3)
