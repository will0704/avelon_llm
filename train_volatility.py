"""
Train the ETH volatility LSTM for Avelon LLM.

Forecasts the next 24 hours of annualized realized volatility from 72 hours of
ETH/PHP price action. The prediction feeds collateral risk sizing — high forecast
volatility means a stake is more likely to breach its liquidation threshold.

Usage:
    python train_volatility.py
    python train_volatility.py --epochs 200 --refresh
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from app.services.volatility_service import (
    VolatilityLSTM,
    SEQUENCE_LENGTH,
    HORIZON_HOURS,
    PERIODS_PER_YEAR,
    WARMUP,
    COINGECKO_URL,
    build_features,
)

# Doubles as the runtime fallback snapshot, so it sits with the model weights.
CACHE_PATH = Path(__file__).parent / "app" / "models" / "eth_php_hourly.json"


def get_args():
    parser = argparse.ArgumentParser(description="Train the ETH volatility forecaster")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--patience", type=int, default=25, help="Early-stopping patience")
    parser.add_argument("--refresh", action="store_true", help="Re-download prices instead of using the cache")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "app" / "models" / "volatility_lstm.pt"),
    )
    return parser.parse_args()


def load_prices(refresh: bool) -> np.ndarray:
    """Hourly ETH/PHP closes, oldest first. Cached so retraining doesn't re-hit the API."""
    if CACHE_PATH.exists() and not refresh:
        payload = json.loads(CACHE_PATH.read_text())
    else:
        import httpx

        print("Downloading ETH/PHP hourly prices from CoinGecko...")
        response = httpx.get(COINGECKO_URL, timeout=60.0)
        response.raise_for_status()
        payload = response.json()
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(payload))
        print(f"Cached {len(payload['prices'])} points to {CACHE_PATH}")

    return np.array([point[1] for point in payload["prices"]], dtype=np.float64)


def build_dataset(prices: np.ndarray):
    """
    Sliding windows of features against the forward realized volatility that followed.

    Targets are log volatility: realized vol is roughly lognormal, and fitting MSE
    on the raw scale lets a few spike hours dominate the loss.
    """
    features = build_features(prices)
    returns = np.diff(np.log(prices))

    # build_features drops the first WARMUP-1 rows, so feature row i describes the
    # series up to and including returns[i + WARMUP - 1].
    offset = len(returns) - len(features)

    sequences, targets = [], []
    for i in range(len(features) - SEQUENCE_LENGTH - HORIZON_HOURS + 1):
        end = i + SEQUENCE_LENGTH
        forward = returns[offset + end : offset + end + HORIZON_HOURS]
        if len(forward) < HORIZON_HOURS:
            break
        realized = forward.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
        sequences.append(features[i:end])
        targets.append(np.log(max(realized, 1e-6)))

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


def baselines(prices: np.ndarray, X_val_raw: np.ndarray, y_val: np.ndarray, train_mean: float):
    """
    What the LSTM has to beat, in log-vol MAE.

    - mean       : always predict the training average
    - persistence: carry the last trailing 24h realized vol forward
    - ewma       : RiskMetrics exponentially weighted variance
    """
    # Feature column 2 is the trailing 24h annualized vol; take it at the last
    # timestep of each validation window.
    persistence = np.log(np.maximum(X_val_raw[:, -1, 2], 1e-6))

    ewma = []
    lam = 0.94 ** (1 / 24)
    for window in X_val_raw:
        r = window[:, 0]
        w = lam ** np.arange(len(r) - 1, -1, -1)
        var = float((w * r**2).sum() / w.sum())
        ewma.append(np.log(max(np.sqrt(var * PERIODS_PER_YEAR), 1e-6)))
    ewma = np.array(ewma)

    return {
        "mean": float(np.abs(train_mean - y_val).mean()),
        "persistence": float(np.abs(persistence - y_val).mean()),
        "ewma": float(np.abs(ewma - y_val).mean()),
    }


def main():
    args = get_args()
    torch.manual_seed(42)
    np.random.seed(42)

    prices = load_prices(args.refresh)
    print(f"Loaded {len(prices)} hourly closes  |  latest ₱{prices[-1]:,.0f}")

    X, y = build_dataset(prices)
    print(f"Windows: {len(X)}  |  realized vol range {np.exp(y).min():.1%} – {np.exp(y).max():.1%}")

    # Chronological split — shuffling a time series leaks the future into training.
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Standardize on training statistics only, then ship them in the checkpoint so
    # inference scales incoming windows the same way.
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std[std < 1e-8] = 1.0

    def to_tensor(a):
        return torch.tensor((a - mean) / std, dtype=torch.float32)

    X_train_t, X_val_t = to_tensor(X_train), to_tensor(X_val)
    y_train_t = torch.tensor(y_train).unsqueeze(1)
    y_val_t = torch.tensor(y_val).unsqueeze(1)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = VolatilityLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8)
    criterion = nn.SmoothL1Loss()

    best_val, best_state, stale = float("inf"), None, 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  train {loss.item():.5f}  val {val_loss:.5f}")

        if stale >= args.patience:
            print(f"Early stop at epoch {epoch} (no val improvement in {args.patience})")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(X_val_t).squeeze(1).numpy()

    mae = float(np.abs(predictions - y_val).mean())
    ref = baselines(prices, X_val, y_val, float(y_train.mean()))

    print(f"\nDone in {time.time() - started:.1f}s     (log-vol MAE, lower is better)")
    print(f"  LSTM        : {mae:.4f}")
    for name, value in ref.items():
        delta = (value - mae) / value
        print(f"  {name:<12}: {value:.4f}   LSTM is {delta:+.1%} vs this")

    # Same error expressed on the volatility scale a reader can interpret.
    pct_error = float(np.abs(np.exp(predictions) - np.exp(y_val)).mean())
    print(f"\n  mean absolute error: {pct_error:.1%} annualized volatility")

    if mae > min(ref.values()):
        print("\n  WARNING: a baseline beats the LSTM. Do not ship this checkpoint.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "sequence_length": SEQUENCE_LENGTH,
            "horizon_hours": HORIZON_HOURS,
            "log_vol_mae": mae,
            "baselines": ref,
            "trained_points": len(prices),
        },
        args.out,
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
