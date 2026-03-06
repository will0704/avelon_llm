"""
Train XGBoost credit scorer model for Avelon LLM.

Generates synthetic credit scoring data and trains an XGBoost regressor
that predicts a 0-100 credit score.

Usage:
    python train_scorer.py
    python train_scorer.py --samples 2000 --output app/models/credit_scorer.json
"""
import os
import sys
import argparse
import random

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
from app.services.scorer_service import ScorerService


def get_args():
    parser = argparse.ArgumentParser(description="Train XGBoost credit scorer")
    parser.add_argument(
        "--samples", type=int, default=2000,
        help="Number of synthetic training samples"
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(os.path.dirname(__file__), "app", "models", "credit_scorer.json"),
        help="Output path for the trained model"
    )
    parser.add_argument("--rounds", type=int, default=100, help="XGBoost boosting rounds")
    return parser.parse_args()


def generate_synthetic_sample() -> tuple:
    """
    Generate a single synthetic credit scoring sample.
    
    Returns (feature_vector, target_score) where target_score
    is computed by the rule-based scorer (ground truth).
    """
    # Randomly generate applicant profile
    gov_id_verified = random.random() > 0.2  # 80% chance
    gov_id_confidence = random.uniform(0.6, 0.99) if gov_id_verified else 0.0
    income_verified = random.random() > 0.3
    address_verified = random.random() > 0.3

    fraud_flags = []
    num_flags = random.choices([0, 1, 2, 3], weights=[0.6, 0.2, 0.1, 0.1])[0]
    for _ in range(num_flags):
        fraud_flags.append({
            "severity": random.choice(["low", "medium", "high"]),
        })

    monthly_income = random.choice([
        0, 5000, 10000, 15000, 20000, 30000, 40000,
        50000, 60000, 80000, 100000, 150000
    ])
    employment_type = random.choice(["permanent", "contract", "self-employed", "part-time", ""])
    years_employed = random.choice([0, 0.5, 1, 2, 3, 5, 7, 10])
    dti_ratio = random.uniform(0, 0.7)

    total_loans = random.randint(0, 10)
    repaid_loans = random.randint(0, total_loans)
    defaulted_loans = random.randint(0, max(total_loans - repaid_loans, 0))
    late_payments = random.randint(0, total_loans * 2)

    wallet_age_days = random.choice([10, 30, 60, 90, 120, 180, 365, 500])
    wallet_tx_count = random.randint(0, 100)
    wallet_balance_eth = random.uniform(0, 3.0)

    # Build inputs for ScorerService
    extracted_data = {
        "verified_documents": {
            "government_id": {"is_verified": gov_id_verified, "confidence": gov_id_confidence},
            "proof_of_income": {"is_verified": income_verified},
            "proof_of_address": {"is_verified": address_verified},
        },
        "fraud_flags": fraud_flags,
        "monthly_income": monthly_income,
        "employment_type": employment_type,
        "years_employed": years_employed,
        "debt_to_income_ratio": dti_ratio,
    }
    wallet_data = {
        "age_days": wallet_age_days,
        "transaction_count": wallet_tx_count,
        "balance_eth": wallet_balance_eth,
    }

    from app.schemas.score import LoanHistory
    loan_history = LoanHistory(
        total_loans=total_loans,
        repaid_loans=repaid_loans,
        defaulted_loans=defaulted_loans,
        late_payments=late_payments,
    ) if total_loans > 0 else None

    # Feature vector (matches ScorerService.FEATURE_NAMES order)
    emp_scores = {"permanent": 4, "contract": 3, "self-employed": 2, "part-time": 1}
    employment_score = emp_scores.get(employment_type, 0) if employment_type else 0

    features = [
        float(gov_id_verified),
        float(gov_id_confidence),
        float(income_verified),
        float(address_verified),
        float(len(fraud_flags)),
        float(sum(1 for f in fraud_flags if f.get("severity") == "high")),
        float(monthly_income),
        float(years_employed),
        float(dti_ratio),
        float(employment_score),
        float(total_loans if loan_history else 0),
        float(repaid_loans if loan_history else 0),
        float(defaulted_loans if loan_history else 0),
        float(late_payments if loan_history else 0),
        float(wallet_age_days),
        float(wallet_tx_count),
        float(wallet_balance_eth),
    ]

    # Target = rule-based score (ground truth for the model to learn)
    scorer = ScorerService(model_path=None)
    score, _, _ = scorer.calculate_score(extracted_data, wallet_data, loan_history)

    # Add small noise to avoid perfect overfitting to rules
    noisy_score = score + random.gauss(0, 2)
    noisy_score = max(0, min(100, noisy_score))

    return features, noisy_score


def main():
    args = get_args()

    print(f"Generating {args.samples} synthetic credit scoring samples...")

    features_list = []
    targets_list = []
    for i in range(args.samples):
        feat, target = generate_synthetic_sample()
        features_list.append(feat)
        targets_list.append(target)
        if (i + 1) % 500 == 0:
            print(f"  Generated: {i + 1}/{args.samples}")

    X = np.array(features_list)
    y = np.array(targets_list)

    print(f"\nDataset shape: {X.shape}")
    print(f"Score range: {y.min():.1f} - {y.max():.1f}, mean: {y.mean():.1f}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create DMatrix with feature names
    feature_names = ScorerService.FEATURE_NAMES
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    # Train XGBoost regressor
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'eval_metric': 'mae',
        'seed': 42,
    }

    print(f"\nTraining XGBoost ({args.rounds} rounds)...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.rounds,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        verbose_eval=20,
    )

    # Evaluate
    y_pred = model.predict(dtest)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nTest MAE: {mae:.2f}")
    print(f"Test R²: {r2:.4f}")

    # Feature importance
    importance = model.get_score(importance_type='gain')
    print("\nFeature Importances (gain):")
    for name, gain in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {name}: {gain:.2f}")

    # Save model
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save_model(output_path)
    print(f"\nModel saved to: {output_path}")
    print(f"Set SCORER_MODEL_PATH={output_path} in your .env")


if __name__ == "__main__":
    main()
