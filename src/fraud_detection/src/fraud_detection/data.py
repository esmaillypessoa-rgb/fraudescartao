from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.config import DEFAULT_RANDOM_STATE


def make_synthetic_creditcard_data(
    rows: int = 5000,
    fraud_rate: float = 0.012,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Create a small anonymized dataset with the same columns as the Kaggle base."""
    rng = np.random.default_rng(random_state)
    is_fraud = rng.binomial(1, fraud_rate, size=rows)

    time = np.sort(rng.integers(0, 172800, size=rows))
    amount = rng.lognormal(mean=3.2, sigma=1.1, size=rows)
    amount = amount + is_fraud * rng.lognormal(mean=3.8, sigma=0.8, size=rows)

    features = {}
    signal_shifts = {3: 1.25, 4: -1.10, 10: 1.45, 11: -1.20, 14: 1.70, 17: -1.35}
    for idx in range(1, 29):
        base = rng.normal(0, 1, size=rows)
        fraud_shift = signal_shifts.get(idx, rng.normal(0.05, 0.05))
        features[f"V{idx}"] = base + is_fraud * fraud_shift

    df = pd.DataFrame({"Time": time, **features, "Amount": amount.round(2), "Class": is_fraud})
    return df


def save_synthetic_dataset(output: Path, rows: int, fraud_rate: float, random_state: int) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    df = make_synthetic_creditcard_data(rows=rows, fraud_rate=fraud_rate, random_state=random_state)
    df.to_csv(output, index=False)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic credit card fraud dataset.")
    parser.add_argument("--output", type=Path, default=Path("data/raw/creditcard_sample.csv"))
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--fraud-rate", type=float, default=0.012)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = save_synthetic_dataset(args.output, args.rows, args.fraud_rate, args.random_state)
    print(f"Synthetic dataset saved to {output}")


if __name__ == "__main__":
    main()
