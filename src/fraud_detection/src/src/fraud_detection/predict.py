from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from fraud_detection.config import DEFAULT_MODEL_PATH, REPORTS_DIR, TARGET_COLUMN


def score_transactions(input_path: Path, model_path: Path, output_path: Path) -> pd.DataFrame:
    artifact = joblib.load(model_path)
    model = artifact["model"]
    threshold = artifact["threshold"]
    feature_columns = artifact["feature_columns"]

    df = pd.read_csv(input_path)
    missing = sorted(set(feature_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    scores = model.predict_proba(df[feature_columns])[:, 1]
    result = df.copy()
    result["fraud_score"] = scores
    result["alert"] = result["fraud_score"] >= threshold
    result["decision"] = result["alert"].map({True: "revisar", False: "aprovar"})

    if TARGET_COLUMN in result.columns:
        result = result.drop(columns=[TARGET_COLUMN])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score new transactions with the trained fraud model.")
    parser.add_argument("--input", type=Path, default=REPORTS_DIR / "predictions_sample.csv")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "scored_transactions.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = score_transactions(args.input, args.model, args.output)
    print(f"Scored {len(result)} transactions. Output saved to {args.output}")


if __name__ == "__main__":
    main()

