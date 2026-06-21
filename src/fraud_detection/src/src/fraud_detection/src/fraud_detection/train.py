from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_RANDOM_STATE,
    MODELS_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
)


@dataclass
class BusinessCosts:
    review_cost: float = 8.0
    fraud_loss: float = 450.0


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Generate a sample with "
            "`python -m fraud_detection.data --output data/raw/creditcard_sample.csv` "
            "or place the Kaggle creditcard.csv in data/raw/."
        )

    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain the target column '{TARGET_COLUMN}'.")

    return df


def split_dataset(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    ordered = df.sort_values("Time").reset_index(drop=True) if "Time" in df.columns else df.reset_index(drop=True)
    x = ordered.drop(columns=[target])
    y = ordered[target].astype(int)

    if "Time" in ordered.columns:
        train_end = int(len(ordered) * 0.70)
        validation_end = int(len(ordered) * 0.85)
        return (
            x.iloc[:train_end],
            x.iloc[train_end:validation_end],
            x.iloc[validation_end:],
            y.iloc[:train_end],
            y.iloc[train_end:validation_end],
            y.iloc[validation_end:],
        )

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=random_state
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def make_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipe, feature_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_models(feature_columns: list[str], random_state: int) -> dict[str, Pipeline]:
    preprocessor = make_preprocessor(feature_columns)
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(feature_columns)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=160,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def predict_scores(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def classification_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    metrics = {
        "average_precision": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    return {key: float(value) for key, value in metrics.items()}


def business_cost(y_true: pd.Series, scores: np.ndarray, threshold: float, costs: BusinessCosts) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    baseline_cost = float((tp + fn) * costs.fraud_loss)
    model_cost = float(fp * costs.review_cost + fn * costs.fraud_loss)
    return {
        "threshold": float(threshold),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "alerts": int(fp + tp),
        "baseline_cost": baseline_cost,
        "model_cost": model_cost,
        "estimated_savings": baseline_cost - model_cost,
    }


def find_best_threshold(y_true: pd.Series, scores: np.ndarray, costs: BusinessCosts) -> dict[str, float]:
    candidate_thresholds = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 200)))
    candidate_thresholds = np.append(candidate_thresholds, [0.5])
    scenarios = [business_cost(y_true, scores, float(threshold), costs) for threshold in candidate_thresholds]
    return max(scenarios, key=lambda item: (item["estimated_savings"], item["true_positives"]))


def get_feature_importance(model: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        values = np.zeros(len(feature_columns))

    return (
        pd.DataFrame({"feature": feature_columns, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def write_business_summary(
    path: Path,
    best_model_name: str,
    test_metrics: dict[str, float],
    test_business: dict[str, float],
    costs: BusinessCosts,
) -> None:
    path.write_text(
        "\n".join(
            [
                "# Resumo executivo",
                "",
                f"Modelo selecionado: **{best_model_name}**",
                f"Limiar recomendado: **{test_business['threshold']:.4f}**",
                "",
                "## Resultado no conjunto de teste",
                "",
                f"- PR-AUC: `{test_metrics['average_precision']:.4f}`",
                f"- ROC-AUC: `{test_metrics['roc_auc']:.4f}`",
                f"- Precisao: `{test_metrics['precision']:.4f}`",
                f"- Recall: `{test_metrics['recall']:.4f}`",
                f"- F1-score: `{test_metrics['f1']:.4f}`",
                "",
                "## Impacto financeiro estimado",
                "",
                f"- Custo de revisao por alerta: `R$ {costs.review_cost:.2f}`",
                f"- Perda media por fraude nao capturada: `R$ {costs.fraud_loss:.2f}`",
                f"- Alertas gerados: `{test_business['alerts']}`",
                f"- Fraudes capturadas: `{test_business['true_positives']}`",
                f"- Fraudes nao capturadas: `{test_business['false_negatives']}`",
                f"- Economia estimada versus nao usar modelo: `R$ {test_business['estimated_savings']:.2f}`",
            ]
        ),
        encoding="utf-8",
    )


def train_and_evaluate(
    data_path: Path,
    model_path: Path,
    report_dir: Path,
    random_state: int,
    costs: BusinessCosts,
) -> dict[str, Any]:
    df = load_dataset(data_path)
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(df, random_state=random_state)
    feature_columns = list(x_train.columns)
    models = make_models(feature_columns, random_state=random_state)

    validation_results: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        fitted_models[name] = model
        scores = predict_scores(model, x_val)
        threshold_result = find_best_threshold(y_val, scores, costs)
        validation_results[name] = {
            "metrics": classification_metrics(y_val, scores, threshold_result["threshold"]),
            "business": threshold_result,
        }

    best_model_name = max(
        validation_results,
        key=lambda item: (
            validation_results[item]["metrics"]["average_precision"],
            validation_results[item]["business"]["estimated_savings"],
        ),
    )
    best_model = fitted_models[best_model_name]
    threshold = validation_results[best_model_name]["business"]["threshold"]

    test_scores = predict_scores(best_model, x_test)
    test_metrics = classification_metrics(y_test, test_scores, threshold)
    test_business = business_cost(y_test, test_scores, threshold, costs)

    report_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": best_model,
        "threshold": threshold,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "best_model_name": best_model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "costs": asdict(costs),
    }
    joblib.dump(artifact, model_path)

    y_pred = (test_scores >= threshold).astype(int)
    pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=[0, 1]),
        index=["real_legitima", "real_fraude"],
        columns=["prevista_legitima", "prevista_fraude"],
    ).to_csv(report_dir / "confusion_matrix.csv")

    scored_sample = x_test.copy()
    scored_sample[TARGET_COLUMN] = y_test.values
    scored_sample["fraud_score"] = test_scores
    scored_sample["predicted_class"] = y_pred
    scored_sample.head(200).to_csv(report_dir / "predictions_sample.csv", index=False)

    get_feature_importance(best_model, feature_columns).to_csv(report_dir / "feature_importance.csv", index=False)

    metrics_payload = {
        "best_model": best_model_name,
        "validation": validation_results,
        "test": {
            "metrics": test_metrics,
            "business": test_business,
        },
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    write_business_summary(report_dir / "business_summary.md", best_model_name, test_metrics, test_business, costs)

    return metrics_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a credit card fraud detection model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--review-cost", type=float, default=8.0)
    parser.add_argument("--fraud-loss", type=float, default=450.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = train_and_evaluate(
        data_path=args.data,
        model_path=args.model,
        report_dir=args.reports,
        random_state=args.random_state,
        costs=BusinessCosts(review_cost=args.review_cost, fraud_loss=args.fraud_loss),
    )
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()

