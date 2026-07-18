#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT.parent / "DATASETS"


RISK_LABEL_ORDER = ["Safe", "Suspicious", "Phishing"]
DEFAULT_TYPE_CONFIDENCE_MIN = 0.40


def resolve_default_split_dir() -> Path:
    candidates = [
        DATASETS_ROOT / "final_with_urdu_v3" / "splits",
        DATASETS_ROOT / "final_with_urdu_v2" / "splits",
        DATASETS_ROOT / "final_with_urdu" / "splits",
    ]
    for candidate in candidates:
        if all((candidate / name).exists() for name in ("train.csv", "val.csv", "test.csv")):
            return candidate
    return candidates[0]


DEFAULT_SPLIT_DIR = resolve_default_split_dir()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)


def build_text_features() -> FeatureUnion:
    return FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="word",
                    token_pattern=r"(?u)\b\w+\b",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=120_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=80_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def build_classifier() -> LogisticRegression:
    return LogisticRegression(
        solver="saga",
        max_iter=600,
        class_weight="balanced",
        C=2.0,
        random_state=42,
        n_jobs=-1,
    )


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("features", build_text_features()),
            ("clf", build_classifier()),
        ]
    )


def find_best_temperature(logits: np.ndarray, y_true: np.ndarray, classes: List[str]) -> float:
    class_to_idx = {label: i for i, label in enumerate(classes)}
    y_idx = np.array([class_to_idx[y] for y in y_true], dtype=np.int64)

    best_t = 1.0
    best_loss = float("inf")
    for t in np.arange(0.6, 3.01, 0.05):
        probs = softmax(logits / t)
        loss = log_loss(y_idx, probs, labels=list(range(len(classes))))
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t


def report_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def compute_calibrated_probs(
    pipeline: Pipeline,
    texts: List[str],
    temperature: float,
) -> np.ndarray:
    logits = pipeline.decision_function(texts)
    return softmax(logits / temperature)


def phishing_scores_from_probs(probs: np.ndarray, risk_classes: List[str]) -> np.ndarray:
    phish_idx = risk_classes.index("Phishing") if "Phishing" in risk_classes else int(np.argmax(probs.mean(axis=0)))
    return probs[:, phish_idx] * 100.0


def expected_calibration_error_binary(
    probs: np.ndarray,
    y_true_binary: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, float]]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = max(len(probs), 1)
    ece = 0.0
    rows: List[Dict[str, float]] = []

    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)

        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": i,
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": 0,
                    "avg_confidence": 0.0,
                    "empirical_positive_rate": 0.0,
                    "gap": 0.0,
                }
            )
            continue

        avg_conf = float(probs[mask].mean())
        avg_true = float(y_true_binary[mask].mean())
        gap = abs(avg_conf - avg_true)
        ece += (count / n) * gap
        rows.append(
            {
                "bin": i,
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "avg_confidence": avg_conf,
                "empirical_positive_rate": avg_true,
                "gap": float(gap),
            }
        )

    return float(ece), rows


def tune_risk_thresholds(val_scores: np.ndarray, val_labels: np.ndarray) -> Dict[str, float]:
    best_macro = -1.0
    best_t1 = 30
    best_t2 = 60
    default_distance_best = float("inf")

    for t1 in range(5, 81):
        for t2 in range(t1 + 5, 96):
            pred = np.where(
                val_scores <= t1,
                "Safe",
                np.where(val_scores <= t2, "Suspicious", "Phishing"),
            )
            macro = f1_score(
                val_labels,
                pred,
                labels=RISK_LABEL_ORDER,
                average="macro",
                zero_division=0,
            )
            default_distance = abs(t1 - 30) + abs(t2 - 60)
            if (
                macro > best_macro + 1e-12
                or (abs(macro - best_macro) <= 1e-12 and default_distance < default_distance_best)
            ):
                best_macro = float(macro)
                best_t1 = int(t1)
                best_t2 = int(t2)
                default_distance_best = float(default_distance)

    phish_scores = val_scores[val_labels == "Phishing"]
    if len(phish_scores) > 0:
        t3 = int(np.percentile(phish_scores, 75))
    else:
        t3 = best_t2 + 15
    t3 = int(np.clip(t3, best_t2 + 5, 99))

    return {
        "low_max": best_t1,
        "medium_max": best_t2,
        "high_max": t3,
        "critical_min": t3 + 1,
        "threshold_objective_macro_f1": best_macro,
    }


def label_from_score(score: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    low_max = thresholds["low_max"]
    medium_max = thresholds["medium_max"]
    return np.where(score <= low_max, "Safe", np.where(score <= medium_max, "Suspicious", "Phishing"))


def tier_from_score(score: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    low_max = thresholds["low_max"]
    medium_max = thresholds["medium_max"]
    high_max = thresholds["high_max"]

    return np.where(
        score <= low_max,
        "Low",
        np.where(score <= medium_max, "Medium", np.where(score <= high_max, "High", "Critical")),
    )


def calibration_report_for_split(
    probs: np.ndarray,
    y_true: np.ndarray,
    risk_classes: List[str],
    n_bins: int = 10,
) -> Dict:
    phish_idx = risk_classes.index("Phishing")
    phish_probs = probs[:, phish_idx]
    y_bin = (y_true == "Phishing").astype(int)
    ece, bins = expected_calibration_error_binary(phish_probs, y_bin, n_bins=n_bins)
    return {
        "target": "Phishing",
        "ece": float(ece),
        "n_bins": int(n_bins),
        "reliability_bins": bins,
    }


def evaluate_risk_model(
    pipeline: Pipeline,
    df: pd.DataFrame,
    temperature: float,
    risk_classes: List[str],
    thresholds: Dict[str, float],
) -> Dict:
    x = df["text"].astype(str).tolist()
    y_true = df["risk_label"].astype(str).to_numpy()

    probs = compute_calibrated_probs(pipeline, x, temperature)
    y_pred = np.array([risk_classes[i] for i in np.argmax(probs, axis=1)])
    scores = phishing_scores_from_probs(probs, risk_classes)
    y_score_pred = label_from_score(scores, thresholds)
    tiers = tier_from_score(scores, thresholds)

    out = report_from_predictions(y_true, y_pred, risk_classes)
    out["rows"] = int(len(df))
    out["temperature"] = float(temperature)
    out["score_threshold_report"] = report_from_predictions(y_true, y_score_pred, RISK_LABEL_ORDER)
    out["severity_distribution"] = {
        str(k): int(v) for k, v in pd.Series(tiers).value_counts().to_dict().items()
    }
    out["calibration"] = calibration_report_for_split(probs, y_true, risk_classes, n_bins=10)

    per_language = {}
    for language, subdf in df.groupby("language"):
        sx = subdf["text"].astype(str).tolist()
        sy_true = subdf["risk_label"].astype(str).to_numpy()
        sprobs = compute_calibrated_probs(pipeline, sx, temperature)
        sy_pred = np.array([risk_classes[i] for i in np.argmax(sprobs, axis=1)])
        sscores = phishing_scores_from_probs(sprobs, risk_classes)
        sy_score_pred = label_from_score(sscores, thresholds)

        per_language[str(language)] = {
            **report_from_predictions(sy_true, sy_pred, risk_classes),
            "rows": int(len(subdf)),
            "score_threshold_report": report_from_predictions(sy_true, sy_score_pred, RISK_LABEL_ORDER),
        }

    out["per_language"] = per_language
    return out


def train_type_pipeline(df: pd.DataFrame) -> Tuple[Pipeline | None, List[str]]:
    if df.empty:
        return None, []
    y = df["scam_type"].astype(str)
    if y.nunique() <= 1:
        return None, []

    pipeline = build_pipeline()
    pipeline.fit(df["text"].astype(str).tolist(), y.to_numpy())
    classes = list(pipeline.named_steps["clf"].classes_)
    return pipeline, classes


def evaluate_type_model(
    pipeline: Pipeline | None,
    df: pd.DataFrame,
    type_labels: List[str],
    description: str,
) -> Dict:
    if pipeline is None:
        return {"available": False, "description": description}

    if df.empty:
        return {"available": False, "description": description, "reason": "No rows in evaluation slice."}

    x = df["text"].astype(str).tolist()
    y_true = df["scam_type"].astype(str).to_numpy()
    y_pred = pipeline.predict(x)

    out = report_from_predictions(y_true, y_pred, type_labels)
    out["available"] = True
    out["description"] = description
    out["rows"] = int(len(df))
    return out


def load_split_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"text", "language", "risk_label", "scam_type"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline TF-IDF + LogisticRegression models.")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_SPLIT_DIR / "train.csv",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=DEFAULT_SPLIT_DIR / "val.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=DEFAULT_SPLIT_DIR / "test.csv",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=ROOT / "data" / "models" / "baseline_model.joblib",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=ROOT / "data" / "models" / "baseline_metrics.json",
    )
    parser.add_argument(
        "--type-confidence-min",
        type=float,
        default=DEFAULT_TYPE_CONFIDENCE_MIN,
        help="Minimum classifier confidence before trusting scam-type head.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="baseline_v3",
        help="Version tag embedded into model artifact and runtime output.",
    )
    args = parser.parse_args()

    train_df = load_split_csv(args.train_csv)
    val_df = load_split_csv(args.val_csv)
    test_df = load_split_csv(args.test_csv)

    risk_pipeline = build_pipeline()
    risk_pipeline.fit(train_df["text"].astype(str).tolist(), train_df["risk_label"].astype(str).to_numpy())
    risk_classes = list(risk_pipeline.named_steps["clf"].classes_)

    val_probs_no_temp = compute_calibrated_probs(
        risk_pipeline,
        val_df["text"].astype(str).tolist(),
        temperature=1.0,
    )
    val_logits = risk_pipeline.decision_function(val_df["text"].astype(str).tolist())
    temperature = find_best_temperature(val_logits, val_df["risk_label"].astype(str).to_numpy(), risk_classes)

    val_probs = compute_calibrated_probs(
        risk_pipeline,
        val_df["text"].astype(str).tolist(),
        temperature=temperature,
    )
    val_scores = phishing_scores_from_probs(val_probs, risk_classes)
    val_labels = val_df["risk_label"].astype(str).to_numpy()
    severity_thresholds = tune_risk_thresholds(val_scores, val_labels)

    # Two-stage type heads.
    train_non_safe_df = train_df[train_df["risk_label"] != "Safe"].copy()
    train_phishing_df = train_df[train_df["risk_label"] == "Phishing"].copy()

    type_pipeline_non_safe, type_classes_non_safe = train_type_pipeline(train_non_safe_df)
    type_pipeline_phishing, type_classes_phishing = train_type_pipeline(train_phishing_df)

    model_out = args.model_out.resolve()
    model_out.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": str(args.model_version),
        "risk_pipeline": risk_pipeline,
        "risk_classes": risk_classes,
        "temperature": float(temperature),
        "severity_thresholds": severity_thresholds,
        "type_pipeline_non_safe": type_pipeline_non_safe,
        "type_classes_non_safe": type_classes_non_safe,
        "type_pipeline_phishing": type_pipeline_phishing,
        "type_classes_phishing": type_classes_phishing,
        "type_confidence_min": float(args.type_confidence_min),
        # Backward compatibility keys.
        "type_pipeline": type_pipeline_non_safe,
        "type_classes": type_classes_non_safe,
        "label_order_hint": RISK_LABEL_ORDER,
    }
    joblib.dump(bundle, model_out)

    risk_metrics = {
        "train": evaluate_risk_model(risk_pipeline, train_df, temperature, risk_classes, severity_thresholds),
        "val": evaluate_risk_model(risk_pipeline, val_df, temperature, risk_classes, severity_thresholds),
        "test": evaluate_risk_model(risk_pipeline, test_df, temperature, risk_classes, severity_thresholds),
    }

    type_metrics = {
        "non_safe": {
            "train": evaluate_type_model(
                type_pipeline_non_safe,
                train_non_safe_df,
                type_classes_non_safe,
                "Type model on non-safe messages",
            ),
            "val": evaluate_type_model(
                type_pipeline_non_safe,
                val_df[val_df["risk_label"] != "Safe"],
                type_classes_non_safe,
                "Type model on non-safe messages",
            ),
            "test": evaluate_type_model(
                type_pipeline_non_safe,
                test_df[test_df["risk_label"] != "Safe"],
                type_classes_non_safe,
                "Type model on non-safe messages",
            ),
        },
        "phishing": {
            "train": evaluate_type_model(
                type_pipeline_phishing,
                train_phishing_df,
                type_classes_phishing,
                "Type model specialized for phishing messages",
            ),
            "val": evaluate_type_model(
                type_pipeline_phishing,
                val_df[val_df["risk_label"] == "Phishing"],
                type_classes_phishing,
                "Type model specialized for phishing messages",
            ),
            "test": evaluate_type_model(
                type_pipeline_phishing,
                test_df[test_df["risk_label"] == "Phishing"],
                type_classes_phishing,
                "Type model specialized for phishing messages",
            ),
        },
    }

    calibration = {
        "val": {
            "before_temperature": calibration_report_for_split(
                val_probs_no_temp,
                val_df["risk_label"].astype(str).to_numpy(),
                risk_classes,
            ),
            "after_temperature": risk_metrics["val"]["calibration"],
        },
        "test": risk_metrics["test"]["calibration"],
    }

    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": str(args.model_version),
        "input_splits": {
            "train_csv": str(args.train_csv.resolve()),
            "val_csv": str(args.val_csv.resolve()),
            "test_csv": str(args.test_csv.resolve()),
        },
        "rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "risk_classes": risk_classes,
        "temperature": float(temperature),
        "severity_thresholds": severity_thresholds,
        "type_confidence_min": float(args.type_confidence_min),
        "risk_metrics": risk_metrics,
        "type_classes_non_safe": type_classes_non_safe,
        "type_classes_phishing": type_classes_phishing,
        "type_metrics": type_metrics,
        "calibration": calibration,
    }

    metrics_out = args.metrics_out.resolve()
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Wrote model: {model_out}")
    print(f"Wrote metrics: {metrics_out}")
    print(f"Risk classes: {risk_classes}")
    print(f"Temperature: {temperature:.2f}")
    print(f"Severity thresholds: {severity_thresholds}")
    print(
        "Test Macro-F1 (argmax risk):",
        metrics["risk_metrics"]["test"]["classification_report"]["macro avg"]["f1-score"],
    )
    print(
        "Test Macro-F1 (score-threshold risk):",
        metrics["risk_metrics"]["test"]["score_threshold_report"]["classification_report"]["macro avg"]["f1-score"],
    )


if __name__ == "__main__":
    main()
