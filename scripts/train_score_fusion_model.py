#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss

from scripts.train_baseline_model import DEFAULT_SPLIT_DIR, RISK_LABEL_ORDER, calibration_report_for_split
from src.bilstm import encode_texts
from src.explainability import detect_cue_tags
from src.model_runtime import (
    MODEL_KINDS,
    _calibrate_risk_probs,
    _probability_to_risk_score,
    _score_to_risk_label,
    _softmax,
    load_model_bundle,
)
from src.transformer_model import batch_predict_logits


MODEL_DIR = ROOT / "data" / "models"
FEATURE_NAMES = ["baseline_score", "bilstm_score", "indicbert_score"]
MODEL_VERSION = "score_fusion_model_v1"


def _load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"text", "risk_label"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def _iter_batches(items: list[str], batch_size: int) -> Iterable[tuple[int, list[str]]]:
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def _scores_from_probs(texts: list[str], probs: np.ndarray, bundle: dict) -> list[int]:
    risk_classes = list(bundle["risk_classes"])
    phish_idx = risk_classes.index("Phishing")
    scores: list[int] = []
    for text, row in zip(texts, probs):
        cue_tags = detect_cue_tags(text)
        calibrated, _ = _calibrate_risk_probs(row.reshape(1, -1), risk_classes, text, cue_tags)
        scores.append(_probability_to_risk_score(float(calibrated[0, phish_idx])))
    return scores


def _predict_scores_for_kind(kind: str, texts: list[str], batch_size: int) -> list[int]:
    bundle = load_model_bundle(kind)
    if bundle is None:
        raise RuntimeError(f"Could not load model bundle for {kind}")

    temp = float(bundle.get("temperature", 1.0))
    if kind == "baseline":
        pipeline = bundle["risk_pipeline"]
        logits = pipeline.decision_function(texts)
        probs = _softmax(logits / max(temp, 1e-6))
        return _scores_from_probs(texts, probs, bundle)

    if kind == "bilstm":
        model = bundle["risk_model"]
        vocab = bundle["risk_vocab"]
        max_length = int(bundle["risk_max_length"])
        out: list[int] = []
        model.eval()
        for _, batch_texts in _iter_batches(texts, batch_size):
            input_ids, lengths = encode_texts(batch_texts, vocab, max_length)
            input_ids_t = torch.as_tensor(input_ids, dtype=torch.long)
            lengths_t = torch.as_tensor(lengths, dtype=torch.long)
            with torch.inference_mode():
                logits = model(input_ids_t, lengths_t).detach().cpu().numpy()
            probs = _softmax(logits / max(temp, 1e-6))
            out.extend(_scores_from_probs(batch_texts, probs, bundle))
        return out

    if kind == "indicbert":
        model = bundle["risk_model"]
        tokenizer = bundle["risk_tokenizer"]
        max_length = int(bundle["risk_max_length"])
        device = next(model.parameters()).device
        out: list[int] = []
        for _, batch_texts in _iter_batches(texts, batch_size):
            logits = batch_predict_logits(model, tokenizer, batch_texts, max_length, batch_size, device)
            probs = _softmax(logits / max(temp, 1e-6))
            out.extend(_scores_from_probs(batch_texts, probs, bundle))
        return out

    raise ValueError(f"Unsupported model kind: {kind}")


def _feature_cache_path(cache_dir: Path, split_name: str) -> Path:
    return cache_dir / f"score_fusion_features_{split_name}.csv"


def build_features(df: pd.DataFrame, split_name: str, cache_dir: Path, batch_size: int, refresh_cache: bool) -> pd.DataFrame:
    cache_path = _feature_cache_path(cache_dir, split_name)
    if cache_path.exists() and not refresh_cache:
        return pd.read_csv(cache_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    texts = df["text"].astype(str).tolist()
    features = pd.DataFrame(
        {
            "risk_label": df["risk_label"].astype(str).to_numpy(),
            "text": texts,
        }
    )
    for kind in MODEL_KINDS:
        print(f"[score-fusion] predicting {split_name} with {kind} ({len(texts)} rows)")
        features[f"{kind}_score"] = _predict_scores_for_kind(kind, texts, batch_size=batch_size)

    features.to_csv(cache_path, index=False)
    return features


def labels_from_scores(scores: np.ndarray) -> np.ndarray:
    thresholds = {"low_max": 30, "medium_max": 60, "high_max": 80, "critical_min": 81}
    return np.asarray([_score_to_risk_label(int(round(score)), thresholds) for score in scores], dtype=object)


def expected_calibration_error(phish_probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    y_bin = (y_true == "Phishing").astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(phish_probs)
    if total == 0:
        return 0.0
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi >= 1.0:
            mask = (phish_probs >= lo) & (phish_probs <= hi)
        else:
            mask = (phish_probs >= lo) & (phish_probs < hi)
        if not np.any(mask):
            continue
        confidence = float(np.mean(phish_probs[mask]))
        accuracy = float(np.mean(y_bin[mask]))
        ece += (float(np.sum(mask)) / total) * abs(confidence - accuracy)
    return float(ece)


def evaluate_scores(name: str, scores: np.ndarray, y_true: np.ndarray) -> dict:
    y_pred = labels_from_scores(scores)
    phish_probs = np.clip(scores / 100.0, 0.0, 1.0)
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=RISK_LABEL_ORDER, average="macro", zero_division=0)),
        "ece": expected_calibration_error(phish_probs, y_true),
        "classification_report": classification_report(y_true, y_pred, labels=RISK_LABEL_ORDER, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=RISK_LABEL_ORDER).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a learned final-score fusion model from baseline/BiLSTM/IndicBERT scores.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--model-out", type=Path, default=MODEL_DIR / "score_fusion_model.joblib")
    parser.add_argument("--metrics-out", type=Path, default=MODEL_DIR / "score_fusion_metrics.json")
    parser.add_argument("--cache-dir", type=Path, default=MODEL_DIR / "score_fusion_cache")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--deploy-if-better", action="store_true", default=True)
    args = parser.parse_args()

    val_df = _load_split(args.split_dir / "val.csv")
    test_df = _load_split(args.split_dir / "test.csv")

    val_features = build_features(val_df, "val", args.cache_dir, args.batch_size, args.refresh_cache)
    test_features = build_features(test_df, "test", args.cache_dir, args.batch_size, args.refresh_cache)

    x_val = val_features[FEATURE_NAMES].to_numpy(dtype=float) / 100.0
    y_val = val_features["risk_label"].astype(str).to_numpy()
    x_test = test_features[FEATURE_NAMES].to_numpy(dtype=float) / 100.0
    y_test = test_features["risk_label"].astype(str).to_numpy()

    y_val_binary = (y_val == "Phishing").astype(int)
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(x_val, y_val_binary)

    fusion_probs = model.predict_proba(x_test)[:, 1]
    fusion_scores = fusion_probs * 100.0
    median_scores = np.median(test_features[FEATURE_NAMES].to_numpy(dtype=float), axis=1)
    average_scores = np.mean(test_features[FEATURE_NAMES].to_numpy(dtype=float), axis=1)

    fusion_eval = evaluate_scores("score_fusion_model_v1", fusion_scores, y_test)
    median_eval = evaluate_scores("median_ensemble_v1", median_scores, y_test)
    average_eval = evaluate_scores("average_ensemble_v1", average_scores, y_test)
    deploy_as_final = bool(
        args.deploy_if_better
        and fusion_eval["macro_f1"] >= median_eval["macro_f1"]
        and fusion_eval["ece"] <= max(0.05, median_eval["ece"] * 1.25)
    )

    bundle = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "model_family": "score_fusion",
        "model_source": "logistic_regression_score_fusion",
        "features": FEATURE_NAMES,
        "model": model,
        "trained_on": "validation split base-model scores",
        "target": "Phishing probability",
        "score_scale": "0-100",
        "deploy_as_final": deploy_as_final,
        "baseline_method": "median_ensemble_v1",
        "metrics": {
            "fusion": fusion_eval,
            "median": median_eval,
            "average": average_eval,
        },
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)

    metrics = {
        "created_at": bundle["created_at"],
        "model_version": MODEL_VERSION,
        "split_dir": str(args.split_dir.resolve()),
        "rows": {"val": int(len(val_features)), "test": int(len(test_features))},
        "features": FEATURE_NAMES,
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "deploy_as_final": deploy_as_final,
        "recommendation": "score_fusion_model_v1" if deploy_as_final else "median_ensemble_v1",
        "fusion": fusion_eval,
        "median": median_eval,
        "average": average_eval,
        "calibration_report": calibration_report_for_split(
            np.column_stack([1.0 - fusion_probs, fusion_probs]),
            y_test,
            ["Safe", "Phishing"],
            n_bins=10,
        ),
    }
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nScore fusion comparison")
    print(f"Median macro-F1: {median_eval['macro_f1']:.4f} | ECE: {median_eval['ece']:.4f}")
    print(f"Fusion macro-F1: {fusion_eval['macro_f1']:.4f} | ECE: {fusion_eval['ece']:.4f}")
    print(f"Recommendation: {metrics['recommendation']}")
    print(f"Artifact: {args.model_out}")


if __name__ == "__main__":
    main()
