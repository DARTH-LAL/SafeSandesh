#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


PATTERN_RULES = [
    ("link_present", re.compile(r"https?://|www\.", re.IGNORECASE)),
    ("account_urgency", re.compile(r"urgent|blocked|suspended|verify|act now|immediately", re.IGNORECASE)),
    ("promo_marketing", re.compile(r"offer|sale|discount|free|recharge|voucher|deal", re.IGNORECASE)),
    ("payment_keyword", re.compile(r"upi|pay|payment|collect request|wallet", re.IGNORECASE)),
    ("otp_kyc", re.compile(r"otp|kyc|aadhaar|aadhar|pan", re.IGNORECASE)),
    ("job_loan", re.compile(r"job|hiring|loan|salary|work from home", re.IGNORECASE)),
    ("lottery_prize", re.compile(r"lottery|prize|winner|jackpot|reward", re.IGNORECASE)),
]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)


def detect_patterns(text: str, language: str) -> List[str]:
    tags: List[str] = []
    if len(text) < 35:
        tags.append("very_short_text")

    digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    if digit_ratio > 0.18:
        tags.append("digit_heavy")

    if language != "en":
        tags.append("non_english_message")

    if re.search(r"\b(?:ok|thanks|call me|hello|meeting|lunch|dinner)\b", text, flags=re.IGNORECASE):
        tags.append("conversational_tone")

    for name, pat in PATTERN_RULES:
        if pat.search(text):
            tags.append(name)

    if not tags:
        tags.append("no_explicit_pattern")
    return tags


def top_examples(df: pd.DataFrame, n: int = 25) -> List[Dict]:
    cols = [
        "record_id",
        "language",
        "risk_label",
        "pred_label",
        "pred_confidence",
        "risk_score",
        "scam_type",
        "predicted_scam_type",
        "patterns",
        "text",
    ]

    use = df.sort_values(["pred_confidence", "risk_score"], ascending=False).head(n).copy()
    for c in cols:
        if c not in use.columns:
            use[c] = None
    return use[cols].to_dict(orient="records")


def pattern_summary(df: pd.DataFrame, top_k: int = 10) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for tags in df["patterns"].tolist():
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return {k: int(v) for k, v in ranked}


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured error analysis for baseline model.")
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("/Users/ajneya/Desktop/FYP/DATASETS/final_with_urdu/splits/test.csv"),
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=Path("/Users/ajneya/Desktop/FYP/scam-webapp/data/models/baseline_model.joblib"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/ajneya/Desktop/FYP/scam-webapp/data/analysis/error_analysis_v1"),
    )
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_csv)
    bundle = joblib.load(args.model_file)

    risk_pipeline = bundle["risk_pipeline"]
    risk_classes = bundle["risk_classes"]
    temperature = float(bundle.get("temperature", 1.0))

    type_pipeline = bundle.get("type_pipeline_non_safe", bundle.get("type_pipeline"))
    type_classes = bundle.get("type_classes_non_safe", bundle.get("type_classes", []))
    thresholds = bundle.get("severity_thresholds", {"low_max": 30, "medium_max": 60})

    texts = test_df["text"].astype(str).tolist()
    logits = risk_pipeline.decision_function(texts)
    probs = softmax(logits / temperature)
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = np.array([risk_classes[i] for i in pred_idx])
    pred_conf = probs[np.arange(len(probs)), pred_idx]

    phish_idx = risk_classes.index("Phishing")
    risk_scores = (probs[:, phish_idx] * 100.0).round().astype(int)

    pred_type = np.array(["Other"] * len(test_df), dtype=object)
    if type_pipeline is not None and len(type_classes) > 0:
        mask = pred_labels != "Safe"
        if mask.any():
            pred_type[mask] = type_pipeline.predict(test_df.loc[mask, "text"].astype(str).tolist())

    work = test_df.copy()
    work["pred_label"] = pred_labels
    work["pred_confidence"] = pred_conf
    work["risk_score"] = risk_scores
    work["predicted_scam_type"] = pred_type
    work["patterns"] = [
        detect_patterns(t, lang)
        for t, lang in zip(work["text"].astype(str).tolist(), work["language"].astype(str).tolist())
    ]

    fp_susp = work[(work["pred_label"] == "Suspicious") & (work["risk_label"] != "Suspicious")].copy()
    fn_susp = work[(work["risk_label"] == "Suspicious") & (work["pred_label"] != "Suspicious")].copy()
    fp_phish = work[(work["pred_label"] == "Phishing") & (work["risk_label"] != "Phishing")].copy()
    fn_phish = work[(work["risk_label"] == "Phishing") & (work["pred_label"] != "Phishing")].copy()

    cls_report = classification_report(
        work["risk_label"].astype(str).to_numpy(),
        work["pred_label"].astype(str).to_numpy(),
        labels=["Safe", "Suspicious", "Phishing"],
        output_dict=True,
        zero_division=0,
    )

    summary = {
        "rows": int(len(work)),
        "temperature": temperature,
        "severity_thresholds": thresholds,
        "risk_report": cls_report,
        "error_counts": {
            "fp_suspicious": int(len(fp_susp)),
            "fn_suspicious": int(len(fn_susp)),
            "fp_phishing": int(len(fp_phish)),
            "fn_phishing": int(len(fn_phish)),
        },
        "pattern_summary": {
            "fp_suspicious": pattern_summary(fp_susp),
            "fn_suspicious": pattern_summary(fn_susp),
            "fp_phishing": pattern_summary(fp_phish),
            "fn_phishing": pattern_summary(fn_phish),
            "all_errors": pattern_summary(work[work["risk_label"] != work["pred_label"]]),
        },
        "top_examples": {
            "fp_suspicious": top_examples(fp_susp, n=25),
            "fn_suspicious": top_examples(fn_susp, n=25),
            "fp_phishing": top_examples(fp_phish, n=25),
            "fn_phishing": top_examples(fn_phish, n=25),
        },
    }

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "error_analysis.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    work[work["risk_label"] != work["pred_label"]].to_csv(out_dir / "all_errors.csv", index=False)
    fp_susp.to_csv(out_dir / "fp_suspicious.csv", index=False)
    fn_susp.to_csv(out_dir / "fn_suspicious.csv", index=False)
    fp_phish.to_csv(out_dir / "fp_phishing.csv", index=False)
    fn_phish.to_csv(out_dir / "fn_phishing.csv", index=False)

    key_patterns = summary["pattern_summary"]["all_errors"]
    top_pattern_lines = [f"{i}. `{name}` -> {count} errors" for i, (name, count) in enumerate(key_patterns.items(), start=1)]

    md = f"""# Structured Error Analysis (v1)

Total rows: **{len(work)}**

## Error counts
- FP Suspicious: **{len(fp_susp)}**
- FN Suspicious: **{len(fn_susp)}**
- FP Phishing: **{len(fp_phish)}**
- FN Phishing: **{len(fn_phish)}**

## Top failure patterns (all errors)
{chr(10).join(top_pattern_lines) if top_pattern_lines else '- None'}

## Notes
- `fn_phishing` are high-risk misses; prioritize these for cue/rule augmentation and threshold review.
- `fp_phishing` can reduce trust; inspect promotional and account-alert style messages.
- Non-English rows appear in error buckets more often due limited non-English phishing supervision.
"""
    (out_dir / "error_analysis.md").write_text(md, encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {out_dir / 'error_analysis.md'}")
    print(f"Wrote CSVs under: {out_dir}")


if __name__ == "__main__":
    main()
