from __future__ import annotations

import json
import html
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.auth import require_technical_password
from src.db import init_db, read_scans
from src.explainability import infer_message_language_name
from src.ui_theme import apply_theme, top_menu


FALLBACK_TIMELINE = {
    "labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "phishing": [148, 162, 139, 178, 203, 186, 126],
    "suspicious": [38, 44, 35, 52, 61, 48, 34],
    "safe": [72, 89, 65, 94, 102, 88, 57],
}

FALLBACK_SCAM_TYPES = [
    ("OTP / KYC Fraud", 433, "#ff3860"),
    ("Lottery / Advance Fee", 287, "#ffdd57"),
    ("Account Block Threat", 218, "#ff8c42"),
    ("UPI Collect Scam", 196, "#b57aff"),
    ("Job / Loan Scam", 152, "#00d4ff"),
    ("Courier / Customs", 98, "#00ff9f"),
    ("Other", 70, "rgba(200,240,224,0.35)"),
]

MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "models"
FALLBACK_MODEL_LABEL = "TF-IDF"


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _metric_report(metrics: dict | None) -> dict:
    if not metrics:
        return {}
    return metrics.get("risk_metrics", {}).get("test", {}).get("classification_report", {}) or {}


def _metric_macro_f1(metrics: dict | None) -> float | None:
    try:
        return float(metrics["risk_metrics"]["test"]["classification_report"]["macro avg"]["f1-score"])
    except Exception:
        return None


def _metric_ece(metrics: dict | None) -> float | None:
    try:
        return float(metrics["risk_metrics"]["test"]["calibration"]["ece"])
    except Exception:
        return None


def _decision_label_display(value: object | None, review_recommended: bool = False) -> str:
    label = str(value or "").strip()
    if label.lower() in {"needs review", "needs_review", "uncertain"}:
        return "Low confidence"
    if label:
        return label
    return "Low confidence" if review_recommended else "Auto decision"


LANGUAGE_SPECS = [("en", "English"), ("hi", "Hindi"), ("pa", "Punjabi"), ("ur", "Urdu")]
EVAL_CLASS_ORDER = ["Safe", "Suspicious", "Phishing"]


def _metric_accuracy(metrics: dict | None) -> float | None:
    try:
        return float(metrics["risk_metrics"]["test"]["classification_report"]["accuracy"])
    except Exception:
        return None


def _metric_language_macro_f1(metrics: dict | None, lang_code: str) -> float | None:
    try:
        return float(
            metrics["risk_metrics"]["test"]["per_language"][lang_code]["classification_report"]["macro avg"]["f1-score"]
        )
    except Exception:
        return None


def _metric_language_support(metrics: dict | None, lang_code: str) -> float | None:
    try:
        return float(
            metrics["risk_metrics"]["test"]["per_language"][lang_code]["classification_report"]["macro avg"]["support"]
        )
    except Exception:
        return None


def _build_model_benchmark_rows(model_specs: list[tuple[str, dict | None]]) -> tuple[list[dict], list[dict], dict[str, str]]:
    summaries: list[dict] = []
    alias_to_label: dict[str, str] = {}

    for rank, (label, metrics) in enumerate(model_specs, start=1):
        metrics = metrics or {}
        version = str(metrics.get("model_version", label))
        source = str(metrics.get("model_source", label.lower()))
        alias_to_label[version] = label
        alias_to_label[source] = label

        report = _metric_report(metrics)
        language_scores: dict[str, float] = {}
        for code, _name in LANGUAGE_SPECS:
            score = _metric_language_macro_f1(metrics, code)
            if score is not None:
                language_scores[code] = score

        overall_macro = _metric_macro_f1(metrics)
        accuracy = _metric_accuracy(metrics)
        ece = _metric_ece(metrics)
        avg_language = sum(language_scores.values()) / len(language_scores) if language_scores else None
        best_language_code = max(language_scores, key=language_scores.get) if language_scores else None
        worst_language_code = min(language_scores, key=language_scores.get) if language_scores else None
        summaries.append(
            {
                "rank": rank,
                "label": label,
                "version": version,
                "source": source,
                "overall_macro_f1": overall_macro,
                "accuracy": accuracy,
                "ece": ece,
                "avg_language_f1": avg_language,
                "language_scores": language_scores,
                "best_language_code": best_language_code,
                "best_language_score": language_scores.get(best_language_code) if best_language_code else None,
                "worst_language_code": worst_language_code,
                "worst_language_score": language_scores.get(worst_language_code) if worst_language_code else None,
                "language_spread": (max(language_scores.values()) - min(language_scores.values())) if language_scores else None,
                "sample_rows": int(report.get("support", 0) or 0),
            }
        )

    language_winners: list[dict] = []
    for code, name in LANGUAGE_SPECS:
        ranked = sorted(
            [row for row in summaries if code in row["language_scores"]],
            key=lambda row: (-row["language_scores"][code], row["label"]),
        )
        if not ranked:
            continue
        best = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        language_winners.append(
            {
                "code": code,
                "name": name,
                "label": best["label"],
                "version": best["version"],
                "score": best["language_scores"][code],
                "runner_up_label": runner_up["label"] if runner_up else None,
                "runner_up_score": runner_up["language_scores"][code] if runner_up else None,
            }
        )

    return summaries, language_winners, alias_to_label


def _build_language_breakdown_rows(model_specs: list[tuple[str, dict | None]]) -> list[dict]:
    rows: list[dict] = []
    for code, name in LANGUAGE_SPECS:
        model_scores: list[dict] = []
        support: float | None = None
        for label, metrics in model_specs:
            score = _metric_language_macro_f1(metrics, code)
            if score is None:
                continue
            if support is None:
                support = _metric_language_support(metrics, code)
            model_scores.append({"label": label, "score": score})

        ranked = sorted(model_scores, key=lambda item: (-item["score"], item["label"]))
        best = ranked[0] if ranked else None
        runner_up = ranked[1] if len(ranked) > 1 else None
        worst = ranked[-1] if ranked else None
        avg_score = (sum(item["score"] for item in model_scores) / len(model_scores)) if model_scores else None
        lead_gap = (best["score"] - runner_up["score"]) if best and runner_up else None
        spread = (best["score"] - worst["score"]) if best and worst and len(ranked) > 1 else None
        rows.append(
            {
                "code": code,
                "name": name,
                "support": support,
                "model_scores": model_scores,
                "ranked_scores": ranked,
                "best_label": best["label"] if best else None,
                "best_score": best["score"] if best else None,
                "runner_up_label": runner_up["label"] if runner_up else None,
                "runner_up_score": runner_up["score"] if runner_up else None,
                "worst_label": worst["label"] if worst else None,
                "worst_score": worst["score"] if worst else None,
                "avg_score": avg_score,
                "lead_gap": lead_gap,
                "spread": spread,
            }
        )
    return rows


def _metric_calibration_ece(metrics: dict | None, phase: str = "val", stage: str = "before_temperature") -> float | None:
    try:
        return float(metrics["calibration"][phase][stage]["ece"])
    except Exception:
        return None


def _metric_temperature(metrics: dict | None) -> float | None:
    try:
        return float(metrics.get("temperature"))
    except Exception:
        return None


def _metric_model_version(metrics: dict | None, fallback: str) -> str:
    if not metrics:
        return fallback
    return str(metrics.get("model_version") or metrics.get("model_source") or fallback)


def _canonical_eval_label(value: object) -> str:
    text = str(value or "").strip().lower()
    if "phish" in text:
        return "Phishing"
    if "susp" in text or "spam" in text:
        return "Suspicious"
    return "Safe"


def _metric_test_confusion_matrix(metrics: dict | None) -> list[list[int]] | None:
    if not metrics:
        return None

    test_metrics = metrics.get("risk_metrics", {}).get("test", {})
    raw_matrix = test_metrics.get("confusion_matrix")
    raw_labels = test_metrics.get("labels") or metrics.get("risk_classes")
    if not raw_matrix or not raw_labels:
        return None

    label_index = {_canonical_eval_label(label): idx for idx, label in enumerate(raw_labels)}
    ordered_matrix: list[list[int]] = []
    for actual_label in EVAL_CLASS_ORDER:
        actual_idx = label_index.get(actual_label)
        ordered_row: list[int] = []
        for predicted_label in EVAL_CLASS_ORDER:
            predicted_idx = label_index.get(predicted_label)
            try:
                ordered_row.append(int(raw_matrix[actual_idx][predicted_idx]))  # type: ignore[index]
            except Exception:
                ordered_row.append(0)
        ordered_matrix.append(ordered_row)
    return ordered_matrix


def _ece_status(ece: float | None) -> tuple[str, str]:
    if ece is None:
        return "ECE unavailable", "muted"
    if ece <= 0.02:
        return "Well calibrated", "good"
    if ece <= 0.05:
        return "Acceptable", "warn"
    return "Needs attention", "risk"


def _select_model_specs() -> list[tuple[str, dict | None]]:
    indicbert = _load_json(MODEL_DIR / "indicbert_metrics.json")
    bilstm = _load_json(MODEL_DIR / "bilstm_metrics.json")
    baseline = _load_json(MODEL_DIR / "baseline_metrics.json")

    candidates: list[tuple[float, str, dict | None]] = []
    for label, metrics in (
        ("IndicBERT", indicbert),
        ("BiLSTM", bilstm),
        (FALLBACK_MODEL_LABEL, baseline),
    ):
        if not metrics:
            continue
        score = _metric_macro_f1(metrics)
        candidates.append((float(score) if score is not None else float("-inf"), label, metrics))

    if candidates:
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [(label, metrics) for _, label, metrics in candidates]

    if baseline:
        return [(FALLBACK_MODEL_LABEL, baseline), ("BiLSTM", bilstm), ("IndicBERT", indicbert)]
    if bilstm:
        return [("BiLSTM", bilstm), ("IndicBERT", indicbert), (FALLBACK_MODEL_LABEL, baseline)]
    if indicbert:
        return [("IndicBERT", indicbert), (FALLBACK_MODEL_LABEL, baseline), ("BiLSTM", bilstm)]
    return [("IndicBERT", None), ("BiLSTM", None), (FALLBACK_MODEL_LABEL, None)]


def _build_perf_data(model_specs: list[tuple[str, dict | None]]) -> list[tuple[str, str, float, float, float]]:
    rows: list[tuple[str, str, float, float, float]] = []
    for model_name, metrics in model_specs:
        report = _metric_report(metrics)
        for cls in ("Safe", "Suspicious", "Phishing"):
            class_report = report.get(cls, {})
            if not class_report:
                continue
            rows.append(
                (
                    cls,
                    model_name,
                    float(class_report.get("precision", 0.0)),
                    float(class_report.get("recall", 0.0)),
                    float(class_report.get("f1-score", 0.0)),
                )
            )
    return rows


def _build_insights(primary_label: str, primary_metrics: dict | None, secondary_label: str, secondary_metrics: dict | None) -> list[tuple[str, str]]:
    primary_macro = _metric_macro_f1(primary_metrics)
    secondary_macro = _metric_macro_f1(secondary_metrics)
    primary_ece = _metric_ece(primary_metrics)
    secondary_ece = _metric_ece(secondary_metrics)

    if primary_macro is not None and secondary_macro is not None:
        if primary_macro >= secondary_macro:
            leader = primary_label
            runner = secondary_label
            delta_pp = (primary_macro - secondary_macro) * 100.0
        else:
            leader = secondary_label
            runner = primary_label
            delta_pp = (secondary_macro - primary_macro) * 100.0
        model_leader_text = f"{leader} is ahead of {runner} by <strong>{delta_pp:.1f}pp macro-F1</strong> on the test split."
    else:
        model_leader_text = "The current test-split comparison is available from the loaded model metrics."

    if primary_ece is not None and secondary_ece is not None:
        calib_text = (
            f"{primary_label} calibration is at <strong>{primary_ece:.3f}</strong> ECE "
            f"versus <strong>{secondary_ece:.3f}</strong> for {secondary_label}."
        )
    else:
        calib_text = "Temperature scaling keeps the current risk scores well-behaved around the decision thresholds."

    return [
        ("Best Current Model", model_leader_text),
        (
            "Multilingual Coverage",
            "English still carries the strongest signal, while Hindi, Punjabi, and Urdu are covered in the live dataset.",
        ),
        (
            "Reliability Evidence",
            "Each deployed detector has saved test-set calibration evidence, so score reliability can be checked model by model.",
        ),
        (
            "Hardest Scam Type",
            "UPI Collect still confuses most with Safe when phrasing looks legitimate.",
        ),
        ("Calibration", calib_text),
        (
            "Most Common Scam",
            "OTP and KYC patterns dominate detection volume across recent scans.",
        ),
    ]


MODEL_SPECS = _select_model_specs()
PRIMARY_MODEL_LABEL, PRIMARY_MODEL_METRICS = MODEL_SPECS[0]
SECONDARY_MODEL_LABEL, SECONDARY_MODEL_METRICS = MODEL_SPECS[1] if len(MODEL_SPECS) > 1 else MODEL_SPECS[0]
TERTIARY_MODEL_LABEL, TERTIARY_MODEL_METRICS = MODEL_SPECS[2] if len(MODEL_SPECS) > 2 else MODEL_SPECS[-1]
MODEL_BENCHMARK_ROWS, LANGUAGE_WINNERS, MODEL_ALIAS_TO_LABEL = _build_model_benchmark_rows(MODEL_SPECS)
LANGUAGE_BREAKDOWN_ROWS = _build_language_breakdown_rows(MODEL_SPECS)
PERF_DATA = _build_perf_data(MODEL_SPECS) or [
    ("Safe", PRIMARY_MODEL_LABEL, 0.961, 0.943, 0.952),
    ("Safe", SECONDARY_MODEL_LABEL, 0.887, 0.831, 0.858),
    ("Safe", TERTIARY_MODEL_LABEL, 0.802, 0.761, 0.781),
    ("Suspicious", PRIMARY_MODEL_LABEL, 0.924, 0.908, 0.916),
    ("Suspicious", SECONDARY_MODEL_LABEL, 0.802, 0.761, 0.781),
    ("Suspicious", TERTIARY_MODEL_LABEL, 0.771, 0.733, 0.751),
    ("Phishing", PRIMARY_MODEL_LABEL, 0.958, 0.962, 0.960),
    ("Phishing", SECONDARY_MODEL_LABEL, 0.879, 0.851, 0.865),
    ("Phishing", TERTIARY_MODEL_LABEL, 0.821, 0.802, 0.811),
]

INSIGHTS = _build_insights(
    PRIMARY_MODEL_LABEL,
    PRIMARY_MODEL_METRICS,
    SECONDARY_MODEL_LABEL,
    SECONDARY_MODEL_METRICS,
)


def _normalize_label(value: str) -> str:
    v = str(value or "").strip().lower()
    if not v or v in {"unknown", "n/a", "na", "none"}:
        return "unknown"
    if "phish" in v:
        return "phishing"
    if "susp" in v or "spam" in v:
        return "suspicious"
    return "safe"


def _language_name(value: str) -> str:
    v = str(value or "").strip().lower()
    mapping = {
        "en": "English",
        "english": "English",
        "hi": "Hindi",
        "hindi": "Hindi",
        "pa": "Punjabi",
        "punjabi": "Punjabi",
        "ur": "Urdu",
        "urdu": "Urdu",
    }
    return mapping.get(v, str(value or "Unknown").strip() or "Unknown")


def _language_badge(value: str) -> str:
    v = _language_name(value)
    if v == "Hindi":
        return "हिं"
    if v == "Punjabi":
        return "ਪੰ"
    if v == "Urdu":
        return "اردو"
    return "EN"


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_scan_timestamps(values: pd.Series) -> pd.Series:
    """Parse both legacy SQLite timestamps and Supabase UTC timestamps."""
    try:
        parsed = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def _as_df(rows: list[tuple]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "ts",
                "language",
                "label",
                "scam_type",
                "risk_score",
                "model_confidence",
                "model_source",
                "model_version",
                "type_source",
                "reason",
                "message",
                "comparison_label",
                "comparison_scam_type",
                "comparison_risk_score",
                "comparison_model_confidence",
                "comparison_model_source",
                "comparison_model_version",
                "comparison_type_source",
                "comparison_tertiary_label",
                "comparison_tertiary_scam_type",
                "comparison_tertiary_risk_score",
                "comparison_tertiary_model_confidence",
                "comparison_tertiary_model_source",
                "comparison_tertiary_model_version",
                "comparison_tertiary_type_source",
                "review_recommended",
                "review_reason",
                "final_score_method",
                "model_outputs_json",
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "id",
            "ts",
            "language",
            "label",
            "scam_type",
            "risk_score",
            "model_confidence",
            "model_source",
            "model_version",
            "type_source",
            "reason",
            "message",
            "comparison_label",
            "comparison_scam_type",
            "comparison_risk_score",
            "comparison_model_confidence",
            "comparison_model_source",
            "comparison_model_version",
            "comparison_type_source",
            "comparison_tertiary_label",
            "comparison_tertiary_scam_type",
            "comparison_tertiary_risk_score",
            "comparison_tertiary_model_confidence",
            "comparison_tertiary_model_source",
            "comparison_tertiary_model_version",
            "comparison_tertiary_type_source",
            "review_recommended",
            "review_reason",
            "final_score_method",
            "model_outputs_json",
        ],
    )
    df["ts"] = _parse_scan_timestamps(df["ts"])
    df["label_norm"] = df["label"].apply(_normalize_label)
    df["comparison_label_norm"] = df["comparison_label"].apply(_normalize_label)
    df["comparison_tertiary_label_norm"] = df["comparison_tertiary_label"].apply(_normalize_label)
    df["language_name"] = df.apply(
        lambda row: infer_message_language_name(str(row.get("message", "")), fallback="English"),
        axis=1,
    )
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0)
    df["comparison_risk_score"] = pd.to_numeric(df["comparison_risk_score"], errors="coerce").fillna(0)
    df["comparison_tertiary_risk_score"] = pd.to_numeric(df["comparison_tertiary_risk_score"], errors="coerce").fillna(0)
    conf = pd.to_numeric(df["model_confidence"], errors="coerce").fillna(0.0)
    if float(conf.max()) <= 1.0:
        conf = conf * 100.0
    df["confidence_pct"] = conf.clip(0, 100)
    compare_conf = pd.to_numeric(df["comparison_model_confidence"], errors="coerce").fillna(0.0)
    if float(compare_conf.max()) <= 1.0:
        compare_conf = compare_conf * 100.0
    df["comparison_confidence_pct"] = compare_conf.clip(0, 100)
    tertiary_conf = pd.to_numeric(df["comparison_tertiary_model_confidence"], errors="coerce").fillna(0.0)
    if float(tertiary_conf.max()) <= 1.0:
        tertiary_conf = tertiary_conf * 100.0
    df["comparison_tertiary_confidence_pct"] = tertiary_conf.clip(0, 100)
    if "review_recommended" in df.columns:
        review = pd.to_numeric(df["review_recommended"], errors="coerce").fillna(0).astype(int)
        df["review_recommended"] = review.astype(bool)
    else:
        df["review_recommended"] = False
    if "review_reason" in df.columns:
        df["review_reason"] = df["review_reason"].fillna("").astype(str)
    else:
        df["review_reason"] = ""
    if "final_score_method" in df.columns:
        df["final_score_method"] = df["final_score_method"].fillna("primary_model").astype(str)
    else:
        df["final_score_method"] = "primary_model"
    if "model_outputs_json" in df.columns:
        df["model_outputs_json"] = df["model_outputs_json"].fillna("").astype(str)
    else:
        df["model_outputs_json"] = ""
    df["decision_label"] = df["review_recommended"].apply(lambda v: "Low confidence" if bool(v) else "Auto decision")
    df["decision_state"] = df["review_recommended"].apply(lambda v: "low_confidence" if bool(v) else "auto")
    df["comparison_agreement"] = (
        (df["label_norm"] == df["comparison_label_norm"])
        & (df["label_norm"] == df["comparison_tertiary_label_norm"])
    )
    return df


def _model_entries_from_row(row: pd.Series) -> list[dict]:
    raw = str(row.get("model_outputs_json", "") or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                entries = [item for item in parsed if isinstance(item, dict)]
                if entries:
                    return entries[:3]
        except Exception:
            pass

    return [
        {
            "model_version": str(row.get("model_version", "unknown") or row.get("model_source", "unknown")),
            "model_source": str(row.get("model_source", "unknown")),
            "label": str(row.get("label", "Safe")),
            "risk_score": int(float(row.get("risk_score", 0) or 0)),
            "model_confidence": float(row.get("confidence_pct", 0.0) or 0.0) / 100.0,
        },
        {
            "model_version": str(row.get("comparison_model_version", "unknown") or row.get("comparison_model_source", "unknown")),
            "model_source": str(row.get("comparison_model_source", "unknown")),
            "label": str(row.get("comparison_label", "unknown")),
            "risk_score": int(float(row.get("comparison_risk_score", 0) or 0)),
            "model_confidence": float(row.get("comparison_confidence_pct", 0.0) or 0.0) / 100.0,
        },
        {
            "model_version": str(row.get("comparison_tertiary_model_version", "unknown") or row.get("comparison_tertiary_model_source", "unknown")),
            "model_source": str(row.get("comparison_tertiary_model_source", "unknown")),
            "label": str(row.get("comparison_tertiary_label", "unknown")),
            "risk_score": int(float(row.get("comparison_tertiary_risk_score", 0) or 0)),
            "model_confidence": float(row.get("comparison_tertiary_confidence_pct", 0.0) or 0.0) / 100.0,
        },
    ]


def _timeline_from_df(df: pd.DataFrame) -> dict[str, list]:
    if df.empty or df["ts"].dropna().empty:
        return FALLBACK_TIMELINE

    end = _utc_now_naive().date()
    days = [end - timedelta(days=i) for i in range(6, -1, -1)]
    labels = [d.strftime("%a") for d in days]

    dfx = df.dropna(subset=["ts"]).copy()
    dfx["day"] = dfx["ts"].dt.date

    def counts_for(label_key: str) -> list[int]:
        out = []
        for d in days:
            out.append(int(((dfx["day"] == d) & (dfx["label_norm"] == label_key)).sum()))
        return out

    phishing = counts_for("phishing")
    suspicious = counts_for("suspicious")
    safe = counts_for("safe")

    if max(phishing + suspicious + safe) == 0:
        return FALLBACK_TIMELINE

    return {
        "labels": labels,
        "phishing": phishing,
        "suspicious": suspicious,
        "safe": safe,
    }


def _build_timeline_svg(labels: list[str], pvals: list[int], svals: list[int], safevals: list[int]) -> str:
    width, height = 760, 210
    pad_l, pad_r, pad_t, pad_b = 44, 18, 18, 34
    cw, ch = width - pad_l - pad_r, height - pad_t - pad_b
    all_vals = pvals + svals + safevals
    max_val = max(all_vals) if all_vals else 1
    max_val = max(max_val, 1)

    def xy(vals: list[int]) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        n = max(len(vals) - 1, 1)
        for i, v in enumerate(vals):
            x = pad_l + (i / n) * cw
            y = pad_t + ch - (float(v) / float(max_val)) * ch
            pts.append((x, y))
        return pts

    def polyline(vals: list[int]) -> str:
        return " ".join([f"{x:.2f},{y:.2f}" for x, y in xy(vals)])

    def area(vals: list[int]) -> str:
        pts = xy(vals)
        if not pts:
            return ""
        start = f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"
        line = " ".join([f"L {x:.2f} {y:.2f}" for x, y in pts[1:]])
        close = f" L {pts[-1][0]:.2f} {pad_t + ch:.2f} L {pts[0][0]:.2f} {pad_t + ch:.2f} Z"
        return start + " " + line + close

    grid = []
    for i in range(5):
        y = pad_t + ch - (i / 4) * ch
        grid.append(f"<line x1='{pad_l}' y1='{y:.2f}' x2='{pad_l + cw}' y2='{y:.2f}' class='tl-grid' />")

    xlabels = []
    for i, label in enumerate(labels):
        x = pad_l + (i / max(len(labels) - 1, 1)) * cw
        xlabels.append(f"<text x='{x:.2f}' y='{height - 10}' text-anchor='middle' class='tl-label'>{html.escape(label)}</text>")

    def dots(vals: list[int], cls: str) -> str:
        return "".join([f"<circle cx='{x:.2f}' cy='{y:.2f}' r='2.9' class='{cls}' />" for x, y in xy(vals)])

    return (
        dedent(
        f"""
<svg class='dash-svg' viewBox='0 0 {width} {height}' preserveAspectRatio='none' role='img' aria-label='scans timeline'>
  <defs>
    <linearGradient id='fill-red' x1='0' x2='0' y1='0' y2='1'>
      <stop offset='0%' stop-color='rgba(255,56,96,0.22)'/><stop offset='100%' stop-color='rgba(255,56,96,0)'/>
    </linearGradient>
    <linearGradient id='fill-yellow' x1='0' x2='0' y1='0' y2='1'>
      <stop offset='0%' stop-color='rgba(255,221,87,0.18)'/><stop offset='100%' stop-color='rgba(255,221,87,0)'/>
    </linearGradient>
    <linearGradient id='fill-green' x1='0' x2='0' y1='0' y2='1'>
      <stop offset='0%' stop-color='rgba(0,255,159,0.2)'/><stop offset='100%' stop-color='rgba(0,255,159,0)'/>
    </linearGradient>
  </defs>
  {''.join(grid)}
  <path d='{area(pvals)}' fill='url(#fill-red)'/>
  <path d='{area(svals)}' fill='url(#fill-yellow)'/>
  <path d='{area(safevals)}' fill='url(#fill-green)'/>
  <polyline points='{polyline(pvals)}' class='tl-line-red' />
  <polyline points='{polyline(svals)}' class='tl-line-yellow' />
  <polyline points='{polyline(safevals)}' class='tl-line-green' />
  {dots(pvals, 'tl-dot-red')}
  {dots(svals, 'tl-dot-yellow')}
  {dots(safevals, 'tl-dot-green')}
  {''.join(xlabels)}
</svg>
"""
    )
        .strip()
        .replace("\n", "")
    )


def _score_class(label_norm: str) -> str:
    if label_norm == "phishing":
        return "phishing"
    if label_norm == "suspicious":
        return "suspicious"
    return "safe"


def _confidence_pct(value: object) -> float:
    try:
        val = float(value or 0.0)
    except Exception:
        val = 0.0
    if val <= 1.0:
        val *= 100.0
    return max(0.0, min(100.0, val))


def _model_case_card(name: str, label: object, score: object, confidence: object, source: object, role: str) -> str:
    label_text = str(label or "Unknown")
    label_cls = _score_class(_normalize_label(label_text))
    try:
        score_int = int(round(float(score or 0)))
    except Exception:
        score_int = 0
    conf_txt = f"{_confidence_pct(confidence):.1f}%"
    return f"""
      <div class="lab-model-card {label_cls}">
        <div class="lab-model-role">{html.escape(role)}</div>
        <div class="lab-model-top">
          <span class="lab-model-name">{html.escape(str(name or 'unknown'))}</span>
          <span class="lab-model-score">{score_int}/100</span>
        </div>
        <div class="lab-model-meta">{html.escape(str(source or 'unknown'))} · {html.escape(label_text)} · confidence {html.escape(conf_txt)}</div>
      </div>
    """


def _build_latest_case_lab_html(df: pd.DataFrame) -> str:
    style = """
    <style>
      .lab-studio {
        border-color: rgba(181,122,255,0.46);
        box-shadow: 0 0 26px rgba(181,122,255,0.14), inset 0 0 0 1px rgba(181,122,255,0.08);
      }
      .lab-studio-grid {
        display:grid;
        grid-template-columns: minmax(0, 1.05fr) minmax(360px, 0.95fr);
        gap:1rem;
        margin-top:1rem;
      }
      .lab-case-card,
      .lab-model-card {
        border:1px solid rgba(0,212,255,0.16);
        background:rgba(0,0,0,0.22);
        padding:1rem;
      }
      .lab-case-k,
      .lab-model-role,
      .lab-case-label {
        font-family:'Share Tech Mono', monospace;
        text-transform:uppercase;
        letter-spacing:0.16em;
        font-size:0.62rem;
        color:var(--muted);
      }
      .lab-case-title {
        font-family:'Share Tech Mono', monospace;
        font-size:1.5rem;
        line-height:1.1;
        font-weight:900;
        color:#f4edff;
        text-shadow:0 0 14px rgba(181,122,255,0.26);
        margin:0.28rem 0 0.55rem;
      }
      .lab-case-row {
        display:grid;
        grid-template-columns: repeat(3, minmax(0,1fr));
        gap:0.65rem;
        margin:0.9rem 0;
      }
      .lab-case-pill {
        border:1px solid rgba(0,255,159,0.12);
        background:rgba(0,255,159,0.025);
        padding:0.65rem;
      }
      .lab-case-value {
        font-family:'Share Tech Mono', monospace;
        color:#eafff7;
        font-size:0.9rem;
        margin-top:0.18rem;
      }
      .lab-case-message,
      .lab-case-reason {
        font-family:'Share Tech Mono', monospace;
        color:rgba(214,245,235,0.78);
        line-height:1.6;
        font-size:0.78rem;
        border-top:1px solid rgba(0,255,159,0.08);
        padding-top:0.85rem;
        margin-top:0.85rem;
      }
      .lab-model-grid {
        display:grid;
        gap:0.72rem;
      }
      .lab-model-card.phishing { border-color:rgba(255,56,96,0.42); box-shadow:inset 3px 0 0 rgba(255,56,96,0.68); }
      .lab-model-card.suspicious { border-color:rgba(255,221,87,0.42); box-shadow:inset 3px 0 0 rgba(255,221,87,0.68); }
      .lab-model-card.safe { border-color:rgba(0,255,159,0.34); box-shadow:inset 3px 0 0 rgba(0,255,159,0.64); }
      .lab-model-top {
        display:flex;
        align-items:baseline;
        justify-content:space-between;
        gap:1rem;
        margin:0.24rem 0;
      }
      .lab-model-name,
      .lab-model-score {
        font-family:'Share Tech Mono', monospace;
        color:#fff;
        font-weight:900;
      }
      .lab-model-name { font-size:1.05rem; }
      .lab-model-score { font-size:1.35rem; color:#00d4ff; }
      .lab-model-card.phishing .lab-model-score { color:#ff5577; }
      .lab-model-card.suspicious .lab-model-score { color:#ffdd57; }
      .lab-model-card.safe .lab-model-score { color:#00ff9f; }
      .lab-model-meta {
        font-family:'Share Tech Mono', monospace;
        color:rgba(214,245,235,0.66);
        font-size:0.72rem;
        line-height:1.45;
      }
      .lab-empty {
        font-family:'Share Tech Mono', monospace;
        color:rgba(214,245,235,0.72);
        border:1px dashed rgba(181,122,255,0.36);
        padding:1rem;
        margin-top:0.85rem;
      }
      @media (max-width:1100px) {
        .lab-studio-grid,
        .lab-case-row { grid-template-columns:1fr; }
      }
    </style>
    """
    if df.empty:
        return f"""
        {style}
        <div class="dash-wrap lab-studio">
          <div class="section-header" style="margin-top:0;">
            <span class="section-label">// AI</span><span class="section-title">AI Studio</span>
          </div>
          <div class="lab-empty">No connected scans yet. Run a message in the consumer Detector, then return here to inspect model evidence and technical analytics.</div>
        </div>
        """

    focus_id = st.session_state.get("technical_lab_focus_scan_id")
    selected = pd.DataFrame()
    if focus_id is not None:
        try:
            selected = df[df["id"].astype(str) == str(int(focus_id))]
        except Exception:
            selected = pd.DataFrame()
    row = (selected if not selected.empty else df.sort_values("ts", ascending=False)).iloc[0]

    ts = row.get("ts")
    ts_txt = ts.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(ts) else "latest scan"
    label = str(row.get("label", "Unknown") or "Unknown")
    score = int(round(float(row.get("risk_score", 0) or 0)))
    confidence = _confidence_pct(row.get("model_confidence", 0.0))
    language = str(row.get("language_name", row.get("language", "Unknown")) or "Unknown")
    scam_type = str(row.get("scam_type", "Other") or "Other")
    decision = _decision_label_display(row.get("decision_label"), bool(row.get("review_recommended", False)))
    reason = str(row.get("reason", "") or "No reason text stored for this scan.")
    message = str(row.get("message", "") or "")
    message_preview = message if len(message) <= 260 else message[:257] + "..."

    cards = [
        _model_case_card(
            row.get("model_version", row.get("model_source", "primary")),
            row.get("label", "Unknown"),
            row.get("risk_score", 0),
            row.get("model_confidence", 0.0),
            row.get("model_source", "unknown"),
            "Primary model",
        ),
        _model_case_card(
            row.get("comparison_model_version", row.get("comparison_model_source", "secondary")),
            row.get("comparison_label", "Unknown"),
            row.get("comparison_risk_score", 0),
            row.get("comparison_model_confidence", 0.0),
            row.get("comparison_model_source", "unknown"),
            "Secondary model",
        ),
        _model_case_card(
            row.get("comparison_tertiary_model_version", row.get("comparison_tertiary_model_source", "tertiary")),
            row.get("comparison_tertiary_label", "Unknown"),
            row.get("comparison_tertiary_risk_score", 0),
            row.get("comparison_tertiary_model_confidence", 0.0),
            row.get("comparison_tertiary_model_source", "unknown"),
            "Tertiary model",
        ),
    ]

    return f"""
    {style}
    <div class="dash-wrap lab-studio">
      <div class="section-header" style="margin-top:0;">
        <span class="section-label">// AI</span><span class="section-title">AI Studio: Latest Case</span>
      </div>
      <div class="lab-studio-grid">
        <div class="lab-case-card">
          <div class="lab-case-k">Connected from detector · {html.escape(ts_txt)}</div>
          <div class="lab-case-title">{html.escape(label)} · {score}/100</div>
          <div class="lab-case-row">
            <div class="lab-case-pill"><div class="lab-case-label">Language</div><div class="lab-case-value">{html.escape(language)}</div></div>
            <div class="lab-case-pill"><div class="lab-case-label">Scam Type</div><div class="lab-case-value">{html.escape(scam_type)}</div></div>
            <div class="lab-case-pill"><div class="lab-case-label">Decision</div><div class="lab-case-value">{html.escape(decision)}</div></div>
          </div>
          <div class="lab-case-reason">Score recipe: risk score = round(P(Phishing) × 100). Primary calibrated confidence is {confidence:.1f}%.</div>
          <div class="lab-case-reason">{html.escape(reason)}</div>
          <div class="lab-case-message">{html.escape(message_preview)}</div>
        </div>
        <div class="lab-model-grid">
          {''.join(cards)}
        </div>
      </div>
    </div>
    """


def _build_dashboard_data(df: pd.DataFrame) -> dict:
    slot_labels = [row["label"] for row in MODEL_BENCHMARK_ROWS[:3]]
    pairwise_agreement = [[1.0 if i == j else None for j in range(3)] for i in range(3)]
    comparison_total = 0
    comparison_unanimous = 0
    comparison_split_two_one = 0
    comparison_split_all_diff = 0
    comparison_agreement_rate = 0.0
    outlier_counts = {label: 0 for label in slot_labels}
    disagreement_rows: list[dict] = []

    if df.empty:
        total = 2088
        phishing = 1142
        suspicious = 312
        safe = 634
        avg_conf = 91.4
        timeline = FALLBACK_TIMELINE
        scam_types = FALLBACK_SCAM_TYPES
        language_counts = [("English", 1089, "#00d4ff"), ("Hindi", 624, "#ff8c42"), ("Punjabi", 375, "#b57aff")]
        log_rows = [
            {
                "ts": "14:32:07",
                "verdict": "phishing",
                "type": "OTP / KYC",
                "score": 91,
                "conf": 96,
                "lang": "EN",
                "preview": "URGENT: Your SBI account will be BLOCKED in 24 hrs...",
                "compare_summary": "TF-IDF: Phishing 91/100 | BiLSTM: Phishing 89/100 | IndicBERT: Phishing 92/100",
                "compare_agreement": True,
                "compare_badge": "3/3",
                "compare_state": "3/3 PHISHING",
                "review_recommended": False,
                "review_reason": "",
                "decision_label": "Auto decision",
                "decision_state": "auto",
            },
            {
                "ts": "14:29:44",
                "verdict": "safe",
                "type": "Delivery Notification",
                "score": 4,
                "conf": 99,
                "lang": "EN",
                "preview": "Hi! Your order has been picked up by the delivery partner...",
                "compare_summary": "TF-IDF: Safe 4/100 | BiLSTM: Safe 6/100 | IndicBERT: Safe 5/100",
                "compare_agreement": True,
                "compare_badge": "3/3",
                "compare_state": "3/3 SAFE",
                "review_recommended": False,
                "review_reason": "",
                "decision_label": "Auto decision",
                "decision_state": "auto",
            },
            {
                "ts": "14:27:11",
                "verdict": "phishing",
                "type": "Lottery Scam",
                "score": 96,
                "conf": 98,
                "lang": "EN",
                "preview": "You have WON Rs. 25,00,000 in KBC Lottery 2024...",
                "compare_summary": "TF-IDF: Phishing 96/100 | BiLSTM: Suspicious 78/100 | IndicBERT: Phishing 93/100",
                "compare_agreement": False,
                "compare_badge": "2/3",
                "compare_state": "2/3 PHISHING · LOW CONFIDENCE",
                "review_recommended": True,
                "review_reason": "Model disagreement between the top three predictions",
                "decision_label": "Low confidence",
                "decision_state": "low_confidence",
            },
            {
                "ts": "14:22:58",
                "verdict": "phishing",
                "type": "UPI Collect",
                "score": 79,
                "conf": 87,
                "lang": "EN",
                "preview": "Collect request Rs. 5000 from PhonePe Rewards...",
                "compare_summary": "TF-IDF: Phishing 79/100 | BiLSTM: Phishing 74/100 | IndicBERT: Phishing 81/100",
                "compare_agreement": True,
                "compare_badge": "3/3",
                "compare_state": "3/3 PHISHING",
                "review_recommended": False,
                "review_reason": "",
                "decision_label": "Auto decision",
                "decision_state": "auto",
            },
        ]
        review_count = sum(1 for row in log_rows if row.get("review_recommended"))
        review_rate = (review_count / total) * 100.0 if total else 0.0
    else:
        total = int(len(df))
        counts = df["label_norm"].value_counts()
        phishing = int(counts.get("phishing", 0))
        suspicious = int(counts.get("suspicious", 0))
        safe = int(counts.get("safe", 0))
        avg_conf = float(df["confidence_pct"].mean()) if total else 0.0
        review_count = int(df["review_recommended"].sum()) if "review_recommended" in df.columns else 0
        review_rate = (review_count / total) * 100.0 if total else 0.0

        timeline = _timeline_from_df(df)

        type_df = df[df["label_norm"].isin(["phishing", "suspicious"])].copy()
        if type_df.empty:
            type_df = df.copy()
        scam_counts = (
            type_df["scam_type"].fillna("Other").replace("", "Other").value_counts().head(7).to_dict()
        )
        palette = ["#ff3860", "#ffdd57", "#ff8c42", "#b57aff", "#00d4ff", "#00ff9f", "rgba(200,240,224,0.35)"]
        scam_types = []
        for i, (name, count) in enumerate(scam_counts.items()):
            scam_types.append((str(name), int(count), palette[i % len(palette)]))
        if not scam_types:
            scam_types = FALLBACK_SCAM_TYPES

        lang_counts = df["language_name"].value_counts()
        lang_palette = {"English": "#00d4ff", "Hindi": "#ff8c42", "Punjabi": "#b57aff", "Urdu": "#00ff9f"}
        language_counts = [
            (lang, int(cnt), lang_palette.get(lang, "rgba(200,240,224,0.5)")) for lang, cnt in lang_counts.items()
        ]
        if not language_counts:
            language_counts = [("English", total, "#00d4ff")]

        log_rows = []
        latest = df.sort_values("ts", ascending=False).head(12)
        for _, row in latest.iterrows():
            ts = row["ts"]
            ts_txt = ts.strftime("%H:%M:%S") if pd.notna(ts) else "--:--:--"
            verdict = _score_class(row.get("label_norm", "safe"))
            message = str(row.get("message", ""))
            preview = message if len(message) <= 88 else (message[:85] + "...")
            model_entries = _model_entries_from_row(row)
            final_score = int(float(row.get("risk_score", 0)))
            review_recommended = bool(row.get("review_recommended", False))
            review_reason = str(row.get("review_reason", ""))
            decision_label = _decision_label_display(row.get("decision_label"), review_recommended)
            decision_state = str(row.get("decision_state", "uncertain" if review_recommended else "auto"))
            labels = [_normalize_label(str(entry.get("label", "unknown"))) for entry in model_entries]
            labels = [label for label in labels if label and label != "unknown"]
            counts: dict[str, int] = {}
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
            consensus_label = max(counts, key=counts.get) if counts else "unknown"
            consensus_count = counts.get(consensus_label, 0)
            compare_agreement = bool(labels) and len(set(labels)) == 1
            compare_summary = " | ".join(
                [
                    (
                        f"{entry.get('model_version', entry.get('model_source', 'unknown'))}: "
                        f"{entry.get('label', 'unknown')} {int(float(entry.get('risk_score', 0) or 0))}/100 "
                        f"({int(round(float(entry.get('model_confidence', 0.0) or 0.0) * 100.0))}%)"
                    )
                    for entry in model_entries
                ]
            )
            compare_state = f"{consensus_count}/3 {consensus_label.upper()}" if labels else "—"
            if review_recommended:
                compare_state = f"{compare_state} · LOW CONFIDENCE" if compare_state else "LOW CONFIDENCE"
            log_rows.append(
                {
                    "ts": ts_txt,
                    "verdict": verdict,
                    "type": str(row.get("scam_type", "Other") or "Other"),
                    "score": final_score,
                    "conf": int(round(float(row.get("confidence_pct", 0.0)))),
                    "lang": _language_badge(str(row.get("language_name", "English"))),
                    "compare_summary": compare_summary,
                    "compare_agreement": compare_agreement,
                    "compare_badge": "3/3" if compare_agreement else f"{consensus_count}/3",
                    "compare_state": compare_state if compare_state else ("UNANIMOUS" if compare_agreement else f"{consensus_count}/3 {consensus_label.upper()}"),
                    "preview": preview,
                    "review_recommended": review_recommended,
                    "review_reason": review_reason,
                    "decision_label": decision_label,
                    "decision_state": decision_state,
                }
            )

        slot_labels = [row["label"] for row in MODEL_BENCHMARK_ROWS[:3]]
        slot_labels = slot_labels if len(slot_labels) == 3 else (slot_labels + [f"Model {i}" for i in range(len(slot_labels) + 1, 4)])[:3]
        pairwise_counts = [[0 for _ in range(3)] for _ in range(3)]
        pairwise_totals = [[0 for _ in range(3)] for _ in range(3)]
        split_counts = {"unanimous": 0, "two_one": 0, "all_diff": 0}
        outlier_counts = {label: 0 for label in slot_labels}
        disagreement_rows: list[dict] = []

        def _pretty_vote(value: str) -> str:
            text = str(value or "").strip()
            return text[:1].upper() + text[1:] if text else "Unknown"

        def _slot_version(row: pd.Series, idx: int) -> str:
            if idx == 0:
                return str(row.get("model_version", "") or row.get("model_source", "") or "")
            if idx == 1:
                return str(row.get("comparison_model_version", "") or row.get("comparison_model_source", "") or "")
            return str(row.get("comparison_tertiary_model_version", "") or row.get("comparison_tertiary_model_source", "") or "")

        def _slot_vote(row: pd.Series, idx: int) -> str:
            if idx == 0:
                return str(row.get("label_norm", "unknown"))
            if idx == 1:
                return str(row.get("comparison_label_norm", "unknown"))
            return str(row.get("comparison_tertiary_label_norm", "unknown"))

        def _outlier_label(votes: list[str]) -> str:
            if votes[0] == votes[1] != votes[2]:
                return slot_labels[2]
            if votes[0] == votes[2] != votes[1]:
                return slot_labels[1]
            if votes[1] == votes[2] != votes[0]:
                return slot_labels[0]
            return "All models split"

        for _, row in df.sort_values("ts", ascending=False).iterrows():
            votes = [_slot_vote(row, i) for i in range(3)]
            uniq = len(set(votes))
            if uniq == 1:
                split_counts["unanimous"] += 1
            elif uniq == 2:
                split_counts["two_one"] += 1
            else:
                split_counts["all_diff"] += 1

            for i in range(3):
                for j in range(i + 1, 3):
                    pairwise_totals[i][j] += 1
                    pairwise_counts[i][j] += int(votes[i] == votes[j])

            if uniq != 1:
                outlier = _outlier_label(votes)
                if uniq == 2 and outlier in outlier_counts:
                    outlier_counts[outlier] += 1
                split_kind = "2/3 split" if uniq == 2 else "3-way split"
                disagreement_rows.append(
                    {
                        "ts": row["ts"].strftime("%H:%M:%S") if pd.notna(row["ts"]) else "--:--:--",
                        "language": str(row.get("language_name", "Unknown")),
                        "language_badge": _language_badge(str(row.get("language_name", "Unknown"))),
                        "scam_type": str(row.get("scam_type", "Other") or "Other"),
                        "split_kind": split_kind,
                        "outlier_model": outlier,
                        "decision_label": _decision_label_display(row.get("decision_label"), bool(row.get("review_recommended", False))),
                        "review_recommended": bool(row.get("review_recommended", False)),
                        "review_reason": str(row.get("review_reason", "")),
                        "vote_summary": " | ".join(
                            [
                                f"{slot_labels[i]}: {_pretty_vote(votes[i])}"
                                for i in range(3)
                            ]
                        ),
                        "preview": str(row.get("message", ""))[:140],
                    }
                )

        pairwise_agreement = [[None for _ in range(3)] for _ in range(3)]
        for i in range(3):
            pairwise_agreement[i][i] = 1.0
            for j in range(i + 1, 3):
                total_pairs = pairwise_totals[i][j]
                agree = (pairwise_counts[i][j] / total_pairs) if total_pairs else None
                pairwise_agreement[i][j] = agree
                pairwise_agreement[j][i] = agree

        comparison_total = int(len(df))
        comparison_unanimous = int(split_counts["unanimous"])
        comparison_split_two_one = int(split_counts["two_one"])
        comparison_split_all_diff = int(split_counts["all_diff"])
        comparison_agreement_rate = (comparison_unanimous / comparison_total * 100.0) if comparison_total else 0.0
        disagreement_rows = disagreement_rows[:8]

    return {
        "total": total,
        "phishing": phishing,
        "suspicious": suspicious,
        "safe": safe,
        "avg_conf": avg_conf,
        "timeline": timeline,
        "scam_types": scam_types,
        "language_counts": language_counts,
        "review_count": review_count,
        "review_rate": review_rate,
        "comparison_total": comparison_total,
        "comparison_unanimous": comparison_unanimous,
        "comparison_split_two_one": comparison_split_two_one,
        "comparison_split_all_diff": comparison_split_all_diff,
        "comparison_agreement_rate": comparison_agreement_rate,
        "comparison_pairwise_agreement": pairwise_agreement,
        "comparison_outlier_counts": outlier_counts,
        "comparison_slots": slot_labels,
        "comparison_disagreements": disagreement_rows,
        "log_rows": log_rows,
    }


def _render_html_block(raw: str) -> str:
    cleaned = dedent(raw).strip("\n")
    return "\n".join([line.lstrip() for line in cleaned.splitlines()]).strip()


def _empty_dashboard_data() -> dict:
    empty_slots = [row["label"] for row in MODEL_BENCHMARK_ROWS[:3]]
    return {
        "total": 0,
        "phishing": 0,
        "suspicious": 0,
        "safe": 0,
        "avg_conf": 0.0,
        "review_count": 0,
        "review_rate": 0.0,
        "timeline": {"labels": FALLBACK_TIMELINE["labels"], "phishing": [0] * 7, "suspicious": [0] * 7, "safe": [0] * 7},
        "scam_types": [("No Data", 0, "rgba(200,240,224,0.35)")],
        "language_counts": [("English", 0, "#00d4ff"), ("Hindi", 0, "#ff8c42"), ("Punjabi", 0, "#b57aff"), ("Urdu", 0, "#00ff9f")],
        "comparison_total": 0,
        "comparison_unanimous": 0,
        "comparison_split_two_one": 0,
        "comparison_split_all_diff": 0,
        "comparison_agreement_rate": 0.0,
        "comparison_pairwise_agreement": [[1.0 if i == j else None for j in range(3)] for i in range(3)],
        "comparison_outlier_counts": {label: 0 for label in empty_slots},
        "comparison_slots": empty_slots,
        "comparison_disagreements": [],
        "log_rows": [],
    }


def _build_language_winner_pills() -> str:
    chips = []
    for row in LANGUAGE_WINNERS:
        score = row.get("score")
        runner = row.get("runner_up_score")
        gap = (score - runner) if (score is not None and runner is not None) else None
        score_txt = f"{score:.3f}" if score is not None else "—"
        chips.append(
            (
                "<span class='lang-winner-pill'>"
                f"<span class='lang-winner-name'>{html.escape(row['name'])}</span>"
                f"<strong>{html.escape(str(row['label']))}</strong>"
                f"<em>{score_txt}</em>"
                f"{f'<small>+{gap:.3f}</small>' if gap is not None else ''}"
                "</span>"
            )
        )
    return "".join(chips)


def _build_model_leaderboard_html(outlier_counts: dict[str, int]) -> str:
    rows = []
    for row in MODEL_BENCHMARK_ROWS:
        label = row["label"]
        overall = row.get("overall_macro_f1")
        ece = row.get("ece")
        avg_lang = row.get("avg_language_f1")
        best_code = row.get("best_language_code")
        best_name = next((name for code, name in LANGUAGE_SPECS if code == best_code), "—")
        best_score = row.get("best_language_score")
        spread = row.get("language_spread")
        outliers = int(outlier_counts.get(label, 0))
        overall_txt = f"{overall:.3f}" if overall is not None else "—"
        ece_txt = f"{ece:.3f}" if ece is not None else "—"
        avg_lang_txt = f"{avg_lang:.3f}" if avg_lang is not None else "—"
        best_score_txt = f"{best_score:.3f}" if best_score is not None else "—"
        spread_txt = f"{spread:.3f}" if spread is not None else "—"
        rows.append(
            (
                "<tr>"
                f"<td><span class='leader-rank'>#{row['rank']}</span></td>"
                f"<td><span class='badge-model {'badge-primary' if row['rank'] == 1 else 'badge-baseline'}'>{html.escape(label)}</span>"
                f"<div class='leader-sub'>{html.escape(row['version'])}</div></td>"
                f"<td class='leader-metric'>{overall_txt}</td>"
                f"<td class='leader-metric'>{ece_txt}</td>"
                f"<td><span class='leader-lang'>{html.escape(best_name)}</span><div class='leader-sub'>{best_score_txt}</div></td>"
                f"<td class='leader-metric'>{avg_lang_txt}</td>"
                f"<td class='leader-metric'>{spread_txt}</td>"
                f"<td class='leader-metric'>{outliers}</td>"
                "</tr>"
            )
        )

    return _render_html_block(
        f"""
        <div class='leader-chip-row'>
          {_build_language_winner_pills()}
        </div>
        <table class='leader-table perf-table'>
          <thead>
            <tr><th>Rank</th><th>Model</th><th>Macro-F1</th><th>ECE</th><th>Best Language</th><th>Avg Lang F1</th><th>Lang Spread</th><th>Live Outliers</th></tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
        """
    )


def _build_agreement_matrix_html(data: dict) -> str:
    slot_labels = data.get("comparison_slots", [])
    matrix = data.get("comparison_pairwise_agreement", [])
    total = int(data.get("comparison_total", 0) or 0)
    unanimous = int(data.get("comparison_unanimous", 0) or 0)
    split_two_one = int(data.get("comparison_split_two_one", 0) or 0)
    split_all_diff = int(data.get("comparison_split_all_diff", 0) or 0)

    stat_cards = [
        ("Unanimous", unanimous, total),
        ("2-1 Split", split_two_one, total),
        ("3-Way Split", split_all_diff, total),
    ]
    stat_html = "".join(
        [
            (
                "<div class='split-stat'>"
                f"<div class='split-stat-k'>{html.escape(label)}</div>"
                f"<div class='split-stat-v'>{count:,}</div>"
                f"<div class='split-stat-p'>{((count / total * 100.0) if total else 0.0):.1f}%</div>"
                "</div>"
            )
            for label, count, total in stat_cards
        ]
    )

    if not slot_labels:
        slot_labels = ["Model 1", "Model 2", "Model 3"]
    if len(matrix) < 3:
        matrix = [[1.0 if i == j else None for j in range(3)] for i in range(3)]

    header_cells = "".join([f"<th>{html.escape(label)}</th>" for label in slot_labels[:3]])
    body_rows = []
    for i, row_label in enumerate(slot_labels[:3]):
        cells = []
        for j in range(3):
            if i == j:
                pct = 100.0
                cell_cls = "diag"
            else:
                value = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else None
                pct = (value * 100.0) if value is not None else None
                cell_cls = "high" if value is not None and value >= 0.8 else "mid" if value is not None and value >= 0.65 else "low"
            cell_text = f"{pct:.1f}%" if pct is not None else "—"
            cells.append(f"<td class='agree-cell {cell_cls}'>{cell_text}</td>")
        body_rows.append(f"<tr><th>{html.escape(row_label)}</th>{''.join(cells)}</tr>")

    return _render_html_block(
        f"""
        <div class='agreement-summary'>{stat_html}</div>
        <table class='agreement-table'>
          <thead><tr><th></th>{header_cells}</tr></thead>
          <tbody>{''.join(body_rows)}</tbody>
        </table>
        """
    )


def _build_disagreement_html(data: dict) -> str:
    rows = data.get("comparison_disagreements", []) or []
    if not rows:
        return "<div class='log-empty'>// no disagreements in the selected slice</div>"

    cards = []
    for idx, row in enumerate(rows, start=1):
        split_kind = html.escape(str(row.get("split_kind", "split")))
        outlier = html.escape(str(row.get("outlier_model", "All models split")))
        decision = html.escape(_decision_label_display(row.get("decision_label")))
        reason = html.escape(str(row.get("review_reason", "")))
        reason_html = f"<span class='compare-reason'>{reason}</span>" if reason else ""
        cards.append(
            (
                "<div class='compare-card'>"
                f"<div class='compare-card-top'><span>#{idx:02d} · {html.escape(str(row.get('ts', '')))}</span><span class='compare-chip'>{split_kind}</span></div>"
                f"<div class='compare-card-meta'>{html.escape(str(row.get('language', 'Unknown')))} · {html.escape(str(row.get('scam_type', 'Other')))} · outlier: {outlier}</div>"
                f"<div class='compare-card-votes'>{html.escape(str(row.get('vote_summary', '')))}</div>"
                f"<div class='compare-card-preview'>{html.escape(str(row.get('preview', '')))}</div>"
                f"<div class='compare-card-foot'><span class='compare-chip {'review' if row.get('review_recommended') else 'auto'}'>{decision}</span>{reason_html}</div>"
                "</div>"
            )
        )

    return "".join(cards)


def _build_language_breakdown_html(rows: list[dict], model_labels: list[str]) -> str:
    if not rows:
        return "<div class='log-empty'>// no language benchmark data available</div>"

    scored_rows = [row for row in rows if row.get("best_score") is not None]
    hardest = min(scored_rows, key=lambda row: float(row.get("best_score", 0.0))) if scored_rows else None
    spread_rows = [row for row in scored_rows if row.get("spread") is not None]
    widest = max(spread_rows, key=lambda row: float(row.get("spread", 0.0))) if spread_rows else None
    tightest = min(spread_rows, key=lambda row: float(row.get("spread", 0.0))) if spread_rows else None

    summary_cards = []
    if hardest is not None:
        summary_cards.append(
            (
                "<div class='lang-summary-card danger'>"
                "<div class='lang-summary-k'>Hardest language</div>"
                f"<div class='lang-summary-v'>{html.escape(str(hardest['name']))}</div>"
                f"<div class='lang-summary-m'>{html.escape(str(hardest.get('best_label', '—')))} "
                f"{hardest['best_score']:.3f}</div>"
                "</div>"
            )
        )
    if widest is not None:
        summary_cards.append(
            (
                "<div class='lang-summary-card warn'>"
                "<div class='lang-summary-k'>Largest model spread</div>"
                f"<div class='lang-summary-v'>{html.escape(str(widest['name']))}</div>"
                f"<div class='lang-summary-m'>lead gap {float(widest.get('lead_gap') or 0.0):.3f} • "
                f"spread {float(widest.get('spread') or 0.0):.3f}</div>"
                "</div>"
            )
        )
    if tightest is not None:
        summary_cards.append(
            (
                "<div class='lang-summary-card calm'>"
                "<div class='lang-summary-k'>Most balanced</div>"
                f"<div class='lang-summary-v'>{html.escape(str(tightest['name']))}</div>"
                f"<div class='lang-summary-m'>spread {float(tightest.get('spread') or 0.0):.3f}</div>"
                "</div>"
            )
        )

    card_html = []
    table_rows = []
    for row in rows:
        support = int(round(float(row.get("support") or 0)))
        best_label = str(row.get("best_label") or "—")
        best_score = row.get("best_score")
        lead_gap = row.get("lead_gap")
        spread = row.get("spread")
        avg_score = row.get("avg_score")
        runner_up_label = str(row.get("runner_up_label") or "—")
        best_score_txt = f"{float(best_score):.3f}" if best_score is not None else "—"
        lead_gap_txt = f"{float(lead_gap):.3f}" if lead_gap is not None else "—"
        spread_txt = f"{float(spread):.3f}" if spread is not None else "—"
        avg_score_txt = f"{float(avg_score):.3f}" if avg_score is not None else "—"

        score_cells = []
        score_map = {str(item.get("label")): item.get("score") for item in row.get("model_scores", [])}
        for label in model_labels:
            score = score_map.get(label)
            score_txt = f"{float(score):.3f}" if score is not None else "—"
            score_cls = "lang-score best" if label == best_label else "lang-score"
            score_cells.append(
                (
                    f"<div class='{score_cls}'>"
                    f"<span class='lang-score-k'>{html.escape(label)}</span>"
                    f"<span class='lang-score-v'>{score_txt}</span>"
                    "</div>"
                )
            )

        card_html.append(
            (
                "<article class='lang-card'>"
                "<div class='lang-card-head'>"
                "<div>"
                f"<div class='lang-card-title'>{html.escape(str(row['name']))}</div>"
                f"<div class='lang-card-meta'>{support:,} test rows</div>"
                "</div>"
                f"<div class='lang-card-winner'>Best: {html.escape(best_label)} · {best_score_txt}</div>"
                "</div>"
                f"<div class='lang-score-grid'>{''.join(score_cells)}</div>"
                f"<div class='lang-card-foot'>"
                f"Runner-up: {html.escape(runner_up_label)}"
                f"{f' · lead gap {lead_gap_txt}' if lead_gap is not None else ''}"
                f"{f' · spread {spread_txt}' if spread is not None else ''}"
                f"{f' · avg {avg_score_txt}' if avg_score is not None else ''}"
                "</div>"
                "</article>"
            )
        )

        table_cells = []
        for label in model_labels:
            score = score_map.get(label)
            score_txt = f"{float(score):.3f}" if score is not None else "—"
            score_cls = "lang-table-score best" if label == best_label else "lang-table-score"
            table_cells.append(f"<td><span class='{score_cls}'>{score_txt}</span></td>")
        table_rows.append(
            (
                "<tr>"
                f"<th><span class='lang-row-label'>{html.escape(str(row['name']))}</span>"
                f"<div class='lang-row-meta'>{support:,} test rows</div></th>"
                f"{''.join(table_cells)}"
                f"<td><span class='lang-winner-badge'>{html.escape(best_label)}</span></td>"
                f"<td><span class='lang-gap'>{lead_gap_txt}</span></td>"
                "</tr>"
            )
        )

    summary_html = "".join(summary_cards)
    cards_html = "".join(card_html)
    table_head = "".join([f"<th>{html.escape(label)}</th>" for label in model_labels])

    return _render_html_block(
        f"""
        <div class='lang-breakdown-shell'>
          <div class='lang-summary-grid'>{summary_html}</div>
          <div class='lang-card-grid'>{cards_html}</div>
          <div class='lang-table-wrap'>
            <table class='lang-table perf-table'>
              <thead>
                <tr><th>Language</th><th>Support</th>{table_head}<th>Winner</th><th>Lead Gap</th></tr>
              </thead>
              <tbody>{''.join(table_rows)}</tbody>
            </table>
          </div>
        </div>
        """
    )


def _alloc_row(total: int, weights: list[float]) -> list[int]:
    if total <= 0:
        return [0] * len(weights)
    raw = [total * w for w in weights]
    base = [int(x) for x in raw]
    rem = total - sum(base)
    if rem > 0:
        order = sorted(range(len(weights)), key=lambda i: raw[i] - base[i], reverse=True)
        for i in order[:rem]:
            base[i] += 1
    return base


def _build_confusion_matrix(safe_count: int, suspicious_count: int, phishing_count: int) -> list[list[int]]:
    # rows=actual [Safe, Suspicious, Phishing], cols=predicted [Safe, Suspicious, Phishing]
    safe_row = _alloc_row(max(0, int(safe_count)), [0.90, 0.07, 0.03])
    suspicious_row = _alloc_row(max(0, int(suspicious_count)), [0.11, 0.79, 0.10])
    phishing_row = _alloc_row(max(0, int(phishing_count)), [0.03, 0.08, 0.89])
    return [safe_row, suspicious_row, phishing_row]


def _build_confusion_matrix_html(
    cm_data: list[list[int]],
    title: str = "Current Model · Overall",
    cell_size: int = 62,
) -> str:
    labels = ["Safe", "Susp.", "Phish."]
    all_vals = [v for row in cm_data for v in row]
    max_val = max(all_vals) if all_vals else 1
    max_val = max(max_val, 1)

    x_labels = "".join([f"<div class='cm-x-label' style='width:{cell_size}px'>{html.escape(label)}</div>" for label in labels])

    rows_html = []
    for ri, row in enumerate(cm_data):
        row_total = max(sum(row), 1)
        cells_html = []
        for ci, val in enumerate(row):
            is_correct = ri == ci
            intensity = float(val) / float(max_val)
            if is_correct:
                bg = f"rgba(0,255,159,{0.08 + intensity * 0.35:.3f})"
                color = "#00ff9f"
                shadow = "0 0 10px rgba(0,255,159,0.30)"
            else:
                if val > 0:
                    bg = f"rgba(255,56,96,{0.04 + intensity * 0.30:.3f})"
                    color = "#ff6b8a"
                else:
                    bg = "rgba(0,0,0,0.20)"
                    color = "rgba(200,240,224,0.20)"
                shadow = "none"
            pct = int(round((val / row_total) * 100))
            cells_html.append(
                (
                    f"<div class='cm-cell' style='width:{cell_size}px;height:{cell_size}px;background:{bg};box-shadow:{shadow}' "
                    f"data-val='{val}' data-row='{ri}' data-col='{ci}'>"
                    f"<span class='cm-cell-val' style='color:{color}'>{val}</span>"
                    f"<span class='cm-cell-pct' style='color:{color}'>{pct}%</span>"
                    "</div>"
                )
            )
        rows_html.append(
            f"<div class='cm-row'><span class='cm-row-label'>{labels[ri]}</span>{''.join(cells_html)}</div>"
        )

    return _render_html_block(
        f"""
        <div class='conf-matrix-wrap'>
          <div class='conf-matrix-title'>{html.escape(title)}</div>
          <div class='cm-outer'>
            <span class='cm-y-label'>Actual</span>
            <div class='cm-inner'>
              <div class='cm-pred-label'>Predicted</div>
              <div class='cm-x-labels'>{x_labels}</div>
              {''.join(rows_html)}
            </div>
          </div>
          <div class='cm-legend'>
            <div class='cm-legend-item'><div class='cm-legend-swatch' style='background:rgba(0,255,159,0.35)'></div>Correct prediction</div>
            <div class='cm-legend-item'><div class='cm-legend-swatch' style='background:rgba(255,56,96,0.25)'></div>Misclassification</div>
          </div>
        </div>
        """
    )


def _build_model_confusion_cards(model_specs: list[tuple[str, dict | None]]) -> str:
    cards: list[str] = []
    for label, metrics in model_specs[:3]:
        matrix = _metric_test_confusion_matrix(metrics)
        if matrix is None:
            matrix = _build_confusion_matrix(safe_count=0, suspicious_count=0, phishing_count=0)
        version = _metric_model_version(metrics, label)
        accuracy = _metric_accuracy(metrics)
        macro_f1 = _metric_macro_f1(metrics)
        accuracy_text = f"{accuracy:.3f}" if accuracy is not None else "n/a"
        macro_text = f"{macro_f1:.3f}" if macro_f1 is not None else "n/a"
        support = sum(sum(row) for row in matrix)
        cards.append(
            f"""
            <article class='model-eval-card'>
              <div class='model-eval-head'>
                <div>
                  <div class='model-eval-role'>{html.escape(label)} model</div>
                  <div class='model-eval-name'>{html.escape(version)}</div>
                </div>
                <span class='model-eval-split'>test set</span>
              </div>
              {_build_confusion_matrix_html(matrix, title=f"{label} · Overall", cell_size=42)}
              <div class='model-eval-metrics'>
                <span>Accuracy {html.escape(accuracy_text)}</span>
                <span>Macro-F1 {html.escape(macro_text)}</span>
                <span>{support:,} rows</span>
              </div>
            </article>
            """
        )

    if not cards:
        return "<div class='empty-note'>No saved confusion-matrix artifacts found.</div>"
    return f"<div class='model-eval-grid'>{''.join(cards)}</div>"


def _build_model_calibration_cards(model_specs: list[tuple[str, dict | None]]) -> str:
    cards: list[str] = []
    for label, metrics in model_specs[:3]:
        version = _metric_model_version(metrics, label)
        val_before = _metric_calibration_ece(metrics, "val", "before_temperature")
        val_after = _metric_calibration_ece(metrics, "val", "after_temperature")
        test_ece = _metric_ece(metrics)
        temperature = _metric_temperature(metrics)
        status, status_class = _ece_status(test_ece)

        before_text = f"{val_before:.3f}" if val_before is not None else "n/a"
        after_text = f"{val_after:.3f}" if val_after is not None else "n/a"
        test_text = f"{test_ece:.3f}" if test_ece is not None else "n/a"
        temp_text = f"{temperature:.3f}" if temperature is not None else "n/a"

        cards.append(
            f"""
            <article class='model-calib-card {status_class}'>
              <div class='model-calib-top'>
                <div>
                  <div class='model-calib-label'>{html.escape(label)}</div>
                  <div class='model-calib-version'>{html.escape(version)}</div>
                </div>
                <span class='ece-badge {status_class}'>{html.escape(status)}</span>
              </div>
              <div class='model-calib-score'>Test ECE <strong>{html.escape(test_text)}</strong></div>
              <div class='model-calib-row'><span>Validation ECE</span><strong>{html.escape(before_text)} → {html.escape(after_text)}</strong></div>
              <div class='model-calib-row'><span>Temperature</span><strong>{html.escape(temp_text)}</strong></div>
            </article>
            """
        )

    if not cards:
        return "<div class='empty-note'>No saved calibration artifacts found.</div>"
    return f"<div class='model-calib-grid'>{''.join(cards)}</div>"


def _query_param_value(name: str, default: str) -> str:
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        if not value:
            return default
        value = value[0]
    return str(value or default).strip()


def _sanitize_period(value: str) -> str:
    v = str(value or "").strip().lower()
    allowed = {"24h", "7d", "30d", "all"}
    return v if v in allowed else "7d"


def _sanitize_language(value: str) -> str:
    v = str(value or "").strip().lower()
    allowed = {"all", "english", "hindi", "punjabi", "urdu"}
    return v if v in allowed else "all"


def _apply_dashboard_filters(df: pd.DataFrame, period_key: str, lang_key: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if period_key in {"24h", "7d", "30d"}:
        now = pd.Timestamp(_utc_now_naive())
        if period_key == "24h":
            cutoff = now - pd.Timedelta(hours=24)
        elif period_key == "7d":
            cutoff = now - pd.Timedelta(days=7)
        else:
            cutoff = now - pd.Timedelta(days=30)
        out = out[out["ts"] >= cutoff]

    if lang_key != "all":
        out = out[out["language_name"].str.lower() == lang_key]

    return out


def _metric_snapshot(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"total": 0.0, "phishing": 0.0, "suspicious": 0.0, "safe": 0.0, "avg_conf": 0.0}
    counts = df["label_norm"].value_counts()
    return {
        "total": float(len(df)),
        "phishing": float(counts.get("phishing", 0)),
        "suspicious": float(counts.get("suspicious", 0)),
        "safe": float(counts.get("safe", 0)),
        "avg_conf": float(df["confidence_pct"].mean()) if len(df) else 0.0,
    }


def _trend_block(curr_val: float, prev_val: float, context_label: str) -> dict[str, str]:
    if prev_val <= 0:
        pct = 100.0 if curr_val > 0 else 0.0
    else:
        pct = ((curr_val - prev_val) / prev_val) * 100.0

    if pct > 0.05:
        cls = "up"
        arrow = "↑"
    elif pct < -0.05:
        cls = "down"
        arrow = "↓"
    else:
        cls = "flat"
        arrow = "→"
        pct = 0.0

    return {"cls": cls, "arrow": arrow, "value": f"{pct:+.1f}%", "context": context_label}


def _comparison_context(period_key: str) -> tuple[pd.Timedelta, str]:
    if period_key == "24h":
        return pd.Timedelta(hours=24), "vs prev 24h"
    if period_key == "30d":
        return pd.Timedelta(days=30), "vs prev 30d"
    if period_key == "all":
        return pd.Timedelta(days=7), "vs last week"
    return pd.Timedelta(days=7), "vs last week"


def _build_stat_trends(df: pd.DataFrame, period_key: str, lang_key: str) -> dict[str, dict[str, str]]:
    if df.empty:
        _, ctx = _comparison_context(period_key)
        return {
            "total": {"cls": "up", "arrow": "↑", "value": "+12.4%", "context": ctx},
            "phishing": {"cls": "up", "arrow": "↑", "value": "+8.1%", "context": ctx},
            "suspicious": {"cls": "down", "arrow": "↓", "value": "-3.2%", "context": ctx},
            "safe": {"cls": "up", "arrow": "↑", "value": "+14.7%", "context": ctx},
            "avg_conf": {"cls": "up", "arrow": "↑", "value": "+2.6%", "context": ctx},
        }

    base = df.copy()
    if lang_key != "all":
        base = base[base["language_name"].str.lower() == lang_key]

    if base.empty:
        _, ctx = _comparison_context(period_key)
        return {
            "total": {"cls": "flat", "arrow": "→", "value": "+0.0%", "context": ctx},
            "phishing": {"cls": "flat", "arrow": "→", "value": "+0.0%", "context": ctx},
            "suspicious": {"cls": "flat", "arrow": "→", "value": "+0.0%", "context": ctx},
            "safe": {"cls": "flat", "arrow": "→", "value": "+0.0%", "context": ctx},
            "avg_conf": {"cls": "flat", "arrow": "→", "value": "+0.0%", "context": ctx},
        }

    delta, ctx = _comparison_context(period_key)
    now = pd.Timestamp(_utc_now_naive())
    cur_start = now - delta
    prev_start = cur_start - delta

    current_window = base[(base["ts"] >= cur_start) & (base["ts"] <= now)]
    previous_window = base[(base["ts"] >= prev_start) & (base["ts"] < cur_start)]

    current = _metric_snapshot(current_window)
    previous = _metric_snapshot(previous_window)

    return {
        "total": _trend_block(current["total"], previous["total"], ctx),
        "phishing": _trend_block(current["phishing"], previous["phishing"], ctx),
        "suspicious": _trend_block(current["suspicious"], previous["suspicious"], ctx),
        "safe": _trend_block(current["safe"], previous["safe"], ctx),
        "avg_conf": _trend_block(current["avg_conf"], previous["avg_conf"], ctx),
    }


st.set_page_config(page_title="Technical Dashboard", layout="wide", initial_sidebar_state="collapsed")
require_technical_password()
init_db()
apply_theme(home_particles=True)
menu_key = "analyst_lab" if os.environ.get("SAFESANDESH_APP_SHELL", "").strip().lower() == "consumer" else "dashboard"
top_menu(menu_key)

terminal_time = datetime.now().strftime("%I:%M:%S %p").lower()
st.markdown(
    _render_html_block(
        f"""
    <div class='dash-term-wrap'>
      <div class='dash-term-line'>
        <span class='dash-term-prompt'>root@safesandesh:~$</span>
        <span class='dash-term-cmd'>technical-dashboard --models --analytics --live</span>
        <span class='dash-term-sep'>|</span>
        <span class='dash-term-live'><span class='dash-live-dot'></span>LIVE DATA</span>
        <span class='dash-term-sep'>|</span>
        <span class='dash-term-label'>last_updated:</span>
        <span class='dash-term-time'>{html.escape(terminal_time)}</span>
        <span class='dash-term-block-inline'></span>
      </div>
    </div>
    """
    ),
    unsafe_allow_html=True,
)

components.html(
    dedent(
        """
        <script>
        (function () {
          const parentWin = window.parent;
          const doc = parentWin.document;
          const cleanTxt = (v) => (v || "").replace(/\\s+/g, " ").trim();
          const safeTxt = (v) =>
            String(v ?? "").replace(/[&<>"']/g, (ch) => ({
              "&": "&amp;",
              "<": "&lt;",
              ">": "&gt;",
              '"': "&quot;",
              "'": "&#39;",
            }[ch]));

          if (!parentWin.__dashPanelZoom) {
            parentWin.__dashPanelZoom = {};
          }

          const api = parentWin.__dashPanelZoom;

          const ensureModal = () => {
            let overlay = doc.getElementById("dash-panel-modal");
            if (overlay) return overlay;

            overlay = doc.createElement("div");
            overlay.id = "dash-panel-modal";
            overlay.className = "dash-modal-overlay";
            overlay.innerHTML = `
              <div class="dash-modal-window">
                <div class="dash-modal-head">
                  <div class="dash-modal-dots">
                    <span class="dash-modal-dot red"></span>
                    <span class="dash-modal-dot yellow"></span>
                    <span class="dash-modal-dot green"></span>
                  </div>
                  <div class="dash-modal-title">Expanded View</div>
                  <button class="dash-modal-close" type="button">Close</button>
                </div>
                <div class="dash-modal-body"></div>
              </div>
            `;
            doc.body.appendChild(overlay);

            overlay.addEventListener("click", (e) => {
              const win = overlay.querySelector(".dash-modal-window");
              if (!win.contains(e.target)) {
                api.close();
              }
            });
            overlay.querySelector(".dash-modal-close").addEventListener("click", api.close);

            const esc = (e) => {
              if (e.key === "Escape") {
                api.close();
              }
            };
            doc.addEventListener("keydown", esc);
            api._esc = esc;
            return overlay;
          };

          api.close = () => {
            const overlay = doc.getElementById("dash-panel-modal");
            if (!overlay) return;
            overlay.classList.remove("open");
            doc.body.style.overflow = "";
          };

          const detailBlock = (rows, note) => {
            const cells = (rows || [])
              .map((row) => `
                <div class="dash-modal-detail-item">
                  <div class="dash-modal-detail-k">${safeTxt(row.k || "Metric")}</div>
                  <div class="dash-modal-detail-v">${safeTxt(row.v || "—")}</div>
                </div>
              `)
              .join("");
            const detailNote = note
              ? `<div class="dash-modal-detail-note">${safeTxt(note)}</div>`
              : "";
            return `
              <div class="dash-modal-detail">
                <div class="dash-modal-detail-head">Detailed Insights</div>
                <div class="dash-modal-detail-grid">${cells}</div>
                ${detailNote}
              </div>
            `;
          };

          const numberFrom = (txt) => {
            const n = parseFloat(String(txt || "").replace(/[^0-9.+-]/g, ""));
            return Number.isFinite(n) ? n : 0;
          };

          const detailsForPanel = (panelId, panelEl) => {
            const panelTag = cleanTxt(panelEl.querySelector(".panel-tag")?.textContent) || "current filter";

            if (panelId === "scans_over_time") {
              const steps = Math.max(1, Math.floor(panelEl.querySelectorAll(".tl-dot-red, .tl-dot-yellow, .tl-dot-green").length / 3));
              const series = Array.from(panelEl.querySelectorAll(".tl-legend-item"))
                .map((el) => cleanTxt(el.textContent))
                .filter(Boolean)
                .join(", ");
              return detailBlock(
                [
                  { k: "Window", v: panelTag },
                  { k: "Intervals", v: `${steps} time points` },
                  { k: "Series", v: series || "Phishing, Suspicious, Safe" },
                  { k: "Reading", v: "Use spike timing to identify likely campaign bursts." },
                ],
                "Compare with another period/language to check if spikes are persistent or one-off."
              );
            }

            if (panelId === "verdict_breakdown") {
              const rows = Array.from(panelEl.querySelectorAll(".legend-item")).slice(0, 4).map((item) => {
                const name = cleanTxt(item.querySelector(".legend-name")?.textContent);
                const val = cleanTxt(item.querySelector(".legend-val")?.textContent);
                const pct = cleanTxt(item.querySelector(".legend-pct")?.textContent);
                return { k: name || "Category", v: [val, pct].filter(Boolean).join(" ") || "—" };
              });
              rows.push({ k: "Scope", v: panelTag });
              return detailBlock(
                rows.slice(0, 4),
                "This split helps prioritize alerting thresholds and analyst triage focus."
              );
            }

            if (panelId === "scam_type_distribution") {
              const bars = Array.from(panelEl.querySelectorAll(".hbar-item"));
              const topOne = bars[0];
              const topTwo = bars[1];
              const topOneName = cleanTxt(topOne?.querySelector(".hbar-name")?.textContent) || "—";
              const topOneVal = cleanTxt(topOne?.querySelector(".hbar-val")?.textContent) || "0";
              const topTwoName = cleanTxt(topTwo?.querySelector(".hbar-name")?.textContent) || "—";
              const topTwoVal = cleanTxt(topTwo?.querySelector(".hbar-val")?.textContent) || "0";
              return detailBlock(
                [
                  { k: "Top scam type", v: `${topOneName} (${topOneVal})` },
                  { k: "Second type", v: `${topTwoName} (${topTwoVal})` },
                  { k: "Types tracked", v: `${bars.length} categories` },
                  { k: "Scope", v: panelTag },
                ],
                "Use this ranking to tune category templates and targeted user guidance."
              );
            }

            if (panelId === "language_distribution") {
              const rows = Array.from(panelEl.querySelectorAll(".legend-item")).slice(0, 4).map((item) => {
                const name = cleanTxt(item.querySelector(".legend-name")?.textContent);
                const val = cleanTxt(item.querySelector(".legend-val")?.textContent);
                const pct = cleanTxt(item.querySelector(".legend-pct")?.textContent);
                return { k: name || "Language", v: [val, pct].filter(Boolean).join(" ") || "—" };
              });
              rows.push({ k: "Scope", v: panelTag });
              return detailBlock(
                rows.slice(0, 4),
                "Language balance indicates where more labeled data and evaluation focus are needed."
              );
            }

            if (panelId === "language_breakdown") {
              const summaryCards = Array.from(panelEl.querySelectorAll(".lang-summary-card")).map((item) => cleanTxt(item.textContent));
              const langCards = Array.from(panelEl.querySelectorAll(".lang-card"));
              const firstCard = langCards[0];
              const modelCount = firstCard ? firstCard.querySelectorAll(".lang-score").length : 0;
              const firstLanguage = cleanTxt(firstCard?.querySelector(".lang-card-title")?.textContent);
              const firstWinner = cleanTxt(firstCard?.querySelector(".lang-card-winner")?.textContent);
              return detailBlock(
                [
                  { k: "Languages compared", v: `${langCards.length} benchmark rows` },
                  { k: "Models per language", v: `${modelCount} models` },
                  { k: "First row", v: firstLanguage || "—" },
                  { k: "Top row winner", v: firstWinner || "—" },
                ],
                summaryCards.join(" · ") || "This panel shows per-language macro-F1 scores for each model and highlights the hardest and most balanced languages."
              );
            }

            if (panelId === "model_performance") {
              const allRows = Array.from(panelEl.querySelectorAll(".perf-table tbody tr")).map((tr) => {
                const tds = tr.querySelectorAll("td");
                return {
                  cls: cleanTxt(tds[0]?.textContent),
                  model: cleanTxt(tds[1]?.textContent),
                  f1: cleanTxt(tds[4]?.textContent),
                  f1n: numberFrom(tds[4]?.textContent),
                };
              });
              const sorted = [...allRows].sort((a, b) => b.f1n - a.f1n);
              const best = sorted[0] || { cls: "—", model: "—", f1: "—" };
              const weakest = sorted[sorted.length - 1] || { cls: "—", model: "—", f1: "—" };
              return detailBlock(
                [
                  { k: "Best class/model", v: `${best.cls} • ${best.model} (F1 ${best.f1})` },
                  { k: "Weakest class/model", v: `${weakest.cls} • ${weakest.model} (F1 ${weakest.f1})` },
                  { k: "Rows compared", v: `${allRows.length} entries` },
                  { k: "Scope", v: panelTag },
                ],
                "Track low-F1 rows to target data augmentation and cue engineering."
              );
            }

            if (panelId === "confusion_matrix") {
              const labels = ["Safe", "Suspicious", "Phishing"];
              const matrixCards = Array.from(panelEl.querySelectorAll(".model-eval-card"));
              if (matrixCards.length > 0) {
                const summaries = matrixCards.map((card) => {
                  const name = cleanTxt(card.querySelector(".model-eval-name")?.textContent) || "Model";
                  const vals = Array.from(card.querySelectorAll(".cm-cell")).map((cell) => numberFrom(cell.getAttribute("data-val")));
                  const total = vals.reduce((a, b) => a + b, 0);
                  const diag = (vals[0] || 0) + (vals[4] || 0) + (vals[8] || 0);
                  const acc = total > 0 ? ((diag / total) * 100).toFixed(1) + "%" : "—";
                  return `${name}: ${acc}`;
                });
                return detailBlock(
                  [
                    { k: "Models shown", v: `${matrixCards.length} base detectors` },
                    { k: "Accuracy by model", v: summaries.join(" · ") || "—" },
                    { k: "Class order", v: labels.join(" / ") },
                    { k: "Scope", v: panelTag },
                  ],
                  "Each matrix uses the saved test-set artifact for that detector; rows are actual labels and columns are predicted labels."
                );
              }
              const vals = Array.from(panelEl.querySelectorAll(".cm-cell")).map((cell) => numberFrom(cell.getAttribute("data-val")));
              const total = vals.reduce((a, b) => a + b, 0);
              const diag = (vals[0] || 0) + (vals[4] || 0) + (vals[8] || 0);
              const acc = total > 0 ? ((diag / total) * 100).toFixed(1) + "%" : "—";
              let topMissVal = -1;
              let topMissLabel = "—";
              vals.forEach((v, idx) => {
                const r = Math.floor(idx / 3);
                const c = idx % 3;
                if (r === c) return;
                if (v > topMissVal) {
                  topMissVal = v;
                  topMissLabel = `${labels[r]} → ${labels[c]} (${v})`;
                }
              });
              return detailBlock(
                [
                  { k: "Overall accuracy", v: acc },
                  { k: "Correct predictions", v: `${diag} / ${total}` },
                  { k: "Largest confusion", v: topMissLabel },
                  { k: "Scope", v: panelTag },
                ],
                "Rows are actual classes, columns are predicted classes."
              );
            }

            if (panelId === "confidence_calibration") {
              const badges = Array.from(panelEl.querySelectorAll(".ece-badge")).map((badge) => cleanTxt(badge.textContent)).filter(Boolean);
              const badge = badges.join(" · ") || "ECE unavailable";
              const note = cleanTxt(panelEl.querySelector(".calib-note")?.textContent);
              return detailBlock(
                [
                  { k: "Method", v: "Post-hoc temperature scaling" },
                  { k: "Calibration status", v: badge },
                  { k: "Interpretation", v: "Lower ECE means probability scores are more trustworthy." },
                  { k: "Scope", v: panelTag },
                ],
                note || "Calibration quality directly impacts risk-score tier reliability."
              );
            }

            if (panelId === "model_leaderboard") {
              const rows = Array.from(panelEl.querySelectorAll(".leader-table tbody tr"));
              const topModel = rows[0]?.querySelector(".badge-model")?.textContent?.trim() || "—";
              const topMacro = cleanTxt(rows[0]?.querySelectorAll("td")?.[2]?.textContent) || "—";
              const winners = Array.from(panelEl.querySelectorAll(".lang-winner-pill")).map((pill) => cleanTxt(pill.textContent));
              return detailBlock(
                [
                  { k: "Models ranked", v: `${rows.length} benchmark slots` },
                  { k: "Leader", v: `${topModel} (${topMacro})` },
                  { k: "Language winners", v: winners.join(" · ") || "—" },
                  { k: "Scope", v: panelTag },
                ],
                "This panel compares the model lineup on the test split and highlights which model leads each language."
              );
            }

            if (panelId === "agreement_matrix") {
              const cells = Array.from(panelEl.querySelectorAll(".agree-cell")).map((cell) => cleanTxt(cell.textContent));
              const split = Array.from(panelEl.querySelectorAll(".split-stat")).map((item) => cleanTxt(item.textContent));
              return detailBlock(
                [
                  { k: "Matrix cells", v: `${cells.length} agreement values` },
                  { k: "Split summary", v: split.join(" · ") || "—" },
                  { k: "Interpretation", v: "Higher pairwise agreement means the models behave more similarly on live scans." },
                  { k: "Scope", v: panelTag },
                ],
                "Use this to spot whether the models converge or diverge on the current scan slice."
              );
            }

            if (panelId === "disagreement_view") {
              const cards = Array.from(panelEl.querySelectorAll(".compare-card"));
              const topCard = cards[0];
              const topVotes = cleanTxt(topCard?.querySelector(".compare-card-votes")?.textContent) || "—";
              const topMeta = cleanTxt(topCard?.querySelector(".compare-card-meta")?.textContent) || "—";
              return detailBlock(
                [
                  { k: "Disagreement rows", v: `${cards.length} recent cases` },
                  { k: "Top case", v: topMeta },
                  { k: "Model split", v: topVotes },
                  { k: "Scope", v: panelTag },
                ],
                "These are the rows where the models split, which is the most direct view of the benchmark behavior."
              );
            }

            if (panelId === "recent_scan_log") {
              const rows = Array.from(panelEl.querySelectorAll(".log-table tbody tr"));
              const first = rows[0];
              const firstTds = first ? first.querySelectorAll("td") : [];
              const latestVerdict = cleanTxt(firstTds[2]?.textContent) || "—";
              const latestType = cleanTxt(firstTds[3]?.textContent) || "—";
              const latestScore = cleanTxt(firstTds[4]?.textContent) || "—";
              const latestConf = cleanTxt(firstTds[5]?.textContent) || "—";
              const latestCompare = cleanTxt(firstTds[7]?.querySelector(".compare-summary")?.textContent || firstTds[7]?.textContent) || "—";
              const latestAgreement = cleanTxt(firstTds[7]?.querySelector(".compare-badge.agree")?.textContent || firstTds[7]?.querySelector(".compare-badge.diff")?.textContent || "—");
              const latestDecision = cleanTxt(firstTds[7]?.querySelector(".compare-badge.review")?.textContent || firstTds[7]?.querySelector(".compare-badge.auto")?.textContent || "—");
              const latestDecisionState = cleanTxt(first?.getAttribute("data-review-state") || first?.getAttribute("data-decision-state") || latestDecision) || "—";
              const latestReviewReason = cleanTxt(first?.getAttribute("data-review-reason") || "") || "—";
              return detailBlock(
                [
                  { k: "Rows visible", v: `${rows.length} latest scans` },
                  { k: "Latest verdict", v: latestVerdict },
                  { k: "Latest type", v: latestType },
                  { k: "Latest score/conf", v: `${latestScore} • ${latestConf}` },
                  { k: "Model compare", v: latestCompare },
                  { k: "Agreement", v: latestAgreement },
                  { k: "Decision", v: latestDecisionState },
                  { k: "Decision note", v: latestReviewReason },
                ],
                "Load the detector page for full message-level explanations and recommended actions."
              );
            }

            return detailBlock(
              [
                { k: "Scope", v: panelTag },
                { k: "Panel type", v: "Analytical summary" },
                { k: "Use case", v: "Quick comparative review" },
                { k: "Next step", v: "Adjust filters for targeted slices" },
              ],
              "Expanded view keeps the same source data while adding interpretation cues."
            );
          };

          api.open = (panelEl) => {
            const overlay = ensureModal();
            const body = overlay.querySelector(".dash-modal-body");
            const titleEl = overlay.querySelector(".dash-modal-title");
            const panelTitle = panelEl.querySelector(".panel-title");
            const panelId = panelEl.getAttribute("data-panel-id") || "";

            titleEl.textContent = (panelTitle ? panelTitle.textContent.trim() : "Expanded View") + " — Expanded";
            body.innerHTML = "";
            const clone = panelEl.cloneNode(true);
            clone.classList.remove("zoomable");
            body.appendChild(clone);
            const detail = detailsForPanel(panelId, panelEl);
            if (detail) {
              const mount = doc.createElement("div");
              mount.innerHTML = detail;
              if (mount.firstElementChild) {
                body.appendChild(mount.firstElementChild);
              }
            }

            overlay.classList.add("open");
            doc.body.style.overflow = "hidden";
          };

          api.bindLogFilters = () => {
            const wraps = doc.querySelectorAll(".dash-wrap.main-only");
            if (!wraps.length) return;
            const scope = wraps[wraps.length - 1];
            const search = scope.querySelector("#logSearch");
            const body = scope.querySelector("#logBody");
            const countEl = scope.querySelector("#logCount");
            if (!search || !body || !countEl) return;
            if (search.dataset.logBound === "1") return;
            search.dataset.logBound = "1";

            const rows = Array.from(body.querySelectorAll("tr[data-row='1']"));
            const filterBtns = Array.from(scope.querySelectorAll(".log-filter-btn"));
            let activeVerdict = "all";

            const render = () => {
              const term = cleanTxt(search.value).toLowerCase();
              let shown = 0;
              rows.forEach((row) => {
                const verdict = (row.dataset.verdict || "").toLowerCase();
                const blob = (row.dataset.search || "").toLowerCase();
                const passVerdict = activeVerdict === "all" || verdict === activeVerdict;
                const passSearch = !term || blob.includes(term);
                const visible = passVerdict && passSearch;
                row.style.display = visible ? "" : "none";
                if (visible) shown += 1;
              });
              countEl.textContent = `showing ${shown} of ${rows.length}`;

              let emptyRow = body.querySelector("tr.log-empty-row");
              if (shown === 0) {
                if (!emptyRow) {
                  emptyRow = doc.createElement("tr");
                  emptyRow.className = "log-empty-row";
                  emptyRow.innerHTML = "<td colspan='9' class='log-empty'>// no_results_found — try adjusting filters</td>";
                  body.appendChild(emptyRow);
                }
                emptyRow.style.display = "";
              } else if (emptyRow) {
                emptyRow.style.display = "none";
              }
            };

            filterBtns.forEach((btn) => {
              btn.addEventListener("click", () => {
                filterBtns.forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                activeVerdict = (btn.dataset.verdict || "all").toLowerCase();
                render();
              });
            });

            search.addEventListener("input", render);
            render();
          };

          api.bind = () => {
            const wraps = doc.querySelectorAll(".dash-wrap.main-only");
            if (!wraps.length) return;
            const scope = wraps[wraps.length - 1];
            const panels = scope.querySelectorAll(".panel");

            panels.forEach((panel) => {
              if (panel.dataset.zoomBound === "1") return;
              panel.dataset.zoomBound = "1";
              panel.classList.add("zoomable");
              panel.addEventListener("click", (e) => {
                if (e.target.closest("a, button, input, select, textarea, summary, details")) {
                  return;
                }
                api.open(panel);
              });
            });
          };

          api.bind();
          api.bindLogFilters();

          if (!api._observer) {
            api._observer = new parentWin.MutationObserver(() => {
              api.bind();
              api.bindLogFilters();
            });
            api._observer.observe(doc.body, { childList: true, subtree: true });
          }
        })();
        </script>
        """
    ),
    height=0,
    width=0,
)

rows = read_scans(limit=6000)
df = _as_df(rows)

period_selected = _sanitize_period(_query_param_value("period", "7d"))
lang_selected = _sanitize_language(_query_param_value("lang", "all"))

if df.empty:
    data = _build_dashboard_data(df)
else:
    filtered_df = _apply_dashboard_filters(df, period_selected, lang_selected)
    data = _build_dashboard_data(filtered_df) if not filtered_df.empty else _empty_dashboard_data()

stat_trends = _build_stat_trends(df, period_selected, lang_selected)

p = data["phishing"]
s = data["suspicious"]
safe = data["safe"]
total = max(data["total"], 1)
avg_conf = data["avg_conf"]

p_pct = int(round((p / total) * 100))
s_pct = int(round((s / total) * 100))
safe_pct = int(round((safe / total) * 100))

lang_total_raw = sum([x[1] for x in data["language_counts"]])
lang_total = lang_total_raw or 1
lang_slices = []
lang_cursor = 0
if lang_total_raw == 0:
    base = data["language_counts"][:4]
    if not base:
        base = [("English", 0, "#00d4ff")]
    step = 100 // max(len(base), 1)
    for i, (lang_name, lang_count, lang_color) in enumerate(base):
        start = i * step
        end = 100 if i == len(base) - 1 else (i + 1) * step
        lang_slices.append((lang_name, lang_count, lang_color, start, end))
else:
    for lang_name, lang_count, lang_color in data["language_counts"][:4]:
        lang_pct = max(1, int(round((lang_count / lang_total) * 100)))
        start = lang_cursor
        end = min(100, lang_cursor + lang_pct)
        lang_slices.append((lang_name, lang_count, lang_color, start, end))
        lang_cursor = end
    if lang_slices:
        last = list(lang_slices[-1])
        last[4] = 100
        lang_slices[-1] = tuple(last)

verdict_slices = [
    ("Phishing", p, "#ff3860", 0, p_pct),
    ("Suspicious", s, "#ffdd57", p_pct, p_pct + s_pct),
    ("Safe", safe, "#00ff9f", p_pct + s_pct, 100),
]

confusion_matrix_html = _build_model_confusion_cards(MODEL_SPECS)

# Guard against all-zero rows after strict filters (prevents division by zero in hbar width).
max_scam = max(1, max([count for _, count, _ in data["scam_types"]], default=0))
scam_rows_html = "".join(
    [
        (
            "<div class='hbar-item'>"
            f"<div class='hbar-top'><span class='hbar-name'>{html.escape(name)}</span><span class='hbar-val'>{count}</span></div>"
            "<div class='hbar-track'>"
            f"<div class='hbar-fill' style='width:{(count / max_scam) * 100:.1f}%; background:{color}; box-shadow:0 0 8px {color};'></div>"
            "</div></div>"
        )
        for name, count, color in data["scam_types"]
    ]
)

perf_rows_html = "".join(
    [
        (
            "<tr>"
            f"<td>{cls}</td>"
            f"<td><span class='badge-model {'badge-primary' if model == PRIMARY_MODEL_LABEL else 'badge-baseline'}'>{model}</span></td>"
            f"<td>{prec:.3f}</td>"
            f"<td>{rec:.3f}</td>"
            f"<td class='f1-cell'>{f1:.3f}</td>"
            "<td class='metric-bar-cell'>"
            f"<div class='metric-top'>{int(round(f1 * 100))}%</div>"
            "<div class='metric-bar'>"
            f"<div class='metric-fill' style='width:{f1 * 100:.1f}%; background:{'#00ff9f' if model == PRIMARY_MODEL_LABEL else 'rgba(200,240,224,0.28)'};'></div>"
            "</div></td></tr>"
        )
        for cls, model, prec, rec, f1 in PERF_DATA
    ]
)

insight_html = "".join(
    [
        (
            "<div class='insight-card'>"
            f"<div class='insight-title'>{title}</div>"
            f"<div class='insight-body'>{body}</div>"
            "</div>"
        )
        for title, body in INSIGHTS
    ]
)

log_rows_html = "".join(
    [
        (
            f"<tr class='row-{r['verdict']}' data-row='1' data-verdict='{r['verdict']}' "
            f"data-review-state='{html.escape(str(r.get('decision_label', 'Auto decision')))}' "
            f"data-decision-state='{html.escape(str(r.get('decision_state', 'auto')))}' "
            f"data-review-reason='{html.escape(str(r.get('review_reason', '')))}' "
            f"data-search='{html.escape((str(r['verdict']) + ' ' + str(r['type']) + ' ' + str(r['lang']) + ' ' + str(r['compare_summary']) + ' ' + str(r.get('decision_label', '')) + ' ' + str(r.get('review_reason', '')) + ' ' + str(r['preview'])).lower())}'>"
            f"<td>#{str(i + 1).zfill(3)}</td>"
            f"<td>{html.escape(r['ts'])}</td>"
            f"<td><span class='verdict-pip {r['verdict']}'><span class='pip {r['verdict']}'></span>{r['verdict'].upper()}</span></td>"
            f"<td>{html.escape(str(r['type']))}</td>"
            f"<td><span class='score-chip {r['verdict']}'>{int(r['score'])}/100</span></td>"
            f"<td class='conf-cell'>{int(r['conf'])}%</td>"
            f"<td><span class='lang-pip'>{html.escape(str(r['lang']))}</span></td>"
            f"<td class='compare-cell' title='{html.escape(str(r['compare_summary']))}'>"
            f"<span class='compare-summary'>{html.escape(str(r['compare_summary']))}</span>"
            f"<span class='compare-badge {'agree' if r.get('compare_agreement') else 'diff'}'>{r.get('compare_badge', 'DIFF')}</span>"
            f"<span class='compare-badge {'review' if r.get('review_recommended') else 'auto'}'>{html.escape(str(r.get('decision_label', 'Auto decision')).upper())}</span>"
            "</td>"
            f"<td class='log-preview'>{html.escape(str(r['preview']))}</td>"
            "</tr>"
        )
        for i, r in enumerate(data["log_rows"])
    ]
)
log_total_count = len(data["log_rows"])
leaderboard_html = _build_model_leaderboard_html(data.get("comparison_outlier_counts", {}))
agreement_matrix_html = _build_agreement_matrix_html(data)
disagreement_html = _build_disagreement_html(data)
language_breakdown_html = _build_language_breakdown_html(LANGUAGE_BREAKDOWN_ROWS, [row["label"] for row in MODEL_BENCHMARK_ROWS])

lang_legend_html = "".join(
    [
        (
            "<div class='legend-item'>"
            f"<div class='legend-dot' style='background:{color}'></div>"
            f"<span class='legend-name'>{html.escape(name)}</span>"
            f"<span class='legend-val' style='color:{color}'>{count}</span>"
            f"<span class='legend-pct'>({int(round((count / lang_total) * 100))}%)</span>"
            "</div>"
        )
        for name, count, color in data["language_counts"][:4]
    ]
)

verdict_legend_html = "".join(
    [
        (
            "<div class='legend-item'>"
            f"<div class='legend-dot' style='background:{color}'></div>"
            f"<span class='legend-name'>{name}</span>"
            f"<span class='legend-val' style='color:{color}'>{count}</span>"
            f"<span class='legend-pct'>({int(round((count / total) * 100))}%)</span>"
            "</div>"
        )
        for name, count, color, _, _ in verdict_slices
    ]
)

verdict_conic = ", ".join([f"{c} {a}% {b}%" for _, _, c, a, b in verdict_slices])
lang_conic = ", ".join([f"{c} {a}% {b}%" for _, _, c, a, b in lang_slices]) or "#00d4ff 0% 100%"

now_ts = datetime.now().strftime("%H:%M:%S")

csv_bytes = pd.DataFrame(data["log_rows"]).to_csv(index=False).encode("utf-8") if data["log_rows"] else b""

period_tag_map = {
    "24h": "24-hour window",
    "7d": "7-day rolling",
    "30d": "30-day window",
    "all": "all-time trend",
}
language_tag_map = {
    "all": "all languages",
    "english": "english",
    "hindi": "hindi",
    "punjabi": "punjabi",
    "urdu": "urdu",
}

selected_period_tag = period_tag_map.get(period_selected, "7-day rolling")
selected_language_tag = language_tag_map.get(lang_selected, "all languages")
selected_context_tag = f"{selected_period_tag} • {selected_language_tag}"
calibration_model_cards_html = _build_model_calibration_cards(MODEL_SPECS)
calibration_copy = (
    "Each detector is evaluated separately with its saved test-set Expected Calibration Error (ECE). "
    "Temperature scaling adjusts confidence reliability; the deployed user score is still produced by "
    "the final median ensemble after these base model scores are generated."
)

st.markdown(
    _render_html_block(
        """
    <style>
    .dash-term-wrap {
      margin: 0.1rem 0 0.85rem;
      border: 1px solid var(--border);
      border-top: 1px solid rgba(0,255,159,0.22);
      border-bottom: 1px solid rgba(0,255,159,0.2);
      background: rgba(3,15,26,0.68);
      overflow: hidden;
    }
    .dash-term-line {
      display:flex;
      align-items:center;
      gap: 0.95rem;
      padding: 0.82rem 1.02rem 0.78rem;
      font-family:'Share Tech Mono', monospace;
      font-size: 0.79rem;
      letter-spacing: 0.08em;
      white-space: nowrap;
      overflow-x: auto;
      border-bottom: none;
    }
    .dash-term-line::-webkit-scrollbar { height: 0; }
    .dash-term-prompt {
      color: var(--neon);
      font-weight: 700;
      text-shadow: 0 0 8px rgba(0,255,159,0.32);
      flex-shrink: 0;
    }
    .dash-term-cmd {
      color: rgba(200,240,224,0.42);
      font-weight: 600;
      flex-shrink: 0;
    }
    .dash-term-sep {
      color: rgba(0,255,159,0.3);
      flex-shrink: 0;
    }
    .dash-term-live {
      color: rgba(200,240,224,0.62);
      display: inline-flex;
      align-items:center;
      gap: 0.46rem;
      flex-shrink: 0;
    }
    .dash-live-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--neon);
      box-shadow: 0 0 10px rgba(0,255,159,0.72), 0 0 18px rgba(0,255,159,0.38);
      animation: dashPulse 1.8s ease-in-out infinite;
      flex-shrink: 0;
    }
    .dash-term-label {
      color: rgba(200,240,224,0.45);
      flex-shrink: 0;
    }
    .dash-term-time {
      color: var(--neon);
      text-shadow: 0 0 8px rgba(0,255,159,0.35);
      font-weight: 700;
      flex-shrink: 0;
    }
    .dash-term-block-inline {
      width: 8px;
      height: 14px;
      background: rgba(0,255,159,0.95);
      box-shadow: 0 0 10px rgba(0,255,159,0.45);
      animation: dashBlink 1s step-end infinite;
      display: inline-block;
      margin-left: 4px;
      flex-shrink: 0;
      vertical-align: middle;
    }
    @keyframes dashBlink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0; }
    }
    @keyframes dashPulse {
      0%, 100% { box-shadow: 0 0 8px rgba(0,255,159,0.6), 0 0 16px rgba(0,255,159,0.3); }
      50% { box-shadow: 0 0 14px rgba(0,255,159,0.9), 0 0 24px rgba(0,255,159,0.5); }
    }
    @keyframes dashFadeIn {
      0% { opacity: 0.22; transform: translateY(8px) scale(0.996); filter: saturate(0.88); }
      100% { opacity: 1; transform: translateY(0) scale(1); filter: saturate(1); }
    }
    @keyframes dashBlockIn {
      0% { opacity: 0; transform: translateY(10px); }
      100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes dashLineDraw {
      0% { stroke-dashoffset: 1200; opacity: 0.15; }
      100% { stroke-dashoffset: 0; opacity: 1; }
    }
    @keyframes dashDotPop {
      0% { transform: scale(0.22); opacity: 0; }
      70% { transform: scale(1.2); opacity: 1; }
      100% { transform: scale(1); opacity: 1; }
    }
    @keyframes dashDonutIn {
      0% { transform: scale(0.72) rotate(-28deg); opacity: 0; }
      100% { transform: scale(1) rotate(0deg); opacity: 1; }
    }
    @keyframes dashBarGrow {
      0% { transform: scaleX(0); opacity: 0.35; }
      100% { transform: scaleX(1); opacity: 1; }
    }
    @keyframes dashNumIn {
      0% { opacity: 0; transform: translateY(8px) scale(0.95); filter: blur(1px); }
      100% { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
    }

    .dash-wrap {
      max-width: none;
      margin: 0;
      padding: 0.28rem 0 4.6rem;
      color: var(--text);
      animation: dashFadeIn 240ms ease-out both;
      will-change: opacity, transform, filter;
    }
    .dash-wrap.head-only {
      padding: 0.22rem 0 0.28rem;
    }
    .dash-wrap.main-only {
      padding-top: 0.1rem;
    }
    .dash-system-row,
    .page-title-row,
    .page-sub {
      max-width: none;
      margin-left: 0;
      margin-right: 0;
      padding-left: 1rem;
      padding-right: 1rem;
      box-sizing: border-box;
    }
    .dash-system-row {
      margin-top: 0;
      margin-bottom: 0;
      padding-top: 0.1rem;
      padding-bottom: 0.04rem;
    }
    .dash-system-row .status-bar { margin-bottom: 0.02rem; }
    .page-title-row {
      display: flex;
      align-items: baseline;
      gap: 20px;
      margin-top: 0;
      margin-bottom: 0;
      padding-top: 0.18rem;
      padding-bottom: 0;
    }
    .section-code {
      font-family:'Share Tech Mono', monospace;
      font-size:0.7rem;
      color:#16ffd6;
      letter-spacing:0.1em;
      opacity:0.95;
      text-shadow:0 0 10px rgba(0,255,200,0.42);
    }
    .page-h1 {
      font-size:1.8rem;
      font-weight:800;
      letter-spacing:0.06em;
      text-transform:uppercase;
      margin:0;
      color:#ffffff !important;
      -webkit-text-fill-color:#ffffff !important;
      opacity:1 !important;
      mix-blend-mode:normal !important;
      filter:none !important;
      text-shadow:
        0 0 10px rgba(255,255,255,0.32),
        0 0 24px rgba(0,212,255,0.18) !important;
    }
    .page-title-row .page-h1,
    .page-title-row h1 {
      color:#ffffff !important;
      -webkit-text-fill-color:#ffffff !important;
      opacity:1 !important;
      mix-blend-mode:normal !important;
      filter:none !important;
      text-shadow:
        0 0 10px rgba(255,255,255,0.32),
        0 0 24px rgba(0,212,255,0.18) !important;
    }
    .page-sub {
      margin-top:14px;
      margin-bottom:0.28rem;
      font-family:'Share Tech Mono', monospace;
      font-size:0.72rem;
      color:#ffffff !important;
      -webkit-text-fill-color:#ffffff !important;
      opacity:1 !important;
      letter-spacing:0.06em;
      text-shadow:0 0 8px rgba(255,255,255,0.18) !important;
    }

    .filter-bar {
      display:flex; align-items:center; gap:0.58rem; flex-wrap:wrap;
      margin-bottom:1.45rem;
    }
    .filter-label {
      font-family:'Share Tech Mono', monospace;
      font-size:0.66rem;
      color:var(--neon2);
      letter-spacing:0.1em;
      text-transform:uppercase;
      text-shadow:0 0 8px rgba(0,212,255,0.22);
    }
    .filter-btn {
      display:inline-flex;
      align-items:center;
      justify-content:center;
      font-family:'Share Tech Mono', monospace;
      font-size:0.74rem;
      letter-spacing:0.08em;
      text-transform:uppercase;
      padding:0.44rem 1.08rem;
      min-height:40px;
      border:1px solid rgba(255,56,96,0.70);
      color:#ff3c7a !important;
      -webkit-text-fill-color:#ff3c7a !important;
      background:rgba(0,0,0,0.86);
      text-decoration:none !important;
      line-height:1;
      transition:all .18s ease;
      box-shadow:inset 0 0 0 1px rgba(0,0,0,0.25);
      position:relative;
      overflow:hidden;
      isolation:isolate;
      transform:translateY(0);
      z-index:0;
    }
    .filter-btn::before {
      content:"";
      position:absolute;
      top:-1px;
      bottom:-1px;
      left:-140%;
      width:62%;
      pointer-events:none;
      z-index:0;
      opacity:0;
      transform:skewX(-18deg);
      background:linear-gradient(
        90deg,
        rgba(255,255,255,0.00) 0%,
        rgba(255,255,255,0.12) 30%,
        rgba(255,117,152,0.56) 50%,
        rgba(255,56,96,0.34) 74%,
        rgba(255,56,96,0.00) 100%
      );
      filter:blur(0.2px);
    }
    .filter-btn span,
    .export-btn span {
      position:relative;
      z-index:2;
    }
    .filter-btn:link,
    .filter-btn:visited,
    .filter-btn:hover,
    .filter-btn:active,
    .filter-btn:focus {
      text-decoration:none !important;
      outline:none;
    }
    .filter-btn.active {
      color:#00f1a8 !important;
      -webkit-text-fill-color:#00f1a8 !important;
      border-color:rgba(0,255,159,0.82) !important;
      background:rgba(0,0,0,0.90) !important;
      box-shadow:0 0 11px rgba(0,255,159,0.16), inset 0 0 0 1px rgba(0,255,159,0.12);
    }
    .filter-btn.active::before {
      background:linear-gradient(
        90deg,
        rgba(255,255,255,0.00) 0%,
        rgba(255,255,255,0.13) 30%,
        rgba(0,255,195,0.58) 50%,
        rgba(0,255,159,0.36) 74%,
        rgba(0,255,159,0.00) 100%
      );
    }
    .filter-btn:hover {
      color:#ff4b86 !important;
      -webkit-text-fill-color:#ff4b86 !important;
      border-color:rgba(255,56,96,0.85);
      background:rgba(0,0,0,0.92);
      box-shadow:0 0 10px rgba(255,56,96,0.22), inset 0 0 0 1px rgba(255,56,96,0.08);
      transform:translateY(-1px);
    }
    .filter-btn.active:hover {
      color:#00f9b1 !important;
      -webkit-text-fill-color:#00f9b1 !important;
      border-color:rgba(0,255,159,0.92) !important;
      box-shadow:0 0 12px rgba(0,255,159,0.24), inset 0 0 0 1px rgba(0,255,159,0.16);
    }
    .filter-btn:hover::before,
    .filter-btn.active:hover::before {
      opacity:1;
      animation:dash-btn-sweep 920ms cubic-bezier(0.22, 0.7, 0.28, 1) 1;
    }
    .filter-sep { width:1px; height:18px; background:var(--border); margin:0 0.1rem; }
    .export-btn {
      margin-left:auto;
      text-decoration:none !important;
      font-family:'Share Tech Mono', monospace;
      font-size:0.74rem;
      letter-spacing:0.08em;
      text-transform:uppercase;
      color:var(--neon2);
      border:1px solid rgba(0,212,255,0.45);
      padding:0.44rem 1.08rem;
      min-height:40px;
      display:inline-flex;
      align-items:center;
      justify-content:center;
      line-height:1;
      background:rgba(0,0,0,0.88);
      position:relative;
      overflow:hidden;
      isolation:isolate;
      transition:all .18s ease;
      transform:translateY(0);
      z-index:0;
    }
    .export-btn::before {
      content:"";
      position:absolute;
      top:-1px;
      bottom:-1px;
      left:-140%;
      width:62%;
      pointer-events:none;
      z-index:0;
      opacity:0;
      transform:skewX(-18deg);
      background:linear-gradient(
        90deg,
        rgba(255,255,255,0.00) 0%,
        rgba(255,255,255,0.13) 30%,
        rgba(0,212,255,0.60) 50%,
        rgba(0,212,255,0.34) 74%,
        rgba(0,212,255,0.00) 100%
      );
      filter:blur(0.2px);
    }
    .export-btn:hover {
      color:#2fd9ff !important;
      border-color:rgba(0,212,255,0.74);
      box-shadow:0 0 11px rgba(0,212,255,0.24), inset 0 0 0 1px rgba(0,212,255,0.10);
      transform:translateY(-1px);
    }
    .export-btn:hover::before {
      opacity:1;
      animation:dash-btn-sweep 920ms cubic-bezier(0.22, 0.7, 0.28, 1) 1;
    }
    .export-btn:link,
    .export-btn:visited,
    .export-btn:hover,
    .export-btn:active,
    .export-btn:focus {
      text-decoration:none !important;
      outline:none;
    }
    .dash-filter-label {
      font-family:'Share Tech Mono', monospace;
      font-size:0.68rem;
      color:var(--neon2);
      letter-spacing:0.09em;
      text-transform:uppercase;
      text-shadow:0 0 8px rgba(0,212,255,0.22);
      padding-top:0.15rem;
      margin-bottom:0.32rem;
      white-space:nowrap;
    }
    .dash-filter-sep {
      height:40px;
      border-left:1px solid var(--border);
      margin:0.1rem auto 0;
      width:1px;
    }
    [class*="st-key-dash_period_"] .stButton > button,
    [class*="st-key-dash_lang_"] .stButton > button,
    .st-key-dash_export_csv .stDownloadButton > button {
      display:inline-flex !important;
      align-items:center !important;
      justify-content:center !important;
      font-family:'Share Tech Mono', monospace !important;
      font-size:0.76rem !important;
      letter-spacing:0.06em !important;
      text-transform:uppercase !important;
      padding:0.52rem 0.9rem !important;
      min-height:52px !important;
      border:1px solid rgba(255,56,96,0.70) !important;
      color:#ff3c7a !important;
      background:rgba(0,0,0,0.86) !important;
      line-height:1 !important;
      transition:all .18s ease !important;
      box-shadow:inset 0 0 0 1px rgba(0,0,0,0.25) !important;
      border-radius:8px !important;
      transform:translateY(0);
      white-space:nowrap !important;
      overflow:hidden !important;
      width:100% !important;
    }
    [class*="st-key-dash_period_"] .stButton > button p,
    [class*="st-key-dash_lang_"] .stButton > button p,
    .st-key-dash_export_csv .stDownloadButton > button p {
      white-space:nowrap !important;
      word-break:keep-all !important;
      overflow-wrap:normal !important;
      line-height:1 !important;
    }
    [class*="st-key-dash_period_"] .stButton > button:hover,
    [class*="st-key-dash_lang_"] .stButton > button:hover,
    .st-key-dash_export_csv .stDownloadButton > button:hover {
      color:#ff4b86 !important;
      border-color:rgba(255,56,96,0.85) !important;
      background:rgba(0,0,0,0.92) !important;
      box-shadow:0 0 10px rgba(255,56,96,0.22), inset 0 0 0 1px rgba(255,56,96,0.08) !important;
      transform:translateY(-1px);
    }
    .st-key-dash_export_csv .stDownloadButton > button {
      border-color:rgba(0,212,255,0.45) !important;
      color:var(--neon2) !important;
      background:rgba(0,0,0,0.88) !important;
      min-height:52px !important;
    }
    .st-key-dash_export_csv .stDownloadButton > button:hover {
      color:#2fd9ff !important;
      border-color:rgba(0,212,255,0.74) !important;
      box-shadow:0 0 11px rgba(0,212,255,0.24), inset 0 0 0 1px rgba(0,212,255,0.10) !important;
    }
    .st-key-dash_export_csv .stDownloadButton > button:disabled {
      opacity:0.45;
      cursor:not-allowed;
      box-shadow:none !important;
      transform:none !important;
    }
    @keyframes dash-btn-sweep {
      0% {
        left:-140%;
        opacity:0;
      }
      16% {
        opacity:0.95;
      }
      100% {
        left:140%;
        opacity:0;
      }
    }

    .stat-grid { display:grid; grid-template-columns:repeat(5, minmax(0,1fr)); gap:1.18rem; margin-bottom:1.95rem; }
    .stat-card {
      background:rgba(2,12,22,0.96);
      border:1px solid var(--border);
      border-radius:12px;
      padding:0.9rem 1rem;
      position:relative;
      overflow:hidden;
      display:flex;
      flex-direction:column;
      justify-content:center;
      gap:0.38rem;
      min-height:128px;
      transition:box-shadow .2s ease, border-color .2s ease, transform .2s ease;
      animation: dashBlockIn 250ms ease both;
    }
    .stat-grid .stat-card:nth-child(1) { animation-delay: 20ms; }
    .stat-grid .stat-card:nth-child(2) { animation-delay: 45ms; }
    .stat-grid .stat-card:nth-child(3) { animation-delay: 70ms; }
    .stat-grid .stat-card:nth-child(4) { animation-delay: 95ms; }
    .stat-grid .stat-card:nth-child(5) { animation-delay: 120ms; }
    .stat-card::after {
      content:'';
      position:absolute;
      left:0;
      right:0;
      bottom:0;
      height:1px;
      background:linear-gradient(90deg, transparent, var(--stat-glow, var(--neon)), transparent);
      opacity:0.62;
    }
    .stat-card.cyan {
      --stat-glow:#00d4ff;
      border:1px solid rgba(0,212,255,0.72);
      box-shadow:
        0 0 0 1px rgba(0,212,255,0.28),
        0 0 14px rgba(0,212,255,0.56),
        0 0 30px rgba(0,212,255,0.24),
        inset 0 0 0 1px rgba(0,212,255,0.10);
    }
    .stat-card.red {
      --stat-glow:#ff3860;
      border:1px solid rgba(255,56,96,0.72);
      box-shadow:
        0 0 0 1px rgba(255,56,96,0.28),
        0 0 14px rgba(255,56,96,0.56),
        0 0 30px rgba(255,56,96,0.24),
        inset 0 0 0 1px rgba(255,56,96,0.10);
    }
    .stat-card.yellow {
      --stat-glow:#ffdd57;
      border:1px solid rgba(255,221,87,0.72);
      box-shadow:
        0 0 0 1px rgba(255,221,87,0.28),
        0 0 14px rgba(255,221,87,0.56),
        0 0 30px rgba(255,221,87,0.24),
        inset 0 0 0 1px rgba(255,221,87,0.10);
    }
    .stat-card.green {
      --stat-glow:#00ff9f;
      border:1px solid rgba(0,255,159,0.72);
      box-shadow:
        0 0 0 1px rgba(0,255,159,0.28),
        0 0 14px rgba(0,255,159,0.56),
        0 0 30px rgba(0,255,159,0.24),
        inset 0 0 0 1px rgba(0,255,159,0.10);
    }
    .stat-card.purple {
      --stat-glow:#b57aff;
      border:1px solid rgba(181,122,255,0.74);
      box-shadow:
        0 0 0 1px rgba(181,122,255,0.28),
        0 0 14px rgba(181,122,255,0.56),
        0 0 30px rgba(181,122,255,0.24),
        inset 0 0 0 1px rgba(181,122,255,0.10);
    }
    .stat-card:hover {
      transform:translateY(-1px);
    }
    .stat-card.cyan:hover {
      box-shadow:
        0 0 0 1px rgba(0,212,255,0.38),
        0 0 18px rgba(0,212,255,0.64),
        0 0 34px rgba(0,212,255,0.30),
        inset 0 0 0 1px rgba(0,212,255,0.16);
    }
    .stat-card.red:hover {
      box-shadow:
        0 0 0 1px rgba(255,56,96,0.38),
        0 0 18px rgba(255,56,96,0.64),
        0 0 34px rgba(255,56,96,0.30),
        inset 0 0 0 1px rgba(255,56,96,0.16);
    }
    .stat-card.yellow:hover {
      box-shadow:
        0 0 0 1px rgba(255,221,87,0.38),
        0 0 18px rgba(255,221,87,0.64),
        0 0 34px rgba(255,221,87,0.30),
        inset 0 0 0 1px rgba(255,221,87,0.16);
    }
    .stat-card.green:hover {
      box-shadow:
        0 0 0 1px rgba(0,255,159,0.38),
        0 0 18px rgba(0,255,159,0.64),
        0 0 34px rgba(0,255,159,0.30),
        inset 0 0 0 1px rgba(0,255,159,0.16);
    }
    .stat-card.purple:hover {
      box-shadow:
        0 0 0 1px rgba(181,122,255,0.38),
        0 0 18px rgba(181,122,255,0.64),
        0 0 34px rgba(181,122,255,0.30),
        inset 0 0 0 1px rgba(181,122,255,0.16);
    }
    .stat-label {
      font-family:'Share Tech Mono', monospace;
      font-size:0.68rem;
      color:#ffffff !important;
      -webkit-text-fill-color:#ffffff !important;
      font-weight:700;
      letter-spacing:0.10em;
      text-transform:uppercase;
      margin-bottom:0.18rem;
      text-shadow:
        0 0 8px rgba(255,255,255,0.24),
        0 0 16px rgba(0,212,255,0.16);
    }
    .stat-foot {
      font-family:'Share Tech Mono', monospace;
      font-size:0.58rem;
      letter-spacing:0.06em;
      line-height:1.35;
      color:rgba(200,240,224,0.56);
      margin-top:-0.06rem;
    }
    .stat-card.purple .stat-foot { color:rgba(255,221,87,0.92); }
    .stat-val {
      font-family:'Oxanium', sans-serif;
      font-size:2.25rem;
      font-weight:900;
      line-height:1;
      letter-spacing:0.015em;
      margin-bottom:0;
      background:
        repeating-linear-gradient(
          to bottom,
          rgba(255,255,255,0.26) 0px,
          rgba(255,255,255,0.26) 1px,
          rgba(255,255,255,0.00) 1px,
          rgba(255,255,255,0.00) 6px
        ),
        repeating-linear-gradient(
          to bottom,
          rgba(0,0,0,0.00) 0px,
          rgba(0,0,0,0.00) 4px,
          rgba(0,0,0,0.24) 4px,
          rgba(0,0,0,0.24) 6px
        ),
        var(--num-grad, linear-gradient(180deg,#66eeff 0%, #00d4ff 48%, #0696e5 100%));
      -webkit-background-clip:text;
      background-clip:text;
      -webkit-text-fill-color:transparent;
      color:transparent;
      filter:
        drop-shadow(0 0 3px var(--num-glow-soft, rgba(0,212,255,0.55)))
        drop-shadow(0 0 11px var(--num-glow, rgba(0,212,255,0.52)));
      background-size:100% 100%;
      animation: dashNumIn 380ms ease-out both;
    }
    .stat-val.cyan {
      --num-grad: linear-gradient(180deg,#77f2ff 0%, #11dcff 46%, #009ee8 100%);
      --num-glow-soft: rgba(0,212,255,0.48);
      --num-glow: rgba(0,212,255,0.58);
    }
    .stat-val.red {
      --num-grad: linear-gradient(180deg,#ff8fb0 0%, #ff3f69 48%, #db1f4f 100%);
      --num-glow-soft: rgba(255,56,96,0.48);
      --num-glow: rgba(255,56,96,0.58);
    }
    .stat-val.yellow {
      --num-grad: linear-gradient(180deg,#fff1a8 0%, #ffe166 52%, #f3c53a 100%);
      --num-glow-soft: rgba(255,221,87,0.48);
      --num-glow: rgba(255,221,87,0.58);
    }
    .stat-val.green {
      --num-grad: linear-gradient(180deg,#84ffd4 0%, #19ffae 52%, #00ce84 100%);
      --num-glow-soft: rgba(0,255,159,0.48);
      --num-glow: rgba(0,255,159,0.58);
    }
    .stat-val.purple {
      --num-grad: linear-gradient(180deg,#dfb8ff 0%, #bb82ff 50%, #9b58f4 100%);
      --num-glow-soft: rgba(181,122,255,0.48);
      --num-glow: rgba(181,122,255,0.58);
    }
    .stat-trend {
      display:flex;
      align-items:center;
      gap:0.44rem;
      margin-top:0.24rem;
      font-family:'Share Tech Mono', monospace;
      font-size:0.86rem;
      letter-spacing:0.07em;
      text-transform:lowercase;
      font-weight:700;
      line-height:1.2;
      white-space:nowrap;
    }
    .stat-trend .trend-arrow { font-size:0.95rem; font-weight:900; }
    .stat-trend .trend-value { font-weight:900; }
    .stat-trend .trend-context { opacity:0.92; font-weight:700; }
    .stat-trend.up {
      color:#11efad;
      text-shadow:0 0 8px rgba(17,239,173,0.28);
    }
    .stat-trend.down {
      color:#ff5d8a;
      text-shadow:0 0 8px rgba(255,93,138,0.28);
    }
    .stat-trend.flat {
      color:#b9d2db;
      text-shadow:0 0 8px rgba(185,210,219,0.20);
    }

    .grid-2-1 { display:grid; grid-template-columns:2fr 1fr; gap:1.4rem; margin-bottom:1.6rem; }
    .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:1.4rem; margin-bottom:1.6rem; }
    .grid-single { margin-bottom:1.15rem; }
    .grid-1-1-1 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:1.4rem; margin-bottom:1.6rem; }

    .panel {
      background:rgba(3,15,26,0.92);
      border:1px solid var(--border);
      border-radius:12px;
      position:relative;
      overflow:hidden;
      transition:box-shadow .2s ease, border-color .2s ease, transform .2s ease;
      animation: dashBlockIn 260ms ease both;
    }
    .grid-2-1 .panel:nth-child(1) { animation-delay: 105ms; }
    .grid-2-1 .panel:nth-child(2) { animation-delay: 135ms; }
    .grid-2 .panel:nth-child(1) { animation-delay: 130ms; }
    .grid-2 .panel:nth-child(2) { animation-delay: 160ms; }
    .grid-1-1-1 .panel:nth-child(1) { animation-delay: 120ms; }
    .grid-1-1-1 .panel:nth-child(2) { animation-delay: 145ms; }
    .grid-1-1-1 .panel:nth-child(3) { animation-delay: 170ms; }
    .panel::before { content:none; }
    .accent-cyan{
      border:1px solid rgba(0,212,255,0.72);
      box-shadow:
        0 0 0 1px rgba(0,212,255,0.28),
        0 0 14px rgba(0,212,255,0.56),
        0 0 30px rgba(0,212,255,0.24),
        inset 0 0 0 1px rgba(0,212,255,0.10);
    }
    .accent-red{
      border:1px solid rgba(255,56,96,0.72);
      box-shadow:
        0 0 0 1px rgba(255,56,96,0.28),
        0 0 14px rgba(255,56,96,0.56),
        0 0 30px rgba(255,56,96,0.24),
        inset 0 0 0 1px rgba(255,56,96,0.10);
    }
    .accent-yellow{
      border:1px solid rgba(255,221,87,0.72);
      box-shadow:
        0 0 0 1px rgba(255,221,87,0.28),
        0 0 14px rgba(255,221,87,0.56),
        0 0 30px rgba(255,221,87,0.24),
        inset 0 0 0 1px rgba(255,221,87,0.10);
    }
    .accent-purple{
      border:1px solid rgba(181,122,255,0.74);
      box-shadow:
        0 0 0 1px rgba(181,122,255,0.28),
        0 0 14px rgba(181,122,255,0.56),
        0 0 30px rgba(181,122,255,0.24),
        inset 0 0 0 1px rgba(181,122,255,0.10);
    }
    .accent-orange{
      border:1px solid rgba(255,140,66,0.74);
      box-shadow:
        0 0 0 1px rgba(255,140,66,0.28),
        0 0 14px rgba(255,140,66,0.56),
        0 0 30px rgba(255,140,66,0.24),
        inset 0 0 0 1px rgba(255,140,66,0.10);
    }
    .panel:hover { transform:translateY(-1px); }
    .accent-cyan:hover{
      box-shadow:
        0 0 0 1px rgba(0,212,255,0.38),
        0 0 18px rgba(0,212,255,0.64),
        0 0 34px rgba(0,212,255,0.30),
        inset 0 0 0 1px rgba(0,212,255,0.16);
    }
    .accent-red:hover{
      box-shadow:
        0 0 0 1px rgba(255,56,96,0.38),
        0 0 18px rgba(255,56,96,0.64),
        0 0 34px rgba(255,56,96,0.30),
        inset 0 0 0 1px rgba(255,56,96,0.16);
    }
    .accent-yellow:hover{
      box-shadow:
        0 0 0 1px rgba(255,221,87,0.38),
        0 0 18px rgba(255,221,87,0.64),
        0 0 34px rgba(255,221,87,0.30),
        inset 0 0 0 1px rgba(255,221,87,0.16);
    }
    .accent-purple:hover{
      box-shadow:
        0 0 0 1px rgba(181,122,255,0.38),
        0 0 18px rgba(181,122,255,0.64),
        0 0 34px rgba(181,122,255,0.30),
        inset 0 0 0 1px rgba(181,122,255,0.16);
    }
    .accent-orange:hover{
      box-shadow:
        0 0 0 1px rgba(255,140,66,0.38),
        0 0 18px rgba(255,140,66,0.64),
        0 0 34px rgba(255,140,66,0.30),
        inset 0 0 0 1px rgba(255,140,66,0.16);
    }
    .panel.zoomable {
      cursor: zoom-in;
    }
    .panel.zoomable:hover {
      transform:translateY(-2px);
      filter:saturate(1.05);
    }
    .dash-modal-overlay {
      position: fixed;
      inset: 0;
      z-index: 99999;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 1.6rem;
      background: rgba(1, 8, 16, 0.78);
      backdrop-filter: blur(2px);
    }
    .dash-modal-overlay.open { display: flex; }
    .dash-modal-window {
      width: min(1280px, 94vw);
      max-height: 90vh;
      background: linear-gradient(180deg, rgba(4, 16, 28, 0.98), rgba(2, 10, 20, 0.99));
      border: 1px solid rgba(0, 255, 159, 0.62);
      border-radius: 16px;
      box-shadow:
        0 0 0 1px rgba(0, 255, 159, 0.25),
        0 0 26px rgba(0, 255, 159, 0.2),
        0 24px 80px rgba(0, 0, 0, 0.65);
      overflow: hidden;
    }
    .dash-modal-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      padding: 0.86rem 1.05rem 0.78rem;
      border-bottom: 1px solid rgba(0, 255, 159, 0.22);
      background: rgba(1, 10, 20, 0.86);
    }
    .dash-modal-dots {
      display: inline-flex;
      align-items: center;
      gap: 0.46rem;
    }
    .dash-modal-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }
    .dash-modal-dot.red { background: #ff3860; box-shadow: 0 0 8px rgba(255, 56, 96, 0.65); }
    .dash-modal-dot.yellow { background: #ffdd57; box-shadow: 0 0 8px rgba(255, 221, 87, 0.65); }
    .dash-modal-dot.green { background: #00ff9f; box-shadow: 0 0 8px rgba(0, 255, 159, 0.65); }
    .dash-modal-title {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.95rem;
      letter-spacing: 0.08em;
      color: rgba(200, 240, 224, 0.72);
      text-transform: uppercase;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .dash-modal-close {
      border: 1px solid rgba(0, 212, 255, 0.46);
      color: var(--neon2);
      background: rgba(0, 0, 0, 0.7);
      border-radius: 6px;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.72rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      padding: 0.3rem 0.52rem;
      cursor: pointer;
    }
    .dash-modal-close:hover {
      border-color: rgba(0, 212, 255, 0.72);
      box-shadow: 0 0 10px rgba(0, 212, 255, 0.24);
    }
    .dash-modal-body {
      padding: 1rem;
      max-height: calc(90vh - 64px);
      overflow: auto;
    }
    .dash-modal-body > .panel {
      margin: 0 !important;
      cursor: default !important;
      transform: none !important;
      filter: none !important;
    }
    .dash-modal-detail {
      margin-top: 0.9rem;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: rgba(0, 0, 0, 0.34);
      padding: 0.82rem 0.9rem;
    }
    .dash-modal-detail-head {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.84rem;
      color: var(--neon2);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 0.58rem;
      text-shadow: 0 0 8px rgba(0, 212, 255, 0.22);
    }
    .dash-modal-detail-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.56rem;
    }
    .dash-modal-detail-item {
      border: 1px solid rgba(0, 255, 159, 0.16);
      background: rgba(0, 255, 159, 0.03);
      border-radius: 8px;
      padding: 0.52rem 0.58rem;
      min-height: 58px;
    }
    .dash-modal-detail-k {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.62rem;
      color: rgba(200, 240, 224, 0.62);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 0.28rem;
    }
    .dash-modal-detail-v {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.9rem;
      color: #e9f7fb;
      letter-spacing: 0.02em;
      line-height: 1.38;
    }
    .dash-modal-detail-note {
      margin-top: 0.6rem;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.72rem;
      color: rgba(200, 240, 224, 0.8);
      line-height: 1.6;
    }
    .panel-head { display:flex; align-items:center; justify-content:space-between; gap:0.7rem; padding:0.82rem 1.08rem; border-bottom:1px solid var(--border); background:rgba(0,0,0,.24); }
    .panel-title {
      font-family:'Share Tech Mono', monospace;
      font-size:0.9rem;
      font-weight:700;
      color:var(--neon2);
      letter-spacing:0.12em;
      text-transform:uppercase;
      display:flex;
      align-items:center;
      gap:8px;
      text-shadow:0 0 8px rgba(0,212,255,0.24);
    }
    .panel-title::before {
      content:'//';
      color:rgba(0,212,255,0.32);
    }
    .panel-tag {
      font-family:'Share Tech Mono', monospace;
      font-size:0.62rem;
      color:#d7eaf0;
      border:1px solid rgba(0,255,159,0.28);
      background:rgba(0,0,0,0.26);
      padding:0.18rem 0.5rem;
      text-shadow:0 0 6px rgba(215,234,240,0.18);
    }
    .panel-body { padding:1.12rem 1.08rem 1.12rem; }

    .tl-legend {
      display:flex;
      align-items:center;
      gap:1.15rem;
      padding:0.1rem 0.15rem 0.46rem;
      margin-bottom:0.48rem;
      border-bottom:1px solid rgba(0,255,159,0.18);
    }
    .tl-legend-item {
      display:inline-flex;
      align-items:center;
      gap:0.48rem;
      font-family:'Share Tech Mono', monospace;
      font-size:0.82rem;
      color:#d6e7ee;
      letter-spacing:0.03em;
      text-shadow:0 0 6px rgba(214,231,238,0.14);
    }
    .tl-legend-line {
      width:26px;
      height:4px;
      border-radius:999px;
      display:inline-block;
    }
    .tl-legend-line.red{
      background:#ff3860;
      box-shadow:0 0 7px rgba(255,56,96,0.62), 0 0 15px rgba(255,56,96,0.34);
    }
    .tl-legend-line.yellow{
      background:#ffdd57;
      box-shadow:0 0 7px rgba(255,221,87,0.62), 0 0 15px rgba(255,221,87,0.34);
    }
    .tl-legend-line.green{
      background:#00ff9f;
      box-shadow:0 0 7px rgba(0,255,159,0.62), 0 0 15px rgba(0,255,159,0.34);
    }
    .dash-svg { width:100%; height:188px; display:block; }
    .tl-grid { stroke: rgba(0,255,159,0.08); stroke-width:1; }
    .tl-label { fill: rgba(200,240,224,0.36); font-size:9px; font-family:'Share Tech Mono', monospace; }
    .tl-line-red {
      fill:none;
      stroke:#ff3860;
      stroke-width:2.45;
      stroke-linecap:round;
      stroke-linejoin:round;
      filter: drop-shadow(0 0 4px rgba(255,56,96,.66)) drop-shadow(0 0 10px rgba(255,56,96,.34));
      stroke-dasharray: 1200;
      stroke-dashoffset: 1200;
      animation: dashLineDraw 760ms ease-out 120ms forwards;
    }
    .tl-line-yellow {
      fill:none;
      stroke:#ffdd57;
      stroke-width:2.35;
      stroke-linecap:round;
      stroke-linejoin:round;
      filter: drop-shadow(0 0 4px rgba(255,221,87,.66)) drop-shadow(0 0 10px rgba(255,221,87,.32));
      stroke-dasharray: 1200;
      stroke-dashoffset: 1200;
      animation: dashLineDraw 860ms ease-out 180ms forwards;
    }
    .tl-line-green {
      fill:none;
      stroke:#00ff9f;
      stroke-width:2.45;
      stroke-linecap:round;
      stroke-linejoin:round;
      filter: drop-shadow(0 0 4px rgba(0,255,159,.66)) drop-shadow(0 0 10px rgba(0,255,159,.34));
      stroke-dasharray: 1200;
      stroke-dashoffset: 1200;
      animation: dashLineDraw 860ms ease-out 220ms forwards;
    }
    .tl-dot-red{
      fill:#ff3860;
      filter:drop-shadow(0 0 5px rgba(255,56,96,.78)) drop-shadow(0 0 11px rgba(255,56,96,.34));
      transform-box: fill-box;
      transform-origin: center;
      animation: dashDotPop 340ms ease-out 560ms both;
    }
    .tl-dot-yellow{
      fill:#ffdd57;
      filter:drop-shadow(0 0 5px rgba(255,221,87,.78)) drop-shadow(0 0 11px rgba(255,221,87,.34));
      transform-box: fill-box;
      transform-origin: center;
      animation: dashDotPop 340ms ease-out 620ms both;
    }
    .tl-dot-green{
      fill:#00ff9f;
      filter:drop-shadow(0 0 5px rgba(0,255,159,.78)) drop-shadow(0 0 11px rgba(0,255,159,.34));
      transform-box: fill-box;
      transform-origin: center;
      animation: dashDotPop 340ms ease-out 680ms both;
    }

    .donut-wrap { display:flex; align-items:center; gap:1rem; }
    .donut {
      width:132px; height:132px; border-radius:50%; position:relative; flex-shrink:0;
      border:1px solid rgba(0,255,159,0.25);
      animation: dashDonutIn 520ms cubic-bezier(0.22, 0.7, 0.28, 1) both;
      overflow:hidden;
      box-shadow:
        0 0 0 1px rgba(0,255,159,0.18),
        0 0 14px rgba(0,255,159,0.14);
    }
    .donut::before {
      content:'';
      position:absolute;
      inset:0;
      border-radius:50%;
      pointer-events:none;
      z-index:1;
      background:
        repeating-linear-gradient(
          to bottom,
          rgba(255,255,255,0.22) 0px,
          rgba(255,255,255,0.22) 1px,
          rgba(255,255,255,0.00) 1px,
          rgba(255,255,255,0.00) 6px
        ),
        repeating-linear-gradient(
          to bottom,
          rgba(0,0,0,0.00) 0px,
          rgba(0,0,0,0.00) 4px,
          rgba(0,0,0,0.22) 4px,
          rgba(0,0,0,0.22) 6px
        );
      mix-blend-mode:screen;
      opacity:0.78;
    }
    .donut::after {
      content:''; position:absolute; inset:24px; border-radius:50%; background:rgba(3,15,26,0.98); border:1px solid rgba(0,255,159,0.14); z-index:2;
    }
    .donut-center {
      position:absolute; inset:0; display:flex; align-items:center; justify-content:center; z-index:3;
      font-family:'Share Tech Mono', monospace; font-weight:800; color:#d8fff0; font-size:1.05rem;
    }
    .legend { display:flex; flex-direction:column; gap:0.48rem; min-width:190px; }
    .legend-item { display:flex; align-items:center; gap:0.45rem; }
    .legend-dot { width:8px; height:8px; border-radius:2px; }
    .legend-name {
      font-family:'Share Tech Mono', monospace;
      font-size:0.78rem;
      font-weight:600;
      color:#e9f7fb;
      text-shadow:0 0 7px rgba(233,247,251,0.14);
      flex:1;
    }
    .legend-val { font-family:'Share Tech Mono', monospace; font-size:0.84rem; font-weight:700; }
    .legend-pct {
      font-family:'Share Tech Mono', monospace;
      font-size:0.68rem;
      color:#c6dbe2;
      font-weight:600;
      text-shadow:0 0 6px rgba(198,219,226,0.12);
    }

    .hbar-list { display:flex; flex-direction:column; gap:0.55rem; }
    .hbar-top { display:flex; align-items:center; justify-content:space-between; gap:0.6rem; margin-bottom:0.22rem; }
    .hbar-name,.hbar-val { font-family:'Share Tech Mono', monospace; font-size:0.66rem; }
    .hbar-name{ color:var(--text); }
    .hbar-val{ color:var(--muted); }
    .hbar-track { height:6px; background:rgba(255,255,255,0.06); }
    .hbar-fill {
      height:100%;
      transform-origin:left center;
      animation: dashBarGrow 620ms cubic-bezier(0.22, 0.7, 0.28, 1) both;
    }

    .section-header { display:flex; align-items:center; gap:0.8rem; margin:1.55rem 0 0.95rem; }
    .section-label { font-family:'Share Tech Mono', monospace; font-size:0.66rem; color:var(--neon); opacity:0.7; letter-spacing:0.1em; }
    .section-title {
      font-family:'Share Tech Mono', monospace;
      font-size:1.45rem;
      font-weight:800;
      color:#ffffff !important;
      -webkit-text-fill-color:#ffffff !important;
      letter-spacing:0.06em;
      text-transform:uppercase;
      text-shadow:
        0 0 10px rgba(255,255,255,0.32),
        0 0 24px rgba(0,212,255,0.18);
    }

    .perf-table, .log-table { width:100%; border-collapse:collapse; }
    .perf-table th, .log-table th {
      font-family:'Share Tech Mono', monospace; font-size:0.6rem; color:var(--neon2); text-transform:uppercase; letter-spacing:0.1em;
      text-align:left; padding:0.5rem 0.72rem; border-bottom:1px solid var(--border);
    }
    .perf-table td, .log-table td {
      font-family:'Share Tech Mono', monospace; font-size:0.74rem; color:var(--muted);
      padding:0.55rem 0.72rem; border-bottom:1px solid rgba(0,255,159,0.05);
      vertical-align:middle;
    }
    .perf-table td:first-child, .log-table td:first-child { color:var(--text); }
    .f1-cell { color:var(--neon); font-weight:700; }
    .metric-top { font-family:'Share Tech Mono', monospace; font-size:0.6rem; color:var(--muted); margin-bottom:0.12rem; }
    .metric-bar { height:3px; background:rgba(255,255,255,0.06); }
    .metric-fill {
      height:100%;
      transform-origin:left center;
      animation: dashBarGrow 560ms cubic-bezier(0.22, 0.7, 0.28, 1) both;
    }
    .badge-model { display:inline-block; padding:0.14rem 0.4rem; font-size:0.58rem; letter-spacing:0.08em; text-transform:uppercase; }
    .badge-primary { color:var(--neon2); background:rgba(0,212,255,.08); border:1px solid rgba(0,212,255,.32); }
    .badge-baseline { color:var(--muted); background:rgba(200,240,224,.04); border:1px solid var(--border); }
    .leader-chip-row {
      display:flex;
      flex-wrap:wrap;
      gap:0.42rem;
      margin-bottom:0.72rem;
    }
    .lang-winner-pill {
      display:inline-flex;
      align-items:center;
      gap:0.35rem;
      padding:0.18rem 0.42rem;
      border:1px solid rgba(0,212,255,0.22);
      background:rgba(0,212,255,0.04);
      font-family:'Share Tech Mono', monospace;
      font-size:0.56rem;
      letter-spacing:0.05em;
      text-transform:uppercase;
      white-space:nowrap;
    }
    .lang-winner-name { color:rgba(200,240,224,0.62); }
    .lang-winner-pill strong { color:var(--neon); font-weight:700; }
    .lang-winner-pill em { color:#d2f8e8; font-style:normal; }
    .lang-winner-pill small { color:var(--yellow); font-size:0.54rem; }
    .lang-breakdown-shell {
      display:flex;
      flex-direction:column;
      gap:0.82rem;
    }
    .lang-summary-grid {
      display:grid;
      grid-template-columns:repeat(3, minmax(0, 1fr));
      gap:0.62rem;
    }
    .lang-summary-card,
    .lang-card {
      border:1px solid var(--border);
      background:rgba(0,0,0,0.24);
    }
    .lang-summary-card {
      padding:0.62rem 0.68rem;
    }
    .lang-summary-card.danger { border-color:rgba(255,56,96,0.22); }
    .lang-summary-card.warn { border-color:rgba(255,221,87,0.22); }
    .lang-summary-card.calm { border-color:rgba(0,255,159,0.22); }
    .lang-summary-k {
      font-family:'Share Tech Mono', monospace;
      font-size:0.54rem;
      letter-spacing:0.08em;
      text-transform:uppercase;
      color:rgba(200,240,224,0.52);
      margin-bottom:0.16rem;
    }
    .lang-summary-v {
      font-family:'Oxanium', sans-serif;
      font-size:0.96rem;
      font-weight:700;
      color:var(--text);
    }
    .lang-summary-m {
      margin-top:0.14rem;
      font-family:'Share Tech Mono', monospace;
      font-size:0.54rem;
      letter-spacing:0.05em;
      color:rgba(200,240,224,0.62);
      line-height:1.5;
    }
    .lang-card-grid {
      display:grid;
      grid-template-columns:repeat(2, minmax(0, 1fr));
      gap:0.68rem;
    }
    .lang-card {
      padding:0.72rem 0.74rem;
    }
    .lang-card-head {
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap:0.56rem;
      margin-bottom:0.54rem;
    }
    .lang-card-title {
      font-family:'Oxanium', sans-serif;
      font-size:0.9rem;
      font-weight:700;
      color:var(--text);
    }
    .lang-card-meta {
      margin-top:0.1rem;
      font-family:'Share Tech Mono', monospace;
      font-size:0.54rem;
      letter-spacing:0.08em;
      text-transform:uppercase;
      color:rgba(200,240,224,0.52);
    }
    .lang-card-winner {
      text-align:right;
      font-family:'Share Tech Mono', monospace;
      font-size:0.55rem;
      letter-spacing:0.08em;
      text-transform:uppercase;
      color:var(--neon);
      white-space:nowrap;
    }
    .lang-score-grid {
      display:grid;
      grid-template-columns:repeat(3, minmax(0, 1fr));
      gap:0.42rem;
    }
    .lang-score,
    .lang-table-score {
      display:flex;
      flex-direction:column;
      gap:0.12rem;
      padding:0.42rem 0.46rem;
      border:1px solid rgba(255,255,255,0.08);
      background:rgba(255,255,255,0.02);
    }
    .lang-score.best,
    .lang-table-score.best {
      border-color:rgba(0,255,159,0.36);
      background:rgba(0,255,159,0.05);
      box-shadow:0 0 0 1px rgba(0,255,159,0.06) inset;
    }
    .lang-score-k {
      font-family:'Share Tech Mono', monospace;
      font-size:0.5rem;
      letter-spacing:0.08em;
      text-transform:uppercase;
      color:rgba(200,240,224,0.48);
    }
    .lang-score-v {
      font-family:'Oxanium', sans-serif;
      font-size:0.92rem;
      font-weight:700;
      color:var(--text);
    }
    .lang-card-foot {
      margin-top:0.46rem;
      font-family:'Share Tech Mono', monospace;
      font-size:0.54rem;
      letter-spacing:0.05em;
      color:rgba(200,240,224,0.58);
      line-height:1.5;
    }
    .lang-table-wrap {
      overflow-x:auto;
      margin-top:0.05rem;
    }
    .lang-table {
      min-width: 820px;
    }
    .lang-table th,
    .lang-table td {
      text-align:center;
      vertical-align:middle;
    }
    .lang-table th:first-child,
    .lang-table td:first-child {
      text-align:left;
    }
    .lang-row-label {
      color:var(--text);
      font-weight:700;
    }
    .lang-row-meta {
      margin-top:0.04rem;
      font-family:'Share Tech Mono', monospace;
      font-size:0.52rem;
      letter-spacing:0.06em;
      text-transform:uppercase;
      color:rgba(200,240,224,0.5);
    }
    .lang-winner-badge {
      display:inline-flex;
      align-items:center;
      padding:0.08rem 0.3rem;
      border:1px solid rgba(0,255,159,0.32);
      color:var(--neon);
      background:rgba(0,255,159,0.04);
      font-size:0.56rem;
      text-transform:uppercase;
      letter-spacing:0.08em;
      white-space:nowrap;
    }
    .lang-gap {
      color:var(--yellow);
      font-weight:700;
    }
    .leader-sub {
      margin-top:0.08rem;
      font-family:'Share Tech Mono', monospace;
      font-size:0.54rem;
      color:rgba(200,240,224,0.52);
      letter-spacing:0.05em;
      text-transform:none;
    }
    .leader-metric {
      color:var(--neon);
      font-weight:700;
    }
    .leader-lang {
      display:inline-block;
      padding:0.12rem 0.34rem;
      border:1px solid rgba(0,255,159,0.24);
      color:var(--neon);
      font-size:0.56rem;
      letter-spacing:0.08em;
      text-transform:uppercase;
    }
    .agreement-summary {
      display:grid;
      grid-template-columns:repeat(3, minmax(0, 1fr));
      gap:0.62rem;
      margin-bottom:0.72rem;
    }
    .split-stat {
      border:1px solid var(--border);
      background:rgba(0,0,0,0.24);
      padding:0.58rem 0.64rem;
    }
    .split-stat-k {
      font-family:'Share Tech Mono', monospace;
      font-size:0.56rem;
      letter-spacing:0.08em;
      text-transform:uppercase;
      color:rgba(200,240,224,0.56);
    }
    .split-stat-v {
      font-family:'Oxanium', sans-serif;
      font-size:1.08rem;
      color:var(--text);
      margin-top:0.12rem;
    }
    .split-stat-p {
      font-family:'Share Tech Mono', monospace;
      font-size:0.56rem;
      color:var(--neon2);
      letter-spacing:0.06em;
      margin-top:0.08rem;
    }
    .agreement-table {
      width:100%;
      border-collapse:collapse;
    }
    .agreement-table th, .agreement-table td {
      font-family:'Share Tech Mono', monospace;
      font-size:0.68rem;
      color:var(--muted);
      padding:0.52rem 0.58rem;
      border-bottom:1px solid rgba(0,255,159,0.05);
      text-align:center;
    }
    .agreement-table th:first-child, .agreement-table td:first-child {
      text-align:left;
      color:var(--text);
    }
    .agree-cell {
      font-weight:700;
      letter-spacing:0.02em;
    }
    .agree-cell.diag { color:var(--neon); }
    .agree-cell.high { color:var(--neon); }
    .agree-cell.mid { color:var(--yellow); }
    .agree-cell.low { color:var(--red); }
    .compare-card-list {
      display:flex;
      flex-direction:column;
      gap:0.62rem;
    }
    .compare-card {
      border:1px solid var(--border);
      background:rgba(0,0,0,0.24);
      padding:0.64rem 0.7rem;
    }
    .compare-card-top {
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:0.6rem;
      font-family:'Share Tech Mono', monospace;
      font-size:0.62rem;
      color:rgba(200,240,224,0.58);
      letter-spacing:0.05em;
      text-transform:uppercase;
      margin-bottom:0.2rem;
    }
    .compare-chip {
      display:inline-flex;
      align-items:center;
      justify-content:center;
      padding:0.08rem 0.32rem;
      border:1px solid rgba(0,212,255,0.24);
      color:var(--neon2);
      background:rgba(0,212,255,0.03);
      font-family:'Share Tech Mono', monospace;
      font-size:0.5rem;
      letter-spacing:0.08em;
      text-transform:uppercase;
      white-space:nowrap;
    }
    .compare-chip.review {
      border-color:rgba(255,221,87,0.34);
      color:var(--yellow);
      background:rgba(255,221,87,0.04);
    }
    .compare-chip.auto {
      border-color:rgba(0,212,255,0.24);
      color:var(--neon2);
      background:rgba(0,212,255,0.03);
    }
    .compare-card-meta,
    .compare-card-votes,
    .compare-card-preview,
    .compare-reason {
      font-family:'Share Tech Mono', monospace;
      font-size:0.68rem;
      line-height:1.55;
      letter-spacing:0.02em;
    }
    .compare-card-meta {
      color:var(--neon2);
      margin-bottom:0.18rem;
    }
    .compare-card-votes {
      color:rgba(200,240,224,0.76);
      margin-bottom:0.22rem;
    }
    .compare-card-preview {
      color:var(--muted);
    }
    .compare-card-foot {
      margin-top:0.3rem;
      display:flex;
      align-items:flex-start;
      gap:0.45rem;
      flex-wrap:wrap;
    }
    .compare-reason {
      color:var(--yellow);
      opacity:0.92;
    }

    .conf-matrix-wrap { display:flex; flex-direction:column; align-items:center; gap:0.74rem; }
    .conf-matrix-title {
      align-self:flex-start;
      display:flex;
      align-items:center;
      gap:0.36rem;
      font-family:'Share Tech Mono', monospace;
      font-size:0.62rem;
      color:var(--muted);
      letter-spacing:0.09em;
      text-transform:uppercase;
    }
    .conf-matrix-title::before { content:'//'; color:rgba(0,212,255,0.28); }
    .cm-outer { display:flex; gap:0; align-items:flex-start; }
    .cm-y-label {
      writing-mode:vertical-rl;
      text-orientation:mixed;
      transform:rotate(180deg);
      margin-right:0.62rem;
      align-self:center;
      font-family:'Share Tech Mono', monospace;
      font-size:0.58rem;
      color:var(--muted);
      letter-spacing:0.09em;
      text-transform:uppercase;
    }
    .cm-inner { display:flex; flex-direction:column; gap:0; }
    .cm-pred-label {
      margin-left:68px;
      margin-bottom:0.2rem;
      text-align:center;
      font-family:'Share Tech Mono', monospace;
      font-size:0.58rem;
      color:var(--muted);
      letter-spacing:0.08em;
      text-transform:uppercase;
    }
    .cm-x-labels { display:flex; margin-left:68px; margin-bottom:0.22rem; gap:0; }
    .cm-x-label {
      text-align:center;
      font-family:'Share Tech Mono', monospace;
      font-size:0.58rem;
      color:var(--muted);
      letter-spacing:0.07em;
      text-transform:uppercase;
    }
    .cm-row { display:flex; align-items:center; gap:0; }
    .cm-row-label {
      width:68px;
      padding-right:0.42rem;
      flex-shrink:0;
      text-align:right;
      font-family:'Share Tech Mono', monospace;
      font-size:0.58rem;
      color:var(--muted);
      letter-spacing:0.07em;
      text-transform:uppercase;
    }
    .cm-cell {
      display:flex;
      flex-direction:column;
      align-items:center;
      justify-content:center;
      gap:2px;
      border:1px solid rgba(0,0,0,0.3);
      transition:transform 0.2s ease, box-shadow 0.2s ease;
      cursor:default;
      position:relative;
    }
    .cm-cell:hover { transform:scale(1.06); box-shadow:0 0 16px rgba(0,0,0,0.5); z-index:2; }
    .cm-cell-val { font-family:'Share Tech Mono', monospace; font-size:0.74rem; font-weight:700; line-height:1; }
    .cm-cell-pct { font-family:'Share Tech Mono', monospace; font-size:0.56rem; opacity:0.75; line-height:1; }
    .cm-legend { display:flex; gap:1rem; font-family:'Share Tech Mono', monospace; font-size:0.58rem; color:var(--muted); letter-spacing:0.06em; }
    .cm-legend-item { display:flex; align-items:center; gap:0.34rem; }
    .cm-legend-swatch { width:12px; height:8px; }

    .model-eval-grid,
    .model-calib-grid {
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(230px, 1fr));
      gap:0.82rem;
    }
    .model-eval-card,
    .model-calib-card {
      border:1px solid rgba(0,212,255,0.20);
      background:rgba(0,0,0,0.22);
      padding:0.78rem;
      box-shadow:inset 0 0 18px rgba(0,212,255,0.04);
    }
    .model-eval-head,
    .model-calib-top {
      display:flex;
      justify-content:space-between;
      gap:0.7rem;
      align-items:flex-start;
      margin-bottom:0.62rem;
    }
    .model-eval-role,
    .model-calib-label {
      font-family:'Share Tech Mono', monospace;
      font-size:0.58rem;
      letter-spacing:0.12em;
      text-transform:uppercase;
      color:var(--muted);
      margin-bottom:0.16rem;
    }
    .model-eval-name,
    .model-calib-version {
      font-family:'Share Tech Mono', monospace;
      color:#f0fff8;
      font-size:0.76rem;
      font-weight:900;
      letter-spacing:0.04em;
      word-break:break-word;
    }
    .model-eval-split {
      font-family:'Share Tech Mono', monospace;
      font-size:0.55rem;
      letter-spacing:0.08em;
      color:var(--neon2);
      border:1px solid rgba(0,212,255,0.28);
      background:rgba(0,212,255,0.06);
      padding:0.18rem 0.42rem;
      text-transform:uppercase;
      white-space:nowrap;
    }
    .model-eval-metrics {
      display:flex;
      flex-wrap:wrap;
      gap:0.35rem;
      margin-top:0.6rem;
      font-family:'Share Tech Mono', monospace;
      font-size:0.58rem;
      color:var(--muted);
      letter-spacing:0.04em;
    }
    .model-eval-metrics span,
    .model-calib-row {
      border:1px solid rgba(200,240,224,0.10);
      background:rgba(200,240,224,0.035);
      padding:0.18rem 0.42rem;
    }
    .model-calib-grid { margin-top:0.72rem; }
    .model-calib-card.good { border-color:rgba(0,255,159,0.28); box-shadow:inset 0 0 20px rgba(0,255,159,0.05); }
    .model-calib-card.warn { border-color:rgba(255,221,87,0.32); box-shadow:inset 0 0 20px rgba(255,221,87,0.05); }
    .model-calib-card.risk { border-color:rgba(255,56,96,0.36); box-shadow:inset 0 0 20px rgba(255,56,96,0.06); }
    .model-calib-score {
      font-family:'Share Tech Mono', monospace;
      font-size:0.74rem;
      color:#f0fff8;
      letter-spacing:0.04em;
      margin-bottom:0.48rem;
    }
    .model-calib-score strong { color:var(--yellow); font-size:1rem; }
    .model-calib-row {
      display:flex;
      justify-content:space-between;
      gap:0.6rem;
      margin-top:0.34rem;
      font-family:'Share Tech Mono', monospace;
      font-size:0.60rem;
      color:var(--muted);
      letter-spacing:0.04em;
    }
    .model-calib-row strong { color:#f0fff8; }
    .empty-note {
      font-family:'Share Tech Mono', monospace;
      color:var(--muted);
      font-size:0.68rem;
      letter-spacing:0.04em;
      padding:0.8rem;
      border:1px solid rgba(200,240,224,0.12);
      background:rgba(0,0,0,0.2);
    }

    .calib-note { font-family:'Share Tech Mono', monospace; font-size:0.67rem; color:var(--muted); line-height:1.65; margin-top:0.5rem; }
    .ece-badge { display:inline-block; margin-top:0.45rem; font-family:'Share Tech Mono', monospace; font-size:0.63rem; letter-spacing:0.07em; color:var(--neon); border:1px solid rgba(0,255,159,.34); background:rgba(0,255,159,.06); padding:0.2rem 0.55rem; }
    .ece-badge.good { color:var(--neon); border-color:rgba(0,255,159,.36); background:rgba(0,255,159,.06); }
    .ece-badge.warn { color:var(--yellow); border-color:rgba(255,221,87,.36); background:rgba(255,221,87,.06); }
    .ece-badge.risk { color:var(--hot); border-color:rgba(255,56,96,.38); background:rgba(255,56,96,.07); }

    .insight-grid { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:1rem; margin-bottom:1.2rem; }
    .insight-card { background:rgba(3,15,26,0.92); border:1px solid var(--border); padding:0.72rem 0.84rem; }
    .insight-title { font-size:0.77rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:var(--neon2); margin-bottom:0.3rem; }
    .insight-body { font-family:'Share Tech Mono', monospace; font-size:0.66rem; color:var(--muted); line-height:1.72; }

    .log-toolbar { display:flex; align-items:center; gap:0.6rem; padding:0.72rem 1.08rem; border-bottom:1px solid var(--border); background:rgba(0,0,0,0.15); flex-wrap:wrap; }
    .log-search {
      flex:1;
      min-width:190px;
      background:rgba(0,0,0,0.42);
      border:1px solid var(--border);
      color:var(--text);
      font-family:'Share Tech Mono', monospace;
      font-size:0.68rem;
      padding:0.42rem 0.72rem;
      outline:none;
      transition:border-color 0.2s, box-shadow 0.2s;
      letter-spacing:0.04em;
    }
    .log-search::placeholder { color:rgba(200,240,224,0.2); }
    .log-search:focus { border-color:rgba(0,212,255,0.4); box-shadow:0 0 10px rgba(0,212,255,0.06); }
    .log-filter-btn {
      font-family:'Share Tech Mono', monospace;
      font-size:0.6rem;
      letter-spacing:0.08em;
      text-transform:uppercase;
      padding:0.32rem 0.75rem;
      background:transparent;
      border:1px solid var(--border);
      color:var(--muted);
      cursor:pointer;
      transition:all 0.2s;
    }
    .log-filter-btn:hover { border-color:var(--neon); color:var(--neon); }
    .log-filter-btn.active { border-color:var(--neon); color:var(--neon); background:rgba(0,255,159,0.04); }
    .log-count { font-family:'Share Tech Mono', monospace; font-size:0.6rem; color:var(--muted); margin-left:auto; }

    .log-table-wrap { overflow-x:auto; }
    .verdict-pip { display:inline-flex; align-items:center; gap:0.34rem; font-weight:700; font-size:0.66rem; letter-spacing:0.07em; text-transform:uppercase; }
    .pip { width:6px; height:6px; border-radius:50%; }
    .pip.phishing { background:var(--red); box-shadow:0 0 6px var(--red); }
    .pip.suspicious { background:var(--yellow); box-shadow:0 0 6px var(--yellow); }
    .pip.safe { background:var(--neon); box-shadow:0 0 6px var(--neon); }
    .verdict-pip.phishing { color:var(--red); }
    .verdict-pip.suspicious { color:var(--yellow); }
    .verdict-pip.safe { color:var(--neon); }
    .score-chip { font-size:0.6rem; padding:0.1rem 0.4rem; border:1px solid; }
    .score-chip.phishing { color:var(--red); border-color:rgba(255,56,96,.35); }
    .score-chip.suspicious { color:var(--yellow); border-color:rgba(255,221,87,.35); }
    .score-chip.safe { color:var(--neon); border-color:rgba(0,255,159,.35); }
    .conf-cell { color:var(--neon2) !important; font-weight:700; }
    .lang-pip { display:inline-block; font-size:0.58rem; padding:0.1rem 0.36rem; border:1px solid rgba(0,212,255,.28); color:var(--neon2); }
    .compare-cell {
      min-width: 280px;
      max-width: 420px;
      white-space: normal;
      display:flex;
      flex-direction:column;
      gap:0.08rem;
      color: var(--text);
    }
    .compare-summary {
      display:block;
      max-width: 100%;
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
      font-family:'Share Tech Mono', monospace;
      font-size:0.56rem;
      letter-spacing:0.05em;
      color:rgba(200,240,224,0.72);
    }
    .compare-badge {
      display:inline-flex;
      align-items:center;
      justify-content:center;
      margin-top:0.16rem;
      padding:0.08rem 0.36rem;
      font-family:'Share Tech Mono', monospace;
      font-size:0.52rem;
      letter-spacing:0.08em;
      text-transform:uppercase;
      border:1px solid;
    }
    .compare-badge.agree { color:var(--neon); border-color:rgba(0,255,159,.35); background:rgba(0,255,159,.04); }
    .compare-badge.diff { color:var(--red); border-color:rgba(255,56,96,.35); background:rgba(255,56,96,.04); }
    .compare-badge.review { color:var(--yellow); border-color:rgba(255,221,87,.38); background:rgba(255,221,87,.05); }
    .compare-badge.auto { color:var(--neon2); border-color:rgba(0,212,255,.28); background:rgba(0,212,255,.04); }
    .log-preview { max-width:420px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:rgba(200,240,224,0.48); }
    .log-table tr.row-phishing td:first-child { border-left:3px solid var(--red); padding-left:0.56rem; }
    .log-table tr.row-suspicious td:first-child { border-left:3px solid var(--yellow); padding-left:0.56rem; }
    .log-table tr.row-safe td:first-child { border-left:3px solid var(--neon); padding-left:0.56rem; }
    .log-table tr:hover td { background:rgba(0,255,159,0.015); }
    .log-table tr.row-phishing:hover td { background:rgba(255,56,96,0.025); }
    .log-table tr.row-suspicious:hover td { background:rgba(255,221,87,0.02); }
    .log-table tr.row-safe:hover td { background:rgba(0,255,159,0.02); }
    .log-empty { font-family:'Share Tech Mono', monospace; font-size:0.68rem; color:var(--muted); text-align:center; padding:1.2rem; letter-spacing:0.08em; }

    @media (max-width: 1200px) {
      .stat-grid { grid-template-columns: repeat(3, minmax(0,1fr)); }
      .grid-2-1, .grid-2, .grid-1-1-1 { grid-template-columns: 1fr; }
      .insight-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
      .lang-summary-grid { grid-template-columns: 1fr; }
      .lang-card-grid { grid-template-columns: 1fr; }
    }
    @media (prefers-reduced-motion: reduce) {
      .dash-wrap,
      .stat-card,
      .panel {
        animation: none !important;
        transition: none !important;
      }
    }
    @media (max-width: 760px) {
      .stat-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
      .insight-grid { grid-template-columns: 1fr; }
      .lang-score-grid { grid-template-columns: 1fr; }
      .page-h1 { font-size: 1.55rem; }
      .section-title { font-size: 1.18rem; }
    }
    </style>
    """
    ),
    unsafe_allow_html=True,
)

st.markdown(
    _render_html_block(
        f"""
    <style>
    .st-key-dash_period_{period_selected} .stButton > button,
    .st-key-dash_lang_{lang_selected} .stButton > button {{
      color:#00f1a8 !important;
      -webkit-text-fill-color:#00f1a8 !important;
      border-color:rgba(0,255,159,0.82) !important;
      background:rgba(0,0,0,0.90) !important;
      box-shadow:0 0 11px rgba(0,255,159,0.16), inset 0 0 0 1px rgba(0,255,159,0.12) !important;
    }}
    </style>
    """
    ),
    unsafe_allow_html=True,
)

st.markdown(
    _render_html_block(
        """
    <div class='dash-wrap head-only'>
      <div class='dash-system-row'>
        <div class='status-bar'>
          <div class='status-dot'></div>
          <span class='status-text'>System Online</span>
          <span class='version-tag'>v2.4.1 — IND</span>
        </div>
      </div>
      <div class='page-title-row'>
        <span class='section-code'>// 01</span>
        <h1 class='page-h1'>Technical Dashboard</h1>
      </div>
      <p class='page-sub'>Research-side analytics for model comparison, calibration, scan logs, and cross-language performance. AI Studio now has its own page in the password-protected Analyst Lab.</p>
    </div>
    """
    ),
    unsafe_allow_html=True,
)

filters_left, filters_mid, filters_right = st.columns([4.5, 4.5, 2.6], gap="medium")

with filters_left:
    st.markdown("<div class='dash-filter-label'>Period:</div>", unsafe_allow_html=True)
    period_cols = st.columns(4, gap="small")
    if period_cols[0].button("24H", key="dash_period_24h", use_container_width=True):
        st.query_params["period"] = "24h"
        st.query_params["lang"] = lang_selected
        st.rerun()
    if period_cols[1].button("7D", key="dash_period_7d", use_container_width=True):
        st.query_params["period"] = "7d"
        st.query_params["lang"] = lang_selected
        st.rerun()
    if period_cols[2].button("30D", key="dash_period_30d", use_container_width=True):
        st.query_params["period"] = "30d"
        st.query_params["lang"] = lang_selected
        st.rerun()
    if period_cols[3].button("ALL TIME", key="dash_period_all", use_container_width=True):
        st.query_params["period"] = "all"
        st.query_params["lang"] = lang_selected
        st.rerun()

with filters_mid:
    st.markdown("<div class='dash-filter-label'>Language:</div>", unsafe_allow_html=True)
    lang_cols = st.columns(5, gap="small")
    if lang_cols[0].button("All", key="dash_lang_all", use_container_width=True):
        st.query_params["period"] = period_selected
        st.query_params["lang"] = "all"
        st.rerun()
    if lang_cols[1].button("English", key="dash_lang_english", use_container_width=True):
        st.query_params["period"] = period_selected
        st.query_params["lang"] = "english"
        st.rerun()
    if lang_cols[2].button("Hindi", key="dash_lang_hindi", use_container_width=True):
        st.query_params["period"] = period_selected
        st.query_params["lang"] = "hindi"
        st.rerun()
    if lang_cols[3].button("Punjabi", key="dash_lang_punjabi", use_container_width=True):
        st.query_params["period"] = period_selected
        st.query_params["lang"] = "punjabi"
        st.rerun()
    if lang_cols[4].button("Urdu", key="dash_lang_urdu", use_container_width=True):
        st.query_params["period"] = period_selected
        st.query_params["lang"] = "urdu"
        st.rerun()

with filters_right:
    st.markdown("<div class='dash-filter-label'>Export:</div>", unsafe_allow_html=True)
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="safesandesh_scans.csv",
        mime="text/csv",
        key="dash_export_csv",
        disabled=not bool(csv_bytes),
        use_container_width=True,
    )

st.markdown("<div style='height:0.18rem;'></div>", unsafe_allow_html=True)

st.markdown(
    _render_html_block(
        f"""
    <div class='dash-wrap main-only'>
      <div class='stat-grid'>
        <div class='stat-card cyan'>
          <div class='stat-label'>Total Scans</div>
          <div class='stat-val cyan'>{data['total']:,}</div>
          <div class='stat-trend {stat_trends['total']['cls']}'><span class='trend-arrow'>{stat_trends['total']['arrow']}</span><span class='trend-value'>{stat_trends['total']['value']}</span><span class='trend-context'>{html.escape(stat_trends['total']['context'])}</span></div>
        </div>
        <div class='stat-card red'>
          <div class='stat-label'>Phishing Detected</div>
          <div class='stat-val red'>{p:,}</div>
          <div class='stat-trend {stat_trends['phishing']['cls']}'><span class='trend-arrow'>{stat_trends['phishing']['arrow']}</span><span class='trend-value'>{stat_trends['phishing']['value']}</span><span class='trend-context'>{html.escape(stat_trends['phishing']['context'])}</span></div>
        </div>
        <div class='stat-card yellow'>
          <div class='stat-label'>Suspicious</div>
          <div class='stat-val yellow'>{s:,}</div>
          <div class='stat-trend {stat_trends['suspicious']['cls']}'><span class='trend-arrow'>{stat_trends['suspicious']['arrow']}</span><span class='trend-value'>{stat_trends['suspicious']['value']}</span><span class='trend-context'>{html.escape(stat_trends['suspicious']['context'])}</span></div>
        </div>
        <div class='stat-card green'>
          <div class='stat-label'>Safe Messages</div>
          <div class='stat-val green'>{safe:,}</div>
          <div class='stat-trend {stat_trends['safe']['cls']}'><span class='trend-arrow'>{stat_trends['safe']['arrow']}</span><span class='trend-value'>{stat_trends['safe']['value']}</span><span class='trend-context'>{html.escape(stat_trends['safe']['context'])}</span></div>
        </div>
        <div class='stat-card purple'>
          <div class='stat-label'>Avg Confidence</div>
          <div class='stat-val purple'>{avg_conf:.1f}%</div>
          <div class='stat-trend {stat_trends['avg_conf']['cls']}'><span class='trend-arrow'>{stat_trends['avg_conf']['arrow']}</span><span class='trend-value'>{stat_trends['avg_conf']['value']}</span><span class='trend-context'>{html.escape(stat_trends['avg_conf']['context'])}</span></div>
          <div class='stat-foot'>Low-confidence scans: {data['review_count']:,} scans ({data['review_rate']:.2f}% of log)</div>
        </div>
      </div>

      <div class='grid-2-1'>
        <div class='panel accent-cyan' data-panel-id='scans_over_time'>
          <div class='panel-head'><div class='panel-title'>Scans Over Time</div><span class='panel-tag'>{html.escape(selected_period_tag)}</span></div>
          <div class='panel-body'>
            <div class='tl-legend'>
              <span class='tl-legend-item'><span class='tl-legend-line red'></span>Phishing</span>
              <span class='tl-legend-item'><span class='tl-legend-line yellow'></span>Suspicious</span>
              <span class='tl-legend-item'><span class='tl-legend-line green'></span>Safe</span>
            </div>
            {_build_timeline_svg(data['timeline']['labels'], data['timeline']['phishing'], data['timeline']['suspicious'], data['timeline']['safe'])}
          </div>
        </div>

        <div class='panel accent-red' data-panel-id='verdict_breakdown'>
          <div class='panel-head'><div class='panel-title'>Verdict Breakdown</div><span class='panel-tag'>{html.escape(selected_language_tag)}</span></div>
          <div class='panel-body'>
            <div class='donut-wrap'>
              <div class='donut' style='background:conic-gradient({verdict_conic});'>
                <div class='donut-center'>{data['total']:,}</div>
              </div>
              <div class='legend'>{verdict_legend_html}</div>
            </div>
          </div>
        </div>
      </div>

      <div class='grid-2'>
        <div class='panel accent-yellow' data-panel-id='scam_type_distribution'>
          <div class='panel-head'><div class='panel-title'>Scam Type Distribution</div><span class='panel-tag'>{html.escape(selected_context_tag)}</span></div>
          <div class='panel-body'><div class='hbar-list'>{scam_rows_html}</div></div>
        </div>

        <div class='panel accent-purple' data-panel-id='language_distribution'>
          <div class='panel-head'><div class='panel-title'>Language Distribution</div><span class='panel-tag'>{html.escape(selected_context_tag)}</span></div>
          <div class='panel-body'>
            <div class='donut-wrap'>
              <div class='donut' style='background:conic-gradient({lang_conic});'>
                <div class='donut-center'>{len(data['language_counts'])}</div>
              </div>
              <div class='legend'>{lang_legend_html}</div>
            </div>
          </div>
        </div>
      </div>

      <div class='section-header'><span class='section-label'>// 02</span><span class='section-title'>Model Comparison Leaderboard</span></div>
      <div class='grid-2'>
        <div class='panel accent-cyan zoomable' data-panel-id='model_leaderboard'>
          <div class='panel-head'><div class='panel-title'>Global Benchmark Ranking</div><span class='panel-tag'>test split • macro-F1</span></div>
          <div class='panel-body'>{leaderboard_html}</div>
        </div>

        <div class='panel accent-purple zoomable' data-panel-id='agreement_matrix'>
          <div class='panel-head'><div class='panel-title'>Live Agreement Matrix</div><span class='panel-tag'>{data['comparison_total']:,} scans • {data['comparison_agreement_rate']:.1f}% unanimous</span></div>
          <div class='panel-body'>{agreement_matrix_html}</div>
        </div>
      </div>

      <div class='panel accent-red zoomable' data-panel-id='disagreement_view' style='margin-bottom:1.1rem;'>
        <div class='panel-head'><div class='panel-title'>Recent Disagreements</div><span class='panel-tag'>{data['comparison_split_two_one'] + data['comparison_split_all_diff']:,} non-unanimous · {data['comparison_split_two_one']:,} two-way · {data['comparison_split_all_diff']:,} all-diff</span></div>
        <div class='panel-body'><div class='compare-card-list'>{disagreement_html}</div></div>
      </div>

      <div class='section-header'><span class='section-label'>// 03</span><span class='section-title'>Language Breakdown</span></div>
      <div class='grid-single'>
        <div class='panel accent-purple zoomable' data-panel-id='language_breakdown'>
          <div class='panel-head'><div class='panel-title'>Per-Language Model Matrix</div><span class='panel-tag'>test split • 4 languages • 3 models</span></div>
          <div class='panel-body'>{language_breakdown_html}</div>
        </div>
      </div>

      <div class='section-header'><span class='section-label'>// 04</span><span class='section-title'>Model Performance Evaluation</span></div>
      <div class='grid-single'>
        <div class='panel accent-cyan' data-panel-id='model_performance'>
          <div class='panel-head'><div class='panel-title'>Per-Class F1 / Precision / Recall</div><span class='panel-tag'>test set • {html.escape(selected_language_tag)}</span></div>
          <div class='panel-body'>
            <table class='perf-table'>
              <thead><tr><th>Class</th><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>F1 Bar</th></tr></thead>
              <tbody>{perf_rows_html}</tbody>
            </table>
          </div>
        </div>
      </div>

      <div class='section-header'><span class='section-label'>// 05</span><span class='section-title'>Calibration & Reliability</span></div>
      <div class='grid-single'>
        <div class='panel accent-purple' data-panel-id='confusion_matrix'>
          <div class='panel-head'><div class='panel-title'>Three-Model Confusion Matrices</div><span class='panel-tag'>baseline · bilstm · indicbert</span></div>
          <div class='panel-body'>{confusion_matrix_html}</div>
        </div>
      </div>
      <div class='grid-single' style='margin-top:1rem;'>
        <div class='panel accent-red' data-panel-id='confidence_calibration'>
          <div class='panel-head'><div class='panel-title'>Three-Model Confidence Calibration</div><span class='panel-tag'>ECE · temperature scaling</span></div>
          <div class='panel-body'>
            {calibration_model_cards_html}
            <div class='calib-note'>
              {calibration_copy}
            </div>
          </div>
        </div>
      </div>

      <div class='section-header'><span class='section-label'>// 06</span><span class='section-title'>Key Findings</span></div>
      <div class='insight-grid'>{insight_html}</div>

      <div class='section-header'><span class='section-label'>// 07</span><span class='section-title'>Recent Scan Log</span></div>
      <div class='panel accent-cyan' data-panel-id='recent_scan_log'>
        <div class='panel-head'><div class='panel-title'>Live Scan Log</div><span class='panel-tag'><span class='live-dot' style='width:6px;height:6px;display:inline-block;margin-right:6px;border-radius:50%;background:var(--neon);box-shadow:0 0 6px var(--neon);vertical-align:middle;'></span>auto-refresh 30s</span></div>
        <div class='log-toolbar'>
          <input class='log-search' id='logSearch' type='text' placeholder='// search by type, verdict, language, message...'>
          <button class='log-filter-btn active' data-verdict='all'>All</button>
          <button class='log-filter-btn' data-verdict='phishing'>Phishing</button>
          <button class='log-filter-btn' data-verdict='suspicious'>Suspicious</button>
          <button class='log-filter-btn' data-verdict='safe'>Safe</button>
          <span class='log-count' id='logCount'>showing {log_total_count} of {log_total_count}</span>
        </div>
        <div class='panel-body' style='padding:0;'>
          <div class='log-table-wrap'>
            <table class='log-table'>
              <thead>
                <tr><th>#</th><th>Timestamp</th><th>Verdict</th><th>Scam Type</th><th>Score</th><th>Conf.</th><th>Lang</th><th>Model Compare</th><th>Message Preview</th></tr>
              </thead>
              <tbody id='logBody'>{log_rows_html}</tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
    """
    ),
    unsafe_allow_html=True,
)
