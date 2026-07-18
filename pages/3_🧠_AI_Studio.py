from __future__ import annotations

import html
import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from textwrap import dedent, shorten

import streamlit as st
import streamlit.components.v1 as components

try:
    import joblib
except Exception:  # pragma: no cover - optional artifact reader
    joblib = None

from src import model_runtime as runtime_utils
from src.db import init_db, read_scans
from src.explainability import cue_labels, detect_cue_tags, infer_message_language_name, language_code
from src.ui_theme import apply_theme, top_menu


ROOT = Path(__file__).resolve().parents[1]
SCORE_FUSION_MODEL_PATH = ROOT / "data" / "models" / "score_fusion_model.joblib"
SCORE_FUSION_METRICS_PATH = ROOT / "data" / "models" / "score_fusion_metrics.json"
BASE_MODEL_ORDER = ("baseline_v3", "bilstm_v3", "indicbert_v3")
ATTRIBUTION_MAX_SEGMENTS = 8


SCAN_COLUMNS = [
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


def _scan_dicts(limit: int = 200) -> list[dict]:
    rows = read_scans(limit=limit)
    return [dict(zip(SCAN_COLUMNS, row)) for row in rows]


@st.cache_data(show_spinner=False)
def _load_score_fusion_metrics() -> dict:
    if not SCORE_FUSION_METRICS_PATH.exists():
        return {}
    try:
        data = json.loads(SCORE_FUSION_METRICS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@st.cache_resource(show_spinner=False)
def _load_score_fusion_bundle() -> dict:
    if joblib is None or not SCORE_FUSION_MODEL_PATH.exists():
        return {}
    try:
        bundle = joblib.load(SCORE_FUSION_MODEL_PATH)
        return bundle if isinstance(bundle, dict) else {}
    except Exception:
        return {}


def _clear_old_browser_refresh_timer() -> None:
    components.html(
        """
        <script>
        (function () {
          try {
            const parentWin = window.parent;
            const timerKey = "__safesandeshAiStudioLiveTimer";
            if (parentWin[timerKey]) {
              parentWin.clearTimeout(parentWin[timerKey]);
              parentWin[timerKey] = null;
            }
            parentWin.sessionStorage.removeItem("safesandesh_ai_studio_scroll_y");
          } catch (err) {}
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def _confidence_pct(value: object) -> float:
    try:
        val = float(value or 0.0)
    except Exception:
        val = 0.0
    if val <= 1.0:
        val *= 100.0
    return max(0.0, min(100.0, val))


def _score_int(value: object) -> int:
    try:
        return max(0, min(100, int(round(float(value or 0)))))
    except Exception:
        return 0


def _label_class(label: object) -> str:
    text = str(label or "").strip().lower()
    if "phish" in text:
        return "phishing"
    if "susp" in text:
        return "suspicious"
    return "safe"


def _score_band(score: int) -> str:
    if score >= 81:
        return "Critical"
    if score >= 61:
        return "High"
    if score >= 31:
        return "Suspicious"
    return "Safe"


def _risk_class_from_score(score: int) -> str:
    if score >= 81:
        return "critical"
    if score >= 61:
        return "phishing"
    if score >= 31:
        return "suspicious"
    return "safe"


def _compact_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _fmt_ts(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "latest scan"
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return text


def _short_time(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown time"
    try:
        return datetime.fromisoformat(text).strftime("%b %d · %H:%M")
    except Exception:
        return text.replace("T", " ")[:16]


def _latest_scan(scans: list[dict]) -> dict | None:
    if not scans:
        return None
    return scans[0]


def _lang_code(lang_name: object) -> str:
    return language_code(str(lang_name or "English"))


def _stored_model_outputs(scan: dict) -> list[dict]:
    raw = str(scan.get("model_outputs_json") or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except Exception:
            pass
    return []


def _is_score_layer(version: object, source: object = "", kind: object = "") -> bool:
    text = " ".join(str(value or "").strip().lower() for value in (version, source, kind))
    return any(marker in text for marker in ("ensemble", "score_fusion", "score fusion", "score_ensemble", "median_"))


def _base_model_sort_key(item: dict) -> tuple[int, str]:
    version = str(item.get("model_version") or item.get("version") or "").lower()
    source = str(item.get("model_source") or item.get("source") or "").lower()
    joined = f"{version} {source}"
    for idx, marker in enumerate(("baseline", "bilstm", "indicbert")):
        if marker in joined:
            return (idx, version or source)
    return (len(BASE_MODEL_ORDER), version or source)


def _base_model_outputs(scan: dict) -> list[dict]:
    stored_outputs = _stored_model_outputs(scan)
    candidates: list[dict] = []
    if stored_outputs:
        candidates.extend(stored_outputs)
    else:
        candidates.extend(
            [
                {
                    "model_source": str(scan.get("model_source") or "unknown"),
                    "model_version": str(scan.get("model_version") or scan.get("model_source") or "unknown"),
                    "label": str(scan.get("label") or "Unknown"),
                    "scam_type": str(scan.get("scam_type") or "Other"),
                    "risk_score": _score_int(scan.get("risk_score")),
                    "model_confidence": _confidence_pct(scan.get("model_confidence")) / 100.0,
                    "type_source": str(scan.get("type_source") or "unknown"),
                    "reason": str(scan.get("reason") or ""),
                },
                {
                    "model_source": str(scan.get("comparison_model_source") or "unknown"),
                    "model_version": str(scan.get("comparison_model_version") or scan.get("comparison_model_source") or "unknown"),
                    "label": str(scan.get("comparison_label") or "Unknown"),
                    "scam_type": str(scan.get("comparison_scam_type") or "Other"),
                    "risk_score": _score_int(scan.get("comparison_risk_score")),
                    "model_confidence": _confidence_pct(scan.get("comparison_model_confidence")) / 100.0,
                    "type_source": str(scan.get("comparison_type_source") or "unknown"),
                    "reason": "",
                },
                {
                    "model_source": str(scan.get("comparison_tertiary_model_source") or "unknown"),
                    "model_version": str(scan.get("comparison_tertiary_model_version") or scan.get("comparison_tertiary_model_source") or "unknown"),
                    "label": str(scan.get("comparison_tertiary_label") or "Unknown"),
                    "scam_type": str(scan.get("comparison_tertiary_scam_type") or "Other"),
                    "risk_score": _score_int(scan.get("comparison_tertiary_risk_score")),
                    "model_confidence": _confidence_pct(scan.get("comparison_tertiary_model_confidence")) / 100.0,
                    "type_source": str(scan.get("comparison_tertiary_type_source") or "unknown"),
                    "reason": "",
                },
            ]
        )

    seen: set[str] = set()
    base_outputs: list[dict] = []
    for item in sorted(candidates, key=_base_model_sort_key):
        version = str(item.get("model_version") or item.get("version") or item.get("model_source") or "unknown")
        source = str(item.get("model_source") or item.get("source") or "unknown")
        kind = str(item.get("model_kind") or "")
        if _is_score_layer(version, source, kind):
            continue
        key = f"{version.lower()}::{source.lower()}"
        if key in seen:
            continue
        seen.add(key)
        base_outputs.append(item)
    return base_outputs[:3]


def _base_model_role(item: dict, idx: int) -> tuple[str, str]:
    text = f"{item.get('model_version', '')} {item.get('model_source', '')}".lower()
    if "baseline" in text:
        return ("Baseline model", "BASELINE")
    if "bilstm" in text:
        return ("BiLSTM model", "BILSTM")
    if "indicbert" in text or "transformer" in text:
        return ("IndicBERT model", "INDICBERT")
    role_names = ["Base model 1", "Base model 2", "Base model 3"]
    role_tags = ["MODEL 1", "MODEL 2", "MODEL 3"]
    return (role_names[idx] if idx < len(role_names) else f"Base model {idx + 1}", role_tags[idx] if idx < len(role_tags) else f"MODEL {idx + 1}")


def _score_recipe(scan: dict, score: int) -> tuple[str, str]:
    method = str(scan.get("final_score_method") or scan.get("model_version") or "primary_model")
    outputs = _base_model_outputs(scan)
    if outputs:
        score_parts = [
            f"{item.get('model_version', item.get('model_source', 'model'))}={_score_int(item.get('risk_score'))}"
            for item in outputs[:3]
        ]
        scores = [_score_int(item.get("risk_score")) for item in outputs[:3]]
        if method.startswith("score_fusion"):
            return (
                "Score Fusion",
                f"score_fusion_model_v1([{', '.join(str(v) for v in scores)}]) = {score}/100",
            )
        return (
            "Median Ensemble",
            f"median({', '.join(score_parts)}) = {score}/100",
        )
    return ("P(Phishing)", f"risk score = round(P(Phishing) × 100) = {score}/100")


def _model_rows(scan: dict, message: str) -> list[dict]:
    base_outputs = _base_model_outputs(scan)
    if base_outputs:
        rows = []
        for idx, item in enumerate(base_outputs[:3]):
            role, tag = _base_model_role(item, idx)
            rows.append(
                {
                    "role": role,
                    "tag": tag,
                    "version": str(item.get("model_version") or item.get("model_source") or "unknown"),
                    "source": str(item.get("model_source") or "unknown"),
                    "label": str(item.get("label") or "Unknown"),
                    "score": _score_int(item.get("risk_score")),
                    "confidence": _confidence_pct(item.get("model_confidence")),
                    "reason": str(item.get("reason") or "Stored base-model vote. Compare it against the final ensemble score."),
                    "cue_labels": cue_labels(list(dict.fromkeys(detect_cue_tags(message))), _lang_code(infer_message_language_name(message, fallback=str(scan.get("language") or "English")))),
                }
            )
        return rows

    shared_tags = list(dict.fromkeys(detect_cue_tags(message)))
    message_lang = infer_message_language_name(message, fallback=str(scan.get("language") or "English"))
    shared_labels = cue_labels(shared_tags, _lang_code(message_lang))
    fallback_reason = "Model vote stored without a separate explanation; compare its score against the primary explanation and cue evidence."
    return [
        {
            "role": "Primary model",
            "tag": "PRIMARY",
            "version": str(scan.get("model_version") or scan.get("model_source") or "unknown"),
            "source": str(scan.get("model_source") or "unknown"),
            "label": str(scan.get("label") or "Unknown"),
            "score": _score_int(scan.get("risk_score")),
            "confidence": _confidence_pct(scan.get("model_confidence")),
            "reason": str(scan.get("reason") or fallback_reason),
            "cue_labels": shared_labels,
        },
        {
            "role": "Secondary model",
            "tag": "SECONDARY",
            "version": str(scan.get("comparison_model_version") or scan.get("comparison_model_source") or "unknown"),
            "source": str(scan.get("comparison_model_source") or "unknown"),
            "label": str(scan.get("comparison_label") or "Unknown"),
            "score": _score_int(scan.get("comparison_risk_score")),
            "confidence": _confidence_pct(scan.get("comparison_model_confidence")),
            "reason": fallback_reason,
            "cue_labels": shared_labels,
        },
        {
            "role": "Tertiary model",
            "tag": "TERTIARY",
            "version": str(scan.get("comparison_tertiary_model_version") or scan.get("comparison_tertiary_model_source") or "unknown"),
            "source": str(scan.get("comparison_tertiary_model_source") or "unknown"),
            "label": str(scan.get("comparison_tertiary_label") or "Unknown"),
            "score": _score_int(scan.get("comparison_tertiary_risk_score")),
            "confidence": _confidence_pct(scan.get("comparison_tertiary_model_confidence")),
            "reason": fallback_reason,
            "cue_labels": shared_labels,
        },
    ]


def _cue_chips_html(labels: list[str], limit: int = 8) -> str:
    clean = [str(label or "").strip() for label in labels if str(label or "").strip()]
    if not clean:
        return "<span class='studio-cue muted'>No explicit cue matched</span>"
    return "".join(f"<span class='studio-cue'>{html.escape(label)}</span>" for label in clean[:limit])


def _model_card(role: str, version: object, source: object, label: object, score: object, confidence: object) -> str:
    label_text = str(label or "Unknown")
    label_cls = _label_class(label_text)
    return f"""
    <article class="studio-model-card {label_cls}">
      <div class="studio-model-role">{html.escape(role)}</div>
      <div class="studio-model-top">
        <span class="studio-model-name">{html.escape(str(version or source or 'unknown'))}</span>
        <span class="studio-model-score">{_score_int(score)}/100</span>
      </div>
      <div class="studio-model-meta">
        {html.escape(str(source or 'unknown'))} · {html.escape(label_text)} · confidence {_confidence_pct(confidence):.1f}%
      </div>
    </article>
    """


def _model_card_from_row(row: dict) -> str:
    return _model_card(
        str(row.get("role") or "Model"),
        row.get("version"),
        row.get("source"),
        row.get("label"),
        row.get("score"),
        row.get("confidence"),
    )


def _median_score(scores: list[int]) -> int:
    if not scores:
        return 0
    ordered = sorted(_score_int(score) for score in scores)
    return ordered[len(ordered) // 2]


def _score_fusion_scan_score(rows: list[dict]) -> int | None:
    bundle = _load_score_fusion_bundle()
    model = bundle.get("model")
    if model is None:
        return None

    by_feature: dict[str, int] = {}
    for row in rows:
        text = f"{row.get('version', '')} {row.get('source', '')}".lower()
        if "baseline" in text:
            by_feature["baseline_score"] = _score_int(row.get("score"))
        elif "bilstm" in text:
            by_feature["bilstm_score"] = _score_int(row.get("score"))
        elif "indicbert" in text or "transformer" in text:
            by_feature["indicbert_score"] = _score_int(row.get("score"))

    features = bundle.get("features") or ["baseline_score", "bilstm_score", "indicbert_score"]
    if any(feature not in by_feature for feature in features):
        return None

    try:
        x = [[float(by_feature[feature]) / 100.0 for feature in features]]
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(x)[0][1])
        else:
            prob = float(model.predict(x)[0])
        return _score_int(prob * 100.0)
    except Exception:
        return None


def _fmt_eval_metric(metrics: dict, key: str, decimals: int = 3) -> str:
    try:
        value = float(metrics.get(key))
    except Exception:
        return "n/a"
    return f"{value:.{decimals}f}"


def _build_eval_card(
    *,
    name: str,
    score: int | None,
    metrics: dict,
    status: str,
    detail: str,
    selected: bool,
) -> str:
    score_text = f"{score}/100" if score is not None else "n/a"
    cls = "selected" if selected else "experimental"
    return f"""
    <article class="studio-eval-card {cls}">
      <div class="studio-eval-head">
        <div>
          <div class="studio-eval-status">{html.escape(status)}</div>
          <div class="studio-eval-name">{html.escape(name)}</div>
        </div>
        <div class="studio-eval-score">{html.escape(score_text)}</div>
      </div>
      <div class="studio-eval-detail">{html.escape(detail)}</div>
      <div class="studio-eval-metrics">
        <span>Accuracy {_fmt_eval_metric(metrics, 'accuracy')}</span>
        <span>Macro-F1 {_fmt_eval_metric(metrics, 'macro_f1')}</span>
        <span>ECE {_fmt_eval_metric(metrics, 'ece', decimals=4)}</span>
      </div>
    </article>
    """


def _build_final_scoring_evaluation_html(scan: dict, message: str) -> str:
    rows = _model_rows(scan, message)
    scores = [row["score"] for row in rows]
    score_parts = ", ".join(f"{row['version']}={row['score']}" for row in rows) or "no base-model scores"
    median_score = _median_score(scores)
    fusion_score = _score_fusion_scan_score(rows)
    final_score = _score_int(scan.get("risk_score"))
    final_method = str(scan.get("final_score_method") or scan.get("model_version") or "primary_model")
    metrics = _load_score_fusion_metrics()
    recommendation = str(metrics.get("recommendation") or "median_ensemble_v1")
    median_metrics = metrics.get("median") if isinstance(metrics.get("median"), dict) else {}
    fusion_metrics = metrics.get("fusion") if isinstance(metrics.get("fusion"), dict) else {}
    fusion_deployed = bool(_load_score_fusion_bundle().get("deploy_as_final", False))

    median_detail = f"Current scan formula: median({score_parts}) = {median_score}/100."
    fusion_detail = (
        "Experimental learned scorer trained on the same three base scores; shown for evaluation, not mixed into the detector stack."
        if fusion_score is not None
        else "Experimental learned scorer artifact is unavailable for this scan."
    )
    note = (
        f"Deployed final score is {final_method} at {final_score}/100. "
        f"Saved evaluation recommendation: {recommendation}. "
        "This compares scoring methods only; the classifier cards above remain the three actual models."
    )
    return f"""
    <section class="studio-panel studio-eval">
      <div class="studio-panel-head">
        <span>Final Scoring Evaluation</span>
        <span>median vs learned fusion</span>
      </div>
      <div class="studio-eval-grid">
        {_build_eval_card(name='median_ensemble_v1', score=median_score, metrics=median_metrics, status='DEPLOYED FINAL' if final_method.startswith('median') else 'EVALUATION', detail=median_detail, selected=final_method.startswith('median'))}
        {_build_eval_card(name='score_fusion_model_v1', score=fusion_score, metrics=fusion_metrics, status='DEPLOYED FINAL' if fusion_deployed else 'EXPERIMENTAL', detail=fusion_detail, selected=fusion_deployed)}
      </div>
      <div class="studio-eval-note">{html.escape(note)}</div>
    </section>
    """


def _cue_html(message: str) -> str:
    lang_name = infer_message_language_name(message)
    labels = cue_labels(list(dict.fromkeys(detect_cue_tags(message))), _lang_code(lang_name))
    return _cue_chips_html(labels)


TRAJECTORY_TAG_WEIGHTS = {
    "OTP_REQUEST": 24.0,
    "KYC_UPDATE": 20.0,
    "PAYMENT_REQUEST": 22.0,
    "ACCOUNT_THREAT": 18.0,
    "LINK_PRESENT": 10.0,
    "LOTTERY_PRIZE": 16.0,
    "DELIVERY_CUSTOMS": 12.0,
    "JOB_LOAN": 10.0,
}

TRAJECTORY_TAG_LABELS = {
    "OTP_REQUEST": "OTP request",
    "KYC_UPDATE": "KYC update",
    "PAYMENT_REQUEST": "Payment/UPI request",
    "ACCOUNT_THREAT": "Account threat",
    "LINK_PRESENT": "Suspicious link",
    "LOTTERY_PRIZE": "Prize/lottery lure",
    "DELIVERY_CUSTOMS": "Delivery/customs pretext",
    "JOB_LOAN": "Job/loan lure",
}

TRAJECTORY_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।॥؟۔])\s+|\n+")
TRAJECTORY_CLAUSE_SPLIT_RE = re.compile(r"[;,؛،:：]\s+")
TRAJECTORY_STEP_MARKER_RE_BY_LANG = {
    "hi": re.compile(
        r"(अत्यावश्यक|तत्काल|तुरंत|अस्थायी\s*रूप\s*से|ब्लॉक|सत्यापित|सत्यापन|"
        r"लिंक|क्लिक\s*करें|ओटीपी|OTP|साझा\s*करें|अपडेट\s*करें|के\s*लिए|और|फिर|तब)",
        re.IGNORECASE,
    ),
    "pa": re.compile(
        r"(ਤੁਰੰਤ|ਅਸਥਾਈ\s*ਤੌਰ\s*ਤੇ|ਬਲਾਕ|ਤਸਦੀਕ|ਲਿੰਕ|ਕਲਿੱਕ\s*ਕਰੋ|ਓਟੀਪੀ|OTP|"
        r"ਸਾਂਝਾ\s*ਕਰੋ|ਅਪਡੇਟ\s*ਕਰੋ|ਲਈ|ਅਤੇ|ਫਿਰ)",
        re.IGNORECASE,
    ),
    "ur": re.compile(
        r"(فوری|فوراً|عارضی\s*طور\s*پر|بلاک|تصدیق|لنک|کلک\s*کریں|"
        r"او\s*ٹی\s*پی|اوٹیپی|OTP|شیئر\s*کریں|اپڈیٹ\s*کریں|کے\s*لیے|اور|پھر)",
        re.IGNORECASE,
    ),
}

TRAJECTORY_URGENT_RE = re.compile(
    r"(urgent|immediately|now|today|blocked|suspended|verify|update|claim|reward|prize|"
    r"अत्यावश्यक|तत्काल|तुरंत|ब्लॉक|सत्यापित|सत्यापन|"
    r"ਤੁਰੰਤ|ਬਲਾਕ|ਤਸਦੀਕ|"
    r"فوری|فوراً|بلاک|تصدیق)",
    re.IGNORECASE,
)
TRAJECTORY_LINK_RE = re.compile(r"https?://|www\.|\b[a-z0-9.-]+\.[a-z]{2,}/[^\s<]*", re.IGNORECASE)
TRAJECTORY_MONEY_RE = re.compile(r"(?:₹|\$|rs\.?|inr)\s?\d|\b\d{4,}\b", re.IGNORECASE)
TRAJECTORY_ACTION_RE = re.compile(
    r"(click|tap|send|share|pay|accept|open|submit|download|install|scan|verify|update|"
    r"क्लिक\s*करें|साझा\s*करें|उपयोग\s*करें|सत्यापित\s*करें|अपडेट\s*करें|"
    r"ਕਲਿੱਕ\s*ਕਰੋ|ਸਾਂਝਾ\s*ਕਰੋ|ਉਪਯੋਗ\s*ਕਰੋ|ਤਸਦੀਕ\s*ਕਰੋ|"
    r"کلک\s*کریں|شیئر\s*کریں|استعمال\s*کریں|تصدیق\s*کریں)",
    re.IGNORECASE,
)


def _split_on_marker_boundaries(text: str, marker_re: re.Pattern) -> list[str]:
    matches = list(marker_re.finditer(text))
    if len(matches) <= 1:
        return [text]

    segments: list[str] = []
    last = 0
    for match in matches:
        piece = text[last : match.start()].strip()
        if piece:
            segments.append(piece)
        last = match.start()
    tail = text[last:].strip()
    if tail:
        segments.append(tail)
    return segments


def _merge_short_segments(segments: list[str], min_chars: int = 12) -> list[str]:
    merged: list[str] = []
    for segment in segments:
        clean = str(segment or "").strip()
        if not clean:
            continue
        if merged and len(clean) <= min_chars and not detect_cue_tags(clean):
            merged[-1] = f"{merged[-1]} {clean}".strip()
        else:
            merged.append(clean)
    return merged


def _split_trajectory_segments(message: str, lang_name: str) -> list[str]:
    text = str(message or "").strip()
    if not text:
        return []

    segments = [seg.strip() for seg in TRAJECTORY_SENTENCE_SPLIT_RE.split(text) if seg and seg.strip()]
    if len(segments) <= 1 and len(text) > 90:
        clause_segments = [seg.strip() for seg in TRAJECTORY_CLAUSE_SPLIT_RE.split(text) if seg and seg.strip()]
        if len(clause_segments) > 1:
            segments = clause_segments

    marker = TRAJECTORY_STEP_MARKER_RE_BY_LANG.get(_lang_code(lang_name))
    if marker is not None:
        refined: list[str] = []
        for segment in segments:
            pieces = _split_on_marker_boundaries(segment, marker) if len(segment) > 18 else [segment]
            refined.extend(pieces)
        segments = _merge_short_segments(refined)

    if len(segments) <= 1 and len(text) > 120:
        words = text.split()
        chunk_count = 4 if len(words) > 28 else 3 if len(words) > 16 else 2
        chunk_size = max(1, (len(words) + chunk_count - 1) // chunk_count)
        segments = [" ".join(words[i : i + chunk_size]).strip() for i in range(0, len(words), chunk_size)]

    if len(segments) > 8:
        segments = segments[:7] + [" ".join(segments[7:])]
    return [seg for seg in segments if seg]


def _segment_evidence_parts(segment: str, tags: list[str]) -> tuple[float, list[str]]:
    text = str(segment or "")
    parts: list[tuple[str, float]] = []
    for tag in tags:
        weight = TRAJECTORY_TAG_WEIGHTS.get(tag, 4.0)
        parts.append((TRAJECTORY_TAG_LABELS.get(tag, tag.replace("_", " ").title()), weight))
    if len(tags) > 1:
        parts.append(("Cue combination", 3.0 * (len(tags) - 1)))
    if TRAJECTORY_URGENT_RE.search(text):
        parts.append(("Urgency/threat wording", 6.0))
    if TRAJECTORY_ACTION_RE.search(text):
        parts.append(("Action request", 3.0))
    if TRAJECTORY_LINK_RE.search(text):
        parts.append(("Link pattern", 5.0))
    if TRAJECTORY_MONEY_RE.search(text):
        parts.append(("Money/code pattern", 2.0))
    if len(text) > 120:
        parts.append(("Long message", 1.5))
    total = sum(value for _, value in parts)
    if not parts:
        return 0.0, ["No explicit cue points"]
    formula = [f"{label} +{value:g}" for label, value in parts]
    return total, formula


def _allocate_proportional_points(raw_weights: list[float], target_score: int) -> list[int]:
    target = max(0, min(100, int(target_score or 0)))
    if not raw_weights:
        return []
    if target <= 0:
        return [0 for _ in raw_weights]

    total = sum(raw_weights)
    if total <= 0:
        base = target // len(raw_weights)
        remainder = target % len(raw_weights)
        return [base + (1 if idx < remainder else 0) for idx in range(len(raw_weights))]

    scaled = [target * (weight / total) for weight in raw_weights]
    floors = [int(value) for value in scaled]
    remainder = target - sum(floors)
    order = sorted(range(len(scaled)), key=lambda idx: (scaled[idx] - floors[idx], raw_weights[idx]), reverse=True)
    for idx in order[:remainder]:
        floors[idx] += 1
    return floors


def _remove_segment_once(message: str, segment: str) -> str:
    source = str(message or "")
    piece = str(segment or "").strip()
    if not piece:
        return source.strip()
    if piece in source:
        reduced = source.replace(piece, " ", 1)
    else:
        reduced = source
    return re.sub(r"\s+", " ", reduced).strip()


@st.cache_data(show_spinner=False)
def _score_occluded_message(message: str, lang_name: str) -> tuple[int, str]:
    text = str(message or "").strip()
    if not text:
        return 0, "empty message"
    try:
        comparison = runtime_utils.compare_all_model_predictions(text, output_language=lang_name)
        final = comparison.get("final") or comparison.get("primary") or {}
        method = str(comparison.get("final_score_method") or final.get("final_score_method") or final.get("model_version") or "model score")
        return _score_int(final.get("risk_score")), method
    except Exception as exc:
        return 0, f"attribution scorer unavailable: {type(exc).__name__}"


def _build_occlusion_attribution(
    *,
    message: str,
    lang_name: str,
    segments: list[str],
    target_score: int,
) -> tuple[list[dict], int, str]:
    target = max(0, min(100, int(target_score or 0)))
    if not segments or target <= 0:
        return [], 0, "No positive final score to attribute."

    attribution_rows: list[dict] = []
    impacts: list[float] = []
    for segment in segments[:ATTRIBUTION_MAX_SEGMENTS]:
        reduced_message = _remove_segment_once(message, segment)
        without_score, method = _score_occluded_message(reduced_message, lang_name)
        impact = max(0, target - without_score)
        impacts.append(float(impact))
        attribution_rows.append(
            {
                "without_score": without_score,
                "impact": impact,
                "method": method,
            }
        )

    impact_total = sum(impacts)
    if impact_total <= 0:
        return attribution_rows, target, (
            "Removing one chunk at a time did not lower the score. "
            "The risk comes from the message pattern as a whole."
        )

    visible_target = min(target, int(round(impact_total)))
    attributed_lifts = _allocate_proportional_points(impacts, visible_target)
    for row, lift in zip(attribution_rows, attributed_lifts):
        row["lift"] = int(lift)

    residual = max(0, target - sum(attributed_lifts))
    if residual:
        note = (
            f"The chunks explain {sum(attributed_lifts)}/100 on their own. "
            f"The remaining {residual}/100 is a combined risk effect: the message is riskier when these parts appear together."
        )
    elif impact_total > target:
        note = "Occlusion impacts exceeded the final score, so they were scaled down to match the deployed score."
    else:
        note = "Occlusion impacts explain the full deployed score."
    return attribution_rows, residual, note


def _build_lift_bridge(raw_weights: list[float], target_score: int) -> tuple[list[int], int, str]:
    target = max(0, min(100, int(target_score or 0)))
    raw_total = sum(raw_weights)
    if target <= 0:
        return [0 for _ in raw_weights], 0, "Final score is 0, so no lift is assigned."
    if raw_total <= 0:
        return [0 for _ in raw_weights], target, "No explicit cue points found; the lift comes from the model probability/ensemble score."

    visible_target = min(target, int(round(raw_total)))
    cue_lifts = _allocate_proportional_points(raw_weights, visible_target)
    residual = max(0, target - sum(cue_lifts))
    if residual:
        note = f"Visible cue points explain {sum(cue_lifts)}/100; model probability explains the remaining {residual}/100."
    elif raw_total > target:
        note = "Visible cue points exceeded the final score, so they were scaled down to match the model score."
    else:
        note = "Visible cue points explain the full final score."
    return cue_lifts, residual, note


def _build_score_breakdown_html(scan: dict, message: str, lang_name: str) -> str:
    score = _score_int(scan.get("risk_score"))
    recipe_tag, recipe_text = _score_recipe(scan, score)
    segments = _split_trajectory_segments(message, lang_name)
    if not segments:
        return """
        <section class="studio-panel studio-breakdown">
          <div class="studio-panel-head"><span>Score Breakdown</span><span>no message chunks</span></div>
          <div class="studio-empty-note">No message text was stored for this scan.</div>
        </section>
        """

    lang = _lang_code(lang_name)
    rows = []
    raw_weights = []
    for idx, segment in enumerate(segments, start=1):
        tags = list(dict.fromkeys(detect_cue_tags(segment)))
        labels = cue_labels(tags, lang)
        raw_weight, formula_parts = _segment_evidence_parts(segment, tags)
        raw_weights.append(raw_weight)
        rows.append(
            {
                "idx": idx,
                "segment": segment,
                "excerpt": shorten(segment, width=130, placeholder="..."),
                "labels": labels,
                "formula_parts": formula_parts,
                "raw_weight": raw_weight,
                "without_score": None,
                "occlusion_impact": 0,
                "attribution_method": "",
                "lift": 0,
                "cumulative": 0,
                "risk_class": "safe",
            }
        )

    attribution_rows, residual_lift, bridge_note = _build_occlusion_attribution(
        message=message,
        lang_name=lang_name,
        segments=segments,
        target_score=score,
    )
    if len(attribution_rows) == len(rows):
        lifts = [int(item.get("lift", 0) or 0) for item in attribution_rows]
        for row, attribution in zip(rows, attribution_rows):
            row["without_score"] = attribution.get("without_score")
            row["occlusion_impact"] = int(attribution.get("impact", 0) or 0)
            row["attribution_method"] = str(attribution.get("method") or "")
    else:
        lifts, residual_lift, bridge_note = _build_lift_bridge(raw_weights, score)
        for row in rows:
            row["attribution_method"] = "cue-weight fallback"
    if not any(lifts) and residual_lift == score and sum(raw_weights) > 0:
        cue_lifts, residual_lift, bridge_note = _build_lift_bridge(raw_weights, score)
        lifts = cue_lifts
        for row in rows:
            row["attribution_method"] = "cue-weight fallback"

    cumulative = 0
    for row, lift in zip(rows, lifts):
        cumulative = min(100, cumulative + int(lift))
        row["lift"] = int(lift)
        row["cumulative"] = cumulative
        row["risk_class"] = _risk_class_from_score(cumulative)

    final_path_points = [str(row["cumulative"]) for row in rows]
    if residual_lift:
        cumulative = min(100, cumulative + residual_lift)
        final_path_points.append(str(cumulative))
    path = "0 → " + " → ".join(final_path_points)
    strongest = max(rows, key=lambda row: (row["lift"], row["occlusion_impact"], row["raw_weight"]))
    strongest_lift = int(strongest["lift"])
    strongest_cues = ", ".join(strongest["labels"][:4]) if strongest["labels"] else "No explicit cue matched"
    if strongest["without_score"] is not None:
        strongest_source = (
            f"Occlusion: {score} - {int(strongest['without_score'])} = "
            f"+{int(strongest['occlusion_impact'])}; {strongest_cues}"
        )
    else:
        strongest_source = f"{strongest_cues} · cue raw {strongest['raw_weight']:.0f}"
    if residual_lift > strongest_lift:
        strongest_lift = residual_lift
        strongest_source = "Combined risk effect: chunks are riskier together"

    chunk_html = []
    for row in rows:
        cue_html = _cue_chips_html(row["labels"], limit=5)
        formula_text = " + ".join(row["formula_parts"])
        if row["raw_weight"] > 0:
            cue_note = f"Cue formula: {formula_text} = raw {row['raw_weight']:.0f}."
        else:
            cue_note = "Cue formula: no explicit cue points = raw 0."
        if row["without_score"] is not None:
            note = (
                f"Occlusion test: original {score}/100 - without this chunk "
                f"{int(row['without_score'])}/100 = +{int(row['occlusion_impact'])} impact. "
                f"Displayed lift +{row['lift']}. {cue_note}"
            )
        else:
            note = f"Displayed lift +{row['lift']} from cue-weight fallback. {cue_note}"
        chunk_html.append(
            f"""
            <article class="studio-step {row['risk_class']}">
              <div class="studio-step-head">
                <div>
                  <div class="studio-step-k">Chunk {row['idx']} · attribution +{row['lift']} · occlusion +{int(row['occlusion_impact'])} · cue raw {row['raw_weight']:.0f}</div>
                  <div class="studio-step-text">{html.escape(row['excerpt'])}</div>
                </div>
                <div class="studio-step-score">{row['cumulative']}/100</div>
              </div>
              <div class="studio-cue-row">{cue_html}</div>
              <div class="studio-step-bar"><span class="{row['risk_class']}" style="width:{row['cumulative']}%"></span></div>
              <div class="studio-step-note">{html.escape(note)}</div>
            </article>
            """
        )

    residual_html = ""
    if residual_lift:
        residual_score = max(0, min(100, score))
        residual_html = f"""
            <article class="studio-step {_risk_class_from_score(residual_score)}">
              <div class="studio-step-head">
                <div>
                  <div class="studio-step-k">Combined risk effect · +{residual_lift}</div>
                  <div class="studio-step-text">This part is not from one single chunk. It means the message became more risky when the chunks were read together: {html.escape(recipe_text)}.</div>
                </div>
                <div class="studio-step-score">{residual_score}/100</div>
              </div>
              <div class="studio-cue-row"><span class='studio-cue'>COMBINED CONTEXT</span><span class='studio-cue'>CHUNKS TOGETHER</span></div>
              <div class="studio-step-bar"><span class="{_risk_class_from_score(residual_score)}" style="width:{residual_score}%"></span></div>
              <div class="studio-step-note">{html.escape(bridge_note)}</div>
            </article>
        """

    return f"""
    <section class="studio-panel studio-breakdown">
      <div class="studio-panel-head">
        <span>Score Breakdown</span>
        <span>{html.escape(recipe_tag)} · {html.escape(_score_band(score))} band</span>
      </div>
      <div class="studio-breakdown-top">
        <div>
          <div class="studio-breakdown-k">Score recipe</div>
          <div class="studio-breakdown-v">{html.escape(recipe_text)}</div>
          <div class="studio-breakdown-m">Chunk numbers use occlusion attribution: remove a chunk, rescore the same ensemble, and measure how much the final score drops. Cues are shown as supporting evidence, not as the full model explanation.</div>
        </div>
        <div class="studio-final-box">
          <span>Final score</span>
          <strong>{score}/100</strong>
          <small>{len(rows)} chunks · {html.escape(_score_band(score))}</small>
        </div>
      </div>
      <div class="studio-breakdown-cards">
        <div><span>Trajectory path</span><strong>{html.escape(path)}</strong><small>{len(rows)} evidence chunks</small></div>
        <div><span>Strongest jump</span><strong>+{strongest_lift} lift</strong><small>{html.escape(strongest_source)}</small></div>
        <div><span>Why it matters</span><strong>Occlusion attribution</strong><small>Shows which parts changed the model score when removed.</small></div>
      </div>
      <div class="studio-scale"><span>0</span><span>30</span><span>60</span><span>80</span><span>100</span></div>
      <div class="studio-step-list">{''.join(chunk_html)}{residual_html}</div>
    </section>
    """


def _build_model_debate_html(scan: dict, message: str) -> str:
    rows = _model_rows(scan, message)
    labels = [row["label"] for row in rows]
    consensus_label = max(set(labels), key=labels.count) if labels else str(scan.get("label") or "Unknown")
    consensus_count = labels.count(consensus_label)
    agreement_label = "unanimous" if len(set(labels)) == 1 else f"{consensus_count}/{len(rows)} agree"
    score_spread = max(row["score"] for row in rows) - min(row["score"] for row in rows)
    debate_rows = sorted(rows, key=lambda row: (row["score"], row["confidence"]), reverse=True)

    cards = []
    for idx, row in enumerate(debate_rows):
        if idx == 0:
            role = "Prosecution"
            cls = "prosecution"
            opening = f"Highest-risk voice: {row['version']} argues the case should stay elevated at {row['score']}/100."
        elif idx == len(debate_rows) - 1:
            role = "Defense"
            cls = "defense"
            opening = f"Lowest-risk voice: {row['version']} pushes back at {row['score']}/100."
        else:
            role = "Bench"
            cls = "bench"
            opening = f"Middle voice: {row['version']} sits between both sides at {row['score']}/100."
        cards.append(
            f"""
            <article class="studio-debate-card {cls}">
              <div class="studio-debate-head">
                <div>
                  <div class="studio-debate-role">{html.escape(role)}</div>
                  <div class="studio-debate-model">{html.escape(row['version'])}</div>
                  <div class="studio-debate-meta">{html.escape(row['source'])} · {html.escape(row['label'])} · {html.escape(row['tag'])}</div>
                </div>
                <div class="studio-debate-score">{row['score']}/100</div>
              </div>
              <div class="studio-debate-opening">{html.escape(opening)}</div>
              <div class="studio-debate-quote">{html.escape(row['reason'])}</div>
              <div class="studio-debate-line">P(Phishing) ≈ {row['score']:.1f}% · confidence {row['confidence']:.1f}% · {_score_band(row['score'])} band</div>
              <div class="studio-cue-row">{_cue_chips_html(row['cue_labels'], limit=5)}</div>
            </article>
            """
        )

    return f"""
    <section class="studio-panel studio-debate">
      <div class="studio-panel-head">
        <span>Base Model Debate</span>
        <span>{html.escape(agreement_label)}</span>
      </div>
      <div class="studio-debate-grid">{''.join(cards)}</div>
      <div class="studio-judge">
        <div class="studio-judge-k">Judge summary</div>
        <div class="studio-judge-v">{html.escape(consensus_label)} · {html.escape(agreement_label)} · {score_spread} point spread</div>
        <div class="studio-judge-m">This compares the three model outputs for the same scan so the technical side can show agreement, disagreement, and risk spread.</div>
      </div>
    </section>
    """


def _build_case_memory_html(scan: dict, scans: list[dict], message: str, lang_name: str) -> str:
    current_id = str(scan.get("id") or "")
    current_tags = list(dict.fromkeys(detect_cue_tags(message)))
    current_tag_set = set(current_tags)
    current_norm = _compact_text(message)
    current_score = _score_int(scan.get("risk_score"))
    current_label = str(scan.get("label") or "")
    current_type = str(scan.get("scam_type") or "")

    candidates: list[dict] = []
    for other in scans:
        if str(other.get("id") or "") == current_id:
            continue
        other_message = str(other.get("message") or "")
        other_tags = list(dict.fromkeys(detect_cue_tags(other_message)))
        other_tag_set = set(other_tags)
        union = current_tag_set | other_tag_set
        cue_overlap = len(current_tag_set & other_tag_set) / len(union) if union else 0.0
        text_overlap = SequenceMatcher(None, current_norm, _compact_text(other_message)).ratio() if current_norm and other_message else 0.0
        score_closeness = 1.0 - min(abs(current_score - _score_int(other.get("risk_score"))) / 100.0, 1.0)
        same_label = 1.0 if current_label == str(other.get("label") or "") else 0.0
        same_type = 1.0 if current_type == str(other.get("scam_type") or "") and current_type else 0.0
        other_lang = infer_message_language_name(other_message, fallback=str(other.get("language") or "English"))
        same_lang = 1.0 if other_lang == lang_name else 0.0
        similarity = (
            0.40 * cue_overlap
            + 0.22 * text_overlap
            + 0.16 * score_closeness
            + 0.10 * same_type
            + 0.08 * same_label
            + 0.04 * same_lang
        )
        candidates.append(
            {
                "scan": other,
                "similarity": max(0.0, min(1.0, similarity)) * 100.0,
                "shared_labels": cue_labels(list(current_tag_set & other_tag_set), _lang_code(lang_name)),
                "language": other_lang,
                "preview": shorten(other_message, width=110, placeholder="...") if other_message else "No message stored.",
            }
        )

    candidates.sort(key=lambda item: item["similarity"], reverse=True)
    top_cases = candidates[:3]
    if not top_cases:
        case_html = "<div class='studio-empty-note'>No previous scans yet. Run more scans to build the case memory.</div>"
    else:
        cards = []
        for idx, case in enumerate(top_cases, start=1):
            other = case["scan"]
            shared = ", ".join(case["shared_labels"][:4]) if case["shared_labels"] else "No shared explicit cues"
            cards.append(
                f"""
                <article class="studio-memory-card {_label_class(other.get('label'))}">
                  <div class="studio-memory-top">
                    <span>Echo #{idx}</span>
                    <strong>{case['similarity']:.0f}% similar</strong>
                  </div>
                  <div class="studio-memory-meta">{html.escape(_short_time(other.get('ts')))} · {html.escape(case['language'])} · {html.escape(str(other.get('label') or 'Unknown'))} {_score_int(other.get('risk_score'))}/100</div>
                  <div class="studio-memory-preview">{html.escape(case['preview'])}</div>
                  <div class="studio-memory-shared">Shared cues: {html.escape(shared)}</div>
                </article>
                """
            )
        case_html = f"<div class='studio-memory-grid'>{''.join(cards)}</div>"

    return f"""
    <section class="studio-panel studio-memory">
      <div class="studio-panel-head">
        <span>Case Memory</span>
        <span>{len(scans)} scans loaded</span>
      </div>
      <div class="studio-memory-intro">Closest historical scans are ranked using shared cues, text similarity, score closeness, scam type, verdict, and language.</div>
      {case_html}
    </section>
    """


def _recent_case_rows(scans: list[dict]) -> str:
    rows = []
    for scan in scans[:8]:
        label = str(scan.get("label", "Unknown") or "Unknown")
        score = _score_int(scan.get("risk_score"))
        language = infer_message_language_name(str(scan.get("message", "") or ""), fallback=str(scan.get("language", "Unknown")))
        preview = str(scan.get("message", "") or "")
        if len(preview) > 90:
            preview = preview[:87] + "..."
        rows.append(
            f"""
            <tr class="{_label_class(label)}">
              <td>{html.escape(_fmt_ts(scan.get("ts")).split(" ")[-1])}</td>
              <td>{html.escape(label)}</td>
              <td>{score}/100</td>
              <td>{html.escape(language)}</td>
              <td>{html.escape(str(scan.get("scam_type", "Other") or "Other"))}</td>
              <td>{html.escape(preview)}</td>
            </tr>
            """
        )
    return "".join(rows) or "<tr><td colspan='6'>No scans yet.</td></tr>"


def _render_empty() -> None:
    st.html(
        dedent(
            """
        <div class="studio-shell">
          <div class="studio-empty">
            <div class="studio-section-k">// AI STUDIO</div>
            <h1>No scan connected yet</h1>
            <p>Run a message in the consumer detector. This page will then show the latest case, model agreement, score recipe, and evidence trail.</p>
          </div>
        </div>
        """
        ).strip()
    )


def _render_studio(scan: dict, scans: list[dict]) -> None:
    message = str(scan.get("message", "") or "")
    label = str(scan.get("label", "Unknown") or "Unknown")
    score = _score_int(scan.get("risk_score"))
    confidence = _confidence_pct(scan.get("model_confidence"))
    language = infer_message_language_name(message, fallback=str(scan.get("language", "Unknown")))
    scam_type = str(scan.get("scam_type", "Other") or "Other")
    reason = str(scan.get("reason", "") or "No reason text stored for this scan.")
    review_reason = str(scan.get("review_reason", "") or "")
    decision = "Low confidence" if bool(scan.get("review_recommended")) else "Auto decision"
    message_preview = message if len(message) <= 420 else message[:417] + "..."
    final_method = str(scan.get("final_score_method") or scan.get("model_version") or "primary_model")
    recipe_tag, recipe_text = _score_recipe(scan, score)
    score_breakdown = _build_score_breakdown_html(scan, message, language)
    model_debate = _build_model_debate_html(scan, message)
    final_scoring_evaluation = _build_final_scoring_evaluation_html(scan, message)
    case_memory = _build_case_memory_html(scan, scans, message, language)

    model_cards = "".join(_model_card_from_row(row) for row in _model_rows(scan, message))

    st.html(
        dedent(
            f"""
        <div class="studio-shell">
          <section class="studio-hero {_label_class(label)}">
            <div>
              <div class="studio-section-k">// AI STUDIO</div>
              <h1>Latest Case Forensics</h1>
              <p>Technical evidence view connected to the consumer detector. This page explains the score, model agreement, and cue evidence without cluttering the user-facing scanner.</p>
            </div>
            <div class="studio-score-card">
              <div class="studio-score-k">Final risk</div>
              <div class="studio-score-v">{score}/100</div>
              <div class="studio-score-m">{html.escape(label)} · confidence {confidence:.1f}%</div>
            </div>
          </section>

          <section class="studio-grid">
            <article class="studio-panel studio-case">
              <div class="studio-panel-head">
                <span>Connected Scan</span>
                <span>{html.escape(_fmt_ts(scan.get("ts")))}</span>
              </div>
              <div class="studio-case-meta">
                <div><span>Language</span><strong>{html.escape(language)}</strong></div>
                <div><span>Scam type</span><strong>{html.escape(scam_type)}</strong></div>
                <div><span>Final method</span><strong>{html.escape(final_method)}</strong></div>
              </div>
              <div class="studio-message">{html.escape(message_preview)}</div>
              <div class="studio-cue-row">{_cue_html(message)}</div>
            </article>

            <article class="studio-panel studio-recipe">
              <div class="studio-panel-head">
                <span>Score Recipe</span>
                <span>{html.escape(recipe_tag)}</span>
              </div>
              <div class="studio-recipe-main">{html.escape(recipe_text)}</div>
              <div class="studio-reason">Final decision: {html.escape(decision)} · method: {html.escape(final_method)}</div>
              <div class="studio-reason">{html.escape(reason)}</div>
              <div class="studio-review">{html.escape(review_reason or 'No extra confidence warning for this scan.')}</div>
            </article>
          </section>

          {score_breakdown}

          <section class="studio-panel">
            <div class="studio-panel-head">
              <span>Three Base Models</span>
              <span>baseline · bilstm · indicbert</span>
            </div>
            <div class="studio-model-grid">{model_cards}</div>
          </section>

          {final_scoring_evaluation}

          {model_debate}

          {case_memory}

          <section class="studio-panel">
            <div class="studio-panel-head">
              <span>Recent Connected Cases</span>
              <span>{len(scans)} loaded</span>
            </div>
            <div class="studio-table-wrap">
              <table class="studio-table">
                <thead><tr><th>Time</th><th>Verdict</th><th>Score</th><th>Lang</th><th>Type</th><th>Preview</th></tr></thead>
                <tbody>{_recent_case_rows(scans)}</tbody>
              </table>
            </div>
          </section>
        </div>
        """,
        ).strip()
    )


def _render_live_studio() -> None:
    all_scans = _scan_dicts()
    selected_scan = _latest_scan(all_scans)
    if selected_scan is None:
        _render_empty()
    else:
        _render_studio(selected_scan, all_scans)


@st.fragment(run_every="10s")
def _render_live_studio_fragment() -> None:
    _render_live_studio()


st.set_page_config(page_title="AI Studio", layout="wide", initial_sidebar_state="collapsed")
init_db()
apply_theme(home_particles=True)
top_menu("ai_studio")
_clear_old_browser_refresh_timer()

st.html(
    dedent(
        """
    <style>
    .studio-shell {
      max-width: 1440px;
      margin: 0 auto;
      padding: 1.3rem 0 2rem;
    }
    .studio-hero,
    .studio-panel,
    .studio-empty {
      border: 1px solid rgba(181,122,255,0.42);
      background:
        linear-gradient(135deg, rgba(181,122,255,0.08), rgba(0,212,255,0.035)),
        rgba(1,9,15,0.82);
      box-shadow: 0 0 28px rgba(181,122,255,0.14), inset 0 0 0 1px rgba(181,122,255,0.08);
    }
    .studio-hero {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 1rem;
      align-items: center;
      padding: 1.15rem 1.25rem;
      margin-bottom: 1rem;
    }
    .studio-section-k,
    .studio-panel-head,
    .studio-score-k,
    .studio-score-m,
    .studio-model-role,
    .studio-model-meta,
    .studio-case-meta span,
    .studio-table th,
    .studio-table td,
    .studio-reason,
    .studio-review,
    .studio-message,
    .studio-cue {
      font-family: 'Share Tech Mono', monospace;
    }
    .studio-section-k {
      color: #b57aff;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 0.68rem;
    }
    .studio-hero h1,
    .studio-empty h1 {
      margin: 0.22rem 0 0.4rem;
      color: #fff;
      font-family: 'Share Tech Mono', monospace;
      font-size: clamp(1.7rem, 4vw, 3.4rem);
      letter-spacing: 0.055em;
      text-transform: uppercase;
      text-shadow: 0 0 18px rgba(181,122,255,0.32);
    }
    .studio-hero p,
    .studio-empty p {
      color: rgba(214,245,235,0.72);
      line-height: 1.7;
      margin: 0;
      max-width: 760px;
    }
    .studio-score-card {
      border: 1px solid rgba(0,212,255,0.24);
      background: rgba(0,0,0,0.28);
      padding: 1rem;
      text-align: right;
    }
    .studio-score-v {
      font-family: 'Share Tech Mono', monospace;
      font-size: 2.4rem;
      line-height: 1;
      font-weight: 900;
      color: #00d4ff;
      text-shadow: 0 0 16px rgba(0,212,255,0.32);
    }
    .studio-hero.phishing .studio-score-v { color: #ff5577; }
    .studio-hero.suspicious .studio-score-v { color: #ffdd57; }
    .studio-hero.safe .studio-score-v { color: #00ff9f; }
    .studio-score-k,
    .studio-score-m {
      color: rgba(214,245,235,0.64);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.64rem;
    }
    .studio-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.08fr) minmax(340px, 0.92fr);
      gap: 1rem;
      margin-bottom: 1rem;
    }
    .studio-panel,
    .studio-empty {
      padding: 1rem;
      margin-bottom: 1rem;
    }
    .studio-panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      color: #eafff7;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.7rem;
      padding-bottom: 0.72rem;
      border-bottom: 1px solid rgba(0,255,159,0.08);
      margin-bottom: 0.85rem;
    }
    .studio-case-meta {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.65rem;
      margin-bottom: 0.85rem;
    }
    .studio-case-meta div {
      border: 1px solid rgba(0,212,255,0.14);
      background: rgba(0,0,0,0.18);
      padding: 0.62rem;
    }
    .studio-case-meta span {
      display: block;
      color: rgba(214,245,235,0.52);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.58rem;
    }
    .studio-case-meta strong {
      display: block;
      margin-top: 0.2rem;
      color: #fff;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.86rem;
    }
    .studio-message,
    .studio-reason,
    .studio-review {
      color: rgba(214,245,235,0.78);
      line-height: 1.7;
      font-size: 0.78rem;
    }
    .studio-review {
      margin-top: 0.8rem;
      color: rgba(255,221,87,0.78);
    }
    .studio-recipe-main {
      color: #fff;
      font-family: 'Share Tech Mono', monospace;
      font-size: 1.08rem;
      margin-bottom: 0.8rem;
    }
    .studio-cue-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      margin-top: 0.8rem;
    }
    .studio-cue {
      border: 1px solid rgba(255,85,119,0.42);
      background: rgba(255,85,119,0.1);
      color: #ff7d98;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      font-size: 0.62rem;
      padding: 0.24rem 0.45rem;
    }
    .studio-cue.muted {
      border-color: rgba(0,212,255,0.22);
      background: rgba(0,212,255,0.04);
      color: rgba(214,245,235,0.52);
    }
    .studio-model-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.85rem;
    }
    .studio-model-card {
      border: 1px solid rgba(0,212,255,0.18);
      background: rgba(0,0,0,0.24);
      padding: 0.9rem;
    }
    .studio-model-card.phishing { border-color: rgba(255,85,119,0.44); box-shadow: inset 3px 0 0 rgba(255,85,119,0.74); }
    .studio-model-card.suspicious { border-color: rgba(255,221,87,0.42); box-shadow: inset 3px 0 0 rgba(255,221,87,0.72); }
    .studio-model-card.safe { border-color: rgba(0,255,159,0.34); box-shadow: inset 3px 0 0 rgba(0,255,159,0.66); }
    .studio-model-role {
      color: rgba(214,245,235,0.54);
      text-transform: uppercase;
      letter-spacing: 0.13em;
      font-size: 0.6rem;
    }
    .studio-model-top {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 0.8rem;
      margin: 0.25rem 0;
    }
    .studio-model-name,
    .studio-model-score {
      color: #fff;
      font-family: 'Share Tech Mono', monospace;
      font-weight: 900;
    }
    .studio-model-score { color: #00d4ff; font-size: 1.3rem; }
    .studio-model-card.phishing .studio-model-score { color: #ff5577; }
    .studio-model-card.suspicious .studio-model-score { color: #ffdd57; }
    .studio-model-card.safe .studio-model-score { color: #00ff9f; }
    .studio-model-meta {
      color: rgba(214,245,235,0.64);
      line-height: 1.45;
      font-size: 0.7rem;
    }
    .studio-eval {
      border-color: rgba(181,122,255,0.36);
      background:
        linear-gradient(135deg, rgba(181,122,255,0.07), rgba(0,212,255,0.03)),
        rgba(1,9,15,0.84);
      box-shadow: 0 0 26px rgba(181,122,255,0.12), inset 0 0 0 1px rgba(181,122,255,0.07);
    }
    .studio-eval-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.85rem;
    }
    .studio-eval-card {
      border: 1px solid rgba(181,122,255,0.24);
      background: rgba(0,0,0,0.26);
      padding: 0.9rem;
    }
    .studio-eval-card.selected {
      border-color: rgba(0,255,159,0.36);
      box-shadow: inset 3px 0 0 rgba(0,255,159,0.68);
    }
    .studio-eval-card.experimental {
      border-color: rgba(255,221,87,0.30);
      box-shadow: inset 3px 0 0 rgba(255,221,87,0.58);
    }
    .studio-eval-head {
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      align-items: flex-start;
      margin-bottom: 0.7rem;
    }
    .studio-eval-status,
    .studio-eval-detail,
    .studio-eval-metrics,
    .studio-eval-note {
      font-family: 'Share Tech Mono', monospace;
    }
    .studio-eval-status {
      color: rgba(214,245,235,0.58);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.58rem;
    }
    .studio-eval-name,
    .studio-eval-score {
      color: #fff;
      font-family: 'Share Tech Mono', monospace;
      font-weight: 900;
    }
    .studio-eval-name {
      margin-top: 0.22rem;
      font-size: 1rem;
    }
    .studio-eval-score {
      color: #b57aff;
      font-size: 1.35rem;
      white-space: nowrap;
      text-shadow: 0 0 14px rgba(181,122,255,0.26);
    }
    .studio-eval-card.selected .studio-eval-score { color: #00ff9f; }
    .studio-eval-card.experimental .studio-eval-score { color: #ffdd57; }
    .studio-eval-detail,
    .studio-eval-note {
      color: rgba(234,252,247,0.80);
      line-height: 1.6;
      font-size: 0.72rem;
    }
    .studio-eval-metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
      margin-top: 0.75rem;
    }
    .studio-eval-metrics span {
      border: 1px solid rgba(0,212,255,0.18);
      background: rgba(0,212,255,0.05);
      color: rgba(214,245,235,0.72);
      padding: 0.24rem 0.45rem;
      font-size: 0.62rem;
    }
    .studio-eval-note {
      margin-top: 0.85rem;
      color: rgba(214,245,235,0.68);
    }
    .studio-breakdown {
      border-color: rgba(255,221,87,0.32);
      background:
        linear-gradient(135deg, rgba(255,221,87,0.07), rgba(0,212,255,0.025)),
        rgba(1,9,15,0.84);
      box-shadow: 0 0 26px rgba(255,221,87,0.12), inset 0 0 0 1px rgba(255,221,87,0.08);
    }
    .studio-breakdown-top {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 220px;
      gap: 1rem;
      align-items: start;
      margin-bottom: 0.9rem;
    }
    .studio-breakdown-k,
    .studio-breakdown-m,
    .studio-breakdown-cards span,
    .studio-breakdown-cards small,
    .studio-scale,
    .studio-step-k,
    .studio-step-note,
    .studio-memory-intro,
    .studio-memory-meta,
    .studio-memory-shared,
    .studio-debate-role,
    .studio-debate-meta,
    .studio-debate-line,
    .studio-judge-k,
    .studio-judge-m,
    .studio-empty-note {
      font-family: 'Share Tech Mono', monospace;
    }
    .studio-breakdown-k,
    .studio-breakdown-cards span,
    .studio-step-k,
    .studio-debate-role,
    .studio-judge-k,
    .studio-memory-top span {
      color: rgba(214,245,235,0.56);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.58rem;
    }
    .studio-breakdown-v {
      color: #fff;
      font-family: 'Share Tech Mono', monospace;
      font-size: 1.05rem;
      line-height: 1.45;
      margin: 0.25rem 0;
    }
    .studio-breakdown-m,
    .studio-step-note,
    .studio-memory-intro,
    .studio-judge-m,
    .studio-empty-note {
      color: rgba(250,255,252,0.92);
      line-height: 1.72;
      font-size: 0.78rem;
    }
    .studio-final-box {
      border: 1px solid rgba(255,221,87,0.28);
      background: rgba(255,221,87,0.06);
      padding: 0.82rem;
      text-align: right;
    }
    .studio-final-box span,
    .studio-final-box small {
      display: block;
      color: rgba(214,245,235,0.58);
      font-family: 'Share Tech Mono', monospace;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.58rem;
    }
    .studio-final-box strong {
      display: block;
      color: #ffdd57;
      font-family: 'Share Tech Mono', monospace;
      font-size: 2rem;
      line-height: 1;
      margin: 0.28rem 0;
      text-shadow: 0 0 14px rgba(255,221,87,0.28);
    }
    .studio-breakdown-cards {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.75rem;
      margin-bottom: 0.75rem;
    }
    .studio-breakdown-cards div,
    .studio-step,
    .studio-judge,
    .studio-memory-card,
    .studio-debate-card {
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(0,0,0,0.26);
    }
    .studio-breakdown-cards div {
      padding: 0.75rem;
    }
    .studio-breakdown-cards strong {
      display: block;
      color: #fff;
      font-family: 'Share Tech Mono', monospace;
      margin: 0.24rem 0;
      line-height: 1.35;
    }
    .studio-breakdown-cards small {
      color: rgba(214,245,235,0.64);
      line-height: 1.45;
    }
    .studio-scale {
      display: flex;
      justify-content: space-between;
      color: rgba(214,245,235,0.48);
      font-size: 0.56rem;
      letter-spacing: 0.12em;
      margin: 0.35rem 0;
    }
    .studio-step-list {
      display: grid;
      gap: 0.75rem;
    }
    .studio-step {
      padding: 0.82rem;
    }
    .studio-step.safe { border-color: rgba(0,255,159,0.26); }
    .studio-step.suspicious { border-color: rgba(255,221,87,0.30); }
    .studio-step.phishing,
    .studio-step.critical { border-color: rgba(255,85,119,0.36); }
    .studio-step-head,
    .studio-debate-head,
    .studio-memory-top {
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      align-items: flex-start;
    }
    .studio-step-text {
      color: #ffffff;
      font-family: 'Share Tech Mono', monospace;
      font-size: 1.02rem;
      line-height: 1.58;
      margin-top: 0.24rem;
    }
    .studio-step-score {
      color: #ffdd57;
      font-family: 'Share Tech Mono', monospace;
      font-size: 1.15rem;
      font-weight: 900;
      white-space: nowrap;
    }
    .studio-step.safe .studio-step-score { color: #00ff9f; }
    .studio-step.phishing .studio-step-score,
    .studio-step.critical .studio-step-score { color: #ff5577; }
    .studio-step-bar {
      height: 8px;
      background: rgba(255,255,255,0.07);
      overflow: hidden;
      margin: 0.65rem 0 0.4rem;
    }
    .studio-step-bar span {
      display: block;
      height: 100%;
    }
    .studio-step-bar span.safe { background: linear-gradient(90deg, #00ff9f, #00d4ff); }
    .studio-step-bar span.suspicious { background: linear-gradient(90deg, #00ff9f, #ffdd57); }
    .studio-step-bar span.phishing,
    .studio-step-bar span.critical { background: linear-gradient(90deg, #ffdd57, #ff5577); }
    .studio-debate {
      border-color: rgba(0,212,255,0.30);
      box-shadow: 0 0 26px rgba(0,212,255,0.12), inset 0 0 0 1px rgba(0,212,255,0.07);
    }
    .studio-debate-grid,
    .studio-memory-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.85rem;
    }
    .studio-debate-card,
    .studio-memory-card {
      padding: 0.86rem;
    }
    .studio-debate-card.prosecution { border-color: rgba(255,85,119,0.36); box-shadow: inset 3px 0 0 rgba(255,85,119,0.68); }
    .studio-debate-card.bench { border-color: rgba(255,221,87,0.32); box-shadow: inset 3px 0 0 rgba(255,221,87,0.62); }
    .studio-debate-card.defense { border-color: rgba(0,255,159,0.32); box-shadow: inset 3px 0 0 rgba(0,255,159,0.62); }
    .studio-debate-model,
    .studio-debate-score,
    .studio-judge-v,
    .studio-memory-top strong {
      color: #fff;
      font-family: 'Share Tech Mono', monospace;
      font-weight: 900;
    }
    .studio-debate-model {
      font-size: 1rem;
      margin-top: 0.2rem;
    }
    .studio-debate-score {
      color: #ffdd57;
      font-size: 1.25rem;
      white-space: nowrap;
    }
    .studio-debate-card.prosecution .studio-debate-score { color: #ff5577; }
    .studio-debate-card.defense .studio-debate-score { color: #00ff9f; }
    .studio-debate-meta,
    .studio-debate-line {
      color: rgba(214,245,235,0.64);
      line-height: 1.5;
      font-size: 0.66rem;
    }
    .studio-debate-opening {
      color: #eafff7;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.72rem;
      line-height: 1.55;
      margin-top: 0.75rem;
    }
    .studio-debate-quote {
      color: rgba(234,252,247,0.86);
      background: rgba(255,255,255,0.035);
      border-left: 2px solid rgba(255,255,255,0.16);
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.72rem;
      line-height: 1.55;
      padding: 0.72rem;
      margin: 0.65rem 0;
    }
    .studio-judge {
      margin-top: 0.85rem;
      padding: 0.8rem;
      border-color: rgba(0,212,255,0.24);
      background: rgba(0,212,255,0.045);
    }
    .studio-judge-v {
      color: #00d4ff;
      margin: 0.25rem 0;
    }
    .studio-memory {
      border-color: rgba(0,255,159,0.28);
      box-shadow: 0 0 26px rgba(0,255,159,0.10), inset 0 0 0 1px rgba(0,255,159,0.06);
    }
    .studio-memory-intro {
      margin-bottom: 0.8rem;
    }
    .studio-memory-card.phishing { border-color: rgba(255,85,119,0.34); }
    .studio-memory-card.suspicious { border-color: rgba(255,221,87,0.30); }
    .studio-memory-card.safe { border-color: rgba(0,255,159,0.28); }
    .studio-memory-top strong {
      color: #00ff9f;
      font-size: 1rem;
    }
    .studio-memory-meta,
    .studio-memory-shared {
      color: rgba(214,245,235,0.64);
      line-height: 1.45;
      font-size: 0.66rem;
      margin-top: 0.48rem;
    }
    .studio-memory-preview {
      color: rgba(234,252,247,0.86);
      font-family: 'Share Tech Mono', monospace;
      line-height: 1.55;
      font-size: 0.74rem;
      margin-top: 0.65rem;
    }
    .studio-table {
      width: 100%;
      border-collapse: collapse;
    }
    .studio-table th,
    .studio-table td {
      text-align: left;
      padding: 0.55rem 0.65rem;
      border-bottom: 1px solid rgba(0,255,159,0.06);
      color: rgba(214,245,235,0.68);
      font-size: 0.7rem;
      vertical-align: top;
    }
    .studio-table th {
      color: #00d4ff;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      font-size: 0.58rem;
    }
    .studio-table tr.phishing td:first-child { border-left: 3px solid #ff5577; }
    .studio-table tr.suspicious td:first-child { border-left: 3px solid #ffdd57; }
    .studio-table tr.safe td:first-child { border-left: 3px solid #00ff9f; }
    @media (max-width: 1100px) {
      .studio-hero,
      .studio-grid,
      .studio-model-grid,
      .studio-eval-grid,
      .studio-case-meta,
      .studio-breakdown-top,
      .studio-breakdown-cards,
      .studio-debate-grid,
      .studio-memory-grid {
        grid-template-columns: 1fr;
      }
      .studio-score-card { text-align: left; }
      .studio-final-box { text-align: left; }
    }
    </style>
    """,
    ).strip()
)

_render_live_studio_fragment()
