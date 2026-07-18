from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import joblib
except Exception:
    joblib = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

try:
    from .bilstm import BiLSTMClassifier, encode_texts
except Exception:
    BiLSTMClassifier = None
    encode_texts = None

try:
    from .transformer_model import batch_predict_logits, tokenize_texts
except Exception:
    batch_predict_logits = None
    tokenize_texts = None

from .explainability import (
    actions_for_scam_type,
    cue_labels,
    detect_cue_tags,
    infer_scam_type_from_tags,
    language_code,
    reason_text,
)
from .mock_model import predict as mock_predict


MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "models"
INDICBERT_MODEL_PATH = MODEL_DIR / "indicbert_model.joblib"
BILSTM_MODEL_PATH = MODEL_DIR / "bilstm_model.joblib"
BASELINE_MODEL_PATH = MODEL_DIR / "baseline_model.joblib"
SCORE_FUSION_MODEL_PATH = MODEL_DIR / "score_fusion_model.joblib"

MODEL_KIND_TO_PATH = {
    "baseline": BASELINE_MODEL_PATH,
    "bilstm": BILSTM_MODEL_PATH,
    "indicbert": INDICBERT_MODEL_PATH,
}

SCAM_TYPES = [
    "OTP",
    "KYC",
    "UPI",
    "Courier/Customs",
    "Job/Loan",
    "Lottery",
    "Account-Block",
    "Other",
]

DEFAULT_THRESHOLDS = {
    "low_max": 30,
    "medium_max": 60,
    "high_max": 80,
    "critical_min": 81,
}
ACTIVE_THRESHOLD_POLICY = "natural_user_facing_v1"
DEFAULT_INDICBERT_MODEL_VERSION = "indicbert_v3"
DEFAULT_BILSTM_MODEL_VERSION = "bilstm_v3"
DEFAULT_BASELINE_MODEL_VERSION = "baseline_v3"
DEFAULT_REVIEW_CONFIDENCE_MIN = 0.70

BENIGN_AUTH_EXPLICIT_RE = re.compile(
    r"\b("
    r"verification code|login code|sign[- ]?in code|security code|passcode|one[- ]?time password|"
    r"authentication code|auth code|"
    r"सत्यापन\s*कोड|पुष्टि\s*कोड|लॉगिन\s*कोड|साइन[- ]?इन\s*कोड|"
    r"ਸਤਿਆਪਨ\s*ਕੋਡ|ਤਸਦੀਕ\s*ਕੋਡ|ਲੌਗਿਨ\s*ਕੋਡ|ਸਾਈਨ[- ]?ਇਨ\s*ਕੋਡ|"
    r"تصدیقی\s*کوڈ|تصدیق\s*کوڈ|لاگ[- ]?اِن\s*کوڈ|سائن[- ]?اِن\s*کوڈ|"
    r"ਸੁਰੱਖਿਆ\s*ਕੋਡ|ਪਾਸਕੋਡ|ਇਕ\s*ਵਾਰ\s*ਦਾ\s*ਪਾਸਵਰਡ|ਪ੍ਰਮਾਣੀਕਰਨ\s*ਕੋਡ|"
    r"ਪ੍ਰਮਾਣਿਕਤਾ\s*ਕੋਡ|ਓਟੀਪੀ|ਓ\s*ਟੀ\s*ਪੀ|OTP|TOTP|اوٹیپی|او\s*ٹی\s*پی"
    r")\b",
    re.IGNORECASE,
)
BENIGN_AUTH_SUPPORT_RE = re.compile(
    r"\b("
    r"use this code|enter this code|complete your sign[- ]?in|complete your login|"
    r"to complete your sign[- ]?in|to verify your account|for login|for sign[- ]?in|"
    r"इस\s*कोड\s*का\s*उपयोग\s*करें|इस\s*कोड\s*से\s*लॉगिन\s*पूरा\s*करें|"
    r"साइन[- ]?इन\s*पूरा\s*करने\s*के\s*लिए|अपने\s*खाते\s*में\s*साइन\s*इन|"
    r"अपने\s*खाते\s*को\s*सत्यापित\s*करने\s*के\s*लिए|यह\s*कोड\s*किसी\s*के\s*साथ\s*साझा\s*न\s*करें|"
    r"ਇਸ\s*ਕੋਡ\s*ਦਾ\s*ਉਪਯੋਗ\s*ਕਰੋ|ਇਸ\s*ਕੋਡ\s*ਨਾਲ\s*ਲੌਗਿਨ\s*ਪੂਰਾ\s*ਕਰੋ|"
    r"ਸਾਈਨ[- ]?ਇਨ\s*ਪੂਰਾ\s*ਕਰਨ\s*ਲਈ|ਆਪਣੇ\s*ਖਾਤੇ\s*ਵਿੱਚ\s*ਸਾਈਨ\s*ਇਨ|"
    r"ਆਪਣੇ\s*ਖਾਤੇ\s*ਦੀ\s*ਪੁਸ਼ਟੀ\s*ਕਰਨ\s*ਲਈ|ਕਿਰਪਾ\s*ਕਰਕੇ\s*ਇਹ\s*ਕੋਡ\s*ਕਿਸੇ\s*ਨਾਲ\s*ਸਾਂਝਾ\s*ਨਾ\s*ਕਰੋ|"
    r"اس\s*کوڈ\s*کا\s*استعمال\s*کریں|اس\s*کوڈ\s*کو\s*استعمال\s*کریں|"
    r"اپنے\s*اکاؤنٹ\s*میں\s*سائن[- ]?اِن\s*مکمل\s*کرنے\s*کے\s*لیے|اپنے\s*کھاتے\s*میں\s*سائن[- ]?اِن\s*مکمل\s*کرنے\s*کے\s*لیے|"
    r"اپنے\s*اکاؤنٹ\s*کی\s*تصدیق\s*کے\s*لیے|براہ\s*مہربانی\s*یہ\s*کوڈ\s*کسی\s*کے\s*ساتھ\s*شیئر\s*نہ\s*کریں|"
    r"براہ\s*کرم\s*یہ\s*کوڈ\s*کسی\s*کے\s*ساتھ\s*شیئر\s*نہ\s*کریں"
    r")\b",
    re.IGNORECASE,
)
BENIGN_AUTH_OTP_CONTEXT_RE = re.compile(
    r"\botp\b.*\b(login|sign[- ]?in|sign in|verify|verification|auth|authentication|code)\b|"
    r"\b(login|sign[- ]?in|sign in|verify|verification|auth|authentication|code)\b.*\botp\b|"
    r"\b(ओटीपी|ओ\s*टी\s*पी)\b.*\b(साइन[- ]?इन|सत्यापन|पुष्टि|लॉगिन|कोड)\b|"
    r"\b(साइन[- ]?इन|सत्यापन|पुष्टि|लॉगिन|कोड)\b.*\b(ओटीपी|ओ\s*टी\s*पी)\b|"
    r"\b(ਓਟੀਪੀ|ਓ\s*ਟੀ\s*ਪੀ)\b.*\b(ਸਾਈਨ[- ]?ਇਨ|ਸਤਿਆਪਨ|ਤਸਦੀਕ|ਲੌਗਿਨ|ਕੋਡ)\b|"
    r"\b(ਸਾਈਨ[- ]?ਇਨ|ਸਤਿਆਪਨ|ਤਸਦੀਕ|ਲੌਗਿਨ|ਕੋਡ)\b.*\b(ਓਟੀਪੀ|ਓ\s*ਟੀ\s*ਪੀ)\b|"
    r"\b(اوٹیپی|او\s*ٹی\s*پی)\b.*\b(سائن[- ]?اِن|تصدیق|تصدیقی|لاگ[- ]?اِن|کوڈ)\b|"
    r"\b(سائن[- ]?اِن|تصدیق|تصدیقی|لاگ[- ]?اِن|کوڈ)\b.*\b(اوٹیپی|او\s*ٹی\s*پی)\b",
    re.IGNORECASE,
)
BENIGN_AUTH_NEGATIVE_RE = re.compile(
    r"\b("
    r"share\s+(?:the\s+)?(?:otp|code|passcode|pin|password)|"
    r"send\s+(?:the\s+)?(?:otp|code|passcode|pin|password)|"
    r"forward\s+(?:the\s+)?(?:otp|code|passcode|pin|password)|"
    r"click\s+|tap\s+|https?://|www\.|"
    r"pay\s+now|upi|kyc|"
    r"blocked|suspended|frozen|locked|reactivate|"
    r"prize|reward|lottery|claim\s+your|"
    r"urgent|immediately|"
    r"तुरंत|अभी|फौरन|जल्द|"
    r"क्लिक\s*करें|लिंक|केवाईसी|यूपीआई|ओटीपी\s*भेजें|"
    r"साझा\s+(?!न\s)(?:करें|कीजिए|करो)|भेजें|अपडेट\s*करें|"
    r"ਸਾਂਝਾ\s+(?!ਨਾ\s*)(?:ਕਰੋ|ਕੀਜੀਏ|ਕਰ)|ਭੇਜੋ|ਅਪਡੇਟ\s*ਕਰੋ|"
    r"شیئر\s+(?!نہ\s*)(?:کریں|کرو|کیجیے)|بھیجیں|اپڈیٹ\s*کریں|"
    r"ब्लॉक|निलंबित|फ्रीज|सस्पेंड|इनाम|लॉटरी|पुरस्कार"
    r")\b",
    re.IGNORECASE,
)
BENIGN_AUTH_RISK_TAGS = {
    "ACCOUNT_THREAT",
    "PAYMENT_REQUEST",
    "KYC_UPDATE",
    "LINK_PRESENT",
    "LOTTERY_PRIZE",
    "DELIVERY_CUSTOMS",
    "JOB_LOAN",
}
HIGH_RISK_ACTION_TAGS = {
    "OTP_REQUEST",
    "PAYMENT_REQUEST",
    "KYC_UPDATE",
    "LINK_PRESENT",
    "LOTTERY_PRIZE",
    "DELIVERY_CUSTOMS",
    "JOB_LOAN",
}
HIGH_RISK_ACTION_TEXT_RE = re.compile(
    r"https?://|www\.|"
    r"\botp\b|\bpin\b|\bupi\b|\bkyc\b|\baadhaar\b|\baadhar\b|\bpan\b|"
    r"share\s+(?:the\s+)?(?:otp|code|pin|password)|"
    r"ओटीपी|पिन|यूपीआई|केवाईसी|आधार|लिंक|क्लिक|साझा\s+(?!न\s)(?:करें|कीजिए|करो)|"
    r"ਓਟੀਪੀ|ਪਿਨ|ਯੂਪੀਆਈ|ਕੇਵਾਈਸੀ|ਆਧਾਰ|ਲਿੰਕ|ਕਲਿੱਕ|ਸਾਂਝਾ\s+(?!ਨਾ\s*)(?:ਕਰੋ|ਕੀਜੀਏ|ਕਰ)|"
    r"او\s*ٹی\s*پی|پن|یو\s*پی\s*آئی|کے\s*وائی\s*سی|آدھار|لنک|کلک|شیئر\s+(?!نہ\s*)(?:کریں|کرو|کیجیے)",
    re.IGNORECASE,
)
SAFE_SECRET_WARNING_RE = re.compile(
    r"(?:do\s+not|don't|never)\s+share[^.!?।॥؟۔]{0,90}\b(?:otp|code|password|pin|passcode)\b|"
    r"\b(?:otp|code|password|pin|passcode)\b[^.!?।॥؟۔]{0,90}(?:do\s+not|don't|never)\s+share|"
    r"(?:OTP|ओटीपी|पासवर्ड|कोड|पिन)[^।॥.!?]{0,110}साझा\s*न\s*करें|"
    r"साझा\s*न\s*करें[^।॥.!?]{0,110}(?:OTP|ओटीपी|पासवर्ड|कोड|पिन)|"
    r"(?:ਓਟੀਪੀ|ਪਾਸਵਰਡ|ਕੋਡ|ਪਿਨ)[^।॥.!?]{0,110}ਸਾਂਝਾ\s*ਨਾ\s*ਕਰੋ|"
    r"ਸਾਂਝਾ\s*ਨਾ\s*ਕਰੋ[^।॥.!?]{0,110}(?:ਓਟੀਪੀ|ਪਾਸਵਰਡ|ਕੋਡ|ਪਿਨ)|"
    r"(?:او\s*ٹی\s*پی|کوڈ|پاس\s*ورڈ|پن)[^.!?؟۔]{0,110}شیئر\s*نہ\s*کریں|"
    r"شیئر\s*نہ\s*کریں[^.!?؟۔]{0,110}(?:او\s*ٹی\s*پی|کوڈ|پاس\s*ورڈ|پن)",
    re.IGNORECASE,
)
SAFE_OFFICIAL_CHANNEL_RE = re.compile(
    r"official\s+(?:bank\s+)?(?:app|website)|only\s+(?:use|open)[^.!?]{0,60}(?:official|bank\s+app)|"
    r"आधिकारिक\s+(?:बैंक\s+)?(?:ऐप|एप|वेबसाइट)|केवल[^।॥.!?]{0,80}(?:आधिकारिक|बैंक\s*(?:ऐप|एप))|"
    r"ਅਧਿਕਾਰਕ\s+(?:ਬੈਂਕ\s+)?(?:ਐਪ|ਵੈਬਸਾਈਟ)|ਕੇਵਲ[^।॥.!?]{0,80}(?:ਅਧਿਕਾਰਕ|ਬੈਂਕ\s*ਐਪ)|"
    r"آفیشل\s+(?:بینک\s+)?(?:ایپ|ویب\s*سائٹ)|صرف[^.!?؟۔]{0,80}(?:آفیشل|بینک\s*ایپ)",
    re.IGNORECASE,
)
SAFE_DELIVERY_BRAND_RE = re.compile(
    r"\b(?:zomato|swiggy|blinkit|zepto|dunzo|amazon|flipkart|myntra)\b|"
    r"\b(?:zomato|swiggy|blinkit|zepto|dunzo|amazon|flipkart|myntra)\.(?:com|in)\b",
    re.IGNORECASE,
)
SAFE_DELIVERY_STATUS_RE = re.compile(
    r"\b(?:order|picked\s+up|delivery\s+partner|estimated\s+delivery|out\s+for\s+delivery|track\s+here|arriving|delivered|eta)\b",
    re.IGNORECASE,
)
DELIVERY_SCAM_ACTION_RE = re.compile(
    r"\b(?:customs?|fee|charge|charges|pay|payment|unpaid|held|release|failed|reschedule|verify|otp|pin|kyc|urgent|blocked)\b|"
    r"https?://|www\.",
    re.IGNORECASE,
)
DELIVERY_PHISHING_ACTION_RE = re.compile(
    r"\b(?:customs?|fee|charge|charges|pay|payment|unpaid|held|release|failed|reschedule|verify|urgent|blocked)\b",
    re.IGNORECASE,
)
UNSAFE_DIRECTIVE_RE = re.compile(
    r"https?://|www\.|"
    r"\b(?:click|tap|open\s+this\s+link|verify\s+now|pay\s+now|send|forward)\b|"
    r"\b(?:share|send|forward)\s+(?:your\s+|the\s+)?(?:otp|code|password|pin|passcode)\b|"
    r"क्लिक\s*करें|लिंक|भेजें|ओटीपी\s*(?:भेजें|साझा\s+(?!न\s)(?:करें|कीजिए|करो))|"
    r"ਕਲਿੱਕ\s*ਕਰੋ|ਲਿੰਕ|ਭੇਜੋ|ਓਟੀਪੀ\s*(?:ਭੇਜੋ|ਸਾਂਝਾ\s+(?!ਨਾ\s*)(?:ਕਰੋ|ਕੀਜੀਏ|ਕਰ))|"
    r"کلک\s*کریں|لنک|بھیجیں|او\s*ٹی\s*پی\s*(?:بھیجیں|شیئر\s+(?!نہ\s*)(?:کریں|کرو|کیجیے))",
    re.IGNORECASE,
)
WEAK_ACCOUNT_ADMIN_RE = re.compile(
    r"security\s+check|account\s+review|profile\s+(?:information|info|details|update)|bank\s+app|"
    r"सुरक्षा\s*(?:जांच|जाँच)|प्रोफ(?:ा|़ा)इल\s*(?:जानकारी|विवरण)|बैंक\s*(?:ऐप|एप)|"
    r"ਸੁਰੱਖਿਆ\s*ਜਾਂਚ|ਪ੍ਰੋਫਾਈਲ\s*(?:ਜਾਣਕਾਰੀ|ਵੇਰਵੇ)|ਬੈਂਕ\s*ਐਪ|"
    r"سیکیورٹی\s*چیک|سکیورٹی\s*چیک|پروفائل\s*(?:معلومات|تفصیلات)|بینک\s*ایپ",
    re.IGNORECASE,
)
SCAM_TYPE_REQUIRED_TAGS = {
    "OTP": {"OTP_REQUEST"},
    "KYC": {"KYC_UPDATE"},
    "UPI": {"PAYMENT_REQUEST"},
    "Courier/Customs": {"DELIVERY_CUSTOMS"},
    "Job/Loan": {"JOB_LOAN"},
    "Lottery": {"LOTTERY_PRIZE"},
    "Account-Block": {"ACCOUNT_THREAT"},
}

MODEL_LOAD_ORDER = (
    (BASELINE_MODEL_PATH, 0),
    (BILSTM_MODEL_PATH, 1),
    (INDICBERT_MODEL_PATH, 2),
)

MODEL_KIND_TO_SOURCE = {
    "baseline": "baseline_ml",
    "bilstm": "bilstm_rnn",
    "indicbert": "indicbert_multilingual",
}

MODEL_KIND_TO_VERSION = {
    "baseline": DEFAULT_BASELINE_MODEL_VERSION,
    "bilstm": DEFAULT_BILSTM_MODEL_VERSION,
    "indicbert": DEFAULT_INDICBERT_MODEL_VERSION,
}

MODEL_KINDS = tuple(MODEL_KIND_TO_PATH.keys())


def _score_confidence(score: int, label: str) -> float:
    score = max(0, min(100, int(score or 0)))
    label_norm = str(label or "").strip().lower()
    if label_norm == "phishing":
        return score / 100.0
    if label_norm == "safe":
        return 1.0 - (score / 100.0)
    # Suspicious is an in-between band, so confidence peaks near the middle.
    return max(0.50, min(0.80, 1.0 - (abs(score - 45) / 100.0)))


def _softmax(logits: np.ndarray) -> np.ndarray:
    if np is None:
        raise RuntimeError("numpy is required for model inference.")
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)


def _resolve_thresholds(bundle: Dict) -> Dict[str, float]:
    # The trained bundles may contain validation-optimized cutoffs, but the live
    # app uses one consistent, explainable 0-100 range for users.
    thresholds = DEFAULT_THRESHOLDS
    out = {
        "low_max": int(thresholds.get("low_max", 30)),
        "medium_max": int(thresholds.get("medium_max", 60)),
        "high_max": int(thresholds.get("high_max", 80)),
        "critical_min": int(thresholds.get("critical_min", 81)),
    }
    if out["medium_max"] <= out["low_max"]:
        out["medium_max"] = out["low_max"] + 1
    if out["high_max"] <= out["medium_max"]:
        out["high_max"] = out["medium_max"] + 1
    out["critical_min"] = out["high_max"] + 1
    return out


def _resolve_review_confidence_min(bundle: Dict) -> float:
    raw = bundle.get("review_confidence_min", DEFAULT_REVIEW_CONFIDENCE_MIN)
    try:
        value = float(raw)
    except Exception:
        value = DEFAULT_REVIEW_CONFIDENCE_MIN
    return float(max(0.5, min(0.99, value)))


def _score_to_tier(score: int, thresholds: Dict[str, float]) -> str:
    if score <= thresholds["low_max"]:
        return "Low"
    if score <= thresholds["medium_max"]:
        return "Medium"
    if score <= thresholds["high_max"]:
        return "High"
    return "Critical"


def _score_to_risk_label(score: int, thresholds: Dict[str, float]) -> str:
    if score <= thresholds["low_max"]:
        return "Safe"
    if score <= thresholds["medium_max"]:
        return "Suspicious"
    return "Phishing"


def _probability_to_risk_score(probability: float) -> int:
    probability = max(0.0, min(1.0, float(probability or 0.0)))
    if probability >= 1.0 - 1e-12:
        return 100
    return max(0, min(99, int(probability * 100.0)))


def _is_benign_auth_flow(text: str, cue_tags: List[str]) -> bool:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return False

    has_explicit_auth = bool(BENIGN_AUTH_EXPLICIT_RE.search(normalized))
    has_support_phrase = bool(BENIGN_AUTH_SUPPORT_RE.search(normalized))
    has_otp_context = bool(BENIGN_AUTH_OTP_CONTEXT_RE.search(normalized))

    if not (has_explicit_auth or has_support_phrase or has_otp_context):
        return False
    if BENIGN_AUTH_NEGATIVE_RE.search(normalized):
        return False

    risk_tags = set(cue_tags)
    if risk_tags & BENIGN_AUTH_RISK_TAGS:
        # Allow a plain OTP/auth flow even when the generic cue extractor leans on
        # account-related wording, but only if the message still reads like a
        # verification step and not a scam call-to-action.
        if not (has_explicit_auth and (has_support_phrase or has_otp_context)):
            return False
        if risk_tags - {"ACCOUNT_THREAT"}:
            return False

    return True


def _is_safe_security_advisory_flow(text: str, cue_tags: List[str]) -> bool:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return False
    if UNSAFE_DIRECTIVE_RE.search(normalized):
        return False
    if set(cue_tags) & HIGH_RISK_ACTION_TAGS:
        return False

    has_secret_warning = bool(SAFE_SECRET_WARNING_RE.search(normalized))
    has_official_channel = bool(SAFE_OFFICIAL_CHANNEL_RE.search(normalized))
    return has_secret_warning and has_official_channel


def _is_safe_delivery_status_flow(text: str, cue_tags: List[str]) -> bool:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return False
    if DELIVERY_SCAM_ACTION_RE.search(normalized):
        return False
    return bool(SAFE_DELIVERY_BRAND_RE.search(normalized) and SAFE_DELIVERY_STATUS_RE.search(normalized))


def _is_explicit_otp_phishing_flow(text: str, cue_tags: List[str]) -> bool:
    normalized = " ".join(str(text or "").split())
    risk_tags = set(cue_tags)
    if "OTP_REQUEST" not in risk_tags:
        return False
    if SAFE_SECRET_WARNING_RE.search(normalized):
        return False
    return bool(risk_tags & {"ACCOUNT_THREAT", "LINK_PRESENT"}) or bool(UNSAFE_DIRECTIVE_RE.search(normalized))


def _is_explicit_delivery_phishing_flow(text: str, cue_tags: List[str]) -> bool:
    normalized = " ".join(str(text or "").split())
    risk_tags = set(cue_tags)
    if "DELIVERY_CUSTOMS" not in risk_tags:
        return False
    has_link = "LINK_PRESENT" in risk_tags or bool(re.search(r"https?://|www\.", normalized, re.IGNORECASE))
    return has_link and bool(DELIVERY_PHISHING_ACTION_RE.search(normalized))


def _is_weak_account_admin_flow(text: str, cue_tags: List[str]) -> bool:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return False

    risk_tags = set(cue_tags)
    if "ACCOUNT_THREAT" not in risk_tags and not WEAK_ACCOUNT_ADMIN_RE.search(normalized):
        return False
    if risk_tags & HIGH_RISK_ACTION_TAGS:
        return False
    if HIGH_RISK_ACTION_TEXT_RE.search(normalized):
        return False

    return bool(WEAK_ACCOUNT_ADMIN_RE.search(normalized))


def _calibrate_risk_probs(
    probs: np.ndarray,
    risk_classes: List[str],
    text: str,
    cue_tags: List[str],
) -> tuple[np.ndarray, str]:
    calibrated = np.array(probs, copy=True)
    calibration_note = ""

    if calibrated.ndim != 2 or calibrated.shape[0] == 0:
        return calibrated, calibration_note

    class_index = {name: idx for idx, name in enumerate(risk_classes)}
    phish_idx = class_index.get("Phishing", int(np.argmax(calibrated[0])))
    safe_idx = class_index.get("Safe", 0)
    suspicious_idx = class_index.get("Suspicious")

    if _is_safe_security_advisory_flow(text, cue_tags):
        phishing_mass = float(calibrated[0, phish_idx])
        safe_mass = float(calibrated[0, safe_idx])
        phish_target = min(max(phishing_mass * 0.05, 0.02), 0.04)
        safe_target = max(safe_mass + phishing_mass * 0.82, 0.84)

        other_indices = [idx for idx in range(calibrated.shape[1]) if idx not in {safe_idx, phish_idx}]
        residual = max(0.0, 1.0 - safe_target - phish_target)
        calibrated[0, safe_idx] = safe_target
        calibrated[0, phish_idx] = phish_target
        if other_indices:
            share = residual / len(other_indices)
            for idx in other_indices:
                calibrated[0, idx] = share

        calibration_note = "official safety advisory with do-not-share wording detected; treated as safe"
    elif _is_safe_delivery_status_flow(text, cue_tags):
        phishing_mass = float(calibrated[0, phish_idx])
        safe_mass = float(calibrated[0, safe_idx])
        phish_target = min(max(phishing_mass * 0.05, 0.02), 0.04)
        safe_target = max(safe_mass + phishing_mass * 0.82, 0.84)

        other_indices = [idx for idx in range(calibrated.shape[1]) if idx not in {safe_idx, phish_idx}]
        residual = max(0.0, 1.0 - safe_target - phish_target)
        calibrated[0, safe_idx] = safe_target
        calibrated[0, phish_idx] = phish_target
        if other_indices:
            share = residual / len(other_indices)
            for idx in other_indices:
                calibrated[0, idx] = share

        calibration_note = "legitimate delivery-status update detected; treated as safe"
    elif _is_benign_auth_flow(text, cue_tags):
        # Strongly re-balance plain login/verification-code messages toward the safe class.
        phishing_mass = float(calibrated[0, phish_idx])
        safe_mass = float(calibrated[0, safe_idx])
        phish_target = min(max(phishing_mass * 0.05, 0.02), 0.04)
        safe_target = max(safe_mass + phishing_mass * 0.82, 0.84)

        other_indices = [idx for idx in range(calibrated.shape[1]) if idx not in {safe_idx, phish_idx}]
        other_raw_total = float(sum(calibrated[0, idx] for idx in other_indices))
        residual = max(0.0, 1.0 - safe_target - phish_target)

        if residual < 0.0:
            safe_target = max(0.5, 1.0 - phish_target)
            residual = max(0.0, 1.0 - safe_target - phish_target)

        calibrated[0, safe_idx] = safe_target
        calibrated[0, phish_idx] = phish_target
        if other_indices:
            if other_raw_total > 0:
                for idx in other_indices:
                    calibrated[0, idx] = residual * (float(calibrated[0, idx]) / other_raw_total)
            else:
                equal_share = residual / len(other_indices) if other_indices else 0.0
                for idx in other_indices:
                    calibrated[0, idx] = equal_share
        calibration_note = "benign verification-code flow detected; treated as safe"
    elif _is_explicit_otp_phishing_flow(text, cue_tags):
        phishing_mass = float(calibrated[0, phish_idx])
        phish_target = max(phishing_mass, 0.72)
        safe_target = min(float(calibrated[0, safe_idx]), 0.08)
        other_indices = [idx for idx in range(calibrated.shape[1]) if idx not in {safe_idx, phish_idx}]
        residual = max(0.0, 1.0 - phish_target - safe_target)

        calibrated[0, :] = 1e-9
        calibrated[0, phish_idx] = phish_target
        calibrated[0, safe_idx] = safe_target
        if other_indices:
            share = residual / len(other_indices)
            for idx in other_indices:
                calibrated[0, idx] = share

        calibration_note = "explicit OTP-sharing scam cues detected; phishing score floor applied"
    elif _is_explicit_delivery_phishing_flow(text, cue_tags):
        phishing_mass = float(calibrated[0, phish_idx])
        phish_target = max(phishing_mass, 0.72)
        safe_target = min(float(calibrated[0, safe_idx]), 0.08)
        other_indices = [idx for idx in range(calibrated.shape[1]) if idx not in {safe_idx, phish_idx}]
        residual = max(0.0, 1.0 - phish_target - safe_target)

        calibrated[0, :] = 1e-9
        calibrated[0, phish_idx] = phish_target
        calibrated[0, safe_idx] = safe_target
        if other_indices:
            share = residual / len(other_indices)
            for idx in other_indices:
                calibrated[0, idx] = share

        calibration_note = "delivery/customs fee scam cues detected; phishing score floor applied"
    elif _is_weak_account_admin_flow(text, cue_tags):
        phishing_mass = float(calibrated[0, phish_idx])
        safe_mass = float(calibrated[0, safe_idx])

        if suspicious_idx is not None and len({safe_idx, suspicious_idx, phish_idx}) == 3:
            # A lone account-admin cue is suspicious, but it should not outrank
            # messages that also ask for an OTP, link click, payment, or KYC.
            phish_target = min(phishing_mass, 0.46)
            safe_target = min(max(safe_mass, 0.05), 0.20)
            suspicious_target = max(0.0, 1.0 - phish_target - safe_target)
            if suspicious_target <= phish_target:
                phish_target = min(phish_target, 0.44)
                suspicious_target = max(0.0, 1.0 - phish_target - safe_target)

            calibrated[0, :] = 1e-9
            calibrated[0, safe_idx] = safe_target
            calibrated[0, suspicious_idx] = suspicious_target
            calibrated[0, phish_idx] = phish_target
        else:
            phish_target = min(phishing_mass, 0.46)
            calibrated[0, phish_idx] = phish_target
            if safe_idx != phish_idx:
                calibrated[0, safe_idx] = max(float(calibrated[0, safe_idx]), 1.0 - phish_target)

        calibration_note = "weak account-admin wording without link, OTP, payment, or KYC cue; treated as suspicious evidence"

    if calibration_note:
        calibrated = np.clip(calibrated, 1e-9, None)
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)

    return calibrated, calibration_note


def _predict_type_with_pipeline(text: str, pipeline, classes: List[str]) -> Tuple[str, float]:
    if pipeline is None or np is None or not classes:
        return "Other", 0.0

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([text])[0]
        idx = int(np.argmax(probs))
        return str(classes[idx]), float(probs[idx])

    pred = str(pipeline.predict([text])[0])
    return pred, 1.0


def _scam_type_supported_by_cues(label: str, cue_tags: List[str]) -> bool:
    required = SCAM_TYPE_REQUIRED_TAGS.get(label)
    if not required:
        return True
    return bool(set(cue_tags) & required)


def _metrics_path_for_artifact(path: Path) -> Path:
    stem = path.stem
    if stem.endswith("_model"):
        stem = stem[: -len("_model")]
    return path.with_name(f"{stem}_metrics.json")


def _kind_from_path(path: Path | None) -> str:
    if path is None:
        return "baseline"

    stem = path.stem.lower()
    if stem.endswith("_model"):
        stem = stem[: -len("_model")]

    if stem in MODEL_KIND_TO_PATH:
        return stem
    if "indicbert" in stem or "indic-bert" in stem or "indic_bert" in stem:
        return "indicbert"
    if "transformer" in stem:
        return "transformer"
    if "bilstm" in stem:
        return "bilstm"
    return "baseline"


def _normalize_model_kind(model_kind: str | None) -> str:
    value = str(model_kind or "best").strip().lower()
    aliases = {
        "auto": "best",
        "default": "best",
        "best": "best",
        "baseline": "baseline",
        "baseline_ml": "baseline",
        "tf-idf": "baseline",
        "tfidf": "baseline",
        "bilstm": "bilstm",
        "bi-lstm": "bilstm",
        "bi_lstm": "bilstm",
        "transformer": "indicbert",
        "indicbert": "indicbert",
        "indic-bert": "indicbert",
        "indic_bert": "indicbert",
    }
    return aliases.get(value, value)


def _model_rankings() -> List[Tuple[str, float]]:
    rankings: List[Tuple[str, float]] = []
    for kind, path in MODEL_KIND_TO_PATH.items():
        if not path.exists():
            continue
        score = _artifact_score(path)
        if score == float("-inf"):
            continue
        rankings.append((kind, score))
    rankings.sort(key=lambda item: (-item[1], item[0]))
    return rankings


def _best_available_kind(exclude: set[str] | None = None) -> str | None:
    excluded = set(exclude or set())
    for kind, _ in _model_rankings():
        if kind not in excluded:
            return kind
    for kind in MODEL_KINDS:
        if kind not in excluded:
            return kind
    return None


def _ordered_model_kinds() -> List[str]:
    ordered = [kind for kind, _ in _model_rankings()]
    for kind in MODEL_KINDS:
        if kind not in ordered:
            ordered.append(kind)
    return ordered


def _load_model_bundle_from_path(path: Path) -> Dict | None:
    if joblib is None or not path.exists():
        return None

    try:
        bundle = joblib.load(path)
    except Exception:
        return None

    bundle["_bundle_path"] = str(path.resolve())
    model_kind = _normalize_model_kind(bundle.get("risk_model_kind") or _kind_from_path(path))

    if model_kind == "indicbert":
        try:
            return _attach_transformer_model(bundle)
        except Exception:
            return None

    if model_kind == "bilstm":
        try:
            return _attach_bilstm_model(bundle)
        except Exception:
            return None

    bundle["risk_model_kind"] = model_kind
    bundle.setdefault("model_source", MODEL_KIND_TO_SOURCE[model_kind])
    bundle.setdefault("model_version", MODEL_KIND_TO_VERSION[model_kind])
    return bundle


def _nested_metric(data: Dict, keys: Tuple[str, ...]) -> float | None:
    current: object = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _artifact_score(path: Path) -> float:
    metrics_path = _metrics_path_for_artifact(path)
    if not metrics_path.exists():
        return float("-inf")

    try:
        metrics = json.loads(metrics_path.read_text())
    except Exception:
        return float("-inf")

    score_candidates = (
        ("risk_metrics", "test", "classification_report", "macro avg", "f1-score"),
        ("risk_metrics", "val", "classification_report", "macro avg", "f1-score"),
        ("evaluation", "test", "risk", "macro_f1"),
    )
    for keys in score_candidates:
        score = _nested_metric(metrics, keys)
        if score is not None:
            return score

    for key in ("test_macro_f1", "best_val_macro_f1", "val_macro_f1"):
        if isinstance(metrics.get(key), (int, float)):
            return float(metrics[key])

    return float("-inf")


def _attach_bilstm_model(bundle: Dict) -> Dict:
    if torch is None or BiLSTMClassifier is None or encode_texts is None:
        raise RuntimeError("PyTorch BiLSTM support is unavailable.")

    config = dict(bundle.get("risk_model_config", {}))
    state_dict = bundle.get("risk_model_state_dict")
    risk_classes = list(bundle.get("risk_classes", []))
    if not config or state_dict is None or not risk_classes:
        raise RuntimeError("BiLSTM bundle is missing required model fields.")

    model = BiLSTMClassifier(
        vocab_size=int(config["vocab_size"]),
        num_classes=len(risk_classes),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config.get("num_layers", 1)),
        dropout=float(config.get("dropout", 0.3)),
        bidirectional=bool(config.get("bidirectional", True)),
        pad_idx=int(config.get("pad_idx", 0)),
    )
    model.load_state_dict(state_dict)
    model.eval()
    bundle["risk_model"] = model
    return bundle


def _resolve_bundle_model_dir(bundle: Dict) -> Path:
    raw_dir = bundle.get("risk_model_dir")
    if raw_dir is None:
        raise RuntimeError("IndicBERT bundle is missing risk_model_dir.")

    path = Path(str(raw_dir))
    if path.is_absolute():
        return path

    bundle_path = bundle.get("_bundle_path")
    base_dir = Path(bundle_path).parent if bundle_path else MODEL_DIR
    return (base_dir / path).resolve()


def _attach_transformer_model(bundle: Dict) -> Dict:
    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        raise RuntimeError("IndicBERT support is unavailable.")

    model_dir = _resolve_bundle_model_dir(bundle)
    if not model_dir.exists():
        raise RuntimeError(f"IndicBERT model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    bundle["risk_tokenizer"] = tokenizer
    bundle["risk_model"] = model
    bundle["risk_model_dir_resolved"] = str(model_dir)
    return bundle


def _predict_risk_probs_bilstm(text: str, bundle: Dict) -> np.ndarray:
    if torch is None or BiLSTMClassifier is None or encode_texts is None:
        raise RuntimeError("PyTorch BiLSTM support is unavailable.")

    model = bundle.get("risk_model")
    if model is None:
        raise RuntimeError("BiLSTM model is not loaded.")

    vocab = bundle.get("risk_vocab")
    max_length = int(bundle.get("risk_max_length", 0))
    if not vocab or max_length <= 0:
        raise RuntimeError("BiLSTM bundle is missing vocabulary metadata.")

    input_ids, lengths = encode_texts([text], vocab, max_length)
    input_ids_t = torch.as_tensor(input_ids, dtype=torch.long)
    lengths_t = torch.as_tensor(lengths, dtype=torch.long)

    with torch.inference_mode():
        logits = model(input_ids_t, lengths_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    return probs


def _predict_risk_probs_transformer(text: str, bundle: Dict) -> np.ndarray:
    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        raise RuntimeError("IndicBERT support is unavailable.")

    model = bundle.get("risk_model")
    tokenizer = bundle.get("risk_tokenizer")
    if model is None or tokenizer is None:
        raise RuntimeError("IndicBERT model is not loaded.")

    max_length = int(bundle.get("risk_max_length", 0))
    if max_length <= 0:
        raise RuntimeError("IndicBERT bundle is missing max_length metadata.")

    if batch_predict_logits is not None and tokenize_texts is not None:
        logits = batch_predict_logits(model, tokenizer, [text], max_length, batch_size=1, device=next(model.parameters()).device)
        probs = torch.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=1).cpu().numpy()
        return probs

    enc = tokenizer(
        [text],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    return probs


def _predict_scam_type(text: str, risk_label: str, bundle: Dict, cue_tags: List[str]) -> Tuple[str, str, float]:
    if risk_label == "Safe":
        return "Other", "risk_gate_safe", 1.0

    type_conf_min = float(bundle.get("type_confidence_min", 0.40))
    non_safe_pipe = bundle.get("type_pipeline_non_safe", bundle.get("type_pipeline"))
    non_safe_classes = bundle.get("type_classes_non_safe", bundle.get("type_classes", []))
    phish_pipe = bundle.get("type_pipeline_phishing")
    phish_classes = bundle.get("type_classes_phishing", [])

    if risk_label == "Phishing" and phish_pipe is not None and phish_classes:
        label, conf = _predict_type_with_pipeline(text, phish_pipe, phish_classes)
        if label in SCAM_TYPES and label != "Other" and conf >= type_conf_min and _scam_type_supported_by_cues(label, cue_tags):
            return label, "phishing_head", conf

    label, conf = _predict_type_with_pipeline(text, non_safe_pipe, non_safe_classes)
    if label in SCAM_TYPES and label != "Other" and conf >= type_conf_min and _scam_type_supported_by_cues(label, cue_tags):
        return label, "non_safe_head", conf

    inferred = infer_scam_type_from_tags(cue_tags)
    if inferred != "Other":
        return inferred, "cue_fallback", max(conf, 0.35)

    return "Other", "fallback_other", conf


@lru_cache(maxsize=8)
def load_model_bundle(model_kind: str = "best") -> Dict | None:
    kind = _normalize_model_kind(model_kind)
    if kind == "best":
        candidates = [
            (path, priority, _artifact_score(path))
            for path, priority in MODEL_LOAD_ORDER
            if path.exists()
        ]
        for path, _, _ in sorted(candidates, key=lambda item: (-item[2], item[1])):
            bundle = _load_model_bundle_from_path(path)
            if bundle is not None:
                return bundle
        return None

    path = MODEL_KIND_TO_PATH.get(kind)
    if path is None:
        return None
    return _load_model_bundle_from_path(path)


@lru_cache(maxsize=1)
def load_score_fusion_bundle() -> Dict | None:
    if joblib is None or not SCORE_FUSION_MODEL_PATH.exists():
        return None
    try:
        bundle = joblib.load(SCORE_FUSION_MODEL_PATH)
    except Exception:
        return None
    if not isinstance(bundle, dict):
        return None
    return bundle


def _score_from_fusion_model(scores: List[int]) -> Tuple[int | None, str]:
    bundle = load_score_fusion_bundle()
    if not bundle or not bool(bundle.get("deploy_as_final", False)):
        return None, "median_ensemble_v1"

    model = bundle.get("model")
    features = bundle.get("features") or ["baseline_score", "bilstm_score", "indicbert_score"]
    if model is None or np is None or len(scores) < len(features):
        return None, "median_ensemble_v1"

    try:
        x = np.asarray([[float(score) / 100.0 for score in scores[: len(features)]]], dtype=float)
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(x)[0, 1])
        else:
            pred = model.predict(x)
            prob = float(pred[0])
        return _probability_to_risk_score(prob), str(bundle.get("model_version") or "score_fusion_model_v1")
    except Exception:
        return None, "median_ensemble_v1"


def _majority_value(values: List[str], fallback: str = "Other") -> str:
    clean = [str(value or "").strip() for value in values if str(value or "").strip() and str(value or "").strip().lower() != "unknown"]
    if not clean:
        return fallback
    counts: Dict[str, int] = {}
    for value in clean:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)


def _model_outputs_for_storage(predictions: List[Dict]) -> List[Dict]:
    outputs: List[Dict] = []
    for entry in predictions:
        pred = entry.get("prediction") or {}
        outputs.append(
            {
                "model_kind": str(entry.get("model_kind", pred.get("model_kind", "unknown"))),
                "model_source": str(entry.get("model_source", pred.get("model_source", "unknown"))),
                "model_version": str(entry.get("model_version", pred.get("model_version", "unknown"))),
                "label": str(entry.get("label", pred.get("label", "Safe"))),
                "scam_type": str(entry.get("scam_type", pred.get("scam_type", "Other"))),
                "risk_score": int(entry.get("risk_score", pred.get("risk_score", 0)) or 0),
                "model_confidence": float(entry.get("model_confidence", pred.get("model_confidence", 0.0)) or 0.0),
                "type_source": str(entry.get("type_source", pred.get("type_source", "unknown"))),
                "reason": str(pred.get("reason", "")),
            }
        )
    return outputs


def _build_final_ensemble_prediction(message: str, output_language: str, predictions: List[Dict]) -> Dict:
    if not predictions:
        return _mock_fallback(message, output_language)

    scores = [int(entry.get("risk_score", 0) or 0) for entry in predictions]
    fusion_score, method = _score_from_fusion_model(scores)
    if fusion_score is None:
        if np is not None:
            final_score = int(round(float(np.median(np.asarray(scores, dtype=float)))))
        else:
            final_score = sorted(scores)[len(scores) // 2]
        method = "median_ensemble_v1"
    else:
        final_score = fusion_score

    primary_pred = predictions[0].get("prediction") or {}
    thresholds = _resolve_thresholds(primary_pred if isinstance(primary_pred, dict) else {})
    final_label = _score_to_risk_label(final_score, thresholds)
    final_tier = _score_to_tier(final_score, thresholds)
    confidence = _score_confidence(final_score, final_label)
    output_lang = language_code(output_language)
    model_outputs = _model_outputs_for_storage(predictions)
    score_summary = ", ".join(
        f"{item['model_version']}={item['risk_score']}/100" for item in model_outputs
    )

    if final_label == "Safe":
        candidate_types = ["Other"]
    else:
        same_label_types = [
            item["scam_type"]
            for item in model_outputs
            if item["label"] == final_label and item["scam_type"] != "Other"
        ]
        candidate_types = same_label_types or [item["scam_type"] for item in model_outputs if item["scam_type"] != "Other"]
    scam_type = _majority_value(candidate_types, fallback=str(primary_pred.get("scam_type", "Other")))
    if scam_type not in SCAM_TYPES:
        scam_type = "Other"

    cue_tags = detect_cue_tags(message)
    cue_label_list = cue_labels(cue_tags, output_lang)
    method_label = "score fusion model" if method.startswith("score_fusion") else "median ensemble"
    reason = (
        f"Final score uses {method_label} across the three model scores ({score_summary}). "
        f"Final risk is {final_score}/100 with {final_label} verdict."
    )
    if cue_label_list:
        reason += f" Matched cues: {', '.join(cue_label_list)}."

    return {
        "label": final_label,
        "risk_label_score_based": final_label,
        "scam_type": scam_type,
        "risk_score": int(final_score),
        "raw_risk_score": int(final_score),
        "severity_tier": final_tier,
        "severity_thresholds": thresholds,
        "threshold_policy": ACTIVE_THRESHOLD_POLICY,
        "reason": reason,
        "actions": actions_for_scam_type(scam_type, output_lang),
        "cue_tags": cue_tags,
        "cue_labels": cue_label_list,
        "type_confidence": 1.0,
        "type_source": "ensemble_consensus",
        "model_confidence": confidence,
        "calibrated_confidence": confidence,
        "calibration_method": method,
        "calibration_reason": "",
        "temperature": 1.0,
        "risk_probs": {"Safe": max(0.0, 1.0 - final_score / 100.0), "Phishing": final_score / 100.0},
        "raw_risk_probs": {"Safe": max(0.0, 1.0 - final_score / 100.0), "Phishing": final_score / 100.0},
        "model_source": "score_ensemble",
        "model_version": method,
        "model_kind": "ensemble",
        "final_score_method": method,
        "model_outputs": model_outputs,
        "language": output_lang,
    }


def _predict_with_bundle(message: str, output_language: str, bundle: Dict) -> Dict:
    if np is None:
        raise RuntimeError("numpy is required for model inference.")

    text = message.strip()
    lang = language_code(output_language)
    model_kind = _normalize_model_kind(bundle.get("risk_model_kind") or _kind_from_path(Path(bundle.get("_bundle_path", ""))))
    if model_kind not in MODEL_KINDS:
        model_kind = "baseline"
    model_source = str(bundle.get("model_source", MODEL_KIND_TO_SOURCE[model_kind]))
    model_version = str(bundle.get("model_version", MODEL_KIND_TO_VERSION[model_kind]))
    temp = float(bundle.get("temperature", 1.0))

    if not text:
        review_confidence_min = _resolve_review_confidence_min(bundle)
        return {
            "label": "Safe",
            "risk_label_score_based": "Safe",
            "scam_type": "Other",
            "risk_score": 0,
            "severity_tier": "Low",
            "severity_thresholds": DEFAULT_THRESHOLDS,
            "reason": reason_text([], confidence=1.0, lang=lang),
            "actions": actions_for_scam_type("Other", lang),
            "cue_tags": [],
            "cue_labels": [],
            "type_source": "risk_gate_safe",
            "model_confidence": 1.0,
            "calibrated_confidence": 1.0,
            "calibration_method": f"temperature_scaling_t={temp:.3f}",
            "temperature": temp,
            "model_source": model_source,
            "model_version": model_version,
            "model_kind": model_kind,
            "review_confidence_min": review_confidence_min,
            "review_recommended": False,
            "abstain": False,
            "review_reason": "",
            "abstain_reason": "",
            "decision_state": "auto",
            "decision_label": "Auto decision",
            "language": lang,
        }

    thresholds = _resolve_thresholds(bundle)
    risk_classes = list(bundle["risk_classes"])

    if model_kind == "indicbert":
        probs = _predict_risk_probs_transformer(text, bundle)
        if abs(temp - 1.0) > 1e-9:
            probs = _softmax(np.log(np.clip(probs, 1e-9, 1.0)) / max(temp, 1e-6))
    elif model_kind == "bilstm":
        probs = _predict_risk_probs_bilstm(text, bundle)
        if abs(temp - 1.0) > 1e-9:
            probs = _softmax(np.log(np.clip(probs, 1e-9, 1.0)) / max(temp, 1e-6))
    else:
        risk_pipeline = bundle["risk_pipeline"]
        logits = risk_pipeline.decision_function([text])
        probs = _softmax(logits / max(temp, 1e-6))

    raw_probs = np.array(probs, copy=True)
    cue_tags = detect_cue_tags(text)
    calibrated_probs, calibration_note = _calibrate_risk_probs(probs, risk_classes, text, cue_tags)
    if calibration_note:
        probs = calibrated_probs

    pred_idx = int(np.argmax(probs[0]))
    risk_label_argmax = risk_classes[pred_idx]

    phish_idx = risk_classes.index("Phishing") if "Phishing" in risk_classes else pred_idx
    risk_score = _probability_to_risk_score(float(probs[0, phish_idx]))
    severity = _score_to_tier(risk_score, thresholds)
    risk_label_score = _score_to_risk_label(risk_score, thresholds)
    scam_type, scam_type_source, type_conf = _predict_scam_type(text, risk_label_score, bundle, cue_tags)

    if scam_type not in SCAM_TYPES:
        scam_type = "Other"

    confidence = float(np.max(probs[0]))
    reason = reason_text(cue_tags, confidence=confidence, lang=lang)
    if calibration_note:
        reason = f"{reason} Calibration: {calibration_note}."

    review_confidence_min = _resolve_review_confidence_min(bundle)
    if calibration_note and risk_score <= 35:
        review_confidence_min = min(review_confidence_min, 0.40)
    review_fields = _build_review_fields(
        primary={"label": risk_label_score, "model_confidence": confidence},
        review_confidence_min=review_confidence_min,
    )

    return {
        "label": risk_label_score,
        "model_class_label": risk_label_argmax,
        "risk_label_score_based": risk_label_score,
        "scam_type": scam_type,
        "risk_score": risk_score,
        "raw_risk_score": _probability_to_risk_score(float(raw_probs[0, phish_idx])),
        "severity_tier": severity,
        "severity_thresholds": thresholds,
        "threshold_policy": ACTIVE_THRESHOLD_POLICY,
        "reason": reason,
        "actions": actions_for_scam_type(scam_type, lang),
        "cue_tags": cue_tags,
        "cue_labels": cue_labels(cue_tags, lang),
        "type_confidence": type_conf,
        "type_source": scam_type_source,
        "model_confidence": confidence,
        "calibrated_confidence": confidence,
        "calibration_method": (
            f"temperature_scaling_t={temp:.3f}+runtime_evidence_guard_v2" if calibration_note else f"temperature_scaling_t={temp:.3f}"
        ),
        "calibration_reason": calibration_note,
        "temperature": temp,
        "risk_probs": {risk_classes[i]: float(probs[0, i]) for i in range(len(risk_classes))},
        "raw_risk_probs": {risk_classes[i]: float(raw_probs[0, i]) for i in range(len(risk_classes))},
        "model_source": model_source,
        "model_version": model_version,
        "model_kind": model_kind,
        **review_fields,
        "language": lang,
    }


def _mock_fallback(message: str, output_language: str) -> Dict:
    lang = language_code(output_language)
    out = mock_predict(message)
    score = int(out.get("risk_score", 0))
    thresholds = DEFAULT_THRESHOLDS
    review_confidence_min = DEFAULT_REVIEW_CONFIDENCE_MIN
    cue_tags = detect_cue_tags(message)
    confidence = 0.5
    out["model_kind"] = "mock"
    out["temperature"] = 1.0
    out["severity_tier"] = _score_to_tier(score, thresholds)
    out["severity_thresholds"] = thresholds
    out["threshold_policy"] = ACTIVE_THRESHOLD_POLICY
    out["risk_label_score_based"] = _score_to_risk_label(score, thresholds)
    out["actions"] = actions_for_scam_type(out.get("scam_type", "Other"), lang)
    out["cue_tags"] = cue_tags
    out["cue_labels"] = cue_labels(cue_tags, lang)
    out["reason"] = reason_text(cue_tags, confidence=confidence, lang=lang)
    out["model_source"] = "mock_model"
    out["model_version"] = "mock_v1"
    out["model_confidence"] = confidence
    out["calibrated_confidence"] = confidence
    out["calibration_method"] = "mock_fallback"
    out["review_confidence_min"] = review_confidence_min
    out["review_recommended"] = False
    out["abstain"] = False
    out["review_reason"] = ""
    out["abstain_reason"] = ""
    out["decision_state"] = "auto"
    out["decision_label"] = "Auto decision"
    out["language"] = lang
    return out


def _build_review_fields(
    *,
    primary: Dict,
    review_confidence_min: float,
    consensus_label: str | None = None,
    consensus_count: int | None = None,
    total_models: int | None = None,
) -> Dict[str, object]:
    confidence_notes: List[str] = []
    confidence = float(primary.get("model_confidence", 0.0) or 0.0)
    label = str(primary.get("label", "Safe"))

    if confidence < review_confidence_min:
        confidence_notes.append(f"lower model confidence after calibration ({confidence * 100:.1f}%)")

    if consensus_count is not None and total_models is not None:
        if consensus_count < 2:
            confidence_notes.append(f"only {consensus_count}/{total_models} models agree")
        elif consensus_label and label != consensus_label:
            confidence_notes.append(f"top model differs from the {consensus_count}/{total_models} model consensus on {consensus_label}")
    elif consensus_label and label != consensus_label:
        confidence_notes.append(f"top model differs from the ensemble consensus on {consensus_label}")

    return {
        "review_confidence_min": float(review_confidence_min),
        "review_recommended": False,
        "abstain": False,
        "review_reason": "",
        "abstain_reason": "",
        "decision_state": "auto",
        "decision_label": "Auto decision",
        "confidence_note": "; ".join(confidence_notes),
    }


def compare_model_predictions(
    message: str,
    output_language: str = "English",
    primary_model_kind: str | None = "best",
    secondary_model_kind: str | None = None,
) -> Dict:
    ranked_kinds = [kind for kind, _ in _model_rankings()]
    if not ranked_kinds:
        ranked_kinds = list(MODEL_KINDS)

    primary_kind = _normalize_model_kind(primary_model_kind)
    if primary_kind == "best":
        primary_kind = ranked_kinds[0] if ranked_kinds else "best"
    if primary_kind not in MODEL_KINDS:
        primary_kind = "best"

    if secondary_model_kind is None:
        candidates = [kind for kind in ranked_kinds if kind != primary_kind]
        if not candidates:
            candidates = [kind for kind in MODEL_KINDS if kind != primary_kind]
        secondary_kind = candidates[0] if candidates else primary_kind
    else:
        secondary_kind = _normalize_model_kind(secondary_model_kind)
        if secondary_kind == "best":
            candidates = [kind for kind in ranked_kinds if kind != primary_kind]
            if not candidates:
                candidates = [kind for kind in MODEL_KINDS if kind != primary_kind]
            secondary_kind = candidates[0] if candidates else primary_kind
    if secondary_kind not in MODEL_KINDS:
        secondary_kind = primary_kind

    primary = predict_message(message, output_language=output_language, model_kind=primary_kind)
    secondary = predict_message(message, output_language=output_language, model_kind=secondary_kind)
    primary_label = str(primary.get("label", "Safe"))
    secondary_label = str(secondary.get("label", "Safe"))
    primary_score = int(primary.get("risk_score", 0) or 0)
    secondary_score = int(secondary.get("risk_score", 0) or 0)
    agreement = primary_label == secondary_label
    primary_version = str(primary.get("model_version", "unknown"))
    secondary_version = str(secondary.get("model_version", "unknown"))
    review_confidence_min = max(
        float(primary.get("review_confidence_min", DEFAULT_REVIEW_CONFIDENCE_MIN) or DEFAULT_REVIEW_CONFIDENCE_MIN),
        float(secondary.get("review_confidence_min", DEFAULT_REVIEW_CONFIDENCE_MIN) or DEFAULT_REVIEW_CONFIDENCE_MIN),
    )
    review_fields = _build_review_fields(
        primary=primary,
        review_confidence_min=review_confidence_min,
        consensus_label=secondary_label,
        consensus_count=2 if agreement else 1,
        total_models=2,
    )

    return {
        "primary": primary,
        "secondary": secondary,
        "primary_model_kind": primary_kind,
        "secondary_model_kind": secondary_kind,
        "agreement": agreement,
        "review_confidence_min": review_confidence_min,
        **review_fields,
        "risk_score_delta": primary_score - secondary_score,
        "confidence_delta": float(primary.get("model_confidence", 0.0) or 0.0) - float(secondary.get("model_confidence", 0.0) or 0.0),
        "summary": f"{primary_version}: {primary_label} {primary_score}/100 | {secondary_version}: {secondary_label} {secondary_score}/100",
    }


def compare_all_model_predictions(message: str, output_language: str = "English") -> Dict:
    ordered_kinds = _ordered_model_kinds()
    predictions = []
    for kind in ordered_kinds[:3]:
        result = predict_message(message, output_language=output_language, model_kind=kind)
        predictions.append(
            {
                "model_kind": kind,
                "prediction": result,
                "label": str(result.get("label", "Safe")),
                "scam_type": str(result.get("scam_type", "Other")),
                "risk_score": int(result.get("risk_score", 0) or 0),
                "model_confidence": float(result.get("model_confidence", 0.0) or 0.0),
                "model_source": str(result.get("model_source", "unknown")),
                "model_version": str(result.get("model_version", "unknown")),
                "type_source": str(result.get("type_source", "unknown")),
            }
        )

    if not predictions:
        fallback = _mock_fallback(message, output_language)
        predictions.append(
            {
                "model_kind": "mock",
                "prediction": fallback,
                "label": str(fallback.get("label", "Safe")),
                "scam_type": str(fallback.get("scam_type", "Other")),
                "risk_score": int(fallback.get("risk_score", 0) or 0),
                "model_confidence": float(fallback.get("model_confidence", 0.0) or 0.0),
                "model_source": str(fallback.get("model_source", "mock_model")),
                "model_version": str(fallback.get("model_version", "mock_v1")),
                "type_source": str(fallback.get("type_source", "unknown")),
            }
        )

    labels = [entry["label"] for entry in predictions]
    unique_labels = list(dict.fromkeys(labels))
    label_counts = {label: labels.count(label) for label in unique_labels}
    consensus_label = max(label_counts, key=label_counts.get) if label_counts else "Safe"
    consensus_count = int(label_counts.get(consensus_label, 0))
    unanimous = len(unique_labels) == 1
    primary = predictions[0]["prediction"]
    secondary = predictions[1]["prediction"] if len(predictions) > 1 else predictions[0]["prediction"]
    tertiary = predictions[2]["prediction"] if len(predictions) > 2 else predictions[-1]["prediction"]
    final = _build_final_ensemble_prediction(message, output_language, predictions)
    review_confidence_min = max(
        [float(entry.get("prediction", {}).get("review_confidence_min", DEFAULT_REVIEW_CONFIDENCE_MIN) or DEFAULT_REVIEW_CONFIDENCE_MIN) for entry in predictions]
        or [DEFAULT_REVIEW_CONFIDENCE_MIN]
    )
    review_fields = _build_review_fields(
        primary=final,
        review_confidence_min=review_confidence_min,
        consensus_label=consensus_label,
        consensus_count=consensus_count,
        total_models=len(predictions),
    )
    final.update(review_fields)
    summary = " | ".join(
        [
            f"{entry['model_version']}: {entry['label']} {entry['risk_score']}/100"
            for entry in predictions
        ]
    )

    return {
        "primary": primary,
        "secondary": secondary,
        "tertiary": tertiary,
        "final": final,
        "predictions": predictions,
        "primary_model_kind": predictions[0]["model_kind"],
        "secondary_model_kind": predictions[1]["model_kind"] if len(predictions) > 1 else predictions[0]["model_kind"],
        "tertiary_model_kind": predictions[2]["model_kind"] if len(predictions) > 2 else predictions[-1]["model_kind"],
        "agreement": unanimous,
        "agreement_label": "unanimous" if unanimous else f"{consensus_count}/{len(predictions)} agree",
        "consensus_label": consensus_label,
        "consensus_count": consensus_count,
        "review_confidence_min": review_confidence_min,
        **review_fields,
        "final_score_method": str(final.get("final_score_method", "median_ensemble_v1")),
        "model_outputs": final.get("model_outputs", _model_outputs_for_storage(predictions)),
        "risk_score_delta": int(final.get("risk_score", 0) or 0) - int(primary.get("risk_score", 0) or 0),
        "confidence_delta": float(final.get("model_confidence", 0.0) or 0.0) - float(primary.get("model_confidence", 0.0) or 0.0),
        "summary": summary,
    }


def predict_message(message: str, output_language: str = "English", model_kind: str = "best") -> Dict:
    try:
        bundle = load_model_bundle(model_kind)
    except Exception:
        bundle = None

    if bundle is None:
        return _mock_fallback(message, output_language)

    try:
        return _predict_with_bundle(message, output_language, bundle)
    except Exception:
        return _mock_fallback(message, output_language)
