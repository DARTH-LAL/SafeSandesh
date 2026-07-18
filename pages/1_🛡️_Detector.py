from __future__ import annotations

import html
import importlib
import json
import os
import re
import textwrap
import time
from difflib import SequenceMatcher
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

from src import db
from src import model_runtime as runtime_model
from src import utils as runtime_utils
from src.explainability import cue_labels as explain_cue_labels
from src.explainability import detect_cue_tags
from src.explainability import infer_message_language_name
from src.ui_theme import apply_theme, top_menu

# Streamlit reruns page code but can keep imported modules alive. Reload this
# small DB helper so schema/signature changes are picked up without a full restart.
db = importlib.reload(db)
runtime_model = importlib.reload(runtime_model)
runtime_utils = importlib.reload(runtime_utils)


APP_SHELL = os.environ.get("SAFESANDESH_APP_SHELL", "combined").strip().lower()
STANDALONE_CONSUMER_APP = APP_SHELL == "consumer"


SAMPLES = {
    "OTP Phishing": (
        "URGENT: Dear Customer, your SBI account will be BLOCKED in 24 hrs. "
        "Update KYC now and verify your OTP to avoid suspension. Click: bit.ly/sbi-kyc-upd"
    ),
    "KYC Fraud": (
        "Dear HDFC user, your account has been temporarily suspended due to incomplete KYC. "
        "Share your Aadhaar number and OTP to reactivate: hdfc-kyc.in/verify"
    ),
    "UPI Collect": (
        "Congratulations! You have received a collect request of Rs. 5000 from PhonePe Rewards. "
        "Accept now before it expires: upi://pay?pa=scam@upi"
    ),
    "Lottery Scam": (
        "You have WON Rs. 25,00,000 in the KBC Lottery 2024! To claim your prize, "
        "send your bank account details and Rs. 500 processing fee to: kbc.winner@gmail.com"
    ),
    "Hindi SMS": (
        "प्रिय ग्राहक, आपका SBI खाता 24 घंटे में बंद हो जाएगा। "
        "अभी अपना OTP साझा करें और KYC अपडेट करें: sbi-kyc.xyz/verify"
    ),
    "Safe Message": (
        "Hi! Your Zomato order #1234 has been picked up by the delivery partner. "
        "Estimated delivery: 30 mins. Track here: zomato.com/track/1234"
    ),
}

SCAM_TYPE_TOOLTIP = {
    "OTP": "Fake urgency + OTP request. Banks never ask for OTP over SMS or chat.",
    "KYC": "Credential-harvesting pretext. Verify only on official bank app/site.",
    "UPI": "Unexpected collect/payment prompts can reverse money flow from your account.",
    "Lottery": "Advance-fee lure. Prize claims with payment demand are fraudulent.",
    "Account-Block": "Account suspension pressure tactic to force rushed actions.",
    "Courier/Customs": "Fake parcel/customs fee demand using panic and quick payment prompts.",
    "Job/Loan": "Upfront fee or fake recruiter/lender identity pattern.",
    "Other": "Suspicious pattern detected; verify through official channels before acting.",
}

SCAN_LINES = [
    "loading india_db_v2.4...",
    "tokenizing message...",
    "running multilingual classifier...",
    "checking scam pattern index...",
    "computing risk score...",
    "calibrating confidence...",
    "generating explanation...",
]

LANG_CODE_TO_NAME = {
    "en": "English",
    "hi": "Hindi",
    "pa": "Punjabi",
    "ur": "Urdu",
}

LANG_NAME_TO_CODE = {v.lower(): k for k, v in LANG_CODE_TO_NAME.items()}

VERDICT_I18N = {
    "hi": {"safe": "सुरक्षित", "suspicious": "संदिग्ध", "phishing": "फ़िशिंग"},
    "pa": {"safe": "ਸੁਰੱਖਿਅਤ", "suspicious": "ਸ਼ੱਕੀ", "phishing": "ਫਿਸ਼ਿੰਗ"},
    "ur": {"safe": "محفوظ", "suspicious": "مشکوک", "phishing": "فشنگ"},
}

TIER_I18N = {
    "hi": {"low": "निम्न", "medium": "मध्यम", "high": "उच्च", "critical": "गंभीर"},
    "pa": {"low": "ਘੱਟ", "medium": "ਦਰਮਿਆਨਾ", "high": "ਉੱਚ", "critical": "ਗੰਭੀਰ"},
    "ur": {"low": "کم", "medium": "درمیانہ", "high": "زیادہ", "critical": "انتہائی"},
}

RESULT_UI_I18N = {
    "en": {
        "risk_analysis": "Risk Analysis",
        "model_confidence": "Model Confidence",
        "message_highlights": "Message Highlights",
        "paste_message": "Paste your message",
        "what_you_should_do": "What You Should Do",
        "output_language": "Output Language",
        "analyze_message": "▶ Analyze Message",
        "reanalyze_message": "↻ Re-Analyze Message",
        "auto_decision": "Auto Decision",
        "low_confidence": "Low Confidence",
        "category_tooltip_title": "What is this category?",
        "scale_low": "LOW",
        "scale_med": "MED",
        "scale_high": "HIGH",
        "scale_crit": "CRIT",
        "legend_high_risk": "High-risk cue",
        "legend_suspicious": "Suspicious phrase",
        "matched_cues": "Matched cues",
        "no_major_cues": "No major scam cues detected in this message pattern.",
        "some_suspicious": "Some suspicious patterns detected; the scanner is staying cautious.",
        "multiple_high_risk": "Multiple high-risk patterns matched this message.",
        "lower_confidence_signal": "Lower-confidence signal; use the score and model comparison together.",
        "auto_decision_text": "Automatic verdict based on the score, cue evidence, and model comparison.",
        "low_confidence_text": "Lower-confidence signal; the final verdict is still generated automatically.",
    },
    "hi": {
        "risk_analysis": "जोखिम विश्लेषण",
        "model_confidence": "मॉडल भरोसा",
        "message_highlights": "संदेश संकेत",
        "paste_message": "अपना संदेश पेस्ट करें",
        "what_you_should_do": "आपको क्या करना चाहिए",
        "output_language": "आउटपुट भाषा",
        "analyze_message": "▶ संदेश जांचें",
        "reanalyze_message": "↻ संदेश फिर जांचें",
        "auto_decision": "स्वचालित निर्णय",
        "low_confidence": "कम भरोसा",
        "category_tooltip_title": "यह श्रेणी क्या है?",
        "scale_low": "कम",
        "scale_med": "मध्यम",
        "scale_high": "उच्च",
        "scale_crit": "गंभीर",
        "legend_high_risk": "उच्च-जोखिम संकेत",
        "legend_suspicious": "संदिग्ध वाक्यांश",
        "matched_cues": "पहचाने गए संकेत",
        "no_major_cues": "इस संदेश में बड़े scam संकेत नहीं मिले।",
        "some_suspicious": "कुछ संदिग्ध संकेत मिले हैं, इसलिए scanner सावधानी दिखा रहा है।",
        "multiple_high_risk": "इस संदेश में कई उच्च-जोखिम संकेत मिले हैं।",
        "lower_confidence_signal": "मॉडल भरोसा कम है; score और model comparison को साथ में देखें।",
        "auto_decision_text": "यह स्वचालित निर्णय score, संकेतों और model comparison पर आधारित है।",
        "low_confidence_text": "मॉडल भरोसा कम है, लेकिन अंतिम verdict अभी भी automatically बनाया गया है।",
    },
    "pa": {
        "risk_analysis": "ਜੋਖਿਮ ਵਿਸ਼ਲੇਸ਼ਣ",
        "model_confidence": "ਮਾਡਲ ਭਰੋਸਾ",
        "message_highlights": "ਸੁਨੇਹਾ ਸੰਕੇਤ",
        "paste_message": "ਆਪਣਾ ਸੁਨੇਹਾ ਪੇਸਟ ਕਰੋ",
        "what_you_should_do": "ਤੁਹਾਨੂੰ ਕੀ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ",
        "output_language": "ਆਉਟਪੁੱਟ ਭਾਸ਼ਾ",
        "analyze_message": "▶ ਸੁਨੇਹਾ ਜਾਂਚੋ",
        "reanalyze_message": "↻ ਸੁਨੇਹਾ ਮੁੜ ਜਾਂਚੋ",
        "auto_decision": "ਆਟੋਮੈਟਿਕ ਫੈਸਲਾ",
        "low_confidence": "ਘੱਟ ਭਰੋਸਾ",
        "category_tooltip_title": "ਇਹ ਸ਼੍ਰੇਣੀ ਕੀ ਹੈ?",
        "scale_low": "ਘੱਟ",
        "scale_med": "ਮੱਧਮ",
        "scale_high": "ਉੱਚ",
        "scale_crit": "ਗੰਭੀਰ",
        "legend_high_risk": "ਉੱਚ-ਜੋਖਿਮ ਸੰਕੇਤ",
        "legend_suspicious": "ਸ਼ੱਕੀ ਵਾਕ",
        "matched_cues": "ਪਛਾਣੇ ਗਏ ਸੰਕੇਤ",
        "no_major_cues": "ਇਸ ਸੁਨੇਹੇ ਵਿੱਚ ਵੱਡੇ scam ਸੰਕੇਤ ਨਹੀਂ ਮਿਲੇ।",
        "some_suspicious": "ਕੁਝ ਸ਼ੱਕੀ ਸੰਕੇਤ ਮਿਲੇ ਹਨ, ਇਸ ਲਈ scanner ਸਾਵਧਾਨ ਹੈ।",
        "multiple_high_risk": "ਇਸ ਸੁਨੇਹੇ ਵਿੱਚ ਕਈ ਉੱਚ-ਜੋਖਿਮ ਸੰਕੇਤ ਮਿਲੇ ਹਨ।",
        "lower_confidence_signal": "ਮਾਡਲ ਭਰੋਸਾ ਘੱਟ ਹੈ; score ਅਤੇ model comparison ਨੂੰ ਇਕੱਠੇ ਵੇਖੋ।",
        "auto_decision_text": "ਇਹ ਆਟੋਮੈਟਿਕ ਫੈਸਲਾ score, ਸੰਕੇਤਾਂ ਅਤੇ model comparison 'ਤੇ ਆਧਾਰਿਤ ਹੈ।",
        "low_confidence_text": "ਮਾਡਲ ਭਰੋਸਾ ਘੱਟ ਹੈ, ਪਰ ਅੰਤਿਮ verdict ਹਾਲੇ ਵੀ automatically ਬਣਾਇਆ ਗਿਆ ਹੈ।",
    },
    "ur": {
        "risk_analysis": "خطرے کا تجزیہ",
        "model_confidence": "ماڈل اعتماد",
        "message_highlights": "پیغام کے اشارے",
        "paste_message": "اپنا پیغام پیسٹ کریں",
        "what_you_should_do": "آپ کو کیا کرنا چاہیے",
        "output_language": "آؤٹ پٹ زبان",
        "analyze_message": "▶ پیغام چیک کریں",
        "reanalyze_message": "↻ پیغام دوبارہ چیک کریں",
        "auto_decision": "خودکار فیصلہ",
        "low_confidence": "کم اعتماد",
        "category_tooltip_title": "یہ زمرہ کیا ہے؟",
        "scale_low": "کم",
        "scale_med": "درمیانہ",
        "scale_high": "زیادہ",
        "scale_crit": "انتہائی",
        "legend_high_risk": "زیادہ خطرے کا اشارہ",
        "legend_suspicious": "مشکوک جملہ",
        "matched_cues": "پہچانے گئے اشارے",
        "no_major_cues": "اس پیغام میں بڑے scam اشارے نہیں ملے۔",
        "some_suspicious": "کچھ مشکوک اشارے ملے ہیں، اس لیے scanner احتیاط دکھا رہا ہے۔",
        "multiple_high_risk": "اس پیغام میں کئی زیادہ خطرے والے اشارے ملے ہیں۔",
        "lower_confidence_signal": "ماڈل اعتماد کم ہے؛ score اور model comparison کو ساتھ دیکھیں۔",
        "auto_decision_text": "یہ خودکار فیصلہ score، اشاروں اور model comparison پر مبنی ہے۔",
        "low_confidence_text": "ماڈل اعتماد کم ہے، لیکن آخری verdict ابھی بھی automatically بنایا گیا ہے۔",
    },
}

CONFIDENCE_LEVEL_I18N = {
    "hi": {"very high": "बहुत उच्च", "high": "उच्च", "medium": "मध्यम", "low": "कम"},
    "pa": {"very high": "ਬਹੁਤ ਉੱਚ", "high": "ਉੱਚ", "medium": "ਮੱਧਮ", "low": "ਘੱਟ"},
    "ur": {"very high": "بہت زیادہ", "high": "زیادہ", "medium": "درمیانہ", "low": "کم"},
}


def _init_state() -> None:
    st.session_state.setdefault("detector_message_input", "")
    st.session_state.setdefault("detector_prefill_text", "")
    st.session_state.setdefault("detector_input_error", "")
    st.session_state.setdefault("detector_lang_code", "en")
    st.session_state.setdefault("detector_scans_today", 1247)
    st.session_state.setdefault("detector_history", [])
    st.session_state.setdefault("detector_panel_state", "waiting")
    st.session_state.setdefault("detector_typewriter_text", "")
    st.session_state.setdefault("detector_last_scan_db_id", None)

    if st.session_state.pop("detector_reset_requested", False):
        st.session_state["detector_message_input"] = ""
        st.session_state["detector_prefill_text"] = ""
        st.session_state["detector_typewriter_text"] = ""
        st.session_state["detector_input_error"] = ""
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_message", None)
        st.session_state.pop("last_language", None)
        st.session_state["detector_panel_state"] = "waiting"
        st.session_state["detector_last_scan_db_id"] = None

    pending_history_message = st.session_state.pop("detector_history_load_message", None)
    pending_history_lang = st.session_state.pop("detector_history_load_lang", None)
    if pending_history_message is not None:
        loaded_message = str(pending_history_message or "")
        # Load should reset the analysis panel so the message can be scanned fresh.
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_message", None)
        st.session_state.pop("last_language", None)
        st.session_state.pop("detector_report_text", None)
        st.session_state["detector_panel_state"] = "waiting"
        st.session_state["detector_last_scan_db_id"] = None
        st.session_state["detector_input_error"] = ""
        st.session_state.pop("detector_auto_analyze", None)
        _start_typewriter(loaded_message)
    if pending_history_lang in LANG_CODE_TO_NAME:
        st.session_state["detector_lang_code"] = pending_history_lang

    if "prefill_message" in st.session_state:
        _start_typewriter(st.session_state.pop("prefill_message"))


def _start_typewriter(text: str) -> None:
    payload = text or ""
    st.session_state["detector_typewriter_text"] = payload
    st.session_state["detector_prefill_text"] = payload
    st.session_state["detector_input_error"] = ""
    # Keep backend state immediately available so Analyze never sees empty input
    # while the frontend typewriter animation is still running.
    st.session_state["detector_message_input"] = payload


def _inject_textarea_typewriter_js(text: str) -> None:
    payload = json.dumps(text or "")
    components.html(
        f"""
        <script>
        (function () {{
          try {{
            const txt = {payload};
            const parentWin = window.parent;
            const doc = parentWin.document;
            const ta = doc.querySelector('textarea[aria-label="Message text"]');
            if (!ta) return;

            const timerKey = "__detectorTextareaTypingTimer";
            if (parentWin[timerKey]) {{
              try {{ parentWin.clearTimeout(parentWin[timerKey]); }} catch (e) {{}}
              parentWin[timerKey] = null;
            }}

            const nativeSetter = Object.getOwnPropertyDescriptor(
              parentWin.HTMLTextAreaElement.prototype,
              "value"
            ).set;
            const setValue = (el, val) => nativeSetter.call(el, val);

            let i = 0;
            const n = txt.length;
            const step = n > 180 ? 4 : (n > 90 ? 3 : 2);

            setValue(ta, "");
            ta.dispatchEvent(new Event("input", {{ bubbles: true }}));

            const tick = () => {{
              i = Math.min(n, i + step);
              setValue(ta, txt.slice(0, i));
              ta.dispatchEvent(new Event("input", {{ bubbles: true }}));
              if (i < n) {{
                parentWin[timerKey] = parentWin.setTimeout(tick, 16);
              }} else {{
                ta.dispatchEvent(new Event("change", {{ bubbles: true }}));
                parentWin[timerKey] = null;
              }}
            }};

            tick();
          }} catch (e) {{}}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def _label_class(label: str) -> str:
    l = (label or "").lower()
    if l == "safe":
        return "safe"
    if l == "suspicious":
        return "suspicious"
    return "phishing"


def _lang_code_from_name(language: str) -> str:
    return LANG_NAME_TO_CODE.get((language or "").strip().lower(), "en")


def _localized_verdict(label: str, lang_code: str) -> str:
    key = (label or "").strip().lower()
    if lang_code == "en":
        # Keep English labels in current title-case style.
        if key == "safe":
            return "Safe"
        if key == "suspicious":
            return "Suspicious"
        return "Phishing"
    return VERDICT_I18N.get(lang_code, {}).get(key, label or "Safe")


def _localized_tier(tier: str, lang_code: str) -> str:
    key = (tier or "").strip().lower()
    if lang_code == "en":
        return (tier or "Low").upper()
    return TIER_I18N.get(lang_code, {}).get(key, tier or "Low")


def _ui_text(lang_code: str, key: str) -> str:
    return RESULT_UI_I18N.get(lang_code, RESULT_UI_I18N["en"]).get(key, RESULT_UI_I18N["en"][key])


def _localized_confidence_level(level: str, lang_code: str) -> str:
    key = (level or "low").strip().lower()
    if lang_code == "en":
        return (level or "LOW").upper()
    return CONFIDENCE_LEVEL_I18N.get(lang_code, {}).get(key, level or "LOW")


def _localized_decision_label(label: str, lang_code: str) -> str:
    normalized = (label or "Auto decision").strip().lower().replace("_", " ")
    if normalized in {"auto decision", "auto", "automatic decision"}:
        return _ui_text(lang_code, "auto_decision")
    if normalized in {"low confidence", "uncertain", "needs review"}:
        return _ui_text(lang_code, "low_confidence")
    return label or _ui_text(lang_code, "auto_decision")


def _score_to_visible_label(score: int) -> str:
    value = max(0, min(100, int(score or 0)))
    if value <= 30:
        return "Safe"
    if value <= 60:
        return "Suspicious"
    return "Phishing"


def _score_to_visible_tier(score: int) -> str:
    value = max(0, min(100, int(score or 0)))
    if value <= 30:
        return "Low"
    if value <= 60:
        return "Medium"
    if value <= 80:
        return "High"
    return "Critical"


def _normalize_visible_result(result: dict) -> dict:
    normalized = dict(result or {})
    score = max(0, min(100, int(normalized.get("risk_score", 0) or 0)))
    normalized["risk_score"] = score
    normalized["label"] = _score_to_visible_label(score)
    normalized["risk_label_score_based"] = normalized["label"]
    normalized["severity_tier"] = _score_to_visible_tier(score)
    normalized["severity_thresholds"] = {
        "low_max": 30,
        "medium_max": 60,
        "high_max": 80,
        "critical_min": 81,
    }
    normalized["threshold_policy"] = "natural_user_facing_v1"
    return normalized


def _confidence_level(conf: float) -> str:
    if conf >= 0.90:
        return "VERY HIGH"
    if conf >= 0.75:
        return "HIGH"
    if conf >= 0.55:
        return "MEDIUM"
    return "LOW"


def _format_confidence_pct(conf: float) -> tuple[str, float]:
    pct = max(0.0, min(100.0, float(conf or 0.0) * 100.0))
    if 99.5 <= pct < 100.0:
        pct = 99.9
    text = f"{pct:.1f}%"
    return text, pct


def _confidence_desc(result: dict, lang_code: str = "en") -> str:
    if result.get("review_recommended"):
        return _ui_text(lang_code, "lower_confidence_signal")

    cue_tags = result.get("cue_tags") or []
    cues = explain_cue_labels(cue_tags, lang_code) if cue_tags else (result.get("cue_labels") or [])
    if cues:
        return f"{_ui_text(lang_code, 'matched_cues')}: {', '.join(cues[:4])}."

    label = (result.get("label") or "").lower()
    if label == "safe":
        return _ui_text(lang_code, "no_major_cues")
    if label == "suspicious":
        return _ui_text(lang_code, "some_suspicious")
    return _ui_text(lang_code, "multiple_high_risk")


SIMPLE_RISK_REASON_COPY = {
    "en": {
        "OTP_REQUEST": "It asks for an OTP or code. Real banks and apps do not ask you to share this in a message.",
        "KYC_UPDATE": "It asks you to update KYC or account details from a message instead of the official app.",
        "PAYMENT_REQUEST": "It mentions payment, UPI, or a collect request, which can be used to take money from you.",
        "ACCOUNT_THREAT": "It creates pressure by saying your account may be blocked, suspended, or needs urgent action.",
        "LINK_PRESENT": "It includes a link. Scam links can open fake pages that steal details.",
        "LOTTERY_PRIZE": "It uses a prize, reward, or lottery offer to make you act quickly.",
        "DELIVERY_CUSTOMS": "It uses a delivery or customs excuse, often to push a payment or verification.",
        "JOB_LOAN": "It uses a job or loan offer, which can lead to fake fees or stolen documents.",
    },
    "hi": {
        "OTP_REQUEST": "यह ओटीपी या कोड मांगता है। असली बैंक और ऐप संदेश में इसे साझा करने को नहीं कहते।",
        "KYC_UPDATE": "यह आधिकारिक ऐप के बजाय संदेश से KYC या खाते की जानकारी अपडेट करने को कहता है।",
        "PAYMENT_REQUEST": "इसमें भुगतान, UPI या collect request का जिक्र है, जिससे आपके पैसे जा सकते हैं।",
        "ACCOUNT_THREAT": "यह खाता बंद, ब्लॉक या तुरंत कार्रवाई जैसी बात कहकर दबाव बनाता है।",
        "LINK_PRESENT": "इसमें लिंक है। नकली लिंक आपकी जानकारी चुराने वाले पेज खोल सकते हैं।",
        "LOTTERY_PRIZE": "यह इनाम, reward या lottery का लालच देकर जल्दी कार्रवाई करवाने की कोशिश करता है।",
        "DELIVERY_CUSTOMS": "यह delivery या customs का बहाना इस्तेमाल करता है, अक्सर भुगतान या verification करवाने के लिए।",
        "JOB_LOAN": "यह नौकरी या loan offer का लालच देता है, जिससे fake fees या document चोरी हो सकती है।",
    },
    "pa": {
        "OTP_REQUEST": "ਇਹ ਓਟੀਪੀ ਜਾਂ ਕੋਡ ਮੰਗਦਾ ਹੈ। ਅਸਲੀ ਬੈਂਕ ਅਤੇ ਐਪ ਸੁਨੇਹੇ ਵਿੱਚ ਇਹ ਸਾਂਝਾ ਕਰਨ ਲਈ ਨਹੀਂ ਕਹਿੰਦੇ।",
        "KYC_UPDATE": "ਇਹ ਅਧਿਕਾਰਕ ਐਪ ਦੀ ਬਜਾਏ ਸੁਨੇਹੇ ਰਾਹੀਂ KYC ਜਾਂ ਖਾਤੇ ਦੀ ਜਾਣਕਾਰੀ ਅਪਡੇਟ ਕਰਨ ਲਈ ਕਹਿੰਦਾ ਹੈ।",
        "PAYMENT_REQUEST": "ਇਸ ਵਿੱਚ ਭੁਗਤਾਨ, UPI ਜਾਂ collect request ਦੀ ਗੱਲ ਹੈ, ਜਿਸ ਨਾਲ ਤੁਹਾਡੇ ਪੈਸੇ ਜਾ ਸਕਦੇ ਹਨ।",
        "ACCOUNT_THREAT": "ਇਹ ਖਾਤਾ ਬੰਦ, ਬਲਾਕ ਜਾਂ ਤੁਰੰਤ ਕਾਰਵਾਈ ਵਰਗੀਆਂ ਗੱਲਾਂ ਨਾਲ ਦਬਾਅ ਬਣਾਉਂਦਾ ਹੈ।",
        "LINK_PRESENT": "ਇਸ ਵਿੱਚ ਲਿੰਕ ਹੈ। ਨਕਲੀ ਲਿੰਕ ਤੁਹਾਡੀ ਜਾਣਕਾਰੀ ਚੋਰੀ ਕਰਨ ਵਾਲੇ ਪੰਨੇ ਖੋਲ੍ਹ ਸਕਦੇ ਹਨ।",
        "LOTTERY_PRIZE": "ਇਹ ਇਨਾਮ, reward ਜਾਂ lottery ਦਾ ਲਾਲਚ ਦੇ ਕੇ ਜਲਦੀ ਕਾਰਵਾਈ ਕਰਵਾਉਣ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰਦਾ ਹੈ।",
        "DELIVERY_CUSTOMS": "ਇਹ delivery ਜਾਂ customs ਦਾ ਬਹਾਨਾ ਵਰਤਦਾ ਹੈ, ਅਕਸਰ ਭੁਗਤਾਨ ਜਾਂ verification ਲਈ।",
        "JOB_LOAN": "ਇਹ ਨੌਕਰੀ ਜਾਂ loan offer ਦਾ ਲਾਲਚ ਦਿੰਦਾ ਹੈ, ਜਿਸ ਨਾਲ fake fees ਜਾਂ document ਚੋਰੀ ਹੋ ਸਕਦੀ ਹੈ।",
    },
    "ur": {
        "OTP_REQUEST": "یہ او ٹی پی یا کوڈ مانگتا ہے۔ اصل بینک اور ایپس پیغام میں اسے شیئر کرنے کو نہیں کہتے۔",
        "KYC_UPDATE": "یہ آفیشل ایپ کے بجائے پیغام سے KYC یا اکاؤنٹ معلومات اپ ڈیٹ کرنے کو کہتا ہے۔",
        "PAYMENT_REQUEST": "اس میں ادائیگی، UPI یا collect request کا ذکر ہے، جس سے آپ کے پیسے جا سکتے ہیں۔",
        "ACCOUNT_THREAT": "یہ اکاؤنٹ بند، بلاک یا فوری کارروائی جیسی باتوں سے دباؤ بناتا ہے۔",
        "LINK_PRESENT": "اس میں لنک ہے۔ جعلی لنک آپ کی معلومات چرانے والے صفحات کھول سکتے ہیں۔",
        "LOTTERY_PRIZE": "یہ انعام، reward یا lottery کا لالچ دے کر جلدی کارروائی کروانے کی کوشش کرتا ہے۔",
        "DELIVERY_CUSTOMS": "یہ delivery یا customs کا بہانہ استعمال کرتا ہے، اکثر ادائیگی یا verification کے لیے۔",
        "JOB_LOAN": "یہ نوکری یا loan offer کا لالچ دیتا ہے، جس سے fake fees یا documents چوری ہو سکتے ہیں۔",
    },
}


SIMPLE_SAFE_REASON_COPY = {
    "en": {
        "with_tags": [
            "This looks safe because it does not ask you to pay, click a suspicious link, or share secrets.",
            "Still, only use the official app or website if you want to check the message.",
        ],
        "no_tags": ["No strong scam signs were found in this message."],
        "suspicious_fallback": ["This message has some warning signs. Check it in the official app before doing anything."],
        "phishing_fallback": ["This message looks similar to known scam patterns. Do not act on it until you verify it directly."],
    },
    "hi": {
        "with_tags": [
            "यह संदेश सुरक्षित लगता है क्योंकि यह भुगतान, संदिग्ध लिंक पर क्लिक या गुप्त जानकारी साझा करने को नहीं कहता।",
            "फिर भी, संदेश जांचने के लिए केवल आधिकारिक ऐप या वेबसाइट का उपयोग करें।",
        ],
        "no_tags": ["इस संदेश में कोई मजबूत scam संकेत नहीं मिला।"],
        "suspicious_fallback": ["इस संदेश में कुछ चेतावनी संकेत हैं। कुछ भी करने से पहले आधिकारिक ऐप में जांचें।"],
        "phishing_fallback": ["यह संदेश known scam pattern जैसा लगता है। सीधे verify करने से पहले कोई कार्रवाई न करें।"],
    },
    "pa": {
        "with_tags": [
            "ਇਹ ਸੁਨੇਹਾ ਸੁਰੱਖਿਅਤ ਲੱਗਦਾ ਹੈ ਕਿਉਂਕਿ ਇਹ ਭੁਗਤਾਨ, ਸੰਦੇਹਜਨਕ ਲਿੰਕ 'ਤੇ ਕਲਿੱਕ ਜਾਂ ਗੁਪਤ ਜਾਣਕਾਰੀ ਸਾਂਝੀ ਕਰਨ ਲਈ ਨਹੀਂ ਕਹਿੰਦਾ।",
            "ਫਿਰ ਵੀ, ਸੁਨੇਹਾ ਚੈਕ ਕਰਨ ਲਈ ਕੇਵਲ ਅਧਿਕਾਰਕ ਐਪ ਜਾਂ ਵੈਬਸਾਈਟ ਵਰਤੋ।",
        ],
        "no_tags": ["ਇਸ ਸੁਨੇਹੇ ਵਿੱਚ ਕੋਈ ਮਜ਼ਬੂਤ scam ਸੰਕੇਤ ਨਹੀਂ ਮਿਲਿਆ।"],
        "suspicious_fallback": ["ਇਸ ਸੁਨੇਹੇ ਵਿੱਚ ਕੁਝ ਚੇਤਾਵਨੀ ਸੰਕੇਤ ਹਨ। ਕੁਝ ਕਰਨ ਤੋਂ ਪਹਿਲਾਂ ਅਧਿਕਾਰਕ ਐਪ ਵਿੱਚ ਚੈਕ ਕਰੋ।"],
        "phishing_fallback": ["ਇਹ ਸੁਨੇਹਾ known scam pattern ਵਰਗਾ ਲੱਗਦਾ ਹੈ। ਸਿੱਧੀ ਤਸਦੀਕ ਤੋਂ ਪਹਿਲਾਂ ਕੋਈ ਕਾਰਵਾਈ ਨਾ ਕਰੋ।"],
    },
    "ur": {
        "with_tags": [
            "یہ پیغام محفوظ لگتا ہے کیونکہ یہ ادائیگی، مشکوک لنک پر کلک یا خفیہ معلومات شیئر کرنے کو نہیں کہتا۔",
            "پھر بھی، پیغام چیک کرنے کے لیے صرف آفیشل ایپ یا ویب سائٹ استعمال کریں۔",
        ],
        "no_tags": ["اس پیغام میں کوئی مضبوط scam اشارہ نہیں ملا۔"],
        "suspicious_fallback": ["اس پیغام میں کچھ warning signs ہیں۔ کچھ کرنے سے پہلے آفیشل ایپ میں چیک کریں۔"],
        "phishing_fallback": ["یہ پیغام known scam pattern جیسا لگتا ہے۔ براہ راست verify کیے بغیر کوئی کارروائی نہ کریں۔"],
    },
}


REASON_HEADING_I18N = {
    "en": {
        "safe": "Why This Looks Safe",
        "suspicious": "Why This Needs Caution",
        "phishing": "Why This Is Risky",
    },
    "hi": {
        "safe": "यह सुरक्षित क्यों लगता है",
        "suspicious": "इसमें सावधानी क्यों चाहिए",
        "phishing": "यह जोखिम भरा क्यों है",
    },
    "pa": {
        "safe": "ਇਹ ਸੁਰੱਖਿਅਤ ਕਿਉਂ ਲੱਗਦਾ ਹੈ",
        "suspicious": "ਇਸ ਵਿੱਚ ਸਾਵਧਾਨੀ ਕਿਉਂ ਚਾਹੀਦੀ ਹੈ",
        "phishing": "ਇਹ ਜੋਖਿਮ ਭਰਿਆ ਕਿਉਂ ਹੈ",
    },
    "ur": {
        "safe": "یہ محفوظ کیوں لگتا ہے",
        "suspicious": "اس میں احتیاط کیوں چاہیے",
        "phishing": "یہ خطرناک کیوں ہے",
    },
}


DEFAULT_ACTION_I18N = {
    "en": ["Cross-check through official channels before acting."],
    "hi": ["कुछ भी करने से पहले आधिकारिक चैनल से जांच करें।"],
    "pa": ["ਕੁਝ ਕਰਨ ਤੋਂ ਪਹਿਲਾਂ ਅਧਿਕਾਰਕ ਚੈਨਲ ਰਾਹੀਂ ਪੁਸ਼ਟੀ ਕਰੋ।"],
    "ur": ["کچھ کرنے سے پہلے آفیشل ذرائع سے تصدیق کریں۔"],
}


def _simple_consumer_reasons(result: dict, message: str, verdict_class: str, lang_code: str = "en") -> list[str]:
    tags = list(dict.fromkeys(result.get("cue_tags") or detect_cue_tags(message or "")))
    reason_copy = SIMPLE_RISK_REASON_COPY.get(lang_code, SIMPLE_RISK_REASON_COPY["en"])
    fallback_copy = SIMPLE_SAFE_REASON_COPY.get(lang_code, SIMPLE_SAFE_REASON_COPY["en"])
    reasons = [reason_copy[tag] for tag in tags if tag in reason_copy]
    if verdict_class == "safe":
        if tags:
            return fallback_copy["with_tags"]
        return fallback_copy["no_tags"]
    if reasons:
        return reasons[:4]
    if verdict_class == "suspicious":
        return fallback_copy["suspicious_fallback"]
    return fallback_copy["phishing_fallback"]


def _consumer_reason_heading(verdict_class: str, lang_code: str = "en") -> str:
    headings = REASON_HEADING_I18N.get(lang_code, REASON_HEADING_I18N["en"])
    if verdict_class == "safe":
        return headings["safe"]
    if verdict_class == "suspicious":
        return headings["suspicious"]
    return headings["phishing"]


def _highlight_message(text: str, cue_tags: list[str] | None = None) -> str:
    out = html.escape(text or "")
    tags = set(cue_tags or detect_cue_tags(text or ""))
    patterns = []
    if "LINK_PRESENT" in tags:
        patterns.append((r"(https?://[^\s<]+|www\.[^\s<]+|\b[a-z0-9.-]+\.[a-z]{2,}/[^\s<]*)", "hl-crit"))
    if "OTP_REQUEST" in tags:
        patterns.append((r"\b(otp|ओटीपी|ਓਟੀਪੀ|او\s*ٹی\s*پی)\b", "hl-crit"))
    if "KYC_UPDATE" in tags:
        patterns.append((r"\b(kyc|aadhaar|aadhar|pan|केवाईसी|आधार|ਕੇਵਾਈਸੀ|ਆਧਾਰ|کے\s*وائی\s*سی|آدھار)\b", "hl-crit"))
    if "PAYMENT_REQUEST" in tags:
        patterns.append((r"\b(upi|gpay|google\s*pay|phonepe|paytm|collect|यूपीआई|ਯੂਪੀਆਈ|یو\s*پی\s*آئی)\b", "hl-crit"))
    if "ACCOUNT_THREAT" in tags:
        patterns.append(
            (
                r"\b(blocked|suspended|urgent|verify now|अत्यावश्यक|तत्काल|तुरंत|ब्लॉक|सत्यापित|सत्यापन|फौरन|जल्द|"
                r"ਤੁਰੰਤ|ਬਲਾਕ|ਤਸਦੀਕ|فوری|فوراً|بلاک|تصدیق)\b",
                "hl-crit",
            )
        )
    if "LOTTERY_PRIZE" in tags:
        patterns.append((r"\b(prize|reward|winner|lottery|इनाम|लॉटरी|ਇਨਾਮ|ਲਾਟਰੀ|انعام|لاٹری)\b", "hl-warn"))
    if "DELIVERY_CUSTOMS" in tags:
        patterns.append((r"\b(courier|parcel|delivery|shipment|customs|कूरियर|ਕੂਰੀਅਰ|کورئیر)\b", "hl-warn"))
    if "JOB_LOAN" in tags:
        patterns.append((r"\b(job|loan|hiring|salary|नौकरी|लोन|ਨੌਕਰੀ|ਲੋਨ|نوکری|قرض)\b", "hl-warn"))
    for pat, cls in patterns:
        out = re.sub(pat, lambda m: f"<span class='{cls}'>{m.group(0)}</span>", out, flags=re.IGNORECASE)
    # Streamlit Markdown can treat indented HTML after raw blank lines as code.
    # Keep multiline messages visually multiline without breaking the result card.
    return out.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def _validate_message_for_analysis(message: str) -> str | None:
    clean = (message or "").strip()
    if not clean:
        return "// paste a message first"
    if len(clean) < 3:
        return "// message too short to analyze"
    return None


def _build_report(result: dict, message: str, lang_name: str, comparison: dict | None = None) -> str:
    result = _normalize_visible_result(result)
    verdict_class = _label_class(str(result.get("label", "Safe")))
    lang_code = _lang_code_from_name(lang_name)
    reasons = _simple_consumer_reasons(result, message, verdict_class, lang_code)
    reasons_text = "\n".join([f"  - {reason}" for reason in reasons])
    actions = result.get("actions", [])
    actions_text = "\n".join([f"  {i + 1}. {x}" for i, x in enumerate(actions)])
    decision_label = result.get("decision_label", "Auto decision")
    review_reason = result.get("review_reason", "")
    review_line = f"Decision Note    : {review_reason}\n" if review_reason else ""
    cue_labels = result.get("cue_labels") or []
    cue_line = ", ".join(cue_labels) if cue_labels else "No explicit cue matched."
    score = int(result.get("risk_score", 0))
    risk_probs = result.get("risk_probs") or {}
    phish_prob = risk_probs.get("Phishing")
    if phish_prob is None:
        phish_prob = score / 100.0
    model_stack = str((comparison or {}).get("summary", "")).strip()
    model_stack_line = f"Model Stack      : {model_stack}\n" if model_stack else ""
    return (
        "SafeSandesh Scan Report\n"
        "=================================\n"
        f"Verdict          : {result.get('label', 'Safe')}\n"
        f"Scam Type        : {result.get('scam_type', 'Other')}\n"
        f"Risk Score       : {score}/100\n"
        f"Risk Basis       : P(Phishing) = {float(phish_prob) * 100:.1f}%\n"
        f"Severity Tier    : {result.get('severity_tier', 'Low')}\n"
        f"Model Confidence : {result.get('model_confidence', 0.0) * 100:.1f}%\n"
        f"Decision State   : {decision_label}\n"
        f"{review_line}"
        f"Output Language  : {lang_name}\n"
        f"{model_stack_line}"
        f"Detected Cues    : {cue_line}\n"
        "---------------------------------\n"
        f"Message          : {message}\n"
        "---------------------------------\n"
        "Why:\n"
        f"{reasons_text}\n"
        "Actions:\n"
        f"{actions_text}\n"
    )


def _build_report_pdf_bytes(report_text: str) -> bytes:
    # Minimal single-font PDF generator for plain text reports.
    wrapped_lines: list[str] = []
    for raw in (report_text or "").splitlines():
        line = (raw or "").replace("\t", "    ")
        chunks = textwrap.wrap(line, width=92) or [""]
        wrapped_lines.extend(chunks)

    max_lines = 54
    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[: max_lines - 1] + ["... (truncated)"]

    text_ops: list[str] = ["BT", "/F1 10 Tf", "50 792 Td", "14 TL"]
    for line in wrapped_lines:
        safe = (
            line.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
            .encode("latin-1", "replace")
            .decode("latin-1")
        )
        text_ops.append(f"({safe}) Tj")
        text_ops.append("T*")
    text_ops.append("ET")
    stream = ("\n".join(text_ops) + "\n").encode("latin-1", "replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        (f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"endstream"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
    ]

    pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{idx} 0 obj\n".encode("latin-1")
        pdf += obj + b"\nendobj\n"

    xref_start = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    pdf += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        pdf += f"{off:010d} 00000 n \n".encode("latin-1")
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("latin-1")
    pdf += f"startxref\n{xref_start}\n%%EOF\n".encode("latin-1")
    return pdf


def _render_block(raw: str) -> str:
    cleaned = textwrap.dedent(raw).strip("\n")
    return "\n".join([line.lstrip() for line in cleaned.splitlines()]).strip()


def _score_band(score: int) -> str:
    if score >= 81:
        return "Critical"
    if score >= 61:
        return "High"
    if score >= 31:
        return "Suspicious"
    return "Safe"


def _risk_class_from_score(score: int) -> str:
    value = int(score or 0)
    if value >= 81:
        return "critical"
    if value >= 61:
        return "phishing"
    if value >= 31:
        return "suspicious"
    return "safe"


def _cue_chips_html(cues: list[str]) -> str:
    if not cues:
        return "<span class='forensics-cue muted'>No explicit cue matched</span>"
    return "".join([f"<span class='forensics-cue'>{html.escape(cue)}</span>" for cue in cues[:5]])


SCAN_ROW_FIELDS = [
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


def _scan_row_to_dict(row: tuple) -> dict:
    values = list(row or [])
    values.extend([None] * max(0, len(SCAN_ROW_FIELDS) - len(values)))
    out = dict(zip(SCAN_ROW_FIELDS, values[: len(SCAN_ROW_FIELDS)]))
    out["id"] = int(out.get("id") or 0)
    out["risk_score"] = int(out.get("risk_score") or 0)
    out["model_confidence"] = float(out.get("model_confidence") or 0.0)
    out["comparison_risk_score"] = int(out.get("comparison_risk_score") or 0)
    out["comparison_model_confidence"] = float(out.get("comparison_model_confidence") or 0.0)
    out["comparison_tertiary_risk_score"] = int(out.get("comparison_tertiary_risk_score") or 0)
    out["comparison_tertiary_model_confidence"] = float(out.get("comparison_tertiary_model_confidence") or 0.0)
    out["review_recommended"] = bool(int(out.get("review_recommended") or 0))
    return out


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _format_case_time(ts: str) -> str:
    raw = str(ts or "").strip()
    try:
        return datetime.fromisoformat(raw).strftime("%b %d · %H:%M")
    except Exception:
        return raw.replace("T", " ")[:16] if raw else "Unknown time"


def _pattern_thread(tags: list[str]) -> str:
    story_map = {
        "ACCOUNT_THREAT": "urgency",
        "OTP_REQUEST": "OTP",
        "KYC_UPDATE": "KYC",
        "PAYMENT_REQUEST": "payment",
        "LINK_PRESENT": "link",
        "LOTTERY_PRIZE": "prize",
        "DELIVERY_CUSTOMS": "parcel",
        "JOB_LOAN": "job/loan bait",
    }
    thread = [story_map[tag] for tag in tags if tag in story_map]
    return " → ".join(thread[:4])


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

TRAJECTORY_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？।॥؟۔])\s+|\n+")
TRAJECTORY_CLAUSE_SPLIT_RE = re.compile(r"[;,؛،:：]\s+")
TRAJECTORY_STEP_MARKER_RE_BY_LANG = {
    "hi": re.compile(
        r"\b(?:"
        r"अत्यावश्यक|तत्काल|तुरंत|अस्थायी\s*रूप\s*से|ब्लॉक|सत्यापित(?:\s*करें)?|सत्यापन|"
        r"लिंक|क्लिक\s*करें|ओटीपी|OTP|साझा\s*करें|अपडेट\s*करें|के\s*लिए|और|फिर|तब"
        r")\b",
        re.IGNORECASE,
    ),
    "pa": re.compile(
        r"\b(?:"
        r"ਤੁਰੰਤ|ਅਸਥਾਈ\s*ਤੌਰ\s*ਤੇ|ਬਲਾਕ|ਤਸਦੀਕ(?:\s*ਕਰੋ)?|ਲਿੰਕ|ਕਲਿੱਕ\s*ਕਰੋ|ਓਟੀਪੀ|OTP|"
        r"ਸਾਂਝਾ\s*ਕਰੋ|ਅਪਡੇਟ\s*ਕਰੋ|ਲਈ|ਅਤੇ|ਫਿਰ"
        r")\b",
        re.IGNORECASE,
    ),
    "ur": re.compile(
        r"\b(?:"
        r"فوری|فوراً|عارضی\s*طور\s*پر|بلاک|تصدیق(?:\s*کریں)?|لنک|کلک\s*کریں|"
        r"او\s*ٹی\s*پی|اوٹیپی|OTP|شیئر\s*کریں|اپڈیٹ\s*کریں|کے\s*لیے|اور|پھر"
        r")\b",
        re.IGNORECASE,
    ),
}

TRAJECTORY_URGENT_RE = re.compile(
    r"\b(urgent|immediately|now|today|act now|blocked|suspended|verify|update|claim|reward|prize|final warning|limited|"
    r"अत्यावश्यक|तत्काल|तुरंत|ब्लॉक|सत्यापित|सत्यापन|फौरन|जल्द|"
    r"ਤੁਰੰਤ|ਬਲਾਕ|ਤਸਦੀਕ|"
    r"فوری|فوراً|بلاک|تصدیق)\b",
    re.IGNORECASE,
)
TRAJECTORY_LINK_RE = re.compile(r"https?://|www\.|\b[a-z0-9.-]+\.[a-z]{2,}/[^\s<]*", re.IGNORECASE)
TRAJECTORY_MONEY_RE = re.compile(r"(?:₹|\$|rs\.?|inr)\s?\d|\b\d{4,}\b", re.IGNORECASE)
TRAJECTORY_ACTION_RE = re.compile(
    r"\b(click|tap|send|share|pay|accept|open|submit|download|install|scan|verify|update|"
    r"क्लिक\s*करें|साझा\s*करें|उपयोग\s*करें|सत्यापित\s*करें|अपडेट\s*करें|"
    r"ਕਲਿੱਕ\s*ਕਰੋ|ਸਾਂਝਾ\s*ਕਰੋ|ਉਪਯੋਗ\s*ਕਰੋ|ਤਸਦੀਕ\s*ਕਰੋ|"
    r"کلک\s*کریں|شیئر\s*کریں|استعمال\s*کریں|تصدیق\s*کریں)\b",
    re.IGNORECASE,
)


def _split_on_marker_boundaries(text: str, marker_re: re.Pattern) -> list[str]:
    matches = list(marker_re.finditer(text))
    if len(matches) <= 1:
        return [text]

    segments: list[str] = []
    last = 0
    for match in matches:
        start = match.start()
        if start < last:
            continue
        piece = text[last:start].strip()
        if piece:
            segments.append(piece)
        last = start

    tail = text[last:].strip()
    if tail:
        segments.append(tail)
    return [seg for seg in segments if seg]


def _merge_short_trajectory_segments(segments: list[str], min_chars: int = 12) -> list[str]:
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


def _split_trajectory_segments(message: str, lang_name: str | None = None) -> list[str]:
    text = str(message or "").strip()
    if not text:
        return []

    lang_code = _lang_code_from_name(lang_name)
    segments = [seg.strip() for seg in TRAJECTORY_SENTENCE_SPLIT_RE.split(text) if seg and seg.strip()]

    if len(segments) <= 1 and len(text) > 100:
        clause_segments = [seg.strip() for seg in TRAJECTORY_CLAUSE_SPLIT_RE.split(text) if seg and seg.strip()]
        if len(clause_segments) > 1:
            segments = clause_segments

    step_splitter = TRAJECTORY_STEP_MARKER_RE_BY_LANG.get(lang_code)
    if step_splitter is not None:
        refined_segments: list[str] = []
        for segment in segments:
            if len(segment) > 18:
                pieces = _split_on_marker_boundaries(segment, step_splitter)
                if len(pieces) > 1:
                    refined_segments.extend(pieces)
                    continue
            refined_segments.append(segment)
        segments = _merge_short_trajectory_segments(refined_segments)

    if len(segments) <= 1 and len(text) > 120:
        words = text.split()
        if len(words) > 12:
            chunk_count = 4 if len(words) > 28 else 3 if len(words) > 16 else 2
            chunk_size = max(1, (len(words) + chunk_count - 1) // chunk_count)
            segments = [" ".join(words[i : i + chunk_size]).strip() for i in range(0, len(words), chunk_size)]
            segments = [seg for seg in segments if seg]

    if len(segments) > 8:
        segments = segments[:7] + [" ".join(segments[7:])]

    return segments


def _segment_evidence_weight(segment: str, tags: list[str]) -> float:
    text = str(segment or "")
    weight = sum(TRAJECTORY_TAG_WEIGHTS.get(tag, 4.0) for tag in tags)

    if len(tags) > 1:
        weight += 3.0 * (len(tags) - 1)
    if TRAJECTORY_URGENT_RE.search(text):
        weight += 6.0
    if TRAJECTORY_ACTION_RE.search(text):
        weight += 3.0
    if TRAJECTORY_LINK_RE.search(text):
        weight += 5.0
    if TRAJECTORY_MONEY_RE.search(text):
        weight += 2.0
    if len(text) > 120:
        weight += 1.5
    return weight


def _allocate_trajectory_lifts(raw_weights: list[float], target_score: int) -> list[int]:
    target = max(0, min(100, int(target_score or 0)))
    if not raw_weights:
        return []

    if target <= 0:
        return [0 for _ in raw_weights]

    total = float(sum(raw_weights))
    if total <= 0:
        base = target // len(raw_weights)
        remainder = target % len(raw_weights)
        return [base + (1 if idx < remainder else 0) for idx in range(len(raw_weights))]

    scaled = [target * (weight / total) for weight in raw_weights]
    floor_vals = [int(value) for value in scaled]
    remainder = target - sum(floor_vals)
    if remainder > 0:
        order = sorted(
            range(len(scaled)),
            key=lambda idx: (scaled[idx] - floor_vals[idx], raw_weights[idx]),
            reverse=True,
        )
        for idx in order[:remainder]:
            floor_vals[idx] += 1

    return floor_vals


def _build_risk_trajectory_html(result: dict, message: str, lang_name: str) -> str:
    lang_code = _lang_code_from_name(lang_name)
    primary_score = max(0, min(100, int(result.get("risk_score", 0) or 0)))
    segments = _split_trajectory_segments(message, lang_name)
    if not segments:
        return textwrap.dedent(
            """
            <div class="evidence-waterfall">
              <div class="evidence-waterfall-head">
                <div>
                  <div class="forensics-title">Risk Trajectory</div>
                  <div class="evidence-waterfall-sub">No message chunks were available for the waterfall trace.</div>
                </div>
              </div>
            </div>
            """
        ).strip()

    rows: list[dict] = []
    raw_weights: list[float] = []
    for idx, segment in enumerate(segments, start=1):
        cue_tags = list(dict.fromkeys(detect_cue_tags(segment)))
        cue_labels = explain_cue_labels(cue_tags, lang_code)
        raw_weight = _segment_evidence_weight(segment, cue_tags)
        rows.append(
            {
                "index": idx,
                "segment": segment,
                "excerpt": textwrap.shorten(segment, width=112, placeholder="…"),
                "cue_tags": cue_tags,
                "cue_labels": cue_labels,
                "raw_weight": raw_weight,
                "lift": 0,
                "cumulative": 0,
                "risk_class": "safe",
            }
        )
        raw_weights.append(raw_weight)

    lifts = _allocate_trajectory_lifts(raw_weights, primary_score)
    cumulative = 0
    for row, lift in zip(rows, lifts):
        cumulative = min(100, cumulative + int(lift))
        row["lift"] = int(lift)
        row["cumulative"] = cumulative
        row["risk_class"] = _risk_class_from_score(cumulative)

    strongest = max(rows, key=lambda row: (row["lift"], row["raw_weight"])) if rows else None
    path_text = "0"
    if rows:
        path_text = "0 → " + " → ".join(str(row["cumulative"]) for row in rows)

    strongest_cues = ", ".join(strongest["cue_labels"][:4]) if strongest and strongest["cue_labels"] else "No explicit cue matched"
    strongest_excerpt = strongest["excerpt"] if strongest else "No message chunk found."
    strongest_lift = int(strongest["lift"]) if strongest else 0
    strongest_raw = int(round(strongest["raw_weight"])) if strongest else 0
    decision_label = str(result.get("decision_label") or "Auto decision")
    if decision_label.strip().lower() in {"uncertain", "needs review", "needs_review"}:
        decision_label = "Auto decision"
    decision_reason = "Automatic verdict based on the score, cue evidence, and model comparison."
    score_band = _score_band(primary_score)

    summary_cards = [
        (
            "Trajectory Path",
            path_text,
            f"Final score {primary_score}/100 · {len(rows)} evidence chunks",
        ),
        (
            "Strongest Jump",
            f"+{strongest_lift} lift",
            f"{strongest_cues} · raw evidence {strongest_raw}",
        ),
        (
            "Decision Note",
            decision_label.upper(),
            decision_reason,
        ),
    ]

    summary_html = "".join(
        [
            f"""
            <div class="evidence-waterfall-card">
              <div class="evidence-waterfall-k">{html.escape(label)}</div>
              <div class="evidence-waterfall-v">{html.escape(value)}</div>
              <div class="evidence-waterfall-m">{html.escape(meta)}</div>
            </div>
            """
            for label, value, meta in summary_cards
        ]
    )

    step_html = []
    for row in rows:
        cue_html = _cue_chips_html(row["cue_labels"]) if row["cue_labels"] else "<span class='forensics-cue muted'>No explicit cue matched</span>"
        step_html.append(
            f"""
            <div class="evidence-waterfall-step {row['risk_class']}">
              <div class="evidence-waterfall-step-head">
                <div>
                  <div class="evidence-waterfall-step-k">Chunk {row['index']} · normalized +{row['lift']} · raw {row['raw_weight']:.0f}</div>
                  <div class="evidence-waterfall-step-v">{html.escape(row['excerpt'])}</div>
                </div>
                <div class="evidence-waterfall-step-score">{row['cumulative']}/100</div>
              </div>
              <div class="evidence-waterfall-meta">{cue_html}</div>
              <div class="evidence-waterfall-bar">
                <div class="evidence-waterfall-bar-fill {row['risk_class']}" style="width:{row['cumulative']}%"></div>
              </div>
              <div class="evidence-waterfall-note">Raw evidence weight {row['raw_weight']:.0f} → normalized lift +{row['lift']} toward the final score.</div>
            </div>
            """
        )

    return textwrap.dedent(
        f"""
        <div class="evidence-waterfall">
          <div class="evidence-waterfall-head">
            <div>
              <div class="forensics-title">Risk Trajectory</div>
              <div class="evidence-waterfall-sub">The message is split into chunks, each chunk earns a cue-based lift, and the cumulative path is normalized into the final 0-100 score.</div>
            </div>
            <div class="evidence-waterfall-hero">
              <div class="evidence-waterfall-hero-k">Final score</div>
              <div class="evidence-waterfall-hero-v">{primary_score}/100</div>
              <div class="evidence-waterfall-hero-m">{len(rows)} chunks · {html.escape(score_band)} band</div>
            </div>
          </div>
          <div class="evidence-waterfall-summary">{summary_html}</div>
          <div class="evidence-waterfall-scale"><span>0</span><span>30</span><span>60</span><span>80</span><span>100</span></div>
          <div class="evidence-waterfall-track">{''.join(step_html)}</div>
          <div class="evidence-waterfall-foot">This waterfall is a transparent cue trace, not a hidden token attribution map. It shows which message chunks pushed the score upward and how those chunks map to the final risk score.</div>
        </div>
        """
    ).strip()


def _build_case_memory_html(result: dict, message: str, comparison: dict | None, lang_name: str) -> str:
    current_scan_id = st.session_state.get("detector_last_scan_db_id")
    current_lang_code = _lang_code_from_name(lang_name)
    current_tags = list(dict.fromkeys((result.get("cue_tags") or detect_cue_tags(message or ""))))
    current_labels = explain_cue_labels(current_tags, current_lang_code)
    current_label = str(result.get("label", "Safe"))
    current_type = str(result.get("scam_type", "Other"))
    current_score = int(result.get("risk_score", 0) or 0)
    current_conf = float(result.get("model_confidence", 0.0) or 0.0)
    current_norm = _compact_text(message)
    current_excerpt = textwrap.shorten(str(message or "").strip(), width=124, placeholder="…") if str(message or "").strip() else "No message text."

    candidates: list[dict] = []
    for row in db.read_scans(limit=120):
        scan = _scan_row_to_dict(row)
        if current_scan_id and scan["id"] == int(current_scan_id):
            continue

        hist_message = str(scan.get("message", "") or "")
        hist_norm = _compact_text(hist_message)
        hist_tags = list(dict.fromkeys(detect_cue_tags(hist_message)))
        shared_tags = [tag for tag in current_tags if tag in hist_tags]
        shared_labels = explain_cue_labels(shared_tags, current_lang_code)
        cue_union = set(current_tags) | set(hist_tags)
        cue_overlap = len(set(current_tags) & set(hist_tags)) / len(cue_union) if cue_union else 0.0
        text_overlap = SequenceMatcher(None, current_norm, hist_norm).ratio() if current_norm and hist_norm else 0.0
        score_closeness = 1.0 - min(abs(current_score - int(scan["risk_score"])) / 100.0, 1.0)
        conf_closeness = 1.0 - min(abs(current_conf - float(scan["model_confidence"] or 0.0)), 1.0)
        same_type = 1.0 if current_type == str(scan.get("scam_type", "Other")) and current_type != "Other" else 0.0
        same_label = 1.0 if current_label == str(scan.get("label", "Safe")) else 0.0
        same_language = 1.0 if str(scan.get("language", "")).strip().lower() == str(lang_name).strip().lower() else 0.0
        similarity = (
            0.42 * cue_overlap
            + 0.24 * text_overlap
            + 0.12 * score_closeness
            + 0.08 * conf_closeness
            + 0.08 * same_type
            + 0.04 * same_label
            + 0.02 * same_language
        )
        if shared_tags:
            similarity += min(0.08, 0.02 * len(shared_tags))
        similarity = max(0.0, min(1.0, similarity))

        candidates.append(
            {
                "id": scan["id"],
                "ts": str(scan.get("ts", "")),
                "language": str(scan.get("language", "English")),
                "label": str(scan.get("label", "Safe")),
                "scam_type": str(scan.get("scam_type", "Other")),
                "risk_score": int(scan.get("risk_score", 0) or 0),
                "model_confidence": float(scan.get("model_confidence", 0.0) or 0.0),
                "model_version": str(scan.get("model_version", "unknown")),
                "model_source": str(scan.get("model_source", "unknown")),
                "reason": str(scan.get("reason", "")),
                "message": hist_message,
                "shared_tags": shared_tags,
                "shared_labels": shared_labels,
                "all_tags": explain_cue_labels(hist_tags, current_lang_code),
                "pattern_thread": _pattern_thread(shared_tags or hist_tags),
                "similarity": similarity * 100.0,
            }
        )

    candidates.sort(key=lambda item: item["similarity"], reverse=True)
    top_cases = candidates[:3]
    best_match = top_cases[0] if top_cases else None
    archive_count = len(candidates)

    if not top_cases:
        return textwrap.dedent(
            f"""
            <div class="case-memory-empty">
              <div class="forensics-title">Case Memory</div>
              <div class="case-memory-empty-k">No prior cases found</div>
              <div class="case-memory-empty-v">Run a few more scans to populate the archive and unlock similarity comparisons.</div>
              <div class="case-memory-empty-m">This panel will surface the closest historical scans, their shared cues, and the attack thread they form.</div>
            </div>
            """
        ).strip()

    node_slots = [
        ("echo-0", "170,120", "170,120"),
        ("echo-1", "830,120", "830,120"),
        ("echo-2", "500,470", "500,470"),
    ]
    line_map = {
        "echo-0": ((500, 280), (170, 120)),
        "echo-1": ((500, 280), (830, 120)),
        "echo-2": ((500, 280), (500, 470)),
    }

    memory_nodes = []
    line_html = []
    for idx, case in enumerate(top_cases):
        slot, _, _ = node_slots[idx]
        start, end = line_map[slot]
        line_html.append(
            f"<line class='memory-line' x1='{start[0]}' y1='{start[1]}' x2='{end[0]}' y2='{end[1]}' />"
        )
        cue_line = ", ".join(case["shared_labels"][:4]) if case["shared_labels"] else "No shared cue labels"
        pattern_thread = case["pattern_thread"] or "No recurring thread"
        message_preview = textwrap.shorten(case["message"], width=86, placeholder="…") if case["message"] else "No preview"
        memory_nodes.append(
            textwrap.dedent(
                f"""
                <div class="memory-node {slot}">
                  <div class="memory-node-k">Memory Echo #{idx + 1}</div>
                  <div class="memory-node-v">{case["similarity"]:.0f}% similar</div>
                  <div class="memory-node-m">{html.escape(case["model_version"])} · {html.escape(case["language"])} · {html.escape(case["label"])} {case["risk_score"]}/100</div>
                  <div class="memory-node-line">Shared cues: {html.escape(cue_line)}</div>
                  <div class="memory-node-line">Pattern thread: {html.escape(pattern_thread)}</div>
                  <div class="memory-node-line">{html.escape(_format_case_time(case["ts"]))}</div>
                  <div class="memory-node-excerpt">{html.escape(message_preview)}</div>
                </div>
                """
            ).strip()
        )

    current_chip_row = "".join(
        [f"<span class='memory-chip'>{html.escape(label)}</span>" for label in current_labels[:4]]
    )
    if not current_chip_row:
        current_chip_row = "<span class='memory-chip muted'>No cue labels</span>"

    best_shared = ", ".join(best_match["shared_labels"][:4]) if best_match and best_match["shared_labels"] else "No shared cues"
    best_thread = best_match["pattern_thread"] if best_match else ""
    best_meta = ""
    if best_match:
        best_meta = (
            f"{best_match['label']} · {best_match['risk_score']}/100 · {best_match['language']} · "
            f"{_format_case_time(best_match['ts'])}"
        )

    story_bits = [x for x in [best_thread, best_shared] if x]
    story_text = " | ".join(story_bits) if story_bits else "No recurring thread surfaced yet."
    top_match_text = f"{best_match['similarity']:.0f}%" if best_match else "0%"

    return textwrap.dedent(
        f"""
        <div class="case-memory">
          <div class="case-memory-head">
            <div>
              <div class="forensics-title">Case Memory Orbit</div>
              <div class="case-memory-sub">The archive pulls the closest prior scans and shows the shared scam cues that tie them together.</div>
            </div>
            <div class="case-memory-badge">Top match {top_match_text}</div>
          </div>
          <div class="case-memory-meta">Archive scans: {archive_count} · Current verdict: {html.escape(_localized_verdict(current_label, current_lang_code))} · {current_score}/100</div>
          <div class="memory-orbit">
            <svg class="memory-lines" viewBox="0 0 1000 560" preserveAspectRatio="none" aria-hidden="true">
              {''.join(line_html)}
            </svg>
            <div class="memory-node current">
              <div class="memory-node-k">Current Scan</div>
              <div class="memory-node-v">{html.escape(_localized_verdict(current_label, current_lang_code))} {current_score}/100</div>
              <div class="memory-node-m">{html.escape(lang_name)} · {current_conf * 100:.1f}% confidence</div>
              <div class="memory-node-line">{html.escape(current_excerpt)}</div>
              <div class="memory-chip-row">{current_chip_row}</div>
            </div>
            {''.join(memory_nodes)}
          </div>
          <div class="memory-story">
            <div class="memory-story-k">Attack thread</div>
            <div class="memory-story-v">{html.escape(story_text)}</div>
            <div class="memory-story-m">{html.escape(best_meta or "Run another scan to surface closer memory echoes.")}</div>
          </div>
        </div>
        """
    ).strip()


def _build_forensics_studio(result: dict, comparison: dict | None, lang_name: str, message: str = "") -> str:
    predictions = list((comparison or {}).get("predictions", []) or [])
    if not predictions:
        predictions = [
            {
                "model_kind": "primary",
                "model_version": str(result.get("model_version", "unknown")),
                "model_source": str(result.get("model_source", "unknown")),
                "prediction": result,
                "label": result.get("label", "Safe"),
                "scam_type": result.get("scam_type", "Other"),
                "risk_score": int(result.get("risk_score", 0)),
                "model_confidence": float(result.get("model_confidence", 0.0)),
                "type_source": str(result.get("type_source", "unknown")),
            }
        ]

    primary_score = int(result.get("risk_score", 0))
    primary_probs = result.get("risk_probs") or {}
    primary_phish_prob = primary_probs.get("Phishing")
    if primary_phish_prob is None:
        primary_phish_prob = primary_score / 100.0
    cue_labels = result.get("cue_labels") or []
    review_reason = str(result.get("review_reason") or result.get("abstain_reason") or "")
    decision_text = review_reason or "Confidence and agreement stayed inside the automatic decision range."
    score_band = _score_band(primary_score)
    stack_summary = str((comparison or {}).get("summary", "")).strip()

    style = textwrap.dedent(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Oxanium:wght@400;600;700;800&family=Share+Tech+Mono&display=swap');
          :root {
            color-scheme: dark;
          }
          * {
            box-sizing: border-box;
          }
          body {
            margin: 0;
            background: transparent;
            color: #eafcf7;
            font-family: 'Share Tech Mono', monospace;
          }
          .forensics-studio {
            background: rgba(6, 11, 16, 0.84);
            border: 1px solid rgba(0, 212, 255, 0.18);
            box-shadow:
              0 0 0 1px rgba(0, 212, 255, 0.06),
              0 0 18px rgba(0, 212, 255, 0.05);
            padding: 14px 14px 12px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            color: #eafcf7;
          }
          .forensics-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
          }
          .forensics-guide {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 10px 12px;
            border: 1px solid rgba(0, 212, 255, 0.16);
            background: rgba(0, 212, 255, 0.04);
          }
          .forensics-guide-chip {
            padding: 7px 10px;
            border: 1px solid rgba(0, 212, 255, 0.18);
            background: rgba(0, 212, 255, 0.05);
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.62rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(200, 240, 224, 0.82);
          }
          .forensics-stack-wrap {
            display: grid;
            grid-template-columns: minmax(180px, 240px) minmax(0, 1fr);
            gap: 12px;
            padding: 12px 14px;
            border: 1px solid rgba(0, 212, 255, 0.18);
            background: rgba(0, 212, 255, 0.04);
          }
          .forensics-stack-copy {
            min-width: 0;
          }
          .forensics-stack-k {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.58rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(200, 240, 224, 0.56);
            white-space: nowrap;
          }
          .forensics-stack-v {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.66rem;
            line-height: 1.55;
            color: rgba(234, 252, 247, 0.76);
            margin-top: 4px;
          }
          .forensics-stack-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: flex-end;
          }
          .forensics-stack-chip {
            min-width: 150px;
            padding: 8px 10px;
            border: 1px solid rgba(0, 212, 255, 0.18);
            background: rgba(0, 0, 0, 0.22);
            display: flex;
            flex-direction: column;
            gap: 3px;
          }
          .forensics-stack-chip-k {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.56rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(200, 240, 224, 0.56);
          }
          .forensics-stack-chip-v {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.64rem;
            line-height: 1.45;
            color: #eafcf7;
          }
          .forensics-title {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.74rem;
            letter-spacing: 0.12em;
            color: rgba(200, 240, 224, 0.72);
            text-transform: uppercase;
            margin-bottom: 4px;
          }
          .forensics-sub,
          .forensics-note,
          .forensics-stack-summary {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.7rem;
            line-height: 1.6;
            color: rgba(234, 252, 247, 0.7);
          }
          .forensics-details {
            border: 1px solid rgba(0, 212, 255, 0.18);
            background: rgba(6, 11, 16, 0.84);
          }
          .forensics-details[open] {
            box-shadow:
              0 0 0 1px rgba(0, 212, 255, 0.05),
              0 0 18px rgba(0, 212, 255, 0.04);
          }
          .forensics-details summary {
            list-style: none;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 12px 14px;
            cursor: pointer;
            background: rgba(0, 0, 0, 0.22);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
          }
          .forensics-details summary::-webkit-details-marker {
            display: none;
          }
          .forensics-details-copy {
            min-width: 0;
          }
          .forensics-details-k {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.58rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(200, 240, 224, 0.56);
          }
          .forensics-details-v {
            margin-top: 3px;
            font-family: 'Oxanium', sans-serif;
            font-size: 0.92rem;
            font-weight: 800;
            line-height: 1.35;
            color: #eafcf7;
          }
          .forensics-details-m {
            margin-top: 4px;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.66rem;
            line-height: 1.55;
            color: rgba(234, 252, 247, 0.72);
          }
          .forensics-details-chip {
            align-self: center;
            padding: 8px 10px;
            border: 1px solid rgba(0, 212, 255, 0.24);
            background: rgba(0, 212, 255, 0.06);
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.62rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #00d4ff;
            white-space: nowrap;
          }
          .forensics-details[open] .forensics-details-chip {
            color: #00ff9f;
            border-color: rgba(0, 255, 159, 0.26);
            background: rgba(0, 255, 159, 0.06);
          }
          .forensics-details-body {
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 12px;
          }
          .forensics-panel {
            padding: 12px 14px;
            border: 1px solid rgba(0, 212, 255, 0.18);
            background: rgba(6, 11, 16, 0.84);
            display: flex;
            flex-direction: column;
            gap: 12px;
          }
          .forensics-panel + .forensics-panel {
            margin-top: 2px;
          }
          .forensics-panel-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
          }
          .forensics-panel-body {
            display: flex;
            flex-direction: column;
            gap: 12px;
          }
          .case-memory {
            padding: 14px;
            border: 1px solid rgba(0, 212, 255, 0.18);
            background: linear-gradient(180deg, rgba(0, 255, 159, 0.06), rgba(0, 212, 255, 0.04));
            display: flex;
            flex-direction: column;
            gap: 10px;
          }
          .case-memory-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
          }
          .case-memory-sub,
          .case-memory-meta,
          .memory-story-m {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.68rem;
            line-height: 1.6;
            color: rgba(234, 252, 247, 0.72);
          }
          .case-memory-badge {
            padding: 8px 10px;
            border: 1px solid rgba(0, 255, 159, 0.28);
            background: rgba(0, 255, 159, 0.06);
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.64rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #00ff9f;
            white-space: nowrap;
          }
          .case-memory-empty {
            padding: 16px;
            border: 1px dashed rgba(0, 212, 255, 0.28);
            background: rgba(0, 212, 255, 0.04);
            display: flex;
            flex-direction: column;
            gap: 6px;
          }
          .case-memory-empty-k,
          .case-memory-empty-v,
          .case-memory-empty-m {
            font-family: 'Share Tech Mono', monospace;
            line-height: 1.6;
          }
          .case-memory-empty-k {
            font-size: 0.58rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(200, 240, 224, 0.56);
          }
          .case-memory-empty-v {
            font-size: 0.82rem;
            color: #eafcf7;
          }
          .case-memory-empty-m {
            font-size: 0.66rem;
            color: rgba(234, 252, 247, 0.7);
          }
          .memory-orbit {
            position: relative;
            min-height: 560px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            background:
              radial-gradient(circle at center, rgba(0, 255, 159, 0.07), rgba(0, 0, 0, 0) 48%),
              rgba(0, 0, 0, 0.24);
            overflow: hidden;
          }
          .memory-lines {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
          }
          .memory-line {
            stroke: rgba(0, 212, 255, 0.28);
            stroke-width: 2.5;
            stroke-dasharray: 8 8;
            filter: drop-shadow(0 0 8px rgba(0, 212, 255, 0.18));
          }
          .memory-node {
            position: absolute;
            width: 280px;
            padding: 12px 12px 10px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(8, 13, 18, 0.92);
            display: flex;
            flex-direction: column;
            gap: 6px;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
          }
          .memory-node.current {
            left: 50%;
            top: 50%;
            width: 350px;
            transform: translate(-50%, -50%);
            border-color: rgba(0, 255, 159, 0.4);
            box-shadow: 0 0 28px rgba(0, 255, 159, 0.16);
          }
          .memory-node.echo-0 {
            left: 16px;
            top: 18px;
          }
          .memory-node.echo-1 {
            right: 16px;
            top: 18px;
          }
          .memory-node.echo-2 {
            left: 50%;
            bottom: 18px;
            width: 350px;
            transform: translateX(-50%);
          }
          .memory-node-k,
          .memory-story-k {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.58rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(200, 240, 224, 0.56);
          }
          .memory-node-v {
            font-family: 'Oxanium', sans-serif;
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.3;
            color: #eafcf7;
          }
          .memory-node-m,
          .memory-node-line,
          .memory-node-excerpt {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.64rem;
            line-height: 1.55;
            color: rgba(234, 252, 247, 0.72);
          }
          .memory-node-line {
            color: rgba(234, 252, 247, 0.84);
          }
          .memory-node-excerpt {
            padding-top: 4px;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            color: rgba(234, 252, 247, 0.58);
          }
          .memory-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
          }
          .memory-chip {
            display: inline-flex;
            align-items: center;
            padding: 4px 7px;
            border: 1px solid rgba(0, 212, 255, 0.18);
            background: rgba(0, 212, 255, 0.06);
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.58rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #eafcf7;
          }
          .memory-chip.muted {
            color: rgba(234, 252, 247, 0.48);
          }
          .memory-story {
            padding: 12px 14px;
            border: 1px solid rgba(0, 255, 159, 0.16);
            background: rgba(0, 255, 159, 0.04);
            display: flex;
            flex-direction: column;
            gap: 6px;
          }
          .evidence-waterfall {
            padding: 14px;
            border: 1px solid rgba(255, 221, 87, 0.16);
            background: linear-gradient(180deg, rgba(255, 221, 87, 0.06), rgba(0, 0, 0, 0.24));
            display: flex;
            flex-direction: column;
            gap: 10px;
          }
          .evidence-waterfall-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
          }
          .evidence-waterfall-sub,
          .evidence-waterfall-foot {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.68rem;
            line-height: 1.6;
            color: rgba(234, 252, 247, 0.72);
          }
          .evidence-waterfall-hero {
            min-width: 190px;
            padding: 10px 12px;
            border: 1px solid rgba(255, 221, 87, 0.22);
            background: rgba(255, 221, 87, 0.05);
            text-align: right;
          }
          .evidence-waterfall-hero-k,
          .evidence-waterfall-k,
          .evidence-waterfall-step-k {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.58rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(200, 240, 224, 0.56);
          }
          .evidence-waterfall-hero-v {
            font-family: 'Oxanium', sans-serif;
            font-size: 1.55rem;
            font-weight: 800;
            color: #ffdd57;
            line-height: 1.05;
            margin-top: 2px;
            text-shadow: 0 0 12px rgba(255, 221, 87, 0.24);
          }
          .evidence-waterfall-hero-m {
            margin-top: 4px;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.6rem;
            color: #ffdd57;
            letter-spacing: 0.06em;
          }
          .evidence-waterfall-summary {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
          }
          .evidence-waterfall-card {
            padding: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(0, 0, 0, 0.26);
            display: flex;
            flex-direction: column;
            gap: 6px;
          }
          .evidence-waterfall-v {
            font-family: 'Oxanium', sans-serif;
            font-size: 1rem;
            font-weight: 800;
            color: #eafcf7;
            line-height: 1.35;
          }
          .evidence-waterfall-m {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.66rem;
            line-height: 1.55;
            color: rgba(234, 252, 247, 0.72);
          }
          .evidence-waterfall-scale {
            display: flex;
            justify-content: space-between;
            gap: 8px;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.56rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(200, 240, 224, 0.54);
          }
          .evidence-waterfall-track {
            display: flex;
            flex-direction: column;
            gap: 8px;
          }
          .evidence-waterfall-step {
            padding: 11px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(0, 0, 0, 0.26);
            display: flex;
            flex-direction: column;
            gap: 8px;
          }
          .evidence-waterfall-step.safe {
            border-color: rgba(0, 255, 159, 0.2);
          }
          .evidence-waterfall-step.suspicious {
            border-color: rgba(255, 221, 87, 0.22);
          }
          .evidence-waterfall-step.phishing {
            border-color: rgba(255, 85, 119, 0.28);
          }
          .evidence-waterfall-step.critical {
            border-color: rgba(255, 85, 119, 0.38);
            box-shadow: 0 0 18px rgba(255, 85, 119, 0.08);
          }
          .evidence-waterfall-step-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 10px;
          }
          .evidence-waterfall-step-v {
            margin-top: 4px;
            font-family: 'Oxanium', sans-serif;
            font-size: 0.86rem;
            line-height: 1.45;
            color: #eafcf7;
          }
          .evidence-waterfall-step-score {
            font-family: 'Oxanium', sans-serif;
            font-size: 1rem;
            font-weight: 800;
            line-height: 1;
            color: #ffdd57;
            white-space: nowrap;
          }
          .evidence-waterfall-step.safe .evidence-waterfall-step-score {
            color: #00ff9f;
          }
          .evidence-waterfall-step.suspicious .evidence-waterfall-step-score {
            color: #ffdd57;
          }
          .evidence-waterfall-step.phishing .evidence-waterfall-step-score,
          .evidence-waterfall-step.critical .evidence-waterfall-step-score {
            color: #ff5577;
          }
          .evidence-waterfall-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
          }
          .evidence-waterfall-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.07);
            overflow: hidden;
          }
          .evidence-waterfall-bar-fill {
            height: 100%;
          }
          .evidence-waterfall-bar-fill.safe {
            background: linear-gradient(90deg, #00ff9f, #00d4ff);
          }
          .evidence-waterfall-bar-fill.suspicious {
            background: linear-gradient(90deg, #00ff9f, #ffdd57);
          }
          .evidence-waterfall-bar-fill.phishing,
          .evidence-waterfall-bar-fill.critical {
            background: linear-gradient(90deg, #ffdd57, #ff5577);
          }
          .evidence-waterfall-note {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.64rem;
            line-height: 1.55;
            color: rgba(234, 252, 247, 0.72);
          }
          .model-debate {
            padding: 14px;
            border: 1px solid rgba(0, 212, 255, 0.18);
            background: linear-gradient(180deg, rgba(6, 11, 16, 0.92), rgba(6, 11, 16, 0.8));
            box-shadow:
              0 0 0 1px rgba(0, 212, 255, 0.05),
              0 0 18px rgba(0, 212, 255, 0.04);
            display: flex;
            flex-direction: column;
            gap: 10px;
          }
          .model-debate-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
          }
          .model-debate-sub,
          .model-debate-judge-m,
          .model-debate-judge-note {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.68rem;
            line-height: 1.6;
            color: rgba(234, 252, 247, 0.72);
          }
          .model-debate-badge {
            min-width: 160px;
            padding: 10px 12px;
            border: 1px solid rgba(0, 212, 255, 0.24);
            background: rgba(0, 212, 255, 0.06);
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.65rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #00d4ff;
            text-align: center;
            white-space: nowrap;
          }
          .model-debate-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
          }
          .model-debate-card {
            padding: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(6, 11, 16, 0.84);
            display: flex;
            flex-direction: column;
            gap: 8px;
          }
          .model-debate-card.prosecution {
            border-color: rgba(255, 85, 119, 0.3);
          }
          .model-debate-card.bench {
            border-color: rgba(255, 221, 87, 0.24);
          }
          .model-debate-card.defense {
            border-color: rgba(0, 255, 159, 0.26);
          }
          .model-debate-card-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 10px;
          }
          .model-debate-role {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.58rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(200, 240, 224, 0.56);
          }
          .model-debate-model {
            margin-top: 3px;
            font-family: 'Oxanium', sans-serif;
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.35;
            color: #eafcf7;
          }
          .model-debate-meta {
            margin-top: 4px;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.62rem;
            line-height: 1.55;
            color: rgba(234, 252, 247, 0.6);
          }
          .model-debate-score {
            font-family: 'Oxanium', sans-serif;
            font-size: 1.25rem;
            font-weight: 800;
            color: #ffdd57;
            line-height: 1;
            white-space: nowrap;
          }
          .model-debate-card.prosecution .model-debate-score {
            color: #ff5577;
          }
          .model-debate-card.defense .model-debate-score {
            color: #00ff9f;
          }
          .model-debate-opening {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.68rem;
            line-height: 1.6;
            color: #eafcf7;
            padding-top: 2px;
          }
          .model-debate-quote {
            font-family: 'Oxanium', sans-serif;
            font-size: 0.88rem;
            line-height: 1.5;
            color: rgba(234, 252, 247, 0.88);
            padding: 10px 11px;
            border-left: 2px solid rgba(255, 255, 255, 0.16);
            background: rgba(255, 255, 255, 0.03);
          }
          .model-debate-card.prosecution .model-debate-quote {
            border-left-color: rgba(255, 85, 119, 0.5);
          }
          .model-debate-card.bench .model-debate-quote {
            border-left-color: rgba(255, 221, 87, 0.5);
          }
          .model-debate-card.defense .model-debate-quote {
            border-left-color: rgba(0, 255, 159, 0.5);
          }
          .model-debate-line {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.66rem;
            line-height: 1.55;
            color: rgba(234, 252, 247, 0.82);
          }
          .model-debate-cues {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
          }
          .model-debate-footer {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.62rem;
            line-height: 1.55;
            color: rgba(234, 252, 247, 0.62);
          }
          .model-debate-judge {
            padding: 12px 14px;
            border: 1px solid rgba(0, 212, 255, 0.2);
            background: rgba(0, 212, 255, 0.04);
            display: flex;
            flex-direction: column;
            gap: 6px;
          }
          .model-debate-judge-k {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.58rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(200, 240, 224, 0.56);
          }
          .model-debate-judge-v {
            font-family: 'Oxanium', sans-serif;
            font-size: 1.12rem;
            font-weight: 800;
            color: #00d4ff;
            line-height: 1.25;
          }
          .forensics-consensus {
            display: flex;
            align-items: stretch;
            justify-content: space-between;
            gap: 12px;
            padding: 12px 14px;
            border: 1px solid rgba(0, 212, 255, 0.18);
            background: linear-gradient(135deg, rgba(0, 255, 159, 0.08), rgba(0, 212, 255, 0.05));
          }
          .forensics-consensus.consensus-agree {
            border-color: rgba(0, 255, 159, 0.34);
            box-shadow: 0 0 16px rgba(0, 255, 159, 0.08);
          }
          .forensics-consensus.consensus-split {
            border-color: rgba(255, 221, 87, 0.34);
            box-shadow: 0 0 16px rgba(255, 221, 87, 0.08);
          }
          .forensics-consensus-copy {
            flex: 1;
            min-width: 0;
          }
          .forensics-consensus-v {
            font-family: 'Oxanium', sans-serif;
            font-size: 1.1rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            color: #eafcf7;
            margin-top: 2px;
          }
          .forensics-consensus-m {
            margin-top: 5px;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.68rem;
            line-height: 1.6;
            color: rgba(234, 252, 247, 0.72);
          }
          .forensics-consensus-chip {
            align-self: center;
            min-width: 160px;
            padding: 10px 12px;
            border: 1px solid rgba(0, 212, 255, 0.28);
            background: rgba(0, 212, 255, 0.06);
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            text-align: center;
            color: #00d4ff;
          }
          .forensics-consensus.consensus-agree .forensics-consensus-chip {
            color: #00ff9f;
            border-color: rgba(0, 255, 159, 0.28);
            background: rgba(0, 255, 159, 0.06);
          }
          .forensics-consensus.consensus-split .forensics-consensus-chip {
            color: #ffdd57;
            border-color: rgba(255, 221, 87, 0.28);
            background: rgba(255, 221, 87, 0.06);
          }
          .forensics-consensus-meter {
            margin-top: 10px;
            height: 7px;
            background: rgba(255, 255, 255, 0.08);
            overflow: hidden;
          }
          .forensics-consensus-fill {
            height: 100%;
          }
          .forensics-consensus.consensus-agree .forensics-consensus-fill {
            background: linear-gradient(90deg, #00ff9f, #00d4ff);
          }
          .forensics-consensus.consensus-split .forensics-consensus-fill {
            background: linear-gradient(90deg, #ffdd57, #ff5577);
          }
          .forensics-hero {
            min-width: 160px;
            padding: 10px 12px;
            border: 1px solid rgba(0, 255, 159, 0.18);
            background: rgba(0, 255, 159, 0.05);
            text-align: right;
          }
          .forensics-hero-k,
          .forensics-trace-k,
          .forensics-card-tag {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.56rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: rgba(200, 240, 224, 0.56);
          }
          .forensics-hero-v {
            font-family: 'Oxanium', sans-serif;
            font-size: 1.5rem;
            font-weight: 800;
            color: #00ff9f;
            text-shadow: 0 0 12px rgba(0, 255, 159, 0.34);
            line-height: 1.05;
            margin-top: 2px;
          }
          .forensics-hero-m {
            margin-top: 4px;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.6rem;
            color: #00d4ff;
            letter-spacing: 0.06em;
          }
          .forensics-trace {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
          }
          .forensics-trace-step {
            padding: 10px 11px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(0, 0, 0, 0.26);
            min-height: 70px;
          }
          .forensics-trace-v {
            margin-top: 8px;
            font-family: 'Oxanium', sans-serif;
            font-size: 0.82rem;
            line-height: 1.45;
            color: #eafcf7;
          }
          .forensics-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
          }
          .forensics-card {
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(6, 11, 16, 0.84);
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 10px;
          }
          .forensics-card.primary {
            border-color: rgba(0, 255, 159, 0.45);
            box-shadow: 0 0 18px rgba(0, 255, 159, 0.08);
          }
          .forensics-card.secondary {
            border-color: rgba(0, 212, 255, 0.28);
          }
          .forensics-card-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 10px;
          }
          .forensics-card-model {
            margin-top: 3px;
            font-family: 'Oxanium', sans-serif;
            font-size: 1rem;
            font-weight: 800;
            color: #eafcf7;
          }
          .forensics-card-meta {
            margin-top: 4px;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.62rem;
            color: rgba(234, 252, 247, 0.6);
          }
          .forensics-decision {
            display: inline-flex;
            align-items: center;
            padding: 5px 8px;
            border: 1px solid currentColor;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.6rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            white-space: nowrap;
          }
          .forensics-decision.review { color: #ffdd57; background: rgba(255, 221, 87, 0.08); }
          .forensics-decision.auto { color: #00d4ff; background: rgba(0, 212, 255, 0.06); }
          .forensics-score-row {
            display: flex;
            gap: 12px;
            align-items: center;
          }
          .forensics-score-circle {
            width: 88px;
            height: 88px;
            border-radius: 50%;
            border: 2px solid rgba(255, 255, 255, 0.14);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            box-shadow: inset 0 0 16px rgba(0, 0, 0, 0.28);
          }
          .forensics-score-circle.safe { border-color: rgba(0, 255, 159, 0.38); }
          .forensics-score-circle.suspicious { border-color: rgba(255, 221, 87, 0.4); }
          .forensics-score-circle.phishing,
          .forensics-score-circle.critical {
            border-color: rgba(255, 85, 119, 0.46);
          }
          .forensics-score-num {
            font-family: 'Oxanium', sans-serif;
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1;
          }
          .forensics-score-circle.safe .forensics-score-num { color: #00ff9f; }
          .forensics-score-circle.suspicious .forensics-score-num { color: #ffdd57; }
          .forensics-score-circle.phishing .forensics-score-num,
          .forensics-score-circle.critical .forensics-score-num { color: #ff5577; }
          .forensics-score-unit {
            font-size: 0.62rem;
            color: rgba(234, 252, 247, 0.65);
          }
          .forensics-score-meta {
            flex: 1;
            min-width: 0;
          }
          .forensics-score-line {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.67rem;
            line-height: 1.55;
            color: #eafcf7;
          }
          .forensics-meter {
            height: 7px;
            background: rgba(255, 255, 255, 0.07);
            overflow: hidden;
            margin-top: 8px;
          }
          .forensics-meter-fill {
            height: 100%;
          }
          .forensics-meter-fill.safe { background: linear-gradient(90deg, #00ff9f, #00d4ff); }
          .forensics-meter-fill.suspicious { background: linear-gradient(90deg, #00ff9f, #ffdd57); }
          .forensics-meter-fill.phishing,
          .forensics-meter-fill.critical { background: linear-gradient(90deg, #ffdd57, #ff5577); }
          .forensics-reason {
            font-size: 0.72rem;
            line-height: 1.55;
            color: rgba(234, 252, 247, 0.88);
          }
          .forensics-cue-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
          }
          .forensics-cue {
            display: inline-flex;
            align-items: center;
            padding: 4px 7px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(0, 212, 255, 0.05);
            color: #eafcf7;
            font-size: 0.6rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
          }
          .forensics-cue.muted {
            color: rgba(234, 252, 247, 0.52);
          }
          .forensics-card-foot {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.62rem;
            color: rgba(234, 252, 247, 0.62);
          }
          .forensics-divergence-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin-top: 8px;
          }
          .forensics-divergence-card {
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(0, 0, 0, 0.26);
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
          }
          .forensics-divergence-card.cautious {
            border-color: rgba(255, 85, 119, 0.26);
          }
          .forensics-divergence-card.aggressive {
            border-color: rgba(0, 255, 159, 0.26);
          }
          .forensics-divergence-card.spread {
            border-color: rgba(0, 212, 255, 0.26);
          }
          .forensics-divergence-k {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.58rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(200, 240, 224, 0.56);
          }
          .forensics-divergence-v {
            font-family: 'Oxanium', sans-serif;
            font-size: 0.95rem;
            font-weight: 800;
            color: #eafcf7;
            line-height: 1.35;
          }
          .forensics-divergence-m {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.66rem;
            line-height: 1.6;
            color: rgba(234, 252, 247, 0.72);
          }
          @media (max-width: 920px) {
            .case-memory {
              gap: 8px;
            }
            .memory-orbit {
              min-height: auto;
              display: flex;
              flex-direction: column;
              gap: 10px;
              padding: 0;
              border: 0;
              background: transparent;
            }
            .memory-lines {
              display: none;
            }
            .memory-node,
            .memory-node.current,
            .memory-node.echo-0,
            .memory-node.echo-1,
            .memory-node.echo-2 {
              position: static;
              width: auto;
              transform: none;
              left: auto;
              right: auto;
              top: auto;
              bottom: auto;
            }
          .forensics-trace,
          .forensics-grid,
          .forensics-divergence-grid,
          .evidence-waterfall-summary,
          .model-debate-grid {
            grid-template-columns: 1fr;
          }
          .forensics-score-row {
            flex-direction: column;
            align-items: flex-start;
          }
          .forensics-hero {
            text-align: left;
          }
          .evidence-waterfall-head {
            flex-direction: column;
          }
          .evidence-waterfall-hero {
            text-align: left;
          }
          .model-debate-head {
            flex-direction: column;
          }
          .model-debate-badge {
            text-align: left;
          }
        }
      </style>
      """
    )

    trace_cards = [
        (
            "Cue Evidence",
            ", ".join(cue_labels) if cue_labels else "No explicit cue matched",
        ),
        (
            "Score Recipe",
            f"Risk score = conservative integer P(Phishing) × 100 = {primary_score}/100; severity band = {score_band} from the current threshold map",
        ),
        (
            "Decision Gate",
            decision_text,
        ),
    ]

    trace_html = "".join(
        [
            (
                "<div class='forensics-trace-step'>"
                f"<div class='forensics-trace-k'>{html.escape(label)}</div>"
                f"<div class='forensics-trace-v'>{html.escape(value)}</div>"
                "</div>"
            )
            for label, value in trace_cards
        ]
    )

    model_cards = []
    model_rows = []
    slot_labels = ["PRIMARY MODEL", "SECONDARY MODEL", "TERTIARY MODEL"]
    for idx, entry in enumerate(predictions[:3]):
        pred = entry.get("prediction") or {}
        label = str(entry.get("label", pred.get("label", "Safe")))
        score = int(pred.get("risk_score", entry.get("risk_score", 0)) or 0)
        confidence = float(pred.get("model_confidence", entry.get("model_confidence", 0.0)) or 0.0)
        phish_prob = (pred.get("risk_probs") or {}).get("Phishing")
        if phish_prob is None:
            phish_prob = score / 100.0
        card_band = _score_band(score)
        cues = pred.get("cue_labels") or []
        reason = str(pred.get("reason") or "No explanation available.")
        decision_label = str(pred.get("decision_label") or "Auto decision")
        review_reason = str(pred.get("review_reason") or pred.get("abstain_reason") or "")
        scam_type = str(entry.get("scam_type", pred.get("scam_type", "Other")))
        version = str(entry.get("model_version", pred.get("model_version", "unknown")))
        source = str(entry.get("model_source", pred.get("model_source", "unknown")))
        label_class = _label_class(label)
        card_cls = "primary" if idx == 0 else "secondary"
        card_tag = slot_labels[idx] if idx < len(slot_labels) else f"MODEL {idx + 1}"
        review_chip_cls = "review" if bool(pred.get("review_recommended")) else "auto"
        score_txt = f"{score}/100"
        phish_txt = f"{float(phish_prob) * 100:.1f}%"
        conf_txt = f"{confidence * 100:.1f}%"
        cue_html = _cue_chips_html(cues)
        review_line = review_reason or (
            "Lower-confidence automatic verdict"
            if bool(pred.get("review_recommended"))
            else "Auto decision"
        )

        model_rows.append(
            {
                "tag": card_tag,
                "label": label,
                "label_class": label_class,
                "version": version,
                "source": source,
                "score": score,
                "confidence": confidence,
                "confidence_text": conf_txt,
                "phish_text": phish_txt,
                "cues": cues,
                "review_line": review_line,
                "decision_label": decision_label,
                "review_chip_cls": review_chip_cls,
                "reason": reason,
            }
        )

        model_cards.append(
            f"""
            <div class="forensics-card {card_cls}">
              <div class="forensics-card-head">
                <div>
                  <div class="forensics-card-tag">{html.escape(card_tag)}</div>
                  <div class="forensics-card-model">{html.escape(version)}</div>
                  <div class="forensics-card-meta">{html.escape(source)} · {html.escape(scam_type)}</div>
                </div>
                <span class="forensics-decision {review_chip_cls}">{html.escape(decision_label.upper())}</span>
              </div>
              <div class="forensics-score-row">
                <div class="forensics-score-circle {label_class}">
                  <span class="forensics-score-num">{score}</span>
                  <span class="forensics-score-unit">/100</span>
                </div>
                <div class="forensics-score-meta">
                  <div class="forensics-score-line">P(Phishing) = {phish_txt} → risk score</div>
                  <div class="forensics-score-line">Model confidence = {conf_txt}</div>
                  <div class="forensics-score-line">Justification: {html.escape(card_band)} band from current thresholds</div>
                </div>
              </div>
              <div class="forensics-meter">
                <div class="forensics-meter-fill {label_class}" style="width:{score:.1f}%"></div>
              </div>
              <div class="forensics-reason">{html.escape(reason)}</div>
              <div class="forensics-cue-row">{cue_html}</div>
              <div class="forensics-card-foot">{html.escape(review_line)}</div>
            </div>
            """
        )

    if model_rows:
        model_scores = [row["score"] for row in model_rows]
        model_confidences = [row["confidence"] for row in model_rows]
    else:
        model_scores = [primary_score]
        model_confidences = [float(result.get("model_confidence", 0.0) or 0.0)]

    total_models = len(model_rows) or len(predictions) or 1
    labels = [row["label"] for row in model_rows] or [str(result.get("label", "Safe"))]
    unique_labels = list(dict.fromkeys(labels))
    label_counts = {label: labels.count(label) for label in unique_labels}
    consensus_label = str(
        (comparison or {}).get(
            "consensus_label",
            max(label_counts, key=label_counts.get) if label_counts else str(result.get("label", "Safe")),
        )
    )
    consensus_count = int((comparison or {}).get("consensus_count", label_counts.get(consensus_label, total_models)))
    agreement_flag = bool((comparison or {}).get("agreement", len(unique_labels) == 1))
    agreement_label = str(
        (comparison or {}).get(
            "agreement_label",
            "unanimous" if agreement_flag else f"{consensus_count}/{total_models} agree",
        )
    )
    consensus_ratio = (consensus_count / total_models) * 100.0 if total_models else 0.0
    score_spread = (max(model_scores) - min(model_scores)) if model_scores else 0
    confidence_spread = ((max(model_confidences) - min(model_confidences)) * 100.0) if model_confidences else 0.0
    consensus_display = _localized_verdict(consensus_label, _lang_code_from_name(lang_name))
    consensus_cls = "consensus-agree" if agreement_flag else "consensus-split"
    consensus_copy = (
        f"All {total_models} models reached the same verdict."
        if agreement_flag
        else f"The stack split on this message, with a {score_spread}-point risk spread and {confidence_spread:.1f}-point confidence spread."
    )
    cautious_row = max(model_rows, key=lambda row: row["score"]) if model_rows else None
    aggressive_row = min(model_rows, key=lambda row: row["score"]) if model_rows else None

    consensus_html = ""
    if model_rows:
        consensus_html = textwrap.dedent(
            f"""
            <div class="forensics-consensus {consensus_cls}">
              <div class="forensics-consensus-copy">
                <div class="forensics-title">Consensus Pulse</div>
                <div class="forensics-consensus-v">{html.escape(agreement_label.upper())}</div>
                <div class="forensics-consensus-m">{html.escape(consensus_copy)}</div>
                <div class="forensics-consensus-meter">
                  <div class="forensics-consensus-fill" style="width:{consensus_ratio:.1f}%"></div>
                </div>
              </div>
              <div class="forensics-consensus-chip">{html.escape(consensus_display.upper())}</div>
            </div>
            """
        ).strip()

    debate_html = ""
    if model_rows:
        debate_rows = sorted(model_rows, key=lambda row: (row["score"], row["confidence"]), reverse=True)
        debate_cards = []
        for idx, row in enumerate(debate_rows):
            if idx == 0:
                role_label = "Prosecution"
                role_cls = "prosecution"
                opening = (
                    f"This voice pushes the case upward because {', '.join(row['cues'][:3]) if row['cues'] else 'the scam cues are still visible'} "
                    f"and the score is already {row['score']}/100."
                )
            elif idx == len(debate_rows) - 1:
                role_label = "Defense"
                role_cls = "defense"
                opening = (
                    f"This voice pushes back, saying the message is not fully locked in and the lowest vote stays at {row['score']}/100."
                )
            else:
                role_label = "Bench"
                role_cls = "bench"
                opening = (
                    f"This voice sits between both sides, keeping the message in dispute at {row['score']}/100 while the cues stay mixed."
                )

            cue_html = _cue_chips_html(row["cues"])
            risk_band = _score_band(row["score"])
            debate_cards.append(
                f"""
                <div class="model-debate-card {role_cls}">
                  <div class="model-debate-card-head">
                    <div>
                      <div class="model-debate-role">{html.escape(role_label.upper())}</div>
                      <div class="model-debate-model">{html.escape(row['version'])}</div>
                      <div class="model-debate-meta">{html.escape(row['source'])} · {html.escape(row['label'])} · {html.escape(row['tag'])}</div>
                    </div>
                    <div class="model-debate-score">{row['score']}/100</div>
                  </div>
                  <div class="model-debate-opening">{html.escape(opening)}</div>
                  <div class="model-debate-quote">{html.escape(row['reason'])}</div>
                  <div class="model-debate-line">P(Phishing) = {row['phish_text']} → {html.escape(risk_band)} band</div>
                  <div class="model-debate-cues">{cue_html}</div>
                  <div class="model-debate-footer">{html.escape(row['review_line'])}</div>
                </div>
                """
            )

        judge_summary = (
            f"Consensus verdict: {consensus_display} · {agreement_label} · score spread {score_spread} points · confidence spread {confidence_spread:.1f} points."
        )
        judge_note_prefix = ""
        if aggressive_row and cautious_row:
            judge_note_prefix = (
                f"Most aggressive: {aggressive_row['version']} at {aggressive_row['score']}/100 · "
                f"Most cautious: {cautious_row['version']} at {cautious_row['score']}/100. "
            )
        judge_note = judge_note_prefix + (stack_summary or decision_text)
        debate_html = textwrap.dedent(
            f"""
            <div class="model-debate">
              <div class="model-debate-head">
                <div>
                  <div class="forensics-title">Model Debate Room</div>
                  <div class="model-debate-sub">The three models cross-examine the same scan, each defending a different risk position before the judge closes the case.</div>
                </div>
                <div class="model-debate-badge">{html.escape(agreement_label.upper())}</div>
              </div>
              <div class="model-debate-grid">
                {''.join(debate_cards)}
              </div>
              <div class="model-debate-judge">
                <div class="model-debate-judge-k">Judge's ruling</div>
                <div class="model-debate-judge-v">{html.escape(consensus_display.upper())}</div>
                <div class="model-debate-judge-m">{html.escape(judge_summary)}</div>
                <div class="model-debate-judge-note">{html.escape(judge_note)}</div>
              </div>
            </div>
            """
        ).strip()

    case_memory_html = _build_case_memory_html(result, message, comparison, lang_name)
    trajectory_html = _build_risk_trajectory_html(result, message, lang_name)

    stack_html = ""
    if stack_summary:
        stack_items = [part.strip() for part in stack_summary.split("|") if part.strip()]
        stack_chips = []
        for part in stack_items:
            if ":" in part:
                model_name, verdict_text = part.split(":", 1)
            else:
                model_name, verdict_text = part, ""
            stack_chips.append(
                "<div class='forensics-stack-chip'>"
                f"<div class='forensics-stack-chip-k'>{html.escape(model_name.strip())}</div>"
                f"<div class='forensics-stack-chip-v'>{html.escape(verdict_text.strip() or 'No verdict')}</div>"
                "</div>"
            )
        stack_html = textwrap.dedent(
            f"""
            <div class="forensics-stack-wrap">
              <div class="forensics-stack-copy">
                <div class="forensics-stack-k">Model stack</div>
                <div class="forensics-stack-v">A compact left-to-right snapshot of each model verdict for the current scan.</div>
              </div>
              <div class="forensics-stack-chips">
                {''.join(stack_chips)}
              </div>
            </div>
            """
        ).strip()

    score_build_html = textwrap.dedent(
        f"""
        <div class="forensics-panel">
          <div class="forensics-panel-head">
            <div>
              <div class="forensics-title">Score Build</div>
              <div class="forensics-details-v">How the score and risk path were built</div>
              <div class="forensics-details-m">Everything is shown directly, with clear spacing between sections.</div>
            </div>
            <div class="forensics-details-chip">VISIBLE</div>
          </div>
          <div class="forensics-panel-body">
            <div class="forensics-trace">{trace_html}</div>
            <div class="forensics-panel">
              <div class="forensics-panel-head">
                <div>
                  <div class="forensics-details-k">Risk Trajectory</div>
                  <div class="forensics-details-v">Chunk-by-chunk path behind the final score</div>
                  <div class="forensics-details-m">Each chunk is shown inline so the score path reads like a report.</div>
                </div>
                <div class="forensics-details-chip">PATH</div>
              </div>
              <div class="forensics-panel-body">
                {trajectory_html}
              </div>
            </div>
          </div>
        </div>
        """
    ).strip()

    comparison_panel_html = textwrap.dedent(
        f"""
        <div class="forensics-panel">
          <div class="forensics-panel-head">
            <div>
              <div class="forensics-title">Model Comparison</div>
              <div class="forensics-details-v">How the three models agreed or diverged</div>
              <div class="forensics-details-m">The debate room is fully visible so the differences are easy to read.</div>
            </div>
            <div class="forensics-details-chip">VISIBLE</div>
          </div>
          <div class="forensics-panel-body">
            {debate_html}
          </div>
        </div>
        """
    ).strip()

    memory_panel_html = textwrap.dedent(
        f"""
        <div class="forensics-panel">
          <div class="forensics-panel-head">
            <div>
              <div class="forensics-title">Case Memory</div>
              <div class="forensics-details-v">How this scan fits into the recent history</div>
              <div class="forensics-details-m">The archive sits in the open so the scan trail stays obvious.</div>
            </div>
            <div class="forensics-details-chip">VISIBLE</div>
          </div>
          <div class="forensics-panel-body">
            {case_memory_html}
          </div>
        </div>
        """
    ).strip()

    return textwrap.dedent(
        f"""
        {style}
        <div class="forensics-studio">
          <div class="forensics-head">
            <div>
              <div class="forensics-title">// AI FORENSICS STUDIO</div>
              <div class="forensics-sub">Verdict first. Open the deep-dive panels only when you need the score build, model comparison, or scan history.</div>
            </div>
            <div class="forensics-hero">
              <div class="forensics-hero-k">Current score</div>
              <div class="forensics-hero-v">{primary_score}/100</div>
              <div class="forensics-hero-m">P(Phishing) = {float(primary_phish_prob) * 100:.1f}%</div>
            </div>
          </div>
          <div class="forensics-guide">
            <div class="forensics-guide-chip">1. Read the verdict</div>
            <div class="forensics-guide-chip">2. Open score build</div>
            <div class="forensics-guide-chip">3. Compare the models</div>
          </div>
          {stack_html}
          {consensus_html}
          {score_build_html}
          {comparison_panel_html}
          {memory_panel_html}
          <div class="forensics-note">Start with the verdict snapshot. Open Score Build for the cue logic, Model Comparison for the three-model debate, and Case Memory only if you want the scan history.</div>
        </div>
        """
    ).strip()


def _model_outputs_for_scan_storage(predictions: list[dict], comparison: dict, result: dict) -> list[dict]:
    outputs = comparison.get("model_outputs") or result.get("model_outputs") or []
    if outputs:
        return [item for item in outputs if isinstance(item, dict)]

    normalized: list[dict] = []
    for entry in predictions[:3]:
        pred = entry.get("prediction") or {}
        normalized.append(
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
    return normalized


def _add_history(result: dict, message: str, language: str, comparison: dict | None = None) -> None:
    compare_summary = str(comparison.get("summary", "")) if comparison else ""
    compare_state = str(comparison.get("agreement_label", "")) if comparison else ""
    review_recommended = False
    review_label = str(result.get("decision_label", "Auto decision"))
    if review_label.strip().lower() in {"uncertain", "needs review", "needs_review"}:
        review_label = "Auto decision"
    item = {
        "label": result.get("label", "Safe"),
        "scam_type": result.get("scam_type", "Other"),
        "score": int(result.get("risk_score", 0)),
        "confidence": float(result.get("model_confidence", 0.0)),
        "message": message,
        "language": language,
        "ts": datetime.now().strftime("%H:%M:%S"),
        "compare_summary": compare_summary,
        "compare_state": compare_state,
        "review_recommended": review_recommended,
        "decision_label": review_label,
        "review_reason": "",
    }
    history = st.session_state.get("detector_history", [])
    history.insert(0, item)
    st.session_state["detector_history"] = history[:8]

def _render_waiting_state() -> None:
    st.markdown(
        """
        <div class="idle-state">
          <div class="idle-icon">🛡</div>
          <div class="idle-text">
            // no message analyzed yet<br>
            // paste a message and click analyze<br>
            // result will appear here
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_scanning_state(line: str) -> str:
    return (
        "<div class='scanning-state scanning-visible'>"
        "<div class='scan-ring'></div>"
        "<div class='scan-log'>"
        "initializing engine...<br>"
        f"<span class='active-line'>{html.escape(line)}</span>"
        "</div>"
        "</div>"
    )


def _render_result_title(title: str) -> str:
    return (
        "<div class='det-marker-result'></div>"
        "<div class='det-panel-titlebar'>"
        "<div class='t-dot t-dot-red'></div>"
        "<div class='t-dot t-dot-yellow'></div>"
        "<div class='t-dot t-dot-green'></div>"
        f"<div class='det-panel-title'>{html.escape(title)}</div>"
        "</div>"
    )


def _render_result_state(result: dict, message: str, lang_name: str) -> None:
    result = _normalize_visible_result(result)
    label = result.get("label", "Safe")
    verdict_class = _label_class(label)
    verdict_icon = "✓" if verdict_class == "safe" else "⚠"
    scam_type = result.get("scam_type", "Other")
    score = int(result.get("risk_score", 0))
    severity = result.get("severity_tier", "Low")
    lang_code = _lang_code_from_name(lang_name)
    verdict_display = _localized_verdict(label, lang_code)
    tier_display = _localized_tier(severity, lang_code)
    confidence = float(result.get("model_confidence", 0.0))
    confidence_text, confidence_pct = _format_confidence_pct(confidence)
    confidence_level = _confidence_level(confidence)
    confidence_level_display = _localized_confidence_level(confidence_level, lang_code)
    tooltip = SCAM_TYPE_TOOLTIP.get(scam_type, SCAM_TYPE_TOOLTIP["Other"])
    review_recommended = bool(result.get("review_recommended", False))
    decision_label = str(result.get("decision_label", "Auto decision"))
    if decision_label.strip().lower() in {"uncertain", "needs review", "needs_review"}:
        decision_label = "Auto decision"
    decision_label_display = _localized_decision_label(decision_label, lang_code)
    review_reason = str(result.get("review_reason") or result.get("abstain_reason") or "")
    decision_text = review_reason or (
        _ui_text(lang_code, "auto_decision_text")
        if not review_recommended
        else _ui_text(lang_code, "low_confidence_text")
    )

    reasons = _simple_consumer_reasons(result, message, verdict_class, lang_code)
    reason_heading = _consumer_reason_heading(verdict_class, lang_code)

    actions = result.get("actions") or DEFAULT_ACTION_I18N.get(lang_code, DEFAULT_ACTION_I18N["en"])
    reasons_html = "".join(
        [
            (
                "<div class='reason-item'>"
                f"<span class='reason-bullet {'safe-bullet' if verdict_class == 'safe' else ''}'>▶</span>"
                f"<span>{html.escape(x)}</span>"
                "</div>"
            )
            for x in reasons
        ]
    )
    actions_html = "".join(
        [
            (
                "<div class='action-item'>"
                f"<span class='action-num'>{i + 1:02d}</span>"
                f"<span>{html.escape(x)}</span>"
                "</div>"
            )
            for i, x in enumerate(actions)
        ]
    )

    st.markdown(
        f"""
        <div class="risk-banner {verdict_class}">
          <div class="risk-verdict">
            <span class="verdict-icon {'no-blink' if verdict_class == 'safe' else ''}">{verdict_icon}</span>
            <span class="verdict-label {verdict_class}">{html.escape(verdict_display)}</span>
          </div>
          <div class="badge-wrap">
            <span class="scam-type-badge {verdict_class}" title="{html.escape(tooltip)}">{html.escape(scam_type)} ⓘ</span>
            <div class="scam-tooltip">
              <div class="tooltip-title">{html.escape(_ui_text(lang_code, "category_tooltip_title"))}</div>
              <div class="tooltip-body">{html.escape(tooltip)}</div>
            </div>
          </div>
        </div>
        <div class="result-body">
          <div>
            <div class="res-label">{html.escape(_ui_text(lang_code, "risk_analysis"))}</div>
            <div class="score-conf-row">
              <div class="score-block">
                <div class="score-circle {verdict_class}">
                  <span class="score-num {verdict_class}">{score}</span>
                  <span class="score-unit">/100</span>
                </div>
                <div class="score-details">
                  <div class="tier-label {verdict_class}">{html.escape(tier_display)}</div>
                  <div class="risk-bar-full"><div class="risk-fill-anim {verdict_class}" style="width:{score}%"></div></div>
                  <div class="risk-ticks"><span>{html.escape(_ui_text(lang_code, "scale_low"))}</span><span>{html.escape(_ui_text(lang_code, "scale_med"))}</span><span>{html.escape(_ui_text(lang_code, "scale_high"))}</span><span>{html.escape(_ui_text(lang_code, "scale_crit"))}</span></div>
                </div>
              </div>
              <div class="conf-block">
                <div class="conf-top">
                  <div class="conf-head">
                    <div class="conf-label">{html.escape(_ui_text(lang_code, "model_confidence"))}</div>
                    <div class="conf-sublabel">{html.escape(confidence_level_display)}</div>
                  </div>
                  <div class="conf-pct">{confidence_text}</div>
                </div>
                <div class="conf-bar-wrap"><div class="conf-fill" style="width:{confidence_pct:.1f}%"></div></div>
                <div class="conf-desc">{html.escape(_confidence_desc(result, lang_code))}</div>
              </div>
            </div>
            <div class="decision-callout {'review' if review_recommended else 'auto'}">
              <div class="decision-title">{html.escape(decision_label_display.upper() if lang_code == "en" else decision_label_display)}</div>
              <div class="decision-text">{html.escape(decision_text)}</div>
            </div>
          </div>

          <hr class="result-divider">

          <div>
            <div class="res-label">{html.escape(_ui_text(lang_code, "message_highlights"))}</div>
            <div class="highlight-box">{_highlight_message(message, result.get("cue_tags") or [])}</div>
            <div class="hl-legend"><span>🔴 {html.escape(_ui_text(lang_code, "legend_high_risk"))}</span><span>🟡 {html.escape(_ui_text(lang_code, "legend_suspicious"))}</span></div>
          </div>

          <hr class="result-divider">

          <div>
            <div class="res-label">{html.escape(reason_heading)}</div>
            <div class="reason-list">{reasons_html}</div>
          </div>

          <hr class="result-divider">

          <div>
            <div class="res-label">{html.escape(_ui_text(lang_code, "what_you_should_do"))}</div>
            <div class="action-list">{actions_html}</div>
          </div>

          <hr class="result-divider">

          <div>
            <div class="res-label">{html.escape(_ui_text(lang_code, "output_language"))}</div>
            <div class="conf-desc">{html.escape(lang_name)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _terminal_state_from_session() -> tuple[str, str]:
    panel_state = st.session_state.get("detector_panel_state", "waiting")
    if panel_state == "scanning":
        return "analyzing_message...", "scanning"

    last = st.session_state.get("last_result")
    if isinstance(last, dict):
        last = _normalize_visible_result(last)
        review_recommended = bool(last.get("review_recommended", False))
        label = str(last.get("label", "SAFE")).upper()
        score = int(last.get("risk_score", 0))
        confidence_text, _ = _format_confidence_pct(float(last.get("model_confidence", 0.0)))
        return f"result:{label} | score:{score}/100 | conf:{confidence_text}", "complete"

    return "awaiting_input...", "ready"


def _render_terminal_strip(status_text: str, mode: str = "ready") -> str:
    scans_today = int(st.session_state.get("detector_scans_today", 0))
    if mode == "scanning":
        engine_cls = "det-engine-scan"
        engine_label = "● ENGINE SCANNING"
    elif mode == "complete":
        engine_cls = "det-engine-done"
        engine_label = "● RESULT READY"
    else:
        engine_cls = "det-engine-ready"
        engine_label = "● ENGINE READY"

    return (
        "<div class='det-term-row terminal-strip'>"
        "<span class='terminal-prompt'>root@safesandesh:~$</span>"
        f"<span class='det-term-status'>{html.escape(status_text)}</span>"
        "<span class='t-pipe'>|</span>"
        f"<span>scans_today: <span class='scan-counter-val'>{scans_today:,}</span></span>"
        "<span class='t-pipe'>|</span>"
        f"<span class='{engine_cls}'>{engine_label}</span>"
        "<span class='cursor'></span>"
        "</div>"
    )


def _run_analysis(
    analyze_trigger: bool,
    message: str,
    lang_name: str,
    panel_placeholder,
    title_placeholder,
    terminal_placeholder,
) -> bool:
    if not analyze_trigger:
        return False

    message_to_analyze = message or ""
    validation_error = _validate_message_for_analysis(message_to_analyze)
    if validation_error:
        st.session_state["detector_input_error"] = validation_error
        return False
    st.session_state["detector_input_error"] = ""

    st.session_state["detector_panel_state"] = "scanning"
    terminal_placeholder.markdown(_render_terminal_strip("analyzing_message...", "scanning"), unsafe_allow_html=True)
    title_placeholder.markdown(_render_result_title("scan_result.json — SCANNING"), unsafe_allow_html=True)
    for line in SCAN_LINES:
        terminal_placeholder.markdown(_render_terminal_strip(line, "scanning"), unsafe_allow_html=True)
        panel_placeholder.markdown(_render_scanning_state(line), unsafe_allow_html=True)
        time.sleep(0.14)

    message_language = infer_message_language_name(message_to_analyze, fallback=lang_name)
    comparison = runtime_utils.compare_all_model_predictions(message_to_analyze, output_language=lang_name)
    predictions = list(comparison.get("predictions", []))
    if not predictions:
        fallback = comparison.get("primary", {})
        predictions = [
            {
                "model_kind": "best",
                "prediction": fallback,
                "label": fallback.get("label", "Safe"),
                "scam_type": fallback.get("scam_type", "Other"),
                "risk_score": fallback.get("risk_score", 0),
                "model_confidence": fallback.get("model_confidence", 0.0),
                "model_source": fallback.get("model_source", "unknown"),
                "model_version": fallback.get("model_version", "unknown"),
                "type_source": fallback.get("type_source", "unknown"),
            }
        ]
    primary_entry = predictions[0]
    secondary_entry = predictions[1] if len(predictions) > 1 else predictions[0]
    tertiary_entry = predictions[2] if len(predictions) > 2 else predictions[-1]
    result = dict(comparison.get("final") or primary_entry["prediction"])
    result.update(
        {
            "review_confidence_min": comparison.get("review_confidence_min", result.get("review_confidence_min", 0.70)),
            "review_recommended": bool(comparison.get("review_recommended", result.get("review_recommended", False))),
            "abstain": bool(comparison.get("abstain", result.get("abstain", False))),
            "review_reason": str(comparison.get("review_reason", result.get("review_reason", ""))),
            "abstain_reason": str(comparison.get("abstain_reason", result.get("abstain_reason", ""))),
            "decision_state": str(comparison.get("decision_state", result.get("decision_state", "auto"))),
            "decision_label": str(comparison.get("decision_label", result.get("decision_label", "Auto decision"))),
            "final_score_method": str(comparison.get("final_score_method", result.get("final_score_method", "median_ensemble_v1"))),
        }
    )
    result = _normalize_visible_result(result)
    model_outputs = _model_outputs_for_scan_storage(predictions, comparison, result)
    model_outputs_json = json.dumps(model_outputs, ensure_ascii=False)
    st.session_state["last_result"] = result
    st.session_state["last_comparison"] = comparison
    st.session_state["last_message"] = message_to_analyze
    st.session_state["last_language"] = lang_name
    st.session_state["detector_prefill_text"] = message_to_analyze
    st.session_state["detector_scans_today"] = int(st.session_state.get("detector_scans_today", 1247)) + 1
    st.session_state["detector_panel_state"] = "complete"
    title_placeholder.markdown(_render_result_title("scan_result.json — COMPLETE"), unsafe_allow_html=True)
    terminal_placeholder.markdown(_render_terminal_strip(*_terminal_state_from_session()), unsafe_allow_html=True)
    scan_row_id = db.insert_scan(
        message_language,
        result.get("label", "Safe"),
        result.get("scam_type", "Other"),
        result.get("risk_score", 0),
        result.get("reason", ""),
        message_to_analyze,
        model_confidence=result.get("model_confidence", 0.0),
        model_source=result.get("model_source", "unknown"),
        model_version=result.get("model_version", "unknown"),
        type_source=result.get("type_source", "unknown"),
        comparison_label=primary_entry["label"],
        comparison_scam_type=primary_entry["scam_type"],
        comparison_risk_score=primary_entry["risk_score"],
        comparison_model_confidence=primary_entry["model_confidence"],
        comparison_model_source=primary_entry["model_source"],
        comparison_model_version=primary_entry["model_version"],
        comparison_type_source=primary_entry["type_source"],
        comparison_tertiary_label=secondary_entry["label"],
        comparison_tertiary_scam_type=secondary_entry["scam_type"],
        comparison_tertiary_risk_score=secondary_entry["risk_score"],
        comparison_tertiary_model_confidence=secondary_entry["model_confidence"],
        comparison_tertiary_model_source=secondary_entry["model_source"],
        comparison_tertiary_model_version=secondary_entry["model_version"],
        comparison_tertiary_type_source=secondary_entry["type_source"],
        review_recommended=bool(comparison.get("review_recommended", False)),
        review_reason=str(comparison.get("review_reason", "")),
        final_score_method=result.get("final_score_method", "median_ensemble_v1"),
        model_outputs_json=model_outputs_json,
    )
    st.session_state["detector_last_scan_db_id"] = scan_row_id
    _add_history(result, message_to_analyze, message_language, comparison)
    return True


def _bind_detector_shell_classes() -> None:
    components.html(
        """
        <script>
        (function () {
          try {
            const parentWin = window.parent;
            const doc = parentWin.document;
            const key = "__detectorShellClassObserver";

            const forceOpaque = (node) => {
              if (!node) return;
              node.style.setProperty("background", "#030f1a", "important");
              node.style.setProperty("background-color", "#030f1a", "important");
              node.style.setProperty("opacity", "1", "important");
            };

            const applyClasses = () => {
              doc.querySelectorAll(".det-marker-input").forEach((el) => {
                const wrap = el.closest('[data-testid="stVerticalBlockBorderWrapper"]');
                const block = el.closest('[data-testid="stVerticalBlock"]');
                if (wrap) {
                  wrap.classList.add("det-shell-input");
                  forceOpaque(wrap);
                  const firstBlock = wrap.querySelector('[data-testid="stVerticalBlock"]');
                  forceOpaque(firstBlock);
                }
                if (block) {
                  block.classList.add("det-shell-fill-input");
                  forceOpaque(block);
                }
              });
              doc.querySelectorAll(".det-marker-result").forEach((el) => {
                const wrap = el.closest('[data-testid="stVerticalBlockBorderWrapper"]');
                const block = el.closest('[data-testid="stVerticalBlock"]');
                if (wrap) {
                  wrap.classList.add("det-shell-result");
                  forceOpaque(wrap);
                  const firstBlock = wrap.querySelector('[data-testid="stVerticalBlock"]');
                  forceOpaque(firstBlock);
                }
                if (block) {
                  block.classList.add("det-shell-fill-result");
                  forceOpaque(block);
                }
              });

              doc.querySelectorAll('[data-testid="stExpander"]').forEach((exp) => {
                exp.classList.remove("det-exp-copy", "det-exp-history");
                const summaryText = (exp.querySelector("summary")?.innerText || "").toLowerCase();
                if (summaryText.includes("copyable report text")) {
                  exp.classList.add("det-exp-copy");
                } else if (summaryText.includes("scan history")) {
                  exp.classList.add("det-exp-history");
                }
              });
            };

            applyClasses();

            if (parentWin[key]) return;
            const obs = new parentWin.MutationObserver(() => applyClasses());
            obs.observe(doc.body, { childList: true, subtree: true });
            parentWin[key] = obs;
          } catch (e) {}
        })();
        </script>
        """,
        height=0,
        width=0,
    )


st.set_page_config(page_title="SafeSandesh", layout="wide", initial_sidebar_state="collapsed")
db.init_db()
_init_state()

apply_theme(home_particles=True)
top_menu("detector")

terminal_placeholder = st.empty()
terminal_placeholder.markdown(_render_terminal_strip(*_terminal_state_from_session()), unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .det-term-row {
      border-bottom: 1px solid var(--border);
      margin-bottom: 0.28rem;
      border-top: 1px solid rgba(0,255,159,0.08);
      background: rgba(3,15,26,0.72);
    }
    .det-term-row .t-pipe { color: rgba(0,255,159,0.25); margin: 0 0.25rem; }
    .det-term-row .scan-counter-val { color: var(--neon2); text-shadow: 0 0 8px rgba(0,212,255,0.35); }
    .det-term-row .det-term-status {
      color: var(--muted);
      letter-spacing: 0.08em;
      font-family: 'Share Tech Mono', monospace;
    }
    .det-term-row .det-engine-ready { color: var(--neon); letter-spacing: 0.08em; }
    .det-term-row .det-engine-scan {
      color: var(--neon2);
      letter-spacing: 0.08em;
      text-shadow: 0 0 8px rgba(0,212,255,0.35);
    }
    .det-term-row .det-engine-done {
      color: #7bffcf;
      letter-spacing: 0.08em;
      text-shadow: 0 0 8px rgba(123,255,207,0.28);
    }
    .det-term-row .cursor {
      display: inline-block;
      width: 8px;
      height: 14px;
      background: var(--neon);
      animation: blink 1s step-end infinite;
      vertical-align: middle;
      margin-left: 4px;
    }
    @keyframes blink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0; }
    }

    .det-system-row,
    .det-page-title-row,
    .det-page-sub,
    .det-examples-label {
      max-width: none;
      margin-left: 0;
      margin-right: 0;
      padding-left: 1rem;
      padding-right: 1rem;
      box-sizing: border-box;
    }

    .det-page-title-row {
      position: relative;
      z-index: 1;
      margin-top: 0;
      margin-bottom: 0;
      padding-top: 0.18rem;
      padding-bottom: 0;
      display: flex;
      align-items: baseline;
      gap: 20px;
    }
    .det-system-row {
      position: relative;
      z-index: 1;
      margin-top: 0;
      margin-bottom: 0;
      padding-top: 0.1rem;
      padding-bottom: 0.04rem;
    }
    .det-system-row .status-bar {
      margin-bottom: 0.02rem;
    }
    .det-section-code {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.7rem;
      color: #16ffd6;
      letter-spacing: 0.1em;
      opacity: 0.95;
      text-shadow: 0 0 10px rgba(0,255,200,0.42);
    }
    .det-page-h1 {
      font-size: 1.8rem;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin: 0;
      color: #ffffff;
      text-shadow:
        0 0 10px rgba(255,255,255,0.22),
        0 0 22px rgba(0,212,255,0.16);
    }
    .det-page-title-row .det-page-h1,
    .det-page-title-row h1 {
      color: #ffffff !important;
      -webkit-text-fill-color: #ffffff !important;
      opacity: 1 !important;
      mix-blend-mode: normal !important;
      filter: none !important;
      text-shadow:
        0 0 10px rgba(255,255,255,0.32),
        0 0 24px rgba(0,212,255,0.18) !important;
    }
    .det-page-sub {
      position: relative;
      z-index: 1;
      margin-top: 14px;
      margin-bottom: 0;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.72rem;
      color: #ffffff;
      letter-spacing: 0.06em;
      text-shadow: 0 0 8px rgba(255,255,255,0.18);
    }

    .det-examples-label {
      position: relative;
      z-index: 1;
      margin-top: 22px;
      margin-bottom: 12px;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.65rem;
      color: #ffffff;
      letter-spacing: 0.08em;
      text-shadow: 0 0 8px rgba(255,255,255,0.18);
    }

    .det-samples-spacer {
      height: 6px;
    }
    .det-page-sub,
    .det-examples-label {
      color: #ffffff !important;
      -webkit-text-fill-color: #ffffff !important;
      opacity: 1 !important;
    }

    .det-two-grid {
      position: relative;
      z-index: 1;
      max-width: 1100px;
      margin: 0 auto;
      padding: 0.6rem 0 1.6rem;
    }

    .det-panel-titlebar {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 16px;
      border-bottom: 1px solid var(--border);
      background: rgba(0,0,0,0.3);
      margin: -0.2rem -0.2rem 1rem -0.2rem;
    }
    .t-dot { width: 10px; height: 10px; border-radius: 50%; }
    .t-dot-red   { background: var(--red); box-shadow: 0 0 6px var(--red); }
    .t-dot-yellow{ background: var(--yellow); box-shadow: 0 0 6px var(--yellow); }
    .t-dot-green { background: var(--neon); box-shadow: 0 0 6px var(--neon); }
    .det-panel-title {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.68rem;
      letter-spacing: 0.1em;
      color: var(--muted);
      margin-left: auto;
      text-transform: none;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.det-marker-input) {
      background: var(--surface);
      border: 1px solid var(--border) !important;
      border-radius: 0 !important;
      position: relative;
      overflow: hidden;
      padding: 0.18rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.det-marker-input)::after {
      content: "";
      position: absolute;
      inset: 0;
      background: #030f1a;
      z-index: 0;
      pointer-events: none;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.det-marker-input)::before {
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      width: 2px;
      height: 100%;
      background: var(--neon);
      box-shadow: 0 0 8px var(--neon);
      pointer-events: none;
      z-index: 4;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.det-marker-input) > div {
      position: relative;
      z-index: 1;
      background: transparent !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.det-marker-result) {
      background: var(--surface);
      border: 1px solid var(--border) !important;
      border-radius: 0 !important;
      position: relative;
      overflow: hidden;
      padding: 0.18rem;
      min-height: 640px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.det-marker-result)::after {
      content: "";
      position: absolute;
      inset: 0;
      background: #030f1a;
      z-index: 0;
      pointer-events: none;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.det-marker-result)::before {
      content: "";
      position: absolute;
      top: 0;
      right: 0;
      width: 2px;
      height: 100%;
      background: var(--neon2);
      box-shadow: 0 0 8px var(--neon2);
      pointer-events: none;
      z-index: 4;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.det-marker-result) > div {
      position: relative;
      z-index: 1;
      background: transparent !important;
    }

    .input-label,
    .res-label {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.9rem;
      font-weight: 700;
      color: var(--neon2);
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .input-label::before,
    .res-label::before {
      content: '//';
      color: rgba(0,212,255,0.3);
    }

    .det-char-row {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.65rem;
      color: var(--muted);
      margin-top: -2px;
      margin-bottom: 12px;
    }
    .det-inline-error {
      display: inline-block;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.78rem;
      color: #ff7f9b;
      letter-spacing: 0.05em;
      border: 1px solid rgba(255,56,96,0.36);
      background: rgba(255,56,96,0.10);
      padding: 0.28rem 0.52rem;
      margin-top: -2px;
      margin-bottom: 0.7rem;
      text-shadow: 0 0 8px rgba(255,56,96,0.16);
    }

    .det-lang-row {
      display: flex;
      align-items: center;
      gap: 12px;
      margin: 0.35rem 0 1rem;
      flex-wrap: wrap;
    }
    .det-lang-label {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.9rem;
      font-weight: 700;
      color: var(--neon2);
      letter-spacing: 0.12em;
      text-transform: uppercase;
      white-space: nowrap;
    }

    [class*="st-key-sample_"] .stButton > button {
      font-family: 'Share Tech Mono', monospace !important;
      font-size: 0.65rem !important;
      letter-spacing: 0.08em !important;
      text-transform: uppercase !important;
      border-radius: 0 !important;
      min-height: 30px !important;
      padding: 0.24rem 0.6rem !important;
    }

    [class*="st-key-sample_"] .stButton > button {
      border-color: rgba(255,56,96,0.24) !important;
      color: rgba(255,56,96,0.72) !important;
      background: #000000 !important;
      box-shadow: none !important;
    }

    [class*="st-key-sample_"] .stButton > button:hover {
      border-color: rgba(255,56,96,0.52) !important;
      color: var(--red) !important;
      background: #000000 !important;
      box-shadow: none !important;
    }

    .st-key-sample_safe .stButton > button {
      border-color: rgba(0,255,159,0.24) !important;
      color: rgba(0,255,159,0.72) !important;
    }

    .st-key-sample_safe .stButton > button:hover {
      border-color: rgba(0,255,159,0.56) !important;
      color: var(--neon) !important;
      background: #000000 !important;
      box-shadow: none !important;
    }

    .st-key-det_lang_en .stButton > button,
    .st-key-det_lang_hi .stButton > button,
    .st-key-det_lang_pa .stButton > button,
    .st-key-det_lang_ur .stButton > button {
      font-family: 'Share Tech Mono', monospace !important;
      font-size: 0.68rem !important;
      letter-spacing: 0.08em !important;
      text-transform: none !important;
      border-radius: 0 !important;
      min-height: 34px !important;
      border-color: var(--border) !important;
      color: var(--muted) !important;
      background: #000000 !important;
      box-shadow: none !important;
    }
    .st-key-det_lang_en .stButton > button:hover,
    .st-key-det_lang_hi .stButton > button:hover,
    .st-key-det_lang_pa .stButton > button:hover,
    .st-key-det_lang_ur .stButton > button:hover {
      border-color: rgba(0,255,159,0.62) !important;
      color: #b9ffe7 !important;
      background: rgba(0,255,159,0.08) !important;
      box-shadow:
        0 0 0 1px rgba(0,255,159,0.22),
        0 0 14px rgba(0,255,159,0.26) !important;
    }

    .st-key-detector_analyze .stButton > button {
      width: 100%;
      background:
        repeating-linear-gradient(
          0deg,
          rgba(0,0,0,0.00) 0px,
          rgba(0,0,0,0.00) 6px,
          rgba(0,0,0,0.10) 6px,
          rgba(0,0,0,0.10) 8px
        ),
        linear-gradient(180deg, #22ffb6 0%, #10ef9d 100%) !important;
      color: #000 !important;
      font-family: 'Oxanium', sans-serif !important;
      font-weight: 900 !important;
      font-size: 1.16rem !important;
      letter-spacing: 0.14em !important;
      text-transform: uppercase !important;
      padding: 0.92rem 1.25rem !important;
      border: none !important;
      clip-path: polygon(10px 0%, 100% 0%, calc(100% - 10px) 100%, 0% 100%);
      box-shadow:
        0 0 20px rgba(0,255,159,0.45),
        0 0 56px rgba(0,255,159,0.16);
    }
    .st-key-detector_analyze .stButton > button p {
      font-size: 1.16rem !important;
      font-weight: 900 !important;
      letter-spacing: 0.14em !important;
      color: #000 !important;
    }

    .st-key-detector_message_input textarea {
      width: 100%;
      min-height: 180px;
      background: #000000 !important;
      border: 1px solid var(--border) !important;
      color: var(--text) !important;
      font-family: 'Share Tech Mono', monospace !important;
      font-size: 0.82rem !important;
      line-height: 1.7 !important;
      padding: 14px !important;
      letter-spacing: 0.03em;
      border-radius: 0 !important;
      box-shadow: none !important;
    }

    /* Solid panel backgrounds so grid doesn't show through controls */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.det-marker-input) {
      background: #030f1a !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.det-marker-input) [data-testid="stVerticalBlock"] {
      background: #030f1a !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.det-marker-result) {
      background: #030f1a !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.det-marker-result) [data-testid="stVerticalBlock"] {
      background: #030f1a !important;
    }

    /* Browser-safe solid fills (applied via JS class binding, no :has dependency) */
    .det-shell-input,
    .det-shell-result {
      background: #030f1a !important;
    }
    .det-shell-input {
      position: relative !important;
      overflow: hidden !important;
      border: 1px solid rgba(0,255,170,0.72) !important;
      box-shadow:
        0 0 0 1px rgba(0,255,170,0.28),
        0 0 14px rgba(0,255,170,0.56),
        0 0 30px rgba(0,255,170,0.24),
        inset 0 0 0 1px rgba(0,255,170,0.10) !important;
    }
    .det-shell-result {
      position: relative !important;
      overflow: hidden !important;
      border: 1px solid rgba(0,212,255,0.72) !important;
      box-shadow:
        0 0 0 1px rgba(0,212,255,0.28),
        0 0 14px rgba(0,212,255,0.56),
        0 0 30px rgba(0,212,255,0.24),
        inset 0 0 0 1px rgba(0,212,255,0.10) !important;
    }
    .det-shell-input::before {
      content: "";
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: 4px;
      background: linear-gradient(180deg, #30ffcc 0%, #00ff9f 55%, #1ee9ff 100%);
      box-shadow:
        0 0 10px rgba(0,255,170,0.92),
        0 0 22px rgba(0,255,170,0.58),
        10px 0 20px rgba(0,255,170,0.18);
      z-index: 8;
      pointer-events: none;
    }
    .det-shell-result::before {
      content: "";
      position: absolute;
      right: 0;
      top: 0;
      bottom: 0;
      width: 4px;
      background: linear-gradient(180deg, #59d8ff 0%, #00c8ff 55%, #27a8ff 100%);
      box-shadow:
        0 0 10px rgba(0,212,255,0.92),
        0 0 22px rgba(0,212,255,0.58),
        -10px 0 20px rgba(0,212,255,0.18);
      z-index: 8;
      pointer-events: none;
    }
    .det-shell-fill-input,
    .det-shell-fill-result {
      background: #030f1a !important;
      background-color: #030f1a !important;
      opacity: 1 !important;
    }
    .det-shell-fill-input {
      border: 1px solid rgba(0,255,170,0.72) !important;
      box-shadow:
        inset 8px 0 18px rgba(0,255,170,0.12),
        inset 0 0 0 1px rgba(0,255,170,0.10) !important;
    }
    .det-shell-fill-result {
      border: 1px solid rgba(0,212,255,0.72) !important;
      box-shadow:
        inset -8px 0 18px rgba(0,212,255,0.12),
        inset 0 0 0 1px rgba(0,212,255,0.10) !important;
    }
    .det-shell-input > div,
    .det-shell-result > div {
      background: transparent !important;
    }

    .st-key-detector_message_input textarea::placeholder {
      color: rgba(200,240,224,0.2) !important;
      font-style: italic;
    }

    .idle-state {
      min-height: 460px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 40px;
      text-align: center;
      gap: 16px;
    }
    .idle-icon { font-size: 3rem; opacity: 0.3; filter: drop-shadow(0 0 10px var(--neon)); }
    .idle-text {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.72rem;
      color: var(--muted);
      letter-spacing: 0.1em;
      line-height: 1.9;
    }

    .scanning-state {
      min-height: 460px;
      display: none;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 40px;
      gap: 20px;
    }
    .scanning-visible { display: flex; }

    .scan-ring {
      width: 80px;
      height: 80px;
      border: 2px solid var(--border);
      border-top-color: var(--neon);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    .scan-log {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.72rem;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-align: center;
      line-height: 2;
    }
    .scan-log .active-line { color: var(--neon); }

    .risk-banner {
      padding: 14px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border);
      margin: -0.2rem -0.2rem 0;
    }
    .risk-banner.phishing { background: rgba(255,56,96,0.08); border-bottom-color: rgba(255,56,96,0.3); }
    .risk-banner.suspicious { background: rgba(255,221,87,0.06); border-bottom-color: rgba(255,221,87,0.25); }
    .risk-banner.safe { background: rgba(0,255,159,0.05); border-bottom-color: rgba(0,255,159,0.2); }

    .risk-verdict { display: flex; align-items: center; gap: 12px; }
    .verdict-icon { font-size: 1.2rem; animation: blink 1.5s step-end infinite; }
    .verdict-icon.no-blink { animation: none; }

    .verdict-label {
      font-size: 1.1rem;
      font-weight: 800;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }
    .verdict-label.phishing { color: var(--red); text-shadow: 0 0 15px rgba(255,56,96,0.6); }
    .verdict-label.suspicious { color: var(--yellow); text-shadow: 0 0 15px rgba(255,221,87,0.4); }
    .verdict-label.safe { color: var(--neon); text-shadow: 0 0 15px rgba(0,255,159,0.4); }

    .badge-wrap { position: relative; display: inline-block; }
    .scam-type-badge {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.7rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      padding: 4px 12px;
      border: 1px solid;
      cursor: help;
      transition: box-shadow 0.2s;
      user-select: none;
    }
    .scam-type-badge.phishing { border-color: rgba(255,56,96,0.4); color: var(--red); background: rgba(255,56,96,0.05); }
    .scam-type-badge.suspicious { border-color: rgba(255,221,87,0.3); color: var(--yellow); background: rgba(255,221,87,0.04); }
    .scam-type-badge.safe { border-color: rgba(0,255,159,0.3); color: var(--neon); background: rgba(0,255,159,0.04); }

    .scam-tooltip {
      display: none;
      position: absolute;
      top: calc(100% + 10px);
      right: 0;
      width: 260px;
      z-index: 300;
      background: #061828;
      border: 1px solid rgba(0,212,255,0.35);
      box-shadow: 0 8px 40px rgba(0,0,0,0.7),0 0 20px rgba(0,212,255,0.08);
      padding: 14px 16px;
    }
    .badge-wrap:hover .scam-tooltip { display: block; }
    .tooltip-title {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.63rem;
      color: var(--neon2);
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .tooltip-body {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.68rem;
      color: var(--muted);
      line-height: 1.75;
    }

    .result-body {
      padding: 20px 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
      max-height: 620px;
      overflow-y: auto;
    }
    .result-body::-webkit-scrollbar { width: 3px; }
    .result-body::-webkit-scrollbar-thumb { background: rgba(0,255,159,0.15); }

    .score-conf-row {
      display: flex;
      flex-direction: column;
      gap: 14px;
      align-items: stretch;
    }
    .score-block {
      display: flex;
      align-items: center;
      gap: 14px;
      width: 100%;
      flex: none;
    }

    .score-circle {
      width: 68px;
      height: 68px;
      border-radius: 50%;
      border: 2px solid var(--border);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .score-circle.phishing { border-color: rgba(255,56,96,0.5); box-shadow: 0 0 20px rgba(255,56,96,0.15); }
    .score-circle.suspicious { border-color: rgba(255,221,87,0.4); box-shadow: 0 0 20px rgba(255,221,87,0.1); }
    .score-circle.safe { border-color: rgba(0,255,159,0.4); box-shadow: 0 0 20px rgba(0,255,159,0.1); }

    .score-num { font-size: 1.35rem; font-weight: 800; line-height: 1; }
    .score-num.phishing { color: var(--red); }
    .score-num.suspicious { color: var(--yellow); }
    .score-num.safe { color: var(--neon); }
    .score-unit { font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; color: var(--muted); }

    .score-details { flex: 1; }
    .tier-label {
      font-size: 0.73rem;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    .tier-label.phishing { color: var(--red); }
    .tier-label.suspicious { color: var(--yellow); }
    .tier-label.safe { color: var(--neon); }

    .risk-bar-full { height: 4px; background: rgba(255,255,255,0.06); position: relative; overflow: hidden; }
    .risk-fill-anim { position: absolute; left: 0; top: 0; bottom: 0; }
    .risk-fill-anim.phishing { background: linear-gradient(90deg,var(--yellow),var(--red)); box-shadow: 0 0 8px var(--red); }
    .risk-fill-anim.suspicious { background: linear-gradient(90deg,var(--neon),var(--yellow)); box-shadow: 0 0 8px var(--yellow); }
    .risk-fill-anim.safe { background: linear-gradient(90deg,var(--neon2),var(--neon)); box-shadow: 0 0 8px var(--neon); }

    .risk-ticks {
      display: flex;
      justify-content: space-between;
      margin-top: 4px;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.52rem;
      color: rgba(200,240,224,0.2);
    }

    .conf-block {
      width: 100%;
      flex: none;
      background: rgba(0,0,0,0.3);
      border: 1px solid rgba(0,212,255,0.12);
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 7px;
    }
    .conf-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; }
    .conf-head { display: flex; flex-direction: column; gap: 2px; }
    .conf-label {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.86rem;
      font-weight: 700;
      color: var(--neon2);
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }
    .conf-sublabel {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.76rem;
      font-weight: 600;
      color: var(--muted);
      letter-spacing: 0.06em;
    }
    .conf-pct {
      font-family: 'Share Tech Mono', monospace;
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--neon2);
      text-shadow: 0 0 10px rgba(0,212,255,0.4);
      line-height: 1;
    }
    .conf-bar-wrap { height: 3px; background: rgba(0,212,255,0.08); position: relative; overflow: hidden; }
    .conf-fill {
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      background: linear-gradient(90deg,rgba(0,212,255,0.5),var(--neon2));
      box-shadow: 0 0 8px rgba(0,212,255,0.4);
    }
    .conf-desc {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.9rem;
      color: #d2f8e8;
      line-height: 1.6;
    }
    .decision-callout {
      margin-top: 10px;
      padding: 10px 12px;
      border: 1px solid rgba(0,255,159,0.15);
      background: rgba(0,255,159,0.04);
      font-family: 'Share Tech Mono', monospace;
      line-height: 1.45;
    }
    .decision-callout.review {
      border-color: rgba(255,221,87,0.34);
      background: rgba(255,221,87,0.05);
      box-shadow: inset 0 0 0 1px rgba(255,221,87,0.06);
    }
    .decision-callout.auto {
      border-color: rgba(0,255,159,0.18);
      background: rgba(0,255,159,0.035);
    }
    .decision-title {
      font-size: 0.65rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 4px;
      color: var(--neon);
    }
    .decision-callout.review .decision-title { color: var(--yellow); }
    .decision-text {
      font-size: 0.82rem;
      color: #d2f8e8;
    }

    .result-divider {
      border: none;
      border-top: 1px solid var(--border);
      margin: 0;
    }

    .highlight-box {
      background: rgba(0,0,0,0.35);
      border: 1px solid var(--border);
      padding: 12px 14px;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.9rem;
      line-height: 1.6;
      letter-spacing: 0.03em;
    }
    .hl-crit { background: rgba(255,56,96,0.18); color: #ff6b8a; padding: 1px 3px; border-radius: 2px; }
    .hl-warn { background: rgba(255,221,87,0.14); color: #ffdd57; padding: 1px 3px; border-radius: 2px; }
    .hl-legend {
      display: flex;
      gap: 16px;
      margin-top: 8px;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.6rem;
      color: var(--muted);
    }

    .reason-list, .action-list { display: flex; flex-direction: column; gap: 8px; }
    .reason-item {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.9rem;
      color: #d2f8e8;
      line-height: 1.6;
    }
    .reason-bullet {
      color: var(--red);
      flex-shrink: 0;
      margin-top: 2px;
      text-shadow: 0 0 8px var(--red);
    }
    .reason-bullet.safe-bullet {
      color: var(--neon);
      text-shadow: 0 0 8px var(--neon);
    }
    .action-item {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      padding: 10px 14px;
      background: rgba(0,0,0,0.25);
      border: 1px solid rgba(0,255,159,0.08);
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.9rem;
      color: var(--text);
      line-height: 1.6;
    }
    .action-num {
      color: var(--neon);
      flex-shrink: 0;
      font-weight: 700;
      text-shadow: 0 0 6px rgba(0,255,159,0.4);
    }

    .forensics-studio {
      background: rgba(6,11,16,0.84);
      border: 1px solid rgba(0,212,255,0.18);
      box-shadow:
        0 0 0 1px rgba(0,212,255,0.06),
        0 0 18px rgba(0,212,255,0.05);
      padding: 14px 14px 12px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .forensics-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    .forensics-title {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.72rem;
      letter-spacing: 0.12em;
      color: rgba(200,240,224,0.72);
      text-transform: uppercase;
      margin-bottom: 4px;
    }
    .forensics-sub,
    .forensics-note,
    .forensics-stack-summary {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.68rem;
      line-height: 1.6;
      color: var(--muted);
    }
    .ai-studio-shell {
      margin: 10px 0 4px;
      padding: 12px;
      border: 1px solid rgba(0,212,255,0.18);
      background: rgba(6,11,16,0.8);
    }
    .ai-studio-shell-head {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.72rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: rgba(200,240,224,0.72);
      margin-bottom: 10px;
    }
    .forensics-hero {
      min-width: 150px;
      padding: 10px 12px;
      border: 1px solid rgba(0,255,159,0.18);
      background: rgba(0,255,159,0.04);
      text-align: right;
    }
    .forensics-hero-k,
    .forensics-trace-k,
    .forensics-card-tag {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.55rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: rgba(200,240,224,0.52);
    }
    .forensics-hero-v {
      font-family: 'Oxanium', sans-serif;
      font-size: 1.4rem;
      font-weight: 800;
      color: var(--neon);
      text-shadow: 0 0 12px rgba(0,255,159,0.34);
      line-height: 1.05;
      margin-top: 2px;
    }
    .forensics-hero-m {
      margin-top: 4px;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.58rem;
      color: var(--neon2);
      letter-spacing: 0.06em;
    }
    .forensics-trace {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .forensics-trace-step {
      padding: 10px 11px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(0,0,0,0.22);
      min-height: 74px;
    }
    .forensics-trace-v {
      margin-top: 5px;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.66rem;
      line-height: 1.55;
      color: #d2f8e8;
    }
    .forensics-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .forensics-card {
      padding: 12px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,0.22);
    }
    .forensics-card.primary {
      border-color: rgba(0,255,159,0.34);
      box-shadow: inset 0 0 0 1px rgba(0,255,159,0.06);
    }
    .forensics-card-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 10px;
      margin-bottom: 8px;
    }
    .forensics-card-model {
      font-family: 'Oxanium', sans-serif;
      font-size: 0.9rem;
      font-weight: 700;
      color: var(--text);
      margin-top: 2px;
    }
    .forensics-card-meta {
      margin-top: 3px;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.56rem;
      color: rgba(200,240,224,0.52);
      letter-spacing: 0.05em;
    }
    .forensics-decision {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.12rem 0.34rem;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.5rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      white-space: nowrap;
      border: 1px solid rgba(0,212,255,0.24);
      color: var(--neon2);
      background: rgba(0,212,255,0.03);
    }
    .forensics-decision.review {
      border-color: rgba(255,221,87,0.34);
      color: var(--yellow);
      background: rgba(255,221,87,0.04);
    }
    .forensics-decision.auto {
      border-color: rgba(0,212,255,0.24);
      color: var(--neon2);
      background: rgba(0,212,255,0.03);
    }
    .forensics-score-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
    }
    .forensics-score-circle {
      width: 58px;
      height: 58px;
      border-radius: 50%;
      border: 2px solid var(--border);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .forensics-score-circle.phishing { border-color: rgba(255,56,96,0.48); box-shadow: 0 0 16px rgba(255,56,96,0.12); }
    .forensics-score-circle.suspicious { border-color: rgba(255,221,87,0.4); box-shadow: 0 0 16px rgba(255,221,87,0.08); }
    .forensics-score-circle.safe { border-color: rgba(0,255,159,0.36); box-shadow: 0 0 16px rgba(0,255,159,0.08); }
    .forensics-score-num {
      font-family: 'Oxanium', sans-serif;
      font-size: 1.1rem;
      font-weight: 800;
      line-height: 1;
      color: var(--text);
    }
    .forensics-score-circle.phishing .forensics-score-num { color: var(--red); }
    .forensics-score-circle.suspicious .forensics-score-num { color: var(--yellow); }
    .forensics-score-circle.safe .forensics-score-num { color: var(--neon); }
    .forensics-score-unit {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.5rem;
      color: var(--muted);
    }
    .forensics-score-meta {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }
    .forensics-score-line {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.58rem;
      line-height: 1.5;
      color: #d2f8e8;
    }
    .forensics-meter {
      height: 4px;
      background: rgba(255,255,255,0.06);
      overflow: hidden;
      position: relative;
      margin-bottom: 8px;
    }
    .forensics-meter-fill {
      height: 100%;
      transform-origin: left center;
      animation: dashBarGrow 620ms cubic-bezier(0.22, 0.7, 0.28, 1) both;
    }
    .forensics-meter-fill.phishing { background: linear-gradient(90deg,var(--yellow),var(--red)); box-shadow: 0 0 8px rgba(255,56,96,0.22); }
    .forensics-meter-fill.suspicious { background: linear-gradient(90deg,var(--neon),var(--yellow)); box-shadow: 0 0 8px rgba(255,221,87,0.16); }
    .forensics-meter-fill.safe { background: linear-gradient(90deg,var(--neon2),var(--neon)); box-shadow: 0 0 8px rgba(0,255,159,0.2); }
    .forensics-reason {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.65rem;
      line-height: 1.55;
      color: #d2f8e8;
      margin-bottom: 8px;
    }
    .forensics-cue-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 8px;
    }
    .forensics-cue {
      display: inline-flex;
      align-items: center;
      padding: 0.12rem 0.32rem;
      border: 1px solid rgba(255,56,96,0.24);
      background: rgba(255,56,96,0.05);
      color: #ff8ca0;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.52rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .forensics-cue.muted {
      border-color: rgba(200,240,224,0.1);
      background: rgba(255,255,255,0.03);
      color: rgba(200,240,224,0.44);
    }
    .forensics-card-foot {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.54rem;
      line-height: 1.5;
      color: rgba(200,240,224,0.58);
      letter-spacing: 0.04em;
    }

    .det-feedback-strip {
      padding: 14px 24px;
      border-top: 1px solid var(--border);
      background: rgba(0,0,0,0.2);
      margin: 0 -0.2rem -0.2rem;
    }
    .st-key-detector_download_pdf .stDownloadButton > button,
    .st-key-detector_download_pdf .stButton > button,
    .st-key-detector_copy .stButton > button,
    .st-key-detector_reset .stButton > button {
      font-size: 0.73rem !important;
      min-height: 34px !important;
      padding: 0.34rem 0.55rem !important;
    }
    .det-history-wrap {
      max-width: 1100px;
      margin: 0 auto;
      padding-bottom: 0.6rem;
    }
    .det-history-card {
      border: 1px solid var(--border);
      background: rgba(0,0,0,0.23);
      padding: 0.55rem 0.65rem;
      margin-bottom: 0.45rem;
    }
    .det-history-top {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      margin-bottom: 4px;
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.82rem;
      color: var(--muted);
    }
    .det-history-meta {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.74rem;
      color: var(--neon2);
      letter-spacing: 0.05em;
      margin-bottom: 4px;
    }
    .det-history-msg {
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.76rem;
      line-height: 1.45;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    [class*="st-key-hist_load_"] .stButton > button,
    .st-key-hist_clear .stButton > button {
      min-height: 30px !important;
      height: 30px !important;
      font-size: 0.7rem !important;
      letter-spacing: 0.06em !important;
      padding: 0.2rem 0.65rem !important;
      width: auto !important;
      border-radius: 0 !important;
      transition: border-color 0.18s ease, box-shadow 0.18s ease, color 0.18s ease, transform 0.18s ease !important;
    }
    [class*="st-key-hist_load_"] .stButton > button {
      border-color: rgba(0,212,255,0.35) !important;
      color: var(--neon2) !important;
      background: #000000 !important;
      box-shadow: none !important;
      min-width: 94px !important;
    }
    .st-key-hist_clear .stButton > button {
      border-color: rgba(255,56,96,0.35) !important;
      color: rgba(255,120,140,0.92) !important;
      background: #000000 !important;
      box-shadow: none !important;
      min-width: 170px !important;
    }
    [class*="st-key-hist_load_"] .stButton > button:hover {
      border-color: rgba(0,212,255,0.78) !important;
      color: #b9f4ff !important;
      box-shadow:
        0 0 0 1px rgba(0,212,255,0.30),
        0 0 16px rgba(0,212,255,0.30) !important;
      transform: translateY(-1px);
    }
    .st-key-hist_clear .stButton > button:hover {
      border-color: rgba(255,56,96,0.82) !important;
      color: #ffc1cf !important;
      box-shadow:
        0 0 0 1px rgba(255,56,96,0.32),
        0 0 16px rgba(255,56,96,0.28) !important;
      transform: translateY(-1px);
    }

    /* Center each Load button against its corresponding message row */
    [data-testid="stHorizontalBlock"]:has([class*="st-key-hist_load_"]) {
      align-items: stretch !important;
    }
    [data-testid="stHorizontalBlock"]:has([class*="st-key-hist_load_"]) > [data-testid="stColumn"]:first-child > div {
      min-height: 100% !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
    }
    [class*="st-key-hist_load_"] {
      width: 100% !important;
      display: flex !important;
      justify-content: center !important;
    }

    /* Match lower expanders to detector panel theme */
    [data-testid="stExpander"] {
      background: #030f1a !important;
      border: 1px solid var(--border) !important;
      border-radius: 0 !important;
      margin-bottom: 0.85rem !important;
      overflow: hidden !important;
      box-shadow:
        0 0 0 1px rgba(0,255,159,0.08),
        inset 0 0 0 1px rgba(0,255,159,0.05) !important;
    }
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] > div {
      background: #030f1a !important;
    }
    [data-testid="stExpander"] summary {
      background: rgba(0,0,0,0.34) !important;
      border-bottom: 1px solid var(--border) !important;
      min-height: 48px !important;
      padding-top: 0.45rem !important;
      padding-bottom: 0.45rem !important;
    }
    [data-testid="stExpander"] summary p {
      font-family: 'Oxanium', sans-serif !important;
      font-size: 0.95rem !important;
      font-weight: 700 !important;
      color: #d7fff1 !important;
      letter-spacing: 0.02em !important;
    }
    [data-testid="stExpander"] summary svg {
      color: var(--neon2) !important;
    }
    [data-testid="stExpander"] [data-testid="stCodeBlock"] {
      background: #000000 !important;
      border: 1px solid var(--border) !important;
      border-radius: 0 !important;
    }
    [data-testid="stExpander"] [data-testid="stCodeBlock"] pre,
    [data-testid="stExpander"] [data-testid="stCodeBlock"] code {
      color: #d7fff1 !important;
      font-family: 'Share Tech Mono', monospace !important;
    }

    /* Expander-specific neon outlines */
    [data-testid="stExpander"].det-exp-copy {
      border: 1px solid rgba(255, 149, 0, 0.72) !important;
      box-shadow:
        0 0 0 1px rgba(255,149,0,0.24),
        0 0 16px rgba(255,149,0,0.28),
        inset 0 0 0 1px rgba(255,149,0,0.10) !important;
    }
    [data-testid="stExpander"].det-exp-copy summary {
      border-bottom: 1px solid rgba(255,149,0,0.42) !important;
      background: rgba(255,149,0,0.06) !important;
    }
    [data-testid="stExpander"].det-exp-copy summary p {
      color: #ffe3b8 !important;
      text-shadow: 0 0 10px rgba(255,149,0,0.24);
    }
    [data-testid="stExpander"].det-exp-copy summary svg {
      color: #ffa645 !important;
      filter: drop-shadow(0 0 6px rgba(255,149,0,0.34));
    }
    [data-testid="stExpander"].det-exp-copy [data-testid="stCodeBlock"] {
      border: 1px solid rgba(255,149,0,0.44) !important;
      box-shadow: inset 0 0 0 1px rgba(255,149,0,0.10) !important;
    }

    [data-testid="stExpander"].det-exp-history {
      border: 1px solid rgba(180, 98, 255, 0.74) !important;
      box-shadow:
        0 0 0 1px rgba(180,98,255,0.24),
        0 0 16px rgba(180,98,255,0.28),
        inset 0 0 0 1px rgba(180,98,255,0.10) !important;
    }
    [data-testid="stExpander"].det-exp-history summary {
      border-bottom: 1px solid rgba(180,98,255,0.42) !important;
      background: rgba(180,98,255,0.06) !important;
    }
    [data-testid="stExpander"].det-exp-history summary p {
      color: #ead5ff !important;
      text-shadow: 0 0 10px rgba(180,98,255,0.24);
    }
    [data-testid="stExpander"].det-exp-history summary svg {
      color: #c889ff !important;
      filter: drop-shadow(0 0 6px rgba(180,98,255,0.34));
    }

    @media (max-width: 860px) {
      .score-conf-row { flex-direction: column; }
      .forensics-trace,
      .forensics-grid {
        grid-template-columns: 1fr;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
_bind_detector_shell_classes()

st.markdown(
    """
    <div class="det-system-row">
      <div class="status-bar">
        <div class="status-dot"></div>
        <span class="status-text">System Online</span>
        <span class="version-tag">v2.4.1 — IND</span>
      </div>
    </div>
    <div class="det-page-title-row">
      <span class="det-section-code">// 01</span>
      <h1 class="det-page-h1">Message Detector</h1>
    </div>
    <p class="det-page-sub">Paste any SMS or WhatsApp message below. The engine will classify, score, and explain the risk in seconds.</p>
    <p class="det-examples-label">▶ Try a sample message (watch it type):</p>
    """,
    unsafe_allow_html=True,
)
st.markdown("<div class='det-samples-spacer'></div>", unsafe_allow_html=True)

sample_cols = st.columns(6, gap="small")
for idx, (label, value) in enumerate(SAMPLES.items()):
    safe_key = "sample_safe" if label == "Safe Message" else f"sample_{idx}"
    with sample_cols[idx]:
        if st.button(
            label,
            key=safe_key,
            use_container_width=True,
        ):
            _start_typewriter(value)
            st.rerun()

if st.session_state.get("detector_lang_code") not in LANG_CODE_TO_NAME:
    st.session_state["detector_lang_code"] = "en"

active_lang = st.session_state.get("detector_lang_code", "en")
st.markdown(
    f"""
    <style>
    .st-key-det_lang_en .stButton > button,
    .st-key-det_lang_hi .stButton > button,
    .st-key-det_lang_pa .stButton > button,
    .st-key-det_lang_ur .stButton > button {{
      border-radius: 0 !important;
      min-height: 34px !important;
    }}

    .st-key-det_lang_{active_lang} .stButton > button {{
      background: rgba(0,255,159,0.06) !important;
      border-color: var(--neon) !important;
      color: var(--neon) !important;
      box-shadow: 0 0 10px rgba(0,255,159,0.16);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='det-two-grid'></div>", unsafe_allow_html=True)
left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="det-marker-input"></div>
            <div class="det-panel-titlebar">
              <div class="t-dot t-dot-red"></div>
              <div class="t-dot t-dot-yellow"></div>
              <div class="t-dot t-dot-green"></div>
              <div class="det-panel-title">message_input.py — DETECTOR</div>
            </div>
            <div class="input-label">{html.escape(_ui_text(active_lang, "paste_message"))}</div>
            """,
            unsafe_allow_html=True,
        )

        st.text_area(
            "Message text",
            key="detector_message_input",
            label_visibility="collapsed",
            height=190,
            placeholder="e.g. Dear Customer, your SBI account will be BLOCKED. Click here to update KYC...",
        )
        pending_typewriter = st.session_state.get("detector_typewriter_text", "")
        if pending_typewriter:
            _inject_textarea_typewriter_js(pending_typewriter)
            st.session_state["detector_typewriter_text"] = ""
        current_msg = st.session_state.get("detector_message_input", "")
        if current_msg.strip():
            st.session_state["detector_prefill_text"] = current_msg
        msg_len = len(st.session_state.get("detector_message_input", ""))

        st.markdown(f"<div class='det-char-row'>{msg_len} chars</div>", unsafe_allow_html=True)
        input_error = st.session_state.get("detector_input_error", "")
        if input_error:
            st.markdown(f"<div class='det-inline-error'>{html.escape(input_error)}</div>", unsafe_allow_html=True)

        st.markdown(
            f"<div class='det-lang-row'><span class='det-lang-label'>{html.escape(_ui_text(active_lang, 'output_language'))}:</span></div>",
            unsafe_allow_html=True,
        )
        lang_cols = st.columns(4, gap="small")
        with lang_cols[0]:
            if st.button("English", key="det_lang_en", use_container_width=True):
                st.session_state["detector_lang_code"] = "en"
                st.rerun()
        with lang_cols[1]:
            if st.button("हिंदी", key="det_lang_hi", use_container_width=True):
                st.session_state["detector_lang_code"] = "hi"
                st.rerun()
        with lang_cols[2]:
            if st.button("ਪੰਜਾਬੀ", key="det_lang_pa", use_container_width=True):
                st.session_state["detector_lang_code"] = "pa"
                st.rerun()
        with lang_cols[3]:
            if st.button("اردو", key="det_lang_ur", use_container_width=True):
                st.session_state["detector_lang_code"] = "ur"
                st.rerun()

        has_existing_result = "last_result" in st.session_state
        if has_existing_result:
            st.markdown(
                """
                <style>
                .st-key-detector_analyze .stButton > button {
                  background:
                    repeating-linear-gradient(
                      0deg,
                      rgba(0,0,0,0.00) 0px,
                      rgba(0,0,0,0.00) 6px,
                      rgba(0,0,0,0.10) 6px,
                      rgba(0,0,0,0.10) 8px
                    ),
                    linear-gradient(180deg, #45d8ff 0%, #1bbaf6 100%) !important;
                  box-shadow:
                    0 0 20px rgba(0,212,255,0.42),
                    0 0 56px rgba(0,212,255,0.16) !important;
                }
                .st-key-detector_analyze .stButton > button p {
                  color: #001420 !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

        analyze_pressed = st.button(
            _ui_text(active_lang, "reanalyze_message") if has_existing_result else _ui_text(active_lang, "analyze_message"),
            key="detector_analyze",
            use_container_width=True,
            disabled=bool(st.session_state.get("detector_typing_active", False)),
        )

with right:
    with st.container(border=True):
        result_title = "scan_result.json — WAITING"
        if st.session_state.get("detector_panel_state") == "scanning":
            result_title = "scan_result.json — SCANNING"
        elif "last_result" in st.session_state:
            result_title = "scan_result.json — COMPLETE"

        title_placeholder = st.empty()
        title_placeholder.markdown(_render_result_title(result_title), unsafe_allow_html=True)

        panel_placeholder = st.empty()

        lang_name = LANG_CODE_TO_NAME.get(st.session_state.get("detector_lang_code", "en"), "English")
        triggered = _run_analysis(
            analyze_pressed or bool(st.session_state.pop("detector_auto_analyze", False)),
            st.session_state.get("detector_message_input", ""),
            lang_name,
            panel_placeholder,
            title_placeholder,
            terminal_placeholder,
        )
        if triggered:
            # Force a fresh render pass so button label/style reflect "has result"
            # immediately after the first successful analysis.
            st.rerun()

        if not triggered:
            if "last_result" not in st.session_state:
                with panel_placeholder:
                    _render_waiting_state()
            else:
                with panel_placeholder:
                    _render_result_state(
                        st.session_state["last_result"],
                        st.session_state.get("last_message", ""),
                        st.session_state.get("last_language", lang_name),
                    )
        else:
            panel_placeholder.empty()
            _render_result_state(
                st.session_state["last_result"],
                st.session_state.get("last_message", ""),
                st.session_state.get("last_language", lang_name),
            )

        st.markdown("<div class='det-feedback-strip'></div>", unsafe_allow_html=True)
        has_result = "last_result" in st.session_state
        report_text = ""
        report_pdf = b""
        report_filename = "safesandesh_report.pdf"
        if has_result:
            report_text = _build_report(
                st.session_state["last_result"],
                st.session_state.get("last_message", ""),
                st.session_state.get("last_language", lang_name),
                comparison=st.session_state.get("last_comparison"),
            )
            report_pdf = _build_report_pdf_bytes(report_text)
            report_filename = f"safesandesh_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        fb1, fb2, fb3 = st.columns([2, 2, 2], gap="small")
        with fb1:
            if st.button("⎘ Copy Text", key="detector_copy", use_container_width=True):
                if has_result:
                    st.session_state["detector_report_text"] = report_text
        with fb2:
            st.download_button(
                "⬇ Download PDF",
                data=report_pdf or b"",
                file_name=report_filename,
                mime="application/pdf",
                key="detector_download_pdf",
                use_container_width=True,
                disabled=not has_result,
            )
        with fb3:
            if st.button("⟳ New Scan", key="detector_reset", use_container_width=True):
                st.session_state["detector_reset_requested"] = True
                st.rerun()

if "detector_report_text" in st.session_state and st.session_state["detector_report_text"]:
    with st.expander("Copyable Report Text", expanded=False):
        st.code(st.session_state["detector_report_text"], language="text")

with st.expander(f"Scan History ({len(st.session_state.get('detector_history', []))})", expanded=False):
    history = st.session_state.get("detector_history", [])
    if not history:
        st.caption("No scans yet.")
    else:
        for idx, item in enumerate(history):
            load_col, meta_col = st.columns([0.55, 4.45], gap="small")
            with load_col:
                if st.button("Load", key=f"hist_load_{idx}", use_container_width=False):
                    try:
                        loaded_message = str(item.get("message", "") or "")
                        if not loaded_message.strip():
                            st.warning("Selected history item has no message text.")
                        else:
                            saved_lang = str(item.get("language", "English") or "English").strip().lower()
                            reverse = {v.lower(): k for k, v in LANG_CODE_TO_NAME.items()}
                            st.session_state["detector_history_load_message"] = loaded_message
                            st.session_state["detector_history_load_lang"] = reverse.get(saved_lang, "en")
                            st.session_state.pop("detector_auto_analyze", None)
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Could not load this scan: {exc}")
            with meta_col:
                compare_html = ""
                if item.get("compare_summary"):
                    state = item.get("compare_state", "")
                    compare_html = (
                        f"<div class='det-history-meta'>"
                        f"Model compare: {html.escape(item.get('compare_summary', ''))}"
                        f"{f' · {html.escape(str(state))}' if state else ''}"
                        f"</div>"
                    )
                st.markdown(
                    (
                        "<div class='det-history-card'>"
                        "<div class='det-history-top'>"
                        f"<span>{html.escape(item.get('label', 'Safe'))}</span>"
                        f"<span>{int(item.get('score', 0))}/100 · {item.get('confidence', 0.0) * 100:.1f}%</span>"
                        "</div>"
                        f"<div class='det-history-meta'>{html.escape(item.get('scam_type', 'Other'))} · {html.escape(item.get('ts', ''))}</div>"
                        f"{compare_html}"
                        f"<div class='det-history-msg'>{html.escape((item.get('message', '') or '')[:130])}</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
        if st.button("Clear History", key="hist_clear"):
            st.session_state["detector_history"] = []
            st.rerun()
