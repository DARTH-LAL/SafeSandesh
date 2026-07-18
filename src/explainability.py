from __future__ import annotations

import re
from typing import Dict, List


LANGUAGE_CODE_MAP = {
    "english": "en",
    "hindi": "hi",
    "punjabi": "pa",
    "urdu": "ur",
    "en": "en",
    "hi": "hi",
    "pa": "pa",
    "ur": "ur",
}

DEFAULT_LANGUAGE = "en"

SCRIPT_LANGUAGE_PATTERNS = [
    ("hi", re.compile(r"[\u0900-\u097F]")),
    ("pa", re.compile(r"[\u0A00-\u0A7F]")),
    ("ur", re.compile(r"[\u0600-\u06FF\u0750-\u077F]")),
]

NEGATED_SECRET_SHARE_RE = re.compile(
    r"(?:do\s+not|don't|never)\s+share[^.!?।॥؟۔]{0,80}\b(?:otp|code|password|pin|passcode)\b|"
    r"\b(?:otp|code|password|pin|passcode)\b[^.!?।॥؟۔]{0,80}(?:do\s+not|don't|never)\s+share|"
    r"(?:OTP|ओटीपी|पासवर्ड|कोड|पिन)[^।॥.!?]{0,100}साझा\s*न\s*करें|"
    r"साझा\s*न\s*करें[^।॥.!?]{0,100}(?:OTP|ओटीपी|पासवर्ड|कोड|पिन)|"
    r"(?:ਓਟੀਪੀ|ਪਾਸਵਰਡ|ਕੋਡ|ਪਿਨ)[^।॥.!?]{0,100}ਸਾਂਝਾ\s*ਨਾ\s*ਕਰੋ|"
    r"ਸਾਂਝਾ\s*ਨਾ\s*ਕਰੋ[^।॥.!?]{0,100}(?:ਓਟੀਪੀ|ਪਾਸਵਰਡ|ਕੋਡ|ਪਿਨ)|"
    r"(?:او\s*ٹی\s*پی|کوڈ|پاس\s*ورڈ|پن)[^.!?؟۔]{0,100}شیئر\s*نہ\s*کریں|"
    r"شیئر\s*نہ\s*کریں[^.!?؟۔]{0,100}(?:او\s*ٹی\s*پی|کوڈ|پاس\s*ورڈ|پن)",
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


CUE_RULES = [
    {
        "tag": "OTP_REQUEST",
        "pattern": re.compile(r"\botp\b|one\s*time\s*password|ओटीपी|ਓਟੀਪੀ|او\s*ٹی\s*پی", re.IGNORECASE),
        "labels": {
            "en": "OTP request",
            "hi": "ओटीपी मांग",
            "pa": "ਓਟੀਪੀ ਮੰਗ",
            "ur": "او ٹی پی کی درخواست",
        },
    },
    {
        "tag": "KYC_UPDATE",
        "pattern": re.compile(r"\bkyc\b|aadhaar|aadhar|\bpan\b|केवाईसी|ਆਧਾਰ|کے\s*وائی\s*سی", re.IGNORECASE),
        "labels": {
            "en": "KYC update request",
            "hi": "केवाईसी अपडेट अनुरोध",
            "pa": "ਕੇਵਾਈਸੀ ਅਪਡੇਟ ਬੇਨਤੀ",
            "ur": "کے وائی سی اپ ڈیٹ کی درخواست",
        },
    },
    {
        "tag": "PAYMENT_REQUEST",
        "pattern": re.compile(
            r"\bupi\b|gpay|google\s*pay|phonepe|paytm|collect\s*request|pay\s*now|upi\s*id|ਯੂਪੀਆਈ|यूपीआई|یو\s*پی\s*آئی",
            re.IGNORECASE,
        ),
        "labels": {
            "en": "Payment/UPI request",
            "hi": "भुगतान/यूपीआई अनुरोध",
            "pa": "ਭੁਗਤਾਨ/ਯੂਪੀਆਈ ਬੇਨਤੀ",
            "ur": "ادائیگی/یو پی آئی کی درخواست",
        },
    },
    {
        "tag": "ACCOUNT_THREAT",
        "pattern": re.compile(
            r"blocked|suspended|frozen|locked|pending|account\s*block|verify\s*account|final warning|act now|urgent|immediately|"
            r"अत्यावश्यक|तत्काल|तुरंत|लंबित|अस्थायी\s*रूप\s*से|ब्लॉक|सत्यापित|सत्यापन|निलंबित|फ्रीज|"
            r"ਤੁਰੰਤ|ਬਕਾਇਆ|ਅਸਥਾਈ\s*ਤੌਰ\s*ਤੇ|ਬਲਾਕ|ਤਸਦੀਕ|ਮੁਅੱਤਲ|ਫ੍ਰੀਜ਼|"
            r"فوری|فوراً|زیر\s*التوا|عارضی\s*طور\s*پر|بلاک|تصدیق|معطل|فریز",
            re.IGNORECASE,
        ),
        "labels": {
            "en": "Account threat/urgency",
            "hi": "खाता खतरा/तत्कालता",
            "pa": "ਖਾਤਾ ਖਤਰਾ/ਤੁਰੰਤਤਾ",
            "ur": "اکاؤنٹ خطرہ/فوری دباؤ",
        },
    },
    {
        "tag": "LINK_PRESENT",
        "pattern": re.compile(r"https?://|www\.|लिंक|ਲਿੰਕ|لنک", re.IGNORECASE),
        "labels": {
            "en": "Suspicious link present",
            "hi": "संदिग्ध लिंक मौजूद",
            "pa": "ਸੰਦੇਹਜਨਕ ਲਿੰਕ ਮੌਜੂਦ",
            "ur": "مشکوک لنک موجود",
        },
    },
    {
        "tag": "LOTTERY_PRIZE",
        "pattern": re.compile(r"lottery|prize|winner|jackpot|reward|lucky draw|इनाम|ਇਨਾਮ|انعام", re.IGNORECASE),
        "labels": {
            "en": "Prize/lottery lure",
            "hi": "इनाम/लॉटरी लालच",
            "pa": "ਇਨਾਮ/ਲਾਟਰੀ ਲਾਲਚ",
            "ur": "انعام/لاٹری کا لالچ",
        },
    },
    {
        "tag": "DELIVERY_CUSTOMS",
        "pattern": re.compile(r"courier|parcel|delivery|shipment|customs|awb|कूरियर|ਕੂਰੀਅਰ|کورئیر", re.IGNORECASE),
        "labels": {
            "en": "Courier/customs pretext",
            "hi": "कूरियर/कस्टम बहाना",
            "pa": "ਕੂਰੀਅਰ/ਕਸਟਮ ਬਹਾਨਾ",
            "ur": "کورئیر/کسٹمز کا بہانہ",
        },
    },
    {
        "tag": "JOB_LOAN",
        "pattern": re.compile(r"\bjob\b|hiring|vacancy|loan|salary|work from home|नौकरी|ਲੋਨ|نوکری|قرض", re.IGNORECASE),
        "labels": {
            "en": "Job/loan bait",
            "hi": "नौकरी/लोन लालच",
            "pa": "ਨੌਕਰੀ/ਲੋਨ ਲਾਲਚ",
            "ur": "نوکری/قرض کا لالچ",
        },
    },
]


REASON_TEMPLATES = {
    "en": {
        "with_cues": "Detected cues: {cues}. Model confidence: {confidence:.2f}.",
        "no_cues": "No strong explicit scam cue detected. Model confidence: {confidence:.2f}.",
    },
    "hi": {
        "with_cues": "पहचाने गए संकेत: {cues}. मॉडल का भरोसा: {confidence:.2f}।",
        "no_cues": "कोई मजबूत स्पष्ट धोखाधड़ी संकेत नहीं मिला। मॉडल का भरोसा: {confidence:.2f}।",
    },
    "pa": {
        "with_cues": "ਪਛਾਣੇ ਗਏ ਸੰਕੇਤ: {cues}. ਮਾਡਲ ਭਰੋਸਾ: {confidence:.2f}.",
        "no_cues": "ਕੋਈ ਮਜ਼ਬੂਤ ਧੋਖਾਧੜੀ ਸੰਕੇਤ ਨਹੀਂ ਮਿਲਿਆ। ਮਾਡਲ ਭਰੋਸਾ: {confidence:.2f}.",
    },
    "ur": {
        "with_cues": "پہچانے گئے اشارے: {cues}۔ ماڈل اعتماد: {confidence:.2f}۔",
        "no_cues": "کوئی مضبوط واضح فراڈ اشارہ نہیں ملا۔ ماڈل اعتماد: {confidence:.2f}۔",
    },
}


ACTION_TEMPLATES = {
    "OTP": {
        "en": ["Do not share OTP with anyone.", "Contact official support if unsure."],
        "hi": ["ओटीपी किसी से साझा न करें।", "संदेह होने पर आधिकारिक सपोर्ट से संपर्क करें।"],
        "pa": ["ਓਟੀਪੀ ਕਿਸੇ ਨਾਲ ਸਾਂਝਾ ਨਾ ਕਰੋ।", "ਸੰਦੇਹ ਹੋਵੇ ਤਾਂ ਅਧਿਕਾਰਕ ਸਹਾਇਤਾ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।"],
        "ur": ["او ٹی پی کسی کے ساتھ شیئر نہ کریں۔", "شبہ ہو تو صرف آفیشل سپورٹ سے رابطہ کریں۔"],
    },
    "KYC": {
        "en": ["Do not update KYC via unknown links.", "Use only official app/website."],
        "hi": ["अनजान लिंक से केवाईसी अपडेट न करें।", "केवल आधिकारिक ऐप/वेबसाइट का उपयोग करें।"],
        "pa": ["ਅਣਜਾਣ ਲਿੰਕ ਰਾਹੀਂ ਕੇਵਾਈਸੀ ਅਪਡੇਟ ਨਾ ਕਰੋ।", "ਕੇਵਲ ਅਧਿਕਾਰਕ ਐਪ/ਵੈਬਸਾਈਟ ਵਰਤੋ।"],
        "ur": ["نامعلوم لنک سے کے وائی سی اپ ڈیٹ نہ کریں۔", "صرف آفیشل ایپ/ویب سائٹ استعمال کریں۔"],
    },
    "UPI": {
        "en": ["Do not approve unknown collect requests.", "Never enter UPI PIN to receive money."],
        "hi": ["अनजान कलेक्ट रिक्वेस्ट स्वीकार न करें।", "पैसे पाने के लिए कभी यूपीआई पिन न डालें।"],
        "pa": ["ਅਣਜਾਣ ਕਲੈਕਟ ਬੇਨਤੀ ਮਨਜ਼ੂਰ ਨਾ ਕਰੋ।", "ਪੈਸੇ ਲੈਣ ਲਈ ਕਦੇ ਵੀ ਯੂਪੀਆਈ ਪਿਨ ਨਾ ਪਾਓ।"],
        "ur": ["نامعلوم کلیکٹ ریکویسٹ منظور نہ کریں۔", "پیسے وصول کرنے کے لیے کبھی یو پی آئی پن نہ ڈالیں۔"],
    },
    "Courier/Customs": {
        "en": ["Verify shipment on official courier site.", "Do not pay fees on unknown links."],
        "hi": ["शिपमेंट स्थिति आधिकारिक साइट पर जांचें।", "अनजान लिंक पर शुल्क भुगतान न करें।"],
        "pa": ["ਸ਼ਿਪਮੈਂਟ ਅਧਿਕਾਰਕ ਸਾਈਟ 'ਤੇ ਚੈਕ ਕਰੋ।", "ਅਣਜਾਣ ਲਿੰਕ 'ਤੇ ਫੀਸ ਨਾ ਭਰੋ।"],
        "ur": ["پارسل اسٹیٹس آفیشل سائٹ پر چیک کریں۔", "نامعلوم لنکس پر فیس ادا نہ کریں۔"],
    },
    "Job/Loan": {
        "en": ["Avoid paying upfront processing fees.", "Verify recruiter/lender independently."],
        "hi": ["पहले से प्रोसेसिंग फीस भुगतान से बचें।", "भर्तीकर्ता/लेंडर की स्वतंत्र पुष्टि करें।"],
        "pa": ["ਪਹਿਲਾਂ ਪ੍ਰੋਸੈਸਿੰਗ ਫੀਸ ਦੇਣ ਤੋਂ ਬਚੋ।", "ਰਿਕਰੂਟਰ/ਲੈਂਡਰ ਦੀ ਖੁਦ ਪੁਸ਼ਟੀ ਕਰੋ।"],
        "ur": ["پہلے سے پروسیسنگ فیس ادا کرنے سے بچیں۔", "ریکروٹر/لینڈر کی الگ سے تصدیق کریں۔"],
    },
    "Lottery": {
        "en": ["Ignore prize claims asking for payment.", "Do not share personal details."],
        "hi": ["भुगतान मांगने वाले इनाम दावों को अनदेखा करें।", "व्यक्तिगत जानकारी साझा न करें।"],
        "pa": ["ਭੁਗਤਾਨ ਮੰਗਣ ਵਾਲੇ ਇਨਾਮ ਦਾਵਿਆਂ ਨੂੰ ਅਣਡਿੱਠਾ ਕਰੋ।", "ਨਿੱਜੀ ਜਾਣਕਾਰੀ ਸਾਂਝੀ ਨਾ ਕਰੋ।"],
        "ur": ["ادائیگی مانگنے والے انعام دعووں کو نظرانداز کریں۔", "ذاتی معلومات شیئر نہ کریں۔"],
    },
    "Account-Block": {
        "en": ["Do not click urgent reactivation links.", "Check account status in official app."],
        "hi": ["तुरंत रिएक्टिवेशन लिंक पर क्लिक न करें।", "खाते की स्थिति आधिकारिक ऐप में देखें।"],
        "pa": ["ਤੁਰੰਤ ਰਿਐਕਟੀਵੇਸ਼ਨ ਲਿੰਕ 'ਤੇ ਕਲਿੱਕ ਨਾ ਕਰੋ।", "ਖਾਤੇ ਦੀ ਸਥਿਤੀ ਅਧਿਕਾਰਕ ਐਪ ਵਿੱਚ ਵੇਖੋ।"],
        "ur": ["فوری ری ایکٹیویشن لنک پر کلک نہ کریں۔", "اکاؤنٹ اسٹیٹس آفیشل ایپ میں چیک کریں۔"],
    },
    "Other": {
        "en": ["Do not share sensitive data without verification.", "Cross-check with official channels."],
        "hi": ["सत्यापन के बिना संवेदनशील जानकारी साझा न करें।", "आधिकारिक चैनल से दोबारा जांच करें।"],
        "pa": ["ਤਸਦੀਕ ਬਿਨਾਂ ਸੰਵੇਦਨਸ਼ੀਲ ਜਾਣਕਾਰੀ ਸਾਂਝੀ ਨਾ ਕਰੋ।", "ਅਧਿਕਾਰਕ ਚੈਨਲ ਨਾਲ ਪੁਸ਼ਟੀ ਕਰੋ।"],
        "ur": ["تصدیق کے بغیر حساس معلومات شیئر نہ کریں۔", "آفیشل ذرائع سے دوبارہ تصدیق کریں۔"],
    },
}


TAG_TO_SCAM_TYPE = {
    "OTP_REQUEST": "OTP",
    "KYC_UPDATE": "KYC",
    "PAYMENT_REQUEST": "UPI",
    "ACCOUNT_THREAT": "Account-Block",
    "LOTTERY_PRIZE": "Lottery",
    "DELIVERY_CUSTOMS": "Courier/Customs",
    "JOB_LOAN": "Job/Loan",
}


def language_code(language: str | None) -> str:
    if language is None:
        return DEFAULT_LANGUAGE
    key = str(language).strip().lower()
    return LANGUAGE_CODE_MAP.get(key, DEFAULT_LANGUAGE)


def infer_message_language_code(text: str, fallback: str | None = None) -> str:
    sample = str(text or "")
    if not sample.strip():
        return language_code(fallback)

    scores = {code: len(pattern.findall(sample)) for code, pattern in SCRIPT_LANGUAGE_PATTERNS}
    best_code, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score > 0:
        return best_code
    return language_code(fallback)


def infer_message_language_name(text: str, fallback: str | None = None) -> str:
    code = infer_message_language_code(text, fallback=fallback)
    return {
        "en": "English",
        "hi": "Hindi",
        "pa": "Punjabi",
        "ur": "Urdu",
    }.get(code, "English")


def detect_cue_tags(text: str) -> List[str]:
    sample = str(text or "")
    tags = [rule["tag"] for rule in CUE_RULES if rule["pattern"].search(sample)]
    if "OTP_REQUEST" in tags and NEGATED_SECRET_SHARE_RE.search(sample):
        tags = [tag for tag in tags if tag != "OTP_REQUEST"]
    if (
        "DELIVERY_CUSTOMS" in tags
        and SAFE_DELIVERY_BRAND_RE.search(sample)
        and SAFE_DELIVERY_STATUS_RE.search(sample)
        and not DELIVERY_SCAM_ACTION_RE.search(sample)
    ):
        tags = [tag for tag in tags if tag != "DELIVERY_CUSTOMS"]
    return tags


def cue_labels(tags: List[str], lang: str) -> List[str]:
    labels = []
    for tag in tags:
        found = next((rule for rule in CUE_RULES if rule["tag"] == tag), None)
        if found is None:
            continue
        labels.append(found["labels"].get(lang, found["labels"]["en"]))
    return labels


def reason_text(tags: List[str], confidence: float, lang: str) -> str:
    tmpl = REASON_TEMPLATES.get(lang, REASON_TEMPLATES["en"])
    if tags:
        labels = cue_labels(tags, lang)
        return tmpl["with_cues"].format(cues=", ".join(labels), confidence=confidence)
    return tmpl["no_cues"].format(confidence=confidence)


def actions_for_scam_type(scam_type: str, lang: str) -> List[str]:
    table = ACTION_TEMPLATES.get(scam_type, ACTION_TEMPLATES["Other"])
    return table.get(lang, table["en"])


def infer_scam_type_from_tags(tags: List[str]) -> str:
    for tag in tags:
        if tag in TAG_TO_SCAM_TYPE:
            return TAG_TO_SCAM_TYPE[tag]
    return "Other"
