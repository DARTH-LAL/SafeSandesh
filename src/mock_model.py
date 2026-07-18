import re
import random

SCAM_TYPES = ["OTP", "KYC", "UPI", "Courier/Customs", "Job/Loan", "Lottery", "Account-Block", "Other"]

def predict(message: str):
    text = message.lower()

    cues = []
    score = 15
    scam_type = "Other"

    if re.search(r"\botp\b|\bone time password\b", text):
        cues.append("OTP request")
        score += 45
        scam_type = "OTP"

    if "kyc" in text or "update kyc" in text:
        cues.append("KYC update")
        score += 30
        scam_type = "KYC"

    if "upi" in text or "collect request" in text or "pay now" in text:
        cues.append("Payment request")
        score += 35
        scam_type = "UPI"

    if "blocked" in text or "suspended" in text or "account" in text and "block" in text:
        cues.append("Threat/urgency")
        score += 20
        scam_type = "Account-Block" if scam_type == "Other" else scam_type

    if "http" in text or "www" in text:
        cues.append("Link present")
        score += 15

    score = max(0, min(100, score + random.randint(-5, 5)))

    if score < 30:
        label = "Safe"
    elif score < 60:
        label = "Suspicious"
    else:
        label = "Phishing"

    reason = " + ".join(cues) if cues else "No strong scam cues detected."
    return {
        "label": label,
        "scam_type": scam_type,
        "risk_score": score,
        "reason": reason
    }
