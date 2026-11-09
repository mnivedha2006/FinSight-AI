"""Shared message rules and text normaliser for FinSight AI.

Used by the generator, the trainer, the dashboard and (later) the fusion engine,
so every part of the project scores a message the same way.
"""
import re

URGENCY = ["urgent", "immediately", "immediate", "blocked", "suspended", "expire", "within 24", "today",
           "tonight", "last chance", "disconnected", "deactivated", "udane", "indru", "ippove", "block aagividum",
           "thundikkapadum", "முடக்க", "உடனடியாக", "இன்று", "तुरंत", "अभी", "आज", "बंद हो", "ब्लॉक", "कट जाएगा",
           "jaldi", "turant", "abhi"]
CREDENTIAL = ["otp", "pin", "password", "cvv", "card number", "upi pin", "ஓடிபி", "கடவுச்சொல்", "பின்",
              "ओटीपी", "पिन", "पासवर्ड"]
LURE = ["reward", "cashback", "prize", "lottery", "refund", "kyc", "gift", "winner", "congratulations",
        "pre-approved", "பரிசு", "வாழ்த்துக்கள்", "इनाम", "बधाई", "रिफंड", "केवाईसी", "parisu", "vazhthukkal"]
COLLECT = ["collect request", "payment request", "approve the request", "accept the request", "request money",
           "enter your pin to receive", "enter your upi pin", "to receive the money", "receive panna",
           "பணம் கோரிக்கை", "पेमेंट रिक्वेस्ट", "पैसे पाने के लिए"]
SAFE_CTX = ["never share", "do not share", "don't share", "not share your otp", "share panna vendaam",
            "பகிர வேண்டாம்", "साझा न करें", "share na karein", "we never ask"]
URL_HINTS = ["http", "bit.ly", "tinyurl", "rb.gy", "verify-", ".xyz", ".top", ".info", "www."]

SIGNALS = [
    ("Urgency language", "Pressure to act before you think.", URGENCY),
    ("Credential request", "Banks never ask for your OTP, PIN or password.", CREDENTIAL),
    ("Suspicious link", "Shortened or look-alike links open fake pages.", URL_HINTS),
    ("Financial lure", "Rewards, refunds and KYC updates are common bait.", LURE),
    ("Collect-request pattern", "Approving a collect request sends money out.", COLLECT),
    ("Safety warning", "Genuine alerts often tell you not to share codes.", SAFE_CTX),
]


def find_signals(text):
    t = str(text).lower()
    return {name: any(w.lower() in t for w in words) for name, _, words in SIGNALS}


def rule_score(f):
    s = (25 if f["Urgency language"] else 0) + (30 if f["Credential request"] else 0) \
        + (25 if f["Suspicious link"] else 0) + (10 if f["Financial lure"] else 0) \
        + (20 if f["Collect-request pattern"] else 0)
    if f["Safety warning"] and not f["Suspicious link"]:
        s = max(0, s - 30)
    return min(100, s)


def rule_score_text(text):
    return rule_score(find_signals(text))


# ---- text normaliser for TF-IDF (module-level so joblib can pickle the pipeline) ----
_URL = re.compile(r"https?://\S+|\b(?:bit\.ly|tinyurl\.com|rb\.gy)/\S+|\b[\w-]+\.(?:xyz|top|info)\b", re.I)
_SUSPICIOUS = re.compile(r"\.(xyz|top|info)\b|bit\.ly|tinyurl|rb\.gy|verify|kyc|secure|update|login", re.I)
_NUM = re.compile(r"\b\d+\b")
# keep word characters, whitespace, Devanagari and Tamil blocks (so matras are not stripped)
_PUNCT = re.compile(r"[^\w\s\u0900-\u097F\u0B80-\u0BFF]")


def normalise(text):
    t = str(text).lower()
    t = _URL.sub(lambda m: " urltoken susurl " if _SUSPICIOUS.search(m.group(0)) else " urltoken ", t)
    t = _NUM.sub(" numtoken ", t)
    t = _PUNCT.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()