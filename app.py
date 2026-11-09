"""FinSight AI - Fraud investigation workspace (Streamlit)

Files:  FinSight-AI/app.py   and   FinSight-AI/.streamlit/config.toml
Run:    streamlit run app.py
"""
import html as H
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "evaluation"
SYNTH_DIR = BASE_DIR / "data" / "synthetic"
PROC_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

# Re-tune after retraining the message classifier on realistic data.
ML_WEIGHT, RULE_WEIGHT = 0.5, 0.5

st.set_page_config(page_title="FinSight AI", page_icon="F", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------- themes
THEMES = {
    "Daybreak": {   # light, tinted blue paper with soft glows
        "bg": "#E6EDF6", "glow1": "rgba(15,110,140,.20)", "glow2": "rgba(110,130,255,.18)",
        "dot": "rgba(20,33,61,.07)", "side": "#0E1830", "side-text": "#B9C4DA",
        "card": "rgba(255,255,255,.88)", "row": "#F3F6FB", "line": "#D3DCE8", "line2": "#E4EAF2",
        "text": "#14213D", "mute": "#5E6C84", "acc": "#0F6E8C", "marker": "#14213D",
        "hi": "#C8372D", "mid": "#B8741A", "mid-text": "#8A5410", "lo": "#2E7D5B",
        "hi-bg": "#FBE9E7", "mid-bg": "#FBF0DA", "lo-bg": "#E3F2EA",
        "z-lo": "#CFE5D9", "z-mid": "#F1DDB0", "z-hi": "#EBC1BC", "track": "#DCE3EC",
        "node": "#14213D", "grey": "#9AA5B4"},
    "Midnight": {   # deep navy with indigo and teal glows
        "bg": "#0A1020", "glow1": "rgba(15,110,140,.34)", "glow2": "rgba(99,102,241,.22)",
        "dot": "rgba(255,255,255,.05)", "side": "#060B16", "side-text": "#A9B6D0",
        "card": "rgba(18,27,46,.86)", "row": "#16213A", "line": "#26365A", "line2": "#1C2943",
        "text": "#E6EBF5", "mute": "#8C9BB8", "acc": "#37B6D9", "marker": "#FFFFFF",
        "hi": "#FF6B5E", "mid": "#F0B04A", "mid-text": "#F0B04A", "lo": "#43C48E",
        "hi-bg": "rgba(255,107,94,.16)", "mid-bg": "rgba(240,176,74,.16)", "lo-bg": "rgba(67,196,142,.16)",
        "z-lo": "rgba(67,196,142,.38)", "z-mid": "rgba(240,176,74,.38)", "z-hi": "rgba(255,107,94,.40)",
        "track": "#223052", "node": "#DCE6F7", "grey": "#5B6B8C"},
    "Monsoon": {    # deep green-teal with a warm gold glow
        "bg": "#071B1A", "glow1": "rgba(30,170,140,.28)", "glow2": "rgba(214,170,70,.16)",
        "dot": "rgba(255,255,255,.045)", "side": "#04100F", "side-text": "#A5C4BE",
        "card": "rgba(13,38,36,.88)", "row": "#123432", "line": "#21504B", "line2": "#173C39",
        "text": "#E4F2EF", "mute": "#86A9A3", "acc": "#3FD0B0", "marker": "#FFFFFF",
        "hi": "#FF7A66", "mid": "#E8B24C", "mid-text": "#E8B24C", "lo": "#4CD19A",
        "hi-bg": "rgba(255,122,102,.16)", "mid-bg": "rgba(232,178,76,.16)", "lo-bg": "rgba(76,209,154,.16)",
        "z-lo": "rgba(76,209,154,.38)", "z-mid": "rgba(232,178,76,.38)", "z-hi": "rgba(255,122,102,.40)",
        "track": "#1B4440", "node": "#D5EFE9", "grey": "#4F7A74"},
}

PAGES = ["Overview", "Check a message", "Investigate a transaction", "Counterfactual lab",
         "Spending insights", "Beneficiary network", "Model results"]

# ---------------------------------------------------------------- sidebar
st.sidebar.markdown(
    '<div class="fs brand"><svg width="26" height="26" viewBox="0 0 26 26"><rect x="2" y="14" width="5" '
    'height="10" rx="2" fill="#5FB3CE"/><rect x="10" y="8" width="5" height="16" rx="2" fill="#8FD0E4"/>'
    '<rect x="18" y="2" width="5" height="22" rx="2" fill="#FFFFFF"/></svg><b>FinSight</b></div>'
    '<div class="fs brand-sub">Fraud investigation workspace</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
theme_name = st.sidebar.selectbox("Theme", list(THEMES), key="theme")
T = THEMES[theme_name]
st.sidebar.markdown('<div class="fs side-note">Prototype running on synthetic data. Every account, '
                    'message and amount is fictional.</div>', unsafe_allow_html=True)

VARS = ":root{" + "".join(f"--{k}:{v};" for k, v in T.items()) + "}"
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700;800&display=swap');
.stApp, .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp textarea, .stApp button,
.stApp input, .fs, .fs *{font-family:'Schibsted Grotesk',system-ui,-apple-system,'Segoe UI',sans-serif;}
.fs{font-variant-numeric:normal;color:var(--text);}
table.ledger td.r, .ev .num, .delta{font-variant-numeric:tabular-nums;}

/* themed canvas */
.stApp{color:var(--text);background-color:var(--bg);
 background-image:radial-gradient(900px 560px at 6% -6%, var(--glow1), transparent 62%),
 radial-gradient(820px 520px at 100% 0%, var(--glow2), transparent 58%),
 radial-gradient(var(--dot) 1px, transparent 1.3px);
 background-size:auto, auto, 26px 26px;background-attachment:fixed;}
[data-testid="stAppViewContainer"], [data-testid="stMain"]{background:transparent;}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer, .stDeployButton, [data-testid="stToolbarActions"], [data-testid="stMainMenu"]{display:none !important;}
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"]{
 visibility:visible !important;opacity:1 !important;}
[data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapsedControl"] *, [data-testid="collapsedControl"] *{
 color:var(--text) !important;fill:var(--text) !important;}
.block-container{padding-top:2.2rem;max-width:1180px;}
.stApp p, .stApp label, [data-testid="stMarkdownContainer"] p{color:var(--text);}
[data-testid="stCaptionContainer"]{color:var(--mute);}

/* page head */
.head h1{font-size:2rem;font-weight:700;letter-spacing:-.6px;margin:0 0 4px 0;color:var(--text);line-height:1.15;}
.head p{margin:0 0 22px 0;color:var(--mute);font-size:1rem;max-width:64ch;line-height:1.5;}
.lead{font-size:1.25rem;line-height:1.45;max-width:56ch;margin:0 0 26px 0;}
.lead b{font-weight:700;}
h3.sec{font-size:1.02rem;font-weight:650;margin:30px 0 12px 0;color:var(--text);letter-spacing:-.1px;}

/* KPI strip */
.strip{display:grid;background:var(--card);border:1px solid var(--line);border-radius:12px;
 margin-bottom:6px;backdrop-filter:blur(6px);}
.strip .cell{padding:18px 22px;border-left:1px solid var(--line);}
.strip .cell:first-child{border-left:none;}
.strip .k{font-size:.82rem;color:var(--mute);}
.strip .v{font-size:1.85rem;font-weight:700;letter-spacing:-.8px;margin-top:2px;}
.strip .n{font-size:.78rem;color:var(--mute);margin-top:1px;}

/* ledger */
table.ledger{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);
 border-radius:12px;overflow:hidden;backdrop-filter:blur(6px);}
table.ledger th{font-weight:500;font-size:.78rem;color:var(--mute);text-align:left;padding:11px 18px;
 border-bottom:1px solid var(--line);background:var(--row);}
table.ledger td{padding:13px 18px;border-bottom:1px solid var(--line2);font-size:.93rem;vertical-align:middle;
 color:var(--text);}
table.ledger tr:last-child td{border-bottom:none;}
table.ledger tbody tr:hover td{background:var(--row);}
table.ledger .r{text-align:right;}
.id{font-weight:600;}
.pill{display:inline-block;padding:3px 11px;border-radius:999px;font-size:.78rem;font-weight:600;}
.pill.hi{background:var(--hi-bg);color:var(--hi);}
.pill.mid{background:var(--mid-bg);color:var(--mid-text);}
.pill.lo{background:var(--lo-bg);color:var(--lo);}
.pill.na{background:var(--line2);color:var(--mute);}
.mini{display:inline-block;width:84px;height:6px;border-radius:3px;background:var(--track);
 vertical-align:middle;margin-right:10px;overflow:hidden;}
.mini i{display:block;height:100%;border-radius:3px;}
.mini i.hi{background:var(--hi);} .mini i.mid{background:var(--mid);} .mini i.lo{background:var(--lo);}

/* risk ruler (signature element) */
.ruler{margin:10px 0 4px 0;}
.ruler .rl{font-size:.8rem;color:var(--mute);margin-bottom:12px;}
.ruler .track{position:relative;display:flex;height:14px;border-radius:7px;}
.ruler .z{height:100%;display:block;}
.ruler .z.lo{background:var(--z-lo);border-radius:7px 0 0 7px;}
.ruler .z.mid{background:var(--z-mid);}
.ruler .z.hi{background:var(--z-hi);border-radius:0 7px 7px 0;}
.ruler .marker{position:absolute;top:-7px;width:4px;height:28px;margin-left:-2px;background:var(--marker);
 border-radius:2px;box-shadow:0 0 0 3px var(--bg);animation:land .9s cubic-bezier(.2,.8,.2,1);}
@keyframes land{from{left:0%;}}
.ruler .ticks{position:relative;height:22px;margin-top:14px;font-size:.76rem;color:var(--mute);}
.ruler .ticks span{position:absolute;}
@media (prefers-reduced-motion: reduce){.ruler .marker{animation:none;}}

.result{display:flex;align-items:flex-end;gap:28px;flex-wrap:wrap;margin:6px 0 10px 0;}
.bignum{font-size:4.6rem;font-weight:800;letter-spacing:-3px;line-height:.9;}
.bignum.hi{color:var(--hi);} .bignum.mid{color:var(--mid);} .bignum.lo{color:var(--lo);}
.verdict{padding-bottom:6px;}
.verdict .t{font-size:1.2rem;font-weight:700;}
.verdict .d{color:var(--mute);margin-top:2px;max-width:46ch;line-height:1.45;}
.cmp{display:flex;align-items:flex-end;gap:26px;flex-wrap:wrap;margin:8px 0 14px 0;}
.cmp .lbl{font-size:.82rem;color:var(--mute);margin-bottom:6px;}
.cmp .sep{width:44px;height:2px;background:var(--line);margin-bottom:26px;}
.delta{font-size:1.3rem;font-weight:700;padding-bottom:6px;}

/* evidence + definition list + notes */
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:6px 22px;
 backdrop-filter:blur(6px);}
.ev{display:grid;grid-template-columns:150px 1fr 48px;gap:16px;align-items:center;padding:12px 0;
 border-bottom:1px solid var(--line2);font-size:.93rem;}
.ev:last-child{border-bottom:none;}
.ev .bar{height:8px;border-radius:4px;background:var(--track);overflow:hidden;}
.ev .bar i{display:block;height:100%;background:var(--acc);border-radius:4px;}
.ev .num{text-align:right;font-weight:600;}
.deflist{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--line);border-radius:12px;
 background:var(--card);overflow:hidden;backdrop-filter:blur(6px);}
.deflist div{padding:14px 18px;border-bottom:1px solid var(--line2);border-right:1px solid var(--line2);}
.deflist .dk{font-size:.76rem;color:var(--mute);}
.deflist .dv{font-weight:600;margin-top:2px;word-break:break-all;}
.cols3{display:grid;grid-template-columns:repeat(3,1fr);gap:28px;margin-top:6px;}
.cols3 div{border-top:2px solid var(--acc);padding-top:10px;}
.cols3 b{display:block;margin-bottom:4px;}
.cols3 span{color:var(--mute);font-size:.9rem;line-height:1.5;}
.fine{color:var(--mute);font-size:.82rem;margin-top:8px;line-height:1.5;}
.note{border-left:3px solid var(--acc);background:var(--card);padding:14px 18px;border-radius:0 10px 10px 0;
 line-height:1.55;font-size:.93rem;border-top:1px solid var(--line);border-right:1px solid var(--line);
 border-bottom:1px solid var(--line);}

/* inputs */
.stButton > button, .stDownloadButton > button{border-radius:8px;border:1px solid var(--line);
 background:var(--card);color:var(--text);font-weight:600;padding:.45rem .9rem;
 transition:border-color .15s, color .15s;}
.stButton > button p, .stDownloadButton > button p{color:inherit;}
.stButton > button:hover, .stDownloadButton > button:hover{border-color:var(--acc);color:var(--acc);}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible{outline:2px solid var(--acc);outline-offset:2px;}
.stTextArea textarea{border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--text);font-size:1rem;}
div[data-baseweb="select"] > div{background:var(--card);border:1px solid var(--line);border-radius:8px;}
div[data-baseweb="select"] *{color:var(--text);}
div[data-baseweb="select"] svg{fill:var(--mute);}
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] [data-baseweb="menu"]{background:var(--row) !important;}
div[data-baseweb="popover"] li{color:var(--text) !important;background:transparent !important;}
div[data-baseweb="popover"] li:hover{background:var(--line2) !important;}
[data-testid="stSlider"] label, [data-testid="stSlider"] div{color:var(--text);}
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"]{color:var(--mute);}
[data-testid="stFileUploaderDropzone"]{background:var(--card);border:1px dashed var(--line);}
[data-testid="stFileUploaderDropzone"] *{color:var(--mute);}
[data-testid="stFileUploaderDropzone"] button{color:var(--text);}
.stTabs [data-baseweb="tab"]{color:var(--mute);}
.stTabs [aria-selected="true"]{color:var(--text);}
.stTabs [data-baseweb="tab-border"]{background:var(--line);}

/* sidebar */
section[data-testid="stSidebar"]{background:var(--side);border-right:none;}
section[data-testid="stSidebar"] *{color:var(--side-text);}
section[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.08);}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div{background:rgba(255,255,255,.07);
 border-color:rgba(255,255,255,.14);}
section[data-testid="stSidebar"] div[data-baseweb="select"] *{color:#E6EBF5;}
div[role="radiogroup"]{gap:2px;}
div[role="radiogroup"] label{padding:9px 12px;border-radius:8px;width:100%;transition:background .15s;}
div[role="radiogroup"] label:hover{background:rgba(255,255,255,.06);}
div[role="radiogroup"] label:has(input:checked){background:rgba(255,255,255,.12);}
div[role="radiogroup"] label:has(input:checked) *{color:#FFFFFF;font-weight:600;}
div[role="radiogroup"] label > div:first-child{display:none;}
.brand{display:flex;align-items:center;gap:10px;padding:4px 4px 2px 4px;}
.brand b{color:#fff !important;font-size:1.25rem;letter-spacing:-.3px;font-weight:700;}
.brand-sub{padding:0 4px 8px 4px;font-size:.8rem;}
.side-note{font-size:.78rem;line-height:1.5;padding:0 4px;}
@media (max-width: 800px){.deflist,.cols3{grid-template-columns:1fr;}}
</style>
"""
st.markdown("<style>" + VARS + "</style>" + CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------- data
@st.cache_data
def load_json(name):
    try:
        with open(EVAL_DIR / name) as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def load_classifier():
    try:
        return joblib.load(MODEL_DIR / "message_classifier.joblib")["pipeline"]
    except Exception:
        return None


@st.cache_resource
def load_behaviour():
    """Isolation Forest + engineered features, with every payment's anomaly score."""
    try:
        art = joblib.load(MODEL_DIR / "fraud_detector.joblib")
        feats = pd.read_csv(PROC_DIR / "transaction_features.csv")
        cols = art["feature_columns"]
        pipe = art["pipeline"]
        imp, mdl = pipe.named_steps["imputer"], pipe.named_steps["model"]
        anomaly = -mdl.decision_function(imp.transform(feats[cols]))
        return {"cols": cols, "imp": imp, "mdl": mdl, "feats": feats,
                "sorted": np.sort(anomaly), "anomaly": anomaly}
    except Exception:
        return None


tx_metrics = load_json("model_metrics.json")
nlp_metrics = load_json("nlp_metrics.json")
graph_metrics = load_json("graph_intelligence_summary.json")
fusion = load_csv(EVAL_DIR / "evidence_fusion_report.csv")
txns = load_csv(SYNTH_DIR / "transactions.csv")
bens = load_csv(SYNTH_DIR / "beneficiaries.csv")


# ---------------------------------------------------------------- helpers
def inr(x):
    """Rupees with Indian digit grouping, e.g. 1,25,000.00"""
    x = float(x)
    whole, dec = f"{abs(x):.2f}".split(".")
    if len(whole) > 3:
        head_, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head_) > 2:
            parts.insert(0, head_[-2:])
            head_ = head_[:-2]
        if head_:
            parts.insert(0, head_)
        whole = ",".join(parts + [tail])
    return f"₹{'-' if x < 0 else ''}{whole}.{dec}"


def level(score):
    return ("hi", "High") if score >= 70 else ("mid", "Medium") if score >= 40 else ("lo", "Low")


DECISION = {"APPROVE": ("lo", "Approve"), "MANUAL_REVIEW": ("mid", "Review"),
            "BLOCK_TRANSACTION": ("hi", "Block")}


def pill(cls, text):
    return f'<span class="pill {cls}">{H.escape(str(text))}</span>'


def head(title, sub=""):
    st.markdown(f'<div class="fs head"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def section(text):
    st.markdown(f'<h3 class="fs sec">{text}</h3>', unsafe_allow_html=True)


def fine(text):
    st.markdown(f'<div class="fs fine">{text}</div>', unsafe_allow_html=True)


def strip(items):
    cells = "".join(f'<div class="cell"><div class="k">{k}</div><div class="v">{v}</div>'
                    f'<div class="n">{n}</div></div>' for k, v, n in items)
    st.markdown(f'<div class="fs strip" style="grid-template-columns:repeat({len(items)},1fr)">{cells}</div>',
                unsafe_allow_html=True)


def ledger(headers, rows, right=()):
    th = "".join(f'<th class="{"r" if i in right else ""}">{h}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"r" if i in right else ""}">{c}</td>'
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    st.markdown(f'<table class="fs ledger"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>',
                unsafe_allow_html=True)


def ruler(score, label=""):
    s = max(0, min(100, float(score)))
    lab = f'<div class="rl">{label}</div>' if label else ""
    st.markdown(
        f'<div class="fs ruler">{lab}<div class="track">'
        '<span class="z lo" style="width:40%"></span><span class="z mid" style="width:30%"></span>'
        '<span class="z hi" style="width:30%"></span>'
        f'<div class="marker" style="left:{s}%"></div></div>'
        '<div class="ticks"><span style="left:0">0</span>'
        '<span style="left:40%;transform:translateX(-50%)">Review from 40</span>'
        '<span style="left:70%;transform:translateX(-50%)">High from 70</span>'
        '<span style="right:0">100</span></div></div>', unsafe_allow_html=True)


def result_band(score, title, detail):
    cls, _ = level(score)
    st.markdown(f'<div class="fs result"><div class="bignum {cls}">{score:.0f}</div>'
                f'<div class="verdict"><div class="t">{title}</div><div class="d">{detail}</div></div></div>',
                unsafe_allow_html=True)


def style_fig(fig, h=320):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Schibsted Grotesk, sans-serif", color=T["text"], size=13),
                      margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=12)))
    fig.update_xaxes(gridcolor=T["line"], zeroline=False, linecolor=T["line"])
    fig.update_yaxes(gridcolor=T["line"], zeroline=False, linecolor=T["line"])
    return fig


# ------------------------------------------------------ message rule engine
URGENCY = ["urgent", "immediately", "immediate", "blocked", "suspended", "expire", "within 24", "today",
           "last chance", "udane", "ippove", "block aagividum", "முடக்க", "உடனடியாக", "तुरंत", "अभी",
           "बंद हो", "ब्लॉक", "jaldi", "turant"]
CREDENTIAL = ["otp", "pin", "password", "cvv", "card number", "upi pin", "ஓடிபி", "கடவுச்சொல்",
              "ओटीपी", "पिन", "पासवर्ड"]
LURE = ["reward", "cashback", "prize", "lottery", "refund", "kyc", "gift", "winner", "பரிசு", "इनाम",
        "रिफंड", "केवाईसी"]
COLLECT = ["collect request", "approve the request", "accept the request", "request money",
           "enter your pin to receive", "to receive money"]
SAFE_CTX = ["never share", "do not share", "don't share", "not share your otp"]
URL_HINTS = ["http", "bit.ly", "tinyurl", "verify-", ".xyz", ".top", "www."]

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


def ml_scores(texts):
    model = load_classifier()
    if model is None:
        return [None] * len(texts)
    try:
        return list(model.predict_proba(list(texts))[:, 1] * 100)
    except Exception:
        return [None] * len(texts)


def combine(ml, rs):
    return round(rs if ml is None else ML_WEIGHT * ml + RULE_WEIGHT * rs, 1)


EXAMPLES = {
    "English scam": "Your bank account will be blocked today. Verify your account immediately at "
                    "http://sbi-verify.xyz and enter your OTP.",
    "Tamil scam": "ungal account udane block aagividum. KYC update seiya inge click pannunga "
                  "http://bit.ly/kyc-now, OTP anuppunga.",
    "Hindi scam": "आपका खाता तुरंत बंद हो जाएगा। केवाईसी अपडेट के लिए http://verify-bank.top पर क्लिक "
                  "करें और OTP भेजें।",
    "Genuine bank alert": "Rs 450.00 debited from A/c XX1234 on 09-Nov to Swiggy via UPI. "
                          "Never share your OTP with anyone.",
}

# ================================================================ OVERVIEW
if page == "Overview":
    head("Overview", "What the system screened, what it flagged, and where to look first.")
    if len(fusion):
        n = len(fusion)
        blocked = int((fusion["system_decision"] == "BLOCK_TRANSACTION").sum())
        review = int((fusion["system_decision"] == "MANUAL_REVIEW").sum())
        st.markdown(f'<p class="fs lead">{n} transactions screened. <b>{blocked}</b> should be blocked '
                    f'and <b>{review}</b> need a human look.</p>', unsafe_allow_html=True)

    n_tx = len(txns) if len(txns) else graph_metrics.get("total_edges", 0)
    n_fraud = int(txns["is_fraud"].sum()) if "is_fraud" in txns else 0
    strip([("Payments in ledger", f"{n_tx:,}", "synthetic UPI data"),
           ("Labelled fraud cases", f"{n_fraud:,}", "three injected scenarios"),
           ("Possible mule accounts", graph_metrics.get("potential_mule_hubs_identified", "-"),
            "beneficiaries with many senders"),
           ("Entities in network", f"{graph_metrics.get('total_nodes', 0):,}", "accounts + beneficiaries")])

    section("Three kinds of evidence, one score")
    st.markdown('<div class="fs cols3"><div><b>Behaviour</b><span>Unusual amount, speed, timing and how '
                'much of the balance leaves the account.</span></div><div><b>Message</b><span>Scam wording '
                'in English, Tamil and Hindi: urgency, OTP requests, look-alike links.</span></div>'
                '<div><b>Network</b><span>Beneficiaries that many unrelated senders pay, and accounts that '
                'share a device.</span></div></div>', unsafe_allow_html=True)

    if len(fusion):
        section("Review queue")
        c1, c2 = st.columns([1, 3])
        floor = c1.slider("Minimum risk score", 0, 100, 0)
        queue = fusion[fusion["unified_risk_score"] * 100 >= floor] \
            .sort_values("unified_risk_score", ascending=False)
        rows = []
        for _, r in queue.head(8).iterrows():
            sc = r["unified_risk_score"] * 100
            c, _ = level(sc)
            dc, dt = DECISION.get(r["system_decision"], ("na", r["system_decision"]))
            rows.append([f'<span class="id">{H.escape(str(r["transaction_id"]))}</span>', inr(r["amount"]),
                         pill(dc, dt), f'<span class="mini"><i class="{c}" style="width:{sc:.0f}%"></i></span>'
                         f'<b>{sc:.0f}</b>'])
        if rows:
            ledger(["Transaction", "Amount", "Decision", "Risk score"], rows, right=(1,))
            fine(f"Showing {len(rows)} of {len(queue)} transactions at or above {floor}.")
            st.download_button("Download queue as CSV", queue.to_csv(index=False), "review_queue.csv", "text/csv")
        else:
            st.info("No transactions at that score. Lower the minimum.")

        section("How the decisions split")
        counts = fusion["system_decision"].value_counts()
        colmap = {"APPROVE": T["lo"], "MANUAL_REVIEW": T["mid"], "BLOCK_TRANSACTION": T["hi"]}
        fig = go.Figure()
        for k in ["APPROVE", "MANUAL_REVIEW", "BLOCK_TRANSACTION"]:
            if k in counts:
                fig.add_trace(go.Bar(y=[""], x=[int(counts[k])], orientation="h", name=DECISION[k][1],
                                     marker_color=colmap[k], text=f"{DECISION[k][1]} {int(counts[k])}",
                                     textposition="inside", insidetextanchor="start",
                                     textfont=dict(color="#fff", size=13)))
        fig.update_layout(barmode="stack", showlegend=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        st.plotly_chart(style_fig(fig, 90))

        section("Where scores fall")
        h = go.Figure(go.Histogram(x=fusion["unified_risk_score"] * 100, nbinsx=20, marker_color=T["acc"]))
        for x0, x1, k in [(0, 40, "z-lo"), (40, 70, "z-mid"), (70, 100, "z-hi")]:
            h.add_vrect(x0=x0, x1=x1, fillcolor=T[k], opacity=.55, layer="below", line_width=0)
        h.update_layout(xaxis_title="Risk score", yaxis_title="Transactions", bargap=.06)
        st.plotly_chart(style_fig(h, 260))
    else:
        st.info("No decision report yet. Run intelligence/evidence_fusion.py to create one.")

# ========================================================= CHECK A MESSAGE
elif page == "Check a message":
    head("Check a message", "Paste an SMS or payment request, or scan a whole file of messages.")
    tab1, tab2 = st.tabs(["Single message", "Batch scan"])

    with tab1:
        if "msg" not in st.session_state:
            st.session_state.msg = EXAMPLES["English scam"]
        cols = st.columns(len(EXAMPLES))
        for col, (name, text) in zip(cols, EXAMPLES.items()):
            if col.button(name, use_container_width=True):
                st.session_state.msg = text
        message = st.text_area("Message", key="msg", height=110, label_visibility="collapsed")
        if message.strip():
            found = find_signals(message)
            rs = rule_score(found)
            ml = ml_scores([message])[0]
            score = combine(ml, rs)
            cls, _ = level(score)
            say = {"hi": ("Likely a scam", "Do not tap the link or share any code. Check with your bank "
                                           "using its official app."),
                   "mid": ("Looks suspicious", "Some scam signals are present. Verify through your bank "
                                               "before acting."),
                   "lo": ("No strong scam signals", "Nothing here matches common scam patterns.")}[cls]
            result_band(score, *say)
            ruler(score)
            section("What the system noticed")
            rows = []
            for sname, why, _ in SIGNALS:
                on = found[sname]
                status = (pill("lo", "Present") if on else pill("na", "Absent")) if sname == "Safety warning" \
                    else (pill("hi", "Found") if on else pill("na", "Not found"))
                rows.append([f"<b>{sname}</b>", why, status])
            ledger(["Signal", "Why it matters", "Result"], rows)
            fine(f'Text classifier: <b>{"unavailable" if ml is None else f"{ml:.0f} / 100"}</b> &nbsp;|&nbsp; '
                 f'Rule engine: <b>{rs} / 100</b>. The score is a risk indicator, not a calibrated probability.')

    with tab2:
        up = st.file_uploader("Upload a CSV with one message per row", type=["csv"])
        if up is not None:
            try:
                df = pd.read_csv(up)
            except Exception:
                df = pd.DataFrame()
            if df.empty:
                st.info("That file could not be read as a CSV with rows.")
            else:
                text_cols = [c for c in df.columns if df[c].dtype == object]
                pref = [c for c in text_cols if c.lower() in ("sender_text", "message", "text", "message_text")]
                options = pref + [c for c in text_cols if c not in pref] or list(df.columns)
                col = st.selectbox("Column that holds the message text", options)
                df = df.head(500)
                texts = df[col].astype(str).tolist()
                mls = ml_scores(texts)
                scores = [combine(m, rule_score(find_signals(t))) for m, t in zip(mls, texts)]
                out = df.copy()
                out["risk_score"] = scores
                out["risk_level"] = [level(s)[1] for s in scores]
                strip([("Messages scanned", len(out), "first 500 rows at most"),
                       ("High risk", int((out["risk_level"] == "High").sum()), "score 70 and above"),
                       ("Medium risk", int((out["risk_level"] == "Medium").sum()), "score 40 to 69")])
                section("Highest risk first")
                top = out.sort_values("risk_score", ascending=False).head(10)
                ledger(["Message", "Risk"],
                       [[H.escape(str(r[col]))[:90], pill(level(r["risk_score"])[0], f'{r["risk_level"]} {r["risk_score"]:.0f}')]
                        for _, r in top.iterrows()])
                st.download_button("Download full results", out.to_csv(index=False), "message_scan.csv", "text/csv")

# ================================================ INVESTIGATE A TRANSACTION
elif page == "Investigate a transaction":
    head("Investigate a transaction", "See the decision, the evidence behind it, and the payment details.")
    if fusion.empty:
        st.info("No decision report yet. Run intelligence/evidence_fusion.py to create one.")
    else:
        a, b = st.columns([1, 2])
        present = [DECISION[k][1] for k in DECISION if k in set(fusion["system_decision"])]
        flt = a.selectbox("Show", ["All decisions"] + present)
        inv = {v[1]: k for k, v in DECISION.items()}
        view = fusion if flt == "All decisions" else fusion[fusion["system_decision"] == inv[flt]]
        tid = b.selectbox("Transaction", view["transaction_id"].tolist())
        r = view[view["transaction_id"] == tid].iloc[0]
        score = round(float(r["unified_risk_score"]) * 100, 1)
        dc, dt = DECISION.get(r["system_decision"], ("na", r["system_decision"]))

        st.markdown(f'<div class="fs" style="display:flex;gap:14px;align-items:center;margin:14px 0 4px 0;">'
                    f'<span style="font-size:1.5rem;font-weight:700;letter-spacing:-.4px">{H.escape(str(tid))}</span>'
                    f'{pill(dc, dt)}<span style="color:var(--mute)">{inr(r["amount"])}</span></div>',
                    unsafe_allow_html=True)
        result_band(score, f"{level(score)[1]} risk", "Unified score from behaviour, message and network evidence.")
        ruler(score)

        section("Evidence behind the score")
        ev = [("Behaviour", r["transaction_risk_score"]), ("Message", r["nlp_risk_score"]),
              ("Network", r["graph_risk_score"])]
        st.markdown('<div class="fs panel">' + "".join(
            f'<div class="ev"><span>{k}</span><div class="bar"><i style="width:{v*100:.0f}%"></i></div>'
            f'<span class="num">{v*100:.0f}</span></div>' for k, v in ev) + '</div>', unsafe_allow_html=True)

        detail_lines = []
        if len(txns) and "transaction_id" in txns:
            t = txns[txns["transaction_id"] == tid]
            if len(t):
                t = t.iloc[0]
                section("Payment details")
                fields = [("Sender", "sender_id"), ("Beneficiary", "receiver_id"), ("Time", "timestamp"),
                          ("Type", "transaction_type"), ("Channel", "payment_channel"), ("Device", "device_id"),
                          ("Location", "location_id"), ("New beneficiary", "is_new_beneficiary"),
                          ("Sender balance", "sender_balance")]
                cells = ""
                for lab, c in fields:
                    if c in t:
                        v = inr(t[c]) if c == "sender_balance" else \
                            (("Yes" if t[c] == 1 else "No") if c == "is_new_beneficiary" else t[c])
                        cells += f'<div><div class="dk">{lab}</div><div class="dv">{H.escape(str(v))}</div></div>'
                        detail_lines.append(f"{lab}: {v}")
                st.markdown(f'<div class="fs deflist">{cells}</div>', unsafe_allow_html=True)
                fine(f'Synthetic ground-truth label: {"fraud" if r.get("true_fraud_label", 0) == 1 else "legitimate"}. '
                     'Shown for evaluation only.')

        section("Case notes")
        notes = st.text_area("Notes", key=f"notes_{tid}", height=90, label_visibility="collapsed",
                             placeholder="What did you check? What should happen next?")
        report = "\n".join(
            ["FINSIGHT AI - CASE REPORT", "=" * 28, f"Transaction: {tid}", f"Decision: {dt}",
             f"Risk score: {score:.0f} / 100 ({level(score)[1]})", f"Amount: {inr(r['amount'])}", "",
             "Evidence (0-100)"] + [f"  {k}: {v*100:.0f}" for k, v in ev] + ["", "Payment details"]
            + [f"  {d}" for d in detail_lines] + ["", "Investigator notes", notes or "(none)", "",
             "Prototype on synthetic data. Scores are risk indicators, not proof of fraud."])
        st.download_button("Download case report", report, f"case_{tid}.txt", "text/plain")

# ====================================================== COUNTERFACTUAL LAB
elif page == "Counterfactual lab":
    head("Counterfactual lab", "Change one thing about a payment and see how its behaviour risk responds. "
                               "The model is re-run on the changed payment; nothing is subtracted by hand.")
    B = load_behaviour()
    if B is None:
        st.info("Needs models/fraud_detector.joblib and data/processed/transaction_features.csv. "
                "Run Day 2 and Day 3 scripts first.")
    else:
        feats, cols, srt = B["feats"], B["cols"], B["sorted"]
        pct = np.searchsorted(srt, B["anomaly"]) / len(srt) * 100
        order = np.argsort(-pct)[:60]
        label = {int(i): f'{feats.iloc[i]["transaction_id"]}  ·  risk {pct[i]:.0f}' for i in order}
        idx = st.selectbox("Payment (highest behaviour risk first)", list(label), format_func=lambda i: label[i])
        base = feats.iloc[idx].to_dict()

        def score_row(row):
            x = pd.DataFrame([{c: row.get(c, 0) for c in cols}])
            a = float(-B["mdl"].decision_function(B["imp"].transform(x))[0])
            return float(np.searchsorted(srt, a) / len(srt) * 100)

        st.markdown('<h3 class="fs sec">What if...</h3>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        known = c1.toggle("Beneficiary is already known", value=False)
        day = c2.toggle("Paid during the day", value=False)
        mult = c3.slider("Amount, as share of original", 0.1, 1.5, 1.0, 0.05)
        vel = c4.slider("Payments in the last hour", 1, 20, int(max(1, base.get("txn_velocity_1h", 1))))

        new = dict(base)
        changes = []
        if known:
            for k in ("is_new_beneficiary", "beneficiary_age_risk"):
                if k in new and new[k] != 0:
                    changes.append((k, new[k], 0)); new[k] = 0
        if day and "night_transaction_flag" in new and new["night_transaction_flag"] != 0:
            changes.append(("night_transaction_flag", new["night_transaction_flag"], 0)); new["night_transaction_flag"] = 0
        if abs(mult - 1.0) > 1e-9:
            for k in ("amount", "balance_depletion_ratio", "amount_vs_account_average"):
                if k in new:
                    changes.append((k, new[k], new[k] * mult)); new[k] = new[k] * mult
            if "amount" in new:
                if "amount_log" in new:
                    changes.append(("amount_log", new["amount_log"], math.log1p(new["amount"])))
                    new["amount_log"] = math.log1p(new["amount"])
                if "high_amount_flag" in new:
                    flag = int(new["amount"] > 10000)
                    if flag != new["high_amount_flag"]:
                        changes.append(("high_amount_flag", new["high_amount_flag"], flag))
                    new["high_amount_flag"] = flag
        if "txn_velocity_1h" in new and vel != new["txn_velocity_1h"]:
            changes.append(("txn_velocity_1h", new["txn_velocity_1h"], vel)); new["txn_velocity_1h"] = vel
            if "txn_velocity_24h" in new and new["txn_velocity_24h"] < vel:
                changes.append(("txn_velocity_24h", new["txn_velocity_24h"], vel)); new["txn_velocity_24h"] = vel

        s0, s1 = score_row(base), score_row(new)
        d = s1 - s0
        dcol = "var(--lo)" if d < -0.5 else "var(--hi)" if d > 0.5 else "var(--mute)"
        c0, c1_ = level(s0)[0], level(s1)[0]
        st.markdown(
            f'<div class="fs cmp"><div><div class="lbl">Original</div><div class="bignum {c0}">{s0:.0f}</div></div>'
            f'<div class="sep"></div><div><div class="lbl">With your changes</div><div class="bignum {c1_}">{s1:.0f}</div></div>'
            f'<div class="delta" style="color:{dcol}">{d:+.0f} points</div></div>', unsafe_allow_html=True)
        ruler(s0, "Original")
        ruler(s1, "With your changes")

        if changes:
            section("What changed in the model's inputs")
            ledger(["Input", "Before", "After"],
                   [[f"<b>{k}</b>", f"{b:,.3g}", f"{a:,.3g}"] for k, b, a in changes], right=(1, 2))
        else:
            fine("Adjust a control above to simulate a change.")
        fine("Behaviour risk is the payment's anomaly rank among all 5,000 payments (100 means more unusual than "
             "every other payment). Message and network evidence are unchanged in this view.")

# ===================================================== SPENDING INSIGHTS
elif page == "Spending insights":
    head("Spending insights", "How one account spends, who it pays, and how much goes to people it has just met.")
    if txns.empty:
        st.info("transactions.csv not found in data/synthetic.")
    else:
        t = txns.copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"])
        top = t["sender_id"].value_counts().head(30)
        acct = st.selectbox("Account", top.index.tolist(), format_func=lambda x: f"{x}  ·  {top[x]} payments")
        s = t[t["sender_id"] == acct].copy()
        if len(bens) and "beneficiary_id" in bens:
            s = s.merge(bens[["beneficiary_id", "category"]], left_on="receiver_id", right_on="beneficiary_id", how="left")
        else:
            s["category"] = "unknown"
        s["category"] = s["category"].fillna("unknown")
        new_share = s.loc[s["is_new_beneficiary"] == 1, "amount"].sum() / max(s["amount"].sum(), 1) * 100 \
            if "is_new_beneficiary" in s else 0
        strip([("Total spent", inr(s["amount"].sum()), f'{len(s)} payments'),
               ("Average payment", inr(s["amount"].mean()), "per payment"),
               ("Largest payment", inr(s["amount"].max()), "single payment"),
               ("Paid to new beneficiaries", f"{new_share:.0f}%", "share of total spend")])

        section("Spend by month")
        m = s.groupby(s["timestamp"].dt.to_period("M").astype(str))["amount"].sum().reset_index()
        fig = go.Figure(go.Bar(x=m["timestamp"], y=m["amount"], marker_color=T["acc"]))
        fig.update_layout(bargap=.45, yaxis_title="Rupees")
        st.plotly_chart(style_fig(fig, 280))

        left, right = st.columns(2)
        with left:
            section("Where the money goes")
            cat = s.groupby("category")["amount"].sum().sort_values()
            fc = go.Figure(go.Bar(x=cat.values, y=[c.replace("_", " ").capitalize() for c in cat.index],
                                  orientation="h", marker_color=T["acc"]))
            st.plotly_chart(style_fig(fc, 260))
        with right:
            section("Top beneficiaries")
            tb = s.groupby("receiver_id")["amount"].agg(["sum", "count"]).sort_values("sum", ascending=False).head(5)
            ledger(["Beneficiary", "Paid", "Times"],
                   [[f'<span class="id">{i}</span>', inr(r["sum"]), int(r["count"])] for i, r in tb.iterrows()],
                   right=(1, 2))
        fine("Categories come from the synthetic beneficiary list. A real build would read them from UPI "
             "merchant codes.")

# ================================================== BENEFICIARY NETWORK
elif page == "Beneficiary network":
    head("Beneficiary network", "Who pays a beneficiary. Red lines are payments labelled as fraud.")
    if txns.empty:
        st.info("transactions.csv not found in data/synthetic.")
    else:
        top = txns["receiver_id"].value_counts().head(25)
        pick = st.selectbox("Beneficiary", top.index.tolist(), format_func=lambda x: f"{x}  ·  {top[x]} payments")
        sub = txns[txns["receiver_id"] == pick]
        senders = sub["sender_id"].unique().tolist()
        pos = {pick: (0.0, 0.0)}
        for i, sd in enumerate(senders):
            ang = 2 * math.pi * i / max(len(senders), 1)
            pos[sd] = (math.cos(ang), math.sin(ang))
        fig = go.Figure()
        for _, r in sub.iterrows():
            x0, y0 = pos[r["sender_id"]]
            bad = r.get("is_fraud", 0) == 1
            fig.add_trace(go.Scatter(x=[x0, 0], y=[y0, 0], mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(color=T["hi"] if bad else T["line"], width=2.6 if bad else 1.4)))
        fig.add_trace(go.Scatter(x=[pos[x][0] for x in senders], y=[pos[x][1] for x in senders],
                                 mode="markers+text", text=senders, textposition="top center",
                                 textfont=dict(size=11, color=T["mute"]), name="Sender",
                                 marker=dict(size=13, color=T["node"])))
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", text=[pick], textposition="bottom center",
                                 textfont=dict(color=T["text"]), name="Beneficiary",
                                 marker=dict(size=24, color=T["acc"])))
        fig.update_xaxes(visible=False, range=[-1.5, 1.5])
        fig.update_yaxes(visible=False, range=[-1.4, 1.4])
        st.plotly_chart(style_fig(fig, 480))
        strip([("Unique senders", len(senders), "distinct accounts"),
               ("Total received", inr(sub["amount"].sum()), "across all payments"),
               ("Fraud-labelled payments", int(sub["is_fraud"].sum()) if "is_fraud" in sub else 0, "synthetic labels")])
        fine("Many senders paying one beneficiary is a reason to look closer. It is not proof of fraud.")

# ======================================================== MODEL RESULTS
elif page == "Model results":
    head("Model results", "Held-out tests on synthetic data. Treat these as indicative, not real-world accuracy.")
    iso, rule = tx_metrics.get("isolation_forest", {}), tx_metrics.get("rule_baseline", {})
    if iso and rule:
        st.markdown(
            f'<div class="fs note">Simple rules catch <b>{rule.get("recall", 0)*100:.0f}%</b> of fraud but flag '
            f'<b>{rule.get("false_positive_rate", 0)*100:.0f}%</b> of normal payments. The Isolation Forest catches '
            f'<b>{iso.get("recall", 0)*100:.0f}%</b> and flags <b>{iso.get("false_positive_rate", 0)*100:.0f}%</b> '
            f'of normal payments. It trades some detection for far fewer false alarms.</div>', unsafe_allow_html=True)
        keys = ["precision", "recall", "f1_score", "pr_auc", "false_positive_rate"]
        names = ["Precision", "Recall", "F1", "PR-AUC", "False-positive rate"]
        section("Transaction model against a rule baseline")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Isolation Forest", x=names, y=[iso.get(k, 0) for k in keys], marker_color=T["acc"]))
        fig.add_trace(go.Bar(name="Rule baseline", x=names, y=[rule.get(k, 0) for k in keys], marker_color=T["grey"]))
        fig.update_layout(barmode="group", bargap=.35)
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(style_fig(fig, 340))
        ledger(["Metric", "Isolation Forest", "Rule baseline"],
               [[n, f"{iso.get(k, 0):.3f}", f"{rule.get(k, 0):.3f}"] for k, n in zip(keys, names)], right=(1, 2))
    else:
        st.info("No metrics found. Run models/train_anomaly_model.py first.")
    if nlp_metrics:
        section("Message classifier")
        strip([(k.replace("_", " ").capitalize(), f"{nlp_metrics.get(k, 0):.2f}", "held-out split")
               for k in ["precision", "recall", "f1_score", "pr_auc"]])
        fine("Scores near 1.00 on templated synthetic text usually mean the task is too easy. They will drop on "
             "realistic, varied messages.")