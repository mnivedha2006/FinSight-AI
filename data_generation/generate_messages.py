"""Regenerate data/synthetic/messages.csv with realistic, varied, multilingual SMS.

Why: the old file had one sentence per category, so a classifier could read the label
straight from the text. This version has ~65 message families in English, Tamil (romanised
and script) and Hindi (script and romanised), genuine messages that look scary (OTP, urgent,
links), subtle scams with few keywords, and typo/leetspeak noise.

Run from the project root:   python data_generation/generate_messages.py
"""
import random
import re
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from intelligence.message_rules import CREDENTIAL, URGENCY, URL_HINTS  # noqa: E402

SEED = 7
N_MESSAGES = 5000
STANDALONE_SCAM_RATE = 0.30
OUT = BASE_DIR / "data" / "synthetic" / "messages.csv"
TX_PATH = BASE_DIR / "data" / "synthetic" / "transactions.csv"

rng = random.Random(SEED)

BANKS = ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak", "Canara Bank", "Indian Bank", "PNB", "IOB",
         "Union Bank"]
MERCHANTS = ["Swiggy", "Zomato", "Amazon", "Flipkart", "BigBasket", "IRCTC", "Ola", "Uber", "Jio", "Airtel",
             "TANGEDCO", "BSNL", "Myntra", "Paytm"]
NAMES = ["Ravi", "Priya", "Karthik", "Meena", "Arjun", "Divya", "Suresh", "Lakshmi", "Rahul", "Anita", "Vijay",
         "Sneha", "Manoj", "Kavya", "Imran", "Pooja"]
BAD_WORDS = ["verify", "kyc", "secure", "update", "login", "claim", "refund", "reward"]
BAD_TLDS = ["xyz", "top", "info", "co", "in", "online", "site"]
UPI_SUFFIX = ["ybl", "paytm", "okaxis", "oksbi", "upi"]

# (template_id, language, category, text); category "Normal" = genuine.
SCAM = [
    ("s_en_kyc", "en", "phishing", "{bank}: Your account will be blocked today. Update KYC immediately: {url_bad}"),
    ("s_en_otp", "en", "impersonation", "Dear customer, we noticed a suspicious login. Share the OTP sent to your mobile to secure your account."),
    ("s_en_collect", "en", "collect scam", "{name} has sent a payment request of Rs {amt}. Enter your UPI PIN to receive the money."),
    ("s_en_cashback", "en", "phishing", "Congratulations! You have won Rs {amt} cashback. Claim now: {url_bad}"),
    ("s_en_refund", "en", "phishing", "Your refund of Rs {amt} is ready. Click {url_bad} and enter your card number and CVV to receive it."),
    ("s_en_power", "en", "impersonation", "Dear consumer, your electricity connection will be disconnected tonight 9:30 PM as last month bill is not updated. Call {phone} immediately."),
    ("s_en_parcel", "en", "phishing", "Your parcel is on hold due to an incomplete address. Pay Rs 25 redelivery fee: {url_bad}"),
    ("s_en_loan", "en", "phishing", "Pre-approved personal loan of Rs {amt}. No documents needed. Apply now {url_bad}. Processing fee Rs 999."),
    ("s_en_job", "en", "impersonation", "Part time job, earn Rs 3000 daily from home. Pay refundable registration Rs 500 to {upi} to start."),
    ("s_en_wrong", "en", "collect scam", "Hi, I sent Rs {amt} to your number by mistake. Please return it to {upi}, it is urgent."),
    ("s_en_sim", "en", "phishing", "Your SIM will be deactivated in 2 hours. Verify your Aadhaar now: {url_bad}"),
    ("s_en_tax", "en", "phishing", "Income tax refund of Rs {amt} approved. Verify your bank details at {url_bad} to credit the amount."),
    ("s_en_card", "en", "impersonation", "Your {bank} card is suspended. Call {phone} and give your PIN to reactivate it."),
    ("s_en_subtle", "en", "impersonation", "Sir this is {name} from {bank}. Your KYC is pending. Please send a screenshot of your debit card on WhatsApp {phone}."),
    ("s_en_lottery", "en", "phishing", "You are the lucky winner of Rs {amt}. To release the amount, pay Rs 1500 tax to {upi}."),
    ("s_ta_kyc", "ta_rom", "phishing", "{bank}: ungal account indru block aagividum. KYC update seiya udane click pannunga {url_bad}"),
    ("s_ta_card", "ta_rom", "impersonation", "Sir ungal card suspend aagiduchu. OTP anuppunga udane, illana account close aagum."),
    ("s_ta_collect", "ta_rom", "collect scam", "{name} ungalukku Rs {amt} request anuppiyirukkaar. Receive panna UPI PIN podunga."),
    ("s_ta_prize", "ta_rom", "phishing", "Vazhthukkal! Ungalukku Rs {amt} cashback parisu kedaichirukku. Claim panna {url_bad} click pannunga."),
    ("s_ta_power", "ta_rom", "impersonation", "TNEB: Ungal min inaippu indru ravu 9.30 manikku thundikkapadum. Udane {phone} ku call pannunga."),
    ("s_ta_wrong", "ta_rom", "collect scam", "Anna, Rs {amt} thappa ungal number-ku anuppitten. Please {upi} ku thirumba anuppunga, urgent."),
    ("s_tn_kyc", "ta", "phishing", "{bank}: உங்கள் கணக்கு இன்று முடக்கப்படும். உடனடியாக KYC புதுப்பிக்க {url_bad} கிளிக் செய்யவும்."),
    ("s_tn_card", "ta", "impersonation", "உங்கள் ATM கார்டு நிறுத்தப்பட்டது. ஓடிபி மற்றும் பின் நம்பரை உடனடியாக அனுப்பவும்."),
    ("s_tn_prize", "ta", "phishing", "வாழ்த்துக்கள்! உங்களுக்கு ரூ {amt} பரிசு கிடைத்துள்ளது. பெற {url_bad} கிளிக் செய்யவும்."),
    ("s_tn_collect", "ta", "collect scam", "{name} உங்களுக்கு ரூ {amt} பணம் கோரிக்கை அனுப்பியுள்ளார். பெற UPI பின் உள்ளிடவும்."),
    ("s_hi_kyc", "hi", "phishing", "{bank}: आपका खाता आज बंद हो जाएगा। तुरंत केवाईसी अपडेट करें: {url_bad}"),
    ("s_hi_card", "hi", "impersonation", "आपका कार्ड ब्लॉक हो गया है। अभी OTP और पिन भेजें वरना खाता बंद हो जाएगा।"),
    ("s_hi_prize", "hi", "phishing", "बधाई हो! आपने Rs {amt} का इनाम जीता है। क्लेम करने के लिए {url_bad} पर क्लिक करें।"),
    ("s_hi_collect", "hi", "collect scam", "{name} ने आपको Rs {amt} का पेमेंट रिक्वेस्ट भेजा है। पैसे पाने के लिए अपना UPI पिन डालें।"),
    ("s_hi_power", "hi", "impersonation", "आपका बिजली कनेक्शन आज रात 9:30 बजे कट जाएगा। तुरंत {phone} पर कॉल करें।"),
    ("s_hg_kyc", "hi_rom", "phishing", "Dear customer aapka {bank} account block ho jayega. Abhi KYC update karein: {url_bad}"),
    ("s_hg_refund", "hi_rom", "phishing", "Aapke account me Rs {amt} refund aaya hai. Receive karne ke liye OTP bhejiye."),
    ("s_hg_wrong", "hi_rom", "collect scam", "Bhai galti se Rs {amt} aapke number pe chala gaya. Please {upi} pe wapas bhej do, jaldi."),
    ("s_hg_prize", "hi_rom", "phishing", "Congratulations aapne Rs {amt} cashback jeeta hai. Claim karne ke liye {url_bad} par click karein."),
]
LEGIT = [
    ("l_en_debit", "en", "Normal", "Rs {amt} debited from A/c XX{last4} on {date} to {merchant} via UPI. Ref {ref}. If not you, call 1800-1234 immediately. -{bank}"),
    ("l_en_credit", "en", "Normal", "Rs {amt} credited to A/c XX{last4} on {date} by {name}. UPI Ref {ref}. -{bank}"),
    ("l_en_otp_txn", "en", "Normal", "{otp} is your OTP for transaction of Rs {amt} at {merchant}. Valid for 10 minutes. Do not share this OTP with anyone. -{bank}"),
    ("l_en_otp_login", "en", "Normal", "Your one time password for login is {otp}. Never share your OTP, PIN or password with anyone, including bank staff."),
    ("l_en_bill", "en", "Normal", "Dear customer, your {merchant} bill of Rs {amt} is due on {date}. Pay securely at {url_good}."),
    ("l_en_emi", "en", "Normal", "Reminder: EMI of Rs {amt} will be auto-debited from your account on {date}. Please maintain sufficient balance. -{bank}"),
    ("l_en_salary", "en", "Normal", "Salary of Rs {amt} credited to your account XX{last4} on {date}. Available balance Rs {bal}. -{bank}"),
    ("l_en_collect", "en", "Normal", "{name} has requested Rs {amt} on UPI. Open your UPI app to approve or decline. Never share your PIN to receive money."),
    ("l_en_kyc", "en", "Normal", "Dear customer, your KYC update is due by {date}. Visit your nearest branch or use the official {bank} app. We never ask for OTP, PIN or card details by call or link."),
    ("l_en_promo", "en", "Normal", "Flat 20% cashback on your next {merchant} order. Use code SAVE20. Offer valid till {date}. T&C apply. {url_good}"),
    ("l_en_delivery", "en", "Normal", "Your {merchant} order is out for delivery. Share OTP {otp} with the delivery partner only at the time of delivery."),
    ("l_en_cardblock", "en", "Normal", "Your {bank} debit card ending {last4} has been blocked as per your request. To unblock, visit a branch or call the number on your card."),
    ("l_en_fraudalert", "en", "Normal", "Alert: Rs {amt} spent on your card at {merchant}. If this was not you, block the card now at {url_good} or call 1800-1234."),
    ("l_en_statement", "en", "Normal", "Your {bank} credit card statement is generated. Total due Rs {amt} by {date}. Pay via the official app."),
    ("l_en_urgent", "en", "Normal", "Last date to pay your {merchant} bill is {date}. Pay now to avoid a late fee. Avoid payment through unknown links."),
    ("l_en_refund", "en", "Normal", "Refund of Rs {amt} for your {merchant} order has been initiated. It will reflect in your account in 5-7 working days."),
    ("l_en_recharge", "en", "Normal", "Recharge of Rs {amt} successful for {phone}. Thank you for using {merchant}."),
    ("l_en_friend", "en", "Normal", "Hi, can you send me Rs {amt} on UPI for dinner yesterday? My id is {upi}. Thanks!"),
    ("l_ta_credit", "ta_rom", "Normal", "Ungal {bank} account-il Rs {amt} credit aagiyullathu. Balance Rs {bal}. -{bank}"),
    ("l_ta_bill", "ta_rom", "Normal", "Ungal {merchant} bill Rs {amt} {date} annikku due. Official app moolam payment seiyyalaam."),
    ("l_ta_otp", "ta_rom", "Normal", "OTP {otp}. Idhai yaarukkum share panna vendaam. -{bank}"),
    ("l_tn_credit", "ta", "Normal", "உங்கள் கணக்கில் ரூ {amt} வரவு வைக்கப்பட்டது. இருப்பு ரூ {bal}. -{bank}"),
    ("l_tn_otp", "ta", "Normal", "உங்கள் OTP {otp}. இதை யாருடனும் பகிர வேண்டாம். -{bank}"),
    ("l_tn_bill", "ta", "Normal", "உங்கள் {merchant} பில் ரூ {amt} {date} அன்று செலுத்த வேண்டும். அதிகாரப்பூர்வ செயலியில் செலுத்தவும்."),
    ("l_hi_debit", "hi", "Normal", "आपके खाते से Rs {amt} {merchant} को UPI द्वारा डेबिट किया गया। रेफ {ref}। -{bank}"),
    ("l_hi_otp", "hi", "Normal", "आपका OTP {otp} है। इसे किसी के साथ साझा न करें। -{bank}"),
    ("l_hi_bill", "hi", "Normal", "आपका {merchant} बिल Rs {amt} {date} तक देय है। आधिकारिक ऐप से भुगतान करें।"),
    ("l_hi_credit", "hi", "Normal", "आपके खाते में Rs {amt} जमा हुए। शेष राशि Rs {bal}। -{bank}"),
    ("l_hg_debit", "hi_rom", "Normal", "Aapke account se Rs {amt} {merchant} ko debit hue. Agar aapne nahi kiya to turant 1800-1234 par call karein. -{bank}"),
    ("l_hg_otp", "hi_rom", "Normal", "Aapka OTP {otp} hai. Kisi ke saath share na karein. -{bank}"),
    ("l_hg_bill", "hi_rom", "Normal", "Aapka {merchant} bill Rs {amt} {date} tak due hai. Official app se pay karein."),
]
LANG_WEIGHT = {"en": 0.45, "ta_rom": 0.14, "ta": 0.09, "hi": 0.14, "hi_rom": 0.18}
PREFIX = ["", "", "", "Dear Customer, ", "Alert: ", "Hi, ", "Important: "]
SUFFIX = ["", "", "", " Thank you.", " Regards."]


def money():
    v = rng.choice([rng.randint(50, 999), rng.randint(1000, 9999), rng.randint(10000, 99999), rng.randint(100000, 500000)])
    return rng.choice([f"{v}", f"{v:,}", f"{v}.00", f"{v}.50"])


def values():
    bank = rng.choice(BANKS)
    slug = re.sub(r"[^a-z]", "", bank.lower())
    word = rng.choice(BAD_WORDS)
    tld = rng.choice(BAD_TLDS)
    code = "".join(rng.choice("abcdefghijkmnpqrstuvwxyz23456789") for _ in range(6))
    bad = rng.choice([f"http://{slug}-{word}.{tld}", f"https://{word}-{slug}.{tld}/{code}", f"http://bit.ly/{code}",
                      f"https://tinyurl.com/{code}", f"rb.gy/{code}", f"{slug}{word}.{tld}"])
    good = rng.choice([f"https://www.{slug}.com", f"{slug}.com/pay", "https://amzn.in/" + code,
                       "https://www.irctc.co.in", f"http://bit.ly/{code}"])  # genuine brands sometimes use shorteners too
    phone = rng.choice([f"{rng.choice('6789')}{rng.randint(10**8, 10**9 - 1)}",
                        f"1800-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"])
    return dict(
        bank=bank, merchant=rng.choice(MERCHANTS), name=rng.choice(NAMES), amt=money(), bal=money(),
        otp=str(rng.randint(100000, 999999)), last4=f"{rng.randint(0, 9999):04d}", ref=str(rng.randint(10**8, 10**9)),
        date=f"{rng.randint(1, 28):02d}-{rng.choice(['Nov', 'Dec', 'Jan'])}", url_bad=bad, url_good=good, phone=phone,
        upi=f"{rng.choice(NAMES).lower()}{rng.randint(1, 99)}@{rng.choice(UPI_SUFFIX)}")


LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}


def add_noise(text, scam):
    if rng.random() < (0.12 if scam else 0.01):  # leetspeak inside a few words
        words = text.split(" ")
        k = rng.randrange(len(words))
        words[k] = "".join(LEET[c] if c in LEET and rng.random() < .5 else c for c in words[k])
        text = " ".join(words)
    if rng.random() < (0.10 if scam else 0.04) and len(text) > 20:  # dropped character
        i = rng.randrange(5, len(text) - 5)
        text = text[:i] + text[i + 1:]
    if rng.random() < 0.15:
        text = text.lower()
    if rng.random() < 0.08:
        text = text.replace(" ", "  ", 1)
    return text


def pick(pool):
    lang = rng.choices(list(LANG_WEIGHT), weights=list(LANG_WEIGHT.values()))[0]
    return rng.choice([t for t in pool if t[1] == lang])


def make(scam):
    tid, lang, cat, tpl = pick(SCAM if scam else LEGIT)
    text = rng.choice(PREFIX) + tpl.format(**values()) + rng.choice(SUFFIX)
    return dict(sender_text=add_noise(text.strip(), scam), message_category=cat, language=lang, template_id=tid,
                label=int(scam))


def main():
    rows = []
    if TX_PATH.exists():
        tx = pd.read_csv(TX_PATH)
        # (probability the payment has a message, probability that message is a scam)
        plan = {"suspicious_collect": (0.90, 0.90), "new_beneficiary_anomaly": (0.60, 0.50),
                "rapid_transfer": (0.50, 0.30), "normal": (0.08, 0.04)}
        for _, t in tx.iterrows():
            p_msg, p_scam = plan.get(t.get("fraud_type", "normal"), plan["normal"])
            if rng.random() < p_msg:
                m = make(rng.random() < p_scam)
                m["transaction_id"] = t["transaction_id"]
                rows.append(m)
    else:
        print("transactions.csv not found; messages will not be linked to payments.")
    while len(rows) < N_MESSAGES:
        m = make(rng.random() < STANDALONE_SCAM_RATE)
        m["transaction_id"] = ""
        rows.append(m)
    rows = rows[:N_MESSAGES]
    rng.shuffle(rows)

    df = pd.DataFrame(rows)
    df.insert(0, "message_id", [f"MSG_{i:05d}" for i in range(1, len(df) + 1)])
    low = df["sender_text"].str.lower()
    # indicator columns come from the TEXT, never from the label
    df["urgency_score"] = low.apply(lambda s: round(min(1.0, sum(w.lower() in s for w in URGENCY) / 3), 2))
    df["suspicious_url"] = low.apply(lambda s: int(any(w in s for w in URL_HINTS)))
    df["credential_request"] = low.apply(lambda s: int(any(w.lower() in s for w in CREDENTIAL)))
    df = df[["message_id", "transaction_id", "sender_text", "message_category", "language", "template_id",
             "urgency_score", "suspicious_url", "credential_request", "label"]]
    df.to_csv(OUT, index=False, encoding="utf-8")

    print(f"Saved {len(df)} messages to {OUT}")
    print("Scam share:", f"{df['label'].mean():.1%}", "| linked to payments:", int((df['transaction_id'] != '').sum()))
    print("\nBy language and label:\n", pd.crosstab(df["language"], df["label"]))
    print("\nSample:")
    for _, r in df.sample(6, random_state=1).iterrows():
        print(f"  [{r['label']}] ({r['language']}) {r['sender_text'][:110]}")


if __name__ == "__main__":
    main()