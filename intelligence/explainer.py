"""Evidence-grounded explanations for a scored payment.

Two layers:
  1. A deterministic rule-based summary built only from the payment's own evidence. Always available.
  2. An optional Gemini rewrite (English, Tamil or Hindi) that is given ONLY that evidence as JSON.
     The reply is checked: any number the model writes that is not in the evidence rejects the reply
     and the rule-based summary is shown instead. The score itself is never produced by the LLM.

Only synthetic data is ever sent to the API. No extra dependencies (standard-library HTTP).
"""
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from intelligence.message_rules import find_signals

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DECISION_WORDS = {"BLOCK_TRANSACTION": "hold the payment", "MANUAL_REVIEW": "send it for human review",
                  "APPROVE": "let it through"}
ACTION = {"BLOCK_TRANSACTION": "Hold the payment and confirm with the sender through the bank's official app or "
                               "number. Do not use any link or phone number from the message.",
          "MANUAL_REVIEW": "Ask a reviewer to confirm the beneficiary with the sender before releasing the payment.",
          "APPROVE": "No action is needed."}


def inr(x):
    whole, dec = f"{abs(float(x)):.2f}".split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return f"Rs {whole}.{dec}"


class Explainer:
    def __init__(self, base_dir=BASE_DIR):
        b = Path(base_dir)
        self.report = pd.read_csv(b / "evaluation" / "evidence_fusion_report.csv").set_index("transaction_id")
        self.tx = pd.read_csv(b / "data" / "synthetic" / "transactions.csv")
        self.feats = pd.read_csv(b / "data" / "processed" / "transaction_features.csv").set_index("transaction_id")
        self.acc_device = pd.read_csv(b / "data" / "synthetic" / "accounts.csv").set_index("account_id")["device_id"]
        msgs = pd.read_csv(b / "data" / "synthetic" / "messages.csv")
        msgs = msgs[msgs["transaction_id"].notna() & (msgs["transaction_id"].astype(str) != "")]
        self.msg_text = msgs.drop_duplicates("transaction_id").set_index("transaction_id")["sender_text"]
        self.n_senders = self.tx.groupby("receiver_id")["sender_id"].nunique()
        self.newrel = self.tx.groupby("receiver_id")["is_new_beneficiary"].mean()
        self.tx = self.tx.set_index("transaction_id")

    # ------------------------------------------------------------------ evidence
    def evidence(self, tid):
        r, t, f = self.report.loc[tid], self.tx.loc[tid], self.feats.loc[tid]
        beh, net = [], []
        ratio = float(f.get("amount_vs_account_average", 0))
        if ratio >= 3:
            beh.append(f"The amount is {ratio:.1f} times this account's usual payment")
        if int(t["is_new_beneficiary"]) == 1:
            beh.append("The beneficiary is new to the sender")
        if int(f.get("night_transaction_flag", 0)) == 1:
            beh.append("The payment was made at night")
        if int(f.get("txn_velocity_1h", 1)) >= 3:
            beh.append(f"{int(f['txn_velocity_1h'])} payments were made by this account in one hour")
        if float(f.get("balance_depletion_ratio", 0)) >= 0.4:
            beh.append(f"The payment uses {float(f['balance_depletion_ratio']) * 100:.0f} percent of the balance")
        if self.acc_device.get(t["sender_id"]) != t["device_id"]:
            beh.append("The device is not the one registered to this account")

        text = self.msg_text.get(tid)
        found = [k for k, v in find_signals(text).items() if v and k != "Safety warning"] if text else []
        msg = {"score": round(float(r["nlp_risk_score"]) * 100), "text": (str(text)[:200] if text else None),
               "indicators": found}

        age, ben = int(t["beneficiary_age_days"]), t["receiver_id"]
        if float(r["net_young"]) == 1:
            net.append(f"The beneficiary account is only {age} days old")
        n, share = int(self.n_senders.get(ben, 0)), float(self.newrel.get(ben, 0))
        if n >= 8 and share >= 0.5:
            net.append(f"{n} different accounts have paid this beneficiary, and {share * 100:.0f} percent of those "
                       "payments were first-time relationships")
        if float(r["net_ring"]) == 1:
            net.append("The sender shares a device with other accounts that also paid this beneficiary")

        contrib = {"Behaviour": float(r["behaviour_contrib"]), "Message": float(r["message_contrib"]),
                   "Network": float(r["network_contrib"])}
        return {"transaction_id": tid, "amount": inr(t["amount"]), "decision": r["system_decision"],
                "risk_score": round(float(r["unified_risk_score"]) * 100),
                "signals": {"behaviour": {"score": round(float(r["transaction_risk_score"]) * 100), "notes": beh},
                            "message": msg,
                            "network": {"score": round(float(r["graph_risk_score"]) * 100), "notes": net}},
                "strongest_signal": max(contrib, key=contrib.get)}

    # ------------------------------------------------------------- rule-based text
    def template(self, ev):
        s = ev["signals"]
        level = "high" if ev["risk_score"] >= 70 else "medium" if ev["risk_score"] >= 40 else "low"
        out = [f"{ev['transaction_id']} ({ev['amount']}) has a {level} risk score of {ev['risk_score']} out of 100. "
               f"The system recommends you {DECISION_WORDS[ev['decision']]}."]
        why = []
        if s["behaviour"]["notes"]:
            why.append("Behaviour: " + "; ".join(s["behaviour"]["notes"]))
        if s["message"]["indicators"]:
            why.append("Message: a linked message shows " + ", ".join(i.lower() for i in s["message"]["indicators"]))
        if s["network"]["notes"]:
            why.append("Network: " + "; ".join(s["network"]["notes"]))
        out.append("Why: " + (". ".join(why) + "." if why else "no signal stands out, so the payment looks routine."))
        if why:
            out.append(f"The {ev['strongest_signal'].lower()} signal pushed the score up the most.")
        out.append(ACTION[ev["decision"]])
        out.append("This is a risk indicator from a prototype trained on synthetic data, not proof of fraud.")
        return " ".join(out)

    # --------------------------------------------------------------------- Gemini
    @staticmethod
    def prompt(ev, language):
        return ("You are helping a fraud investigator. Explain the decision below in plain " + language +
                ", in 80 to 120 words, for a non-expert.\nRules:\n- Use ONLY facts in the evidence JSON. Never invent facts.\n"
                "- Copy numbers exactly as written. Do not calculate new numbers.\n"
                "- Say 'risk indicator', not 'fraud'. Do not accuse anyone.\n"
                "- End with one safe next step. Never ask anyone for an OTP, PIN or card details.\n"
                f"- Write the whole answer in {language}.\nEvidence JSON:\n" + json.dumps(ev, ensure_ascii=False))

    @staticmethod
    def call_gemini(prompt, api_key, model=MODEL, timeout=25):
        cfg = {"temperature": 0.2, "maxOutputTokens": 600}
        if "2.5" in model:
            cfg["thinkingConfig"] = {"thinkingBudget": 0}
        body = json.dumps({"contents": [{"parts": [{"text": prompt}]}], "generationConfig": cfg}).encode()
        req = urllib.request.Request(ENDPOINT.format(model=model), data=body, method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"]).strip()

    @staticmethod
    def numbers(text):
        out = set()
        for m in re.findall(r"\d[\d,]*\.?\d*", text):
            try:
                out.add(round(float(m.replace(",", "").rstrip(".")), 1))
            except ValueError:
                pass
        return out

    def grounded(self, text, ev):
        allowed = self.numbers(json.dumps(ev, ensure_ascii=False)) | {0.0, 1.0, 2.0, 3.0, 40.0, 70.0, 100.0}
        bad = sorted(n for n in self.numbers(text) if n not in allowed)
        return not bad, bad

    def explain(self, tid, language="English", api_key=None):
        ev = self.evidence(tid)
        base = self.template(ev)
        if not api_key:
            note = "Add GEMINI_API_KEY to enable AI explanations."
            if language != "English":
                note = f"{language} explanations need the Gemini key. Showing the English summary."
            return {"text": base, "source": "Rule-based", "note": note, "evidence": ev}
        try:
            text = self.call_gemini(self.prompt(ev, language), api_key)
        except urllib.error.HTTPError as e:
            why = "rate limit reached" if e.code == 429 else f"HTTP {e.code}"
            return {"text": base, "source": "Rule-based", "note": f"Gemini unavailable ({why}).", "evidence": ev}
        except Exception as e:  # network, timeout, bad JSON
            return {"text": base, "source": "Rule-based", "note": f"Gemini unavailable ({type(e).__name__}).",
                    "evidence": ev}
        ok, bad = self.grounded(text, ev)
        if not text or not ok:
            return {"text": base, "source": "Rule-based",
                    "note": f"The AI reply used numbers not in the evidence ({', '.join(map(str, bad))}), so it was "
                            "rejected.", "evidence": ev}
        return {"text": text, "source": "Gemini", "note": "Checked: every number comes from the evidence.", "evidence": ev}