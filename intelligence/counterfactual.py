"""Counterfactual engine: change one thing about a payment and re-run the WHOLE fusion.

Nothing is subtracted by hand. Every what-if rebuilds the three signals and passes them through the same
fitted fusion model that scored the payment originally:
  Behaviour : engineered features are edited, then the Isolation Forest scores the edited payment
  Message   : the scam message can be removed (message score becomes 0)
  Network   : the beneficiary-age, hub and shared-device components are edited, then recombined
"""
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
NAMES = ["Behaviour", "Message", "Network"]


DISPLAY_K = 0.4   # softness of the 0-100 display scale; larger = steeper


def display_score(p, t, k=DISPLAY_K):
    """Monotone display scale: the validation threshold t shows as 40, and the score is a sigmoid of the
    log-odds distance from t. It spreads payments across 0-100 so a single what-if visibly moves the score;
    it never changes the ranking or which side of the threshold a payment falls on."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit, lt = np.log(p / (1 - p)), np.log(t / (1 - t))
    return 100 / (1 + np.exp(-(k * (logit - lt) + math.log(0.4 / 0.6))))


def decision_of(score):
    return "BLOCK_TRANSACTION" if score >= 70 else "MANUAL_REVIEW" if score >= 40 else "APPROVE"


class CounterfactualEngine:
    def __init__(self, base_dir=BASE_DIR):
        base = Path(base_dir)
        art = joblib.load(base / "models" / "fraud_detector.joblib")
        fus = joblib.load(base / "models" / "fusion_model.joblib")
        self.cols, self.pipe = art["feature_columns"], art["pipeline"]
        self.coef, self.intercept, self.t = np.asarray(fus["coef"]), float(fus["intercept"]), float(fus["threshold"])
        self.feats = pd.read_csv(base / "data" / "processed" / "transaction_features.csv").set_index("transaction_id")
        self.report = pd.read_csv(base / "evaluation" / "evidence_fusion_report.csv").set_index("transaction_id")
        anomaly = -self.pipe.named_steps["model"].decision_function(
            self.pipe.named_steps["imputer"].transform(self.feats[self.cols]))
        self.sorted_anomaly = np.sort(anomaly)

    # --- the three signals -------------------------------------------------
    def behaviour(self, row):
        x = pd.DataFrame([{c: row.get(c, 0) for c in self.cols}])
        a = float(-self.pipe.named_steps["model"].decision_function(self.pipe.named_steps["imputer"].transform(x))[0])
        return float(np.searchsorted(self.sorted_anomaly, a, side="left") / (len(self.sorted_anomaly) - 1) * 100)

    @staticmethod
    def network(young, hub, ring):
        return 100 * (0.40 * young + 0.35 * hub + 0.25 * ring)

    def fuse(self, B, M, N):
        x = np.array([B, M, N]) / 100
        contrib = self.coef * x
        p = 1 / (1 + math.exp(-(self.intercept + contrib.sum())))
        score = float(display_score(p, self.t))
        return {"B": B, "M": M, "N": N, "p": p, "score": score, "decision": decision_of(score),
                "contrib": dict(zip(NAMES, contrib))}

    # --- what-if -----------------------------------------------------------
    def simulate(self, tid, known=False, established=False, drop_message=False, clear_device=False,
                 amount_share=1.0, velocity=None, daytime=False):
        row = self.feats.loc[tid].to_dict()
        rep = self.report.loc[tid]
        young, hub, ring = float(rep["net_young"]), float(rep["net_hub"]), float(rep["net_ring"])
        M = float(rep["nlp_risk_score"]) * 100
        before = self.fuse(self.behaviour(row), M, self.network(young, hub, ring))

        new, changes = dict(row), []

        def set_(k, v, label=None):
            if k in new and not math.isclose(float(new[k]), float(v), rel_tol=1e-9, abs_tol=1e-9):
                changes.append((label or k, float(new[k]), float(v)))
                new[k] = v

        if known:
            set_("is_new_beneficiary", 0, "Beneficiary is new to sender")
        if established:
            set_("beneficiary_age_risk", 0, "Beneficiary under 15 days old")
            if young:
                changes.append(("Network: young beneficiary", young, 0.0))
                young = 0.0
        if clear_device and ring:
            changes.append(("Network: shared-device link", ring, 0.0))
            ring = 0.0
        if drop_message and M > 0:
            changes.append(("Message score", M, 0.0))
            M = 0.0
        if daytime:
            set_("night_transaction_flag", 0, "Paid at night")
        if abs(amount_share - 1.0) > 1e-9:
            for k in ("amount", "balance_depletion_ratio", "amount_vs_account_average"):
                if k in new:
                    changes.append((k, float(new[k]), float(new[k]) * amount_share))
                    new[k] = float(new[k]) * amount_share
            if "amount_log" in new:
                new["amount_log"] = math.log1p(new["amount"])
            if "high_amount_flag" in new:
                set_("high_amount_flag", int(new["amount"] > 10000), "High-amount flag")
        if velocity is not None:
            set_("txn_velocity_1h", velocity, "Payments in the last hour")
            if "txn_velocity_24h" in new and new["txn_velocity_24h"] < velocity:
                set_("txn_velocity_24h", velocity, "Payments in the last 24 hours")

        after = self.fuse(self.behaviour(new), M, self.network(young, hub, ring))
        return {"before": before, "after": after, "changes": changes, "summary": self.explain(before, after)}

    @staticmethod
    def explain(before, after):
        drop = before["score"] - after["score"]
        if abs(drop) < 0.5:
            return "These changes do not move the score. The signals that drive this payment are still present."
        deltas = {n: after["contrib"][n] - before["contrib"][n] for n in NAMES}
        lead = max(deltas, key=lambda n: abs(deltas[n]))
        word = "lowers" if drop > 0 else "raises"
        verdict = ""
        if before["decision"] != after["decision"]:
            nice = {"APPROVE": "approve", "MANUAL_REVIEW": "review", "BLOCK_TRANSACTION": "block"}
            verdict = f" The decision moves from {nice[before['decision']]} to {nice[after['decision']]}."
        return (f"This {word} the risk score from {before['score']:.0f} to {after['score']:.0f}. "
                f"The {lead.lower()} signal accounts for most of the change.{verdict}")