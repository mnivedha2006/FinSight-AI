"""Real risk fusion for FinSight AI.

Combines three independent signals into one score per payment, for ALL payments:
  Behaviour : Isolation Forest anomaly rank (models/fraud_detector.joblib)
  Message   : scam score of the SMS linked to the payment (TF-IDF + LogReg + rules),
              scored out-of-fold by message family, so no message is scored by a model that saw its family
  Network   : unsupervised indicators from the ledger (young beneficiary, many new-relationship senders,
              accounts on a shared device paying the same beneficiary)
Fraud labels are used ONLY to fit/evaluate the final logistic-regression weights, with a 60/20/20
train / validation / test split. The threshold is chosen on validation; results are reported on test.

Run from the project root:   python intelligence/risk_fusion.py
Writes: evaluation/evidence_fusion_report.csv  (read by the dashboard)
        evaluation/fusion_metrics.json
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from intelligence.message_rules import rule_score_text  # noqa: E402
from models.train_nlp_model import ML_W, RULE_W, build_pipeline  # noqa: E402

warnings.filterwarnings("ignore")
SYN, PROC, EVAL, MODELS = (BASE_DIR / "data" / "synthetic", BASE_DIR / "data" / "processed",
                           BASE_DIR / "evaluation", BASE_DIR / "models")
SEED = 42


# ------------------------------------------------------------------ behaviour
def behaviour_scores(tx):
    art = joblib.load(MODELS / "fraud_detector.joblib")
    feats = pd.read_csv(PROC / "transaction_features.csv")
    pipe = art["pipeline"]
    x = feats[art["feature_columns"]]
    anomaly = -pipe.named_steps["model"].decision_function(pipe.named_steps["imputer"].transform(x))
    score = (rankdata(anomaly) - 1) / (len(anomaly) - 1) * 100          # 100 = most unusual payment
    return tx[["transaction_id"]].merge(pd.DataFrame({"transaction_id": feats["transaction_id"], "B": score}),
                                        on="transaction_id", how="left")["B"].fillna(0).values


# -------------------------------------------------------------------- message
def message_scores(tx):
    msgs = pd.read_csv(SYN / "messages.csv")
    texts = msgs["sender_text"].astype(str)
    y = msgs["label"].astype(int)
    ml = np.zeros(len(msgs))
    for tr, te in GroupKFold(n_splits=5).split(texts, y, msgs["template_id"]):   # out-of-family scoring
        ml[te] = build_pipeline().fit(texts.iloc[tr], y.iloc[tr]).predict_proba(texts.iloc[te])[:, 1] * 100
    rule = np.array([rule_score_text(t) for t in texts], dtype=float)
    msgs["M"] = ML_W * ml + RULE_W * rule
    linked = msgs[msgs["transaction_id"].notna() & (msgs["transaction_id"].astype(str) != "")]
    per_tx = linked.groupby("transaction_id")["M"].max()
    return tx["transaction_id"].map(per_tx).fillna(0.0).values


# -------------------------------------------------------------------- network
def network_scores(tx):
    acc = pd.read_csv(SYN / "accounts.csv").set_index("account_id")["device_id"]
    dev_members = acc.groupby(acc).groups                                  # device -> accounts on it
    members = {d: list(idx) for d, idx in dev_members.items()}
    payers = tx.groupby("receiver_id")["sender_id"].agg(set)
    n_senders = tx.groupby("receiver_id")["sender_id"].nunique()
    newrel = tx.groupby("receiver_id")["is_new_beneficiary"].mean()
    young = (tx["beneficiary_age_days"] < 15).astype(float).values
    hub = (tx["receiver_id"].map(newrel) * np.minimum(1.0, tx["receiver_id"].map(n_senders) / 8)).values
    ring = np.zeros(len(tx))
    for i, (s, r) in enumerate(zip(tx["sender_id"], tx["receiver_id"])):
        mates = [m for m in members.get(acc.get(s), []) if m != s]
        ring[i] = float(any(m in payers[r] for m in mates))
    return 100 * (0.40 * young + 0.35 * hub + 0.25 * ring)


# ------------------------------------------------------------------- metrics
def metrics(y, flag, score):
    tn, fp, fn, tp = confusion_matrix(y, flag, labels=[0, 1]).ravel()
    return {"precision": round(float(precision_score(y, flag, zero_division=0)), 3),
            "recall": round(float(recall_score(y, flag, zero_division=0)), 3),
            "f1_score": round(float(f1_score(y, flag, zero_division=0)), 3),
            "pr_auc": round(float(average_precision_score(y, score)), 3),
            "false_positive_rate": round(float(fp / (fp + tn)), 3)}


def best_threshold(y, score, grid):
    return max(grid, key=lambda t: f1_score(y, (score >= t).astype(int), zero_division=0))


def display(p, t):
    """Rescale probability so the validation-chosen threshold t shows as 40 (review) and the midpoint
    between t and 1 shows as 70 (block)."""
    tb = t + (1 - t) / 2
    return np.where(p < t, 40 * p / t, np.where(p < tb, 40 + 30 * (p - t) / (tb - t), 70 + 30 * (p - tb) / (1 - tb)))


def main():
    tx = pd.read_csv(SYN / "transactions.csv")
    print(f"Scoring {len(tx)} payments...")
    B, M, N = behaviour_scores(tx), message_scores(tx), network_scores(tx)
    y = tx["is_fraud"].astype(int).values
    X = np.column_stack([B, M, N]) / 100

    idx = np.arange(len(tx))
    tr, rest = train_test_split(idx, test_size=0.4, random_state=SEED, stratify=y)
    va, te = train_test_split(rest, test_size=0.5, random_state=SEED, stratify=y[rest])
    lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000).fit(X[tr], y[tr])
    p = lr.predict_proba(X)[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    t_fusion = float(best_threshold(y[va], p[va], grid))
    shown = display(p, t_fusion)

    signals = {"Behaviour only": B, "Message only": M, "Network only": N}
    thr = {k: float(best_threshold(y[va], v[va], np.linspace(1, 99, 99))) for k, v in signals.items()}
    results, flags = {}, {}
    for k, v in signals.items():
        flags[k] = (v >= thr[k]).astype(int)
        results[k] = metrics(y[te], flags[k][te], v[te])
    flags["Fusion"] = (p >= t_fusion).astype(int)
    results["Fusion"] = metrics(y[te], flags["Fusion"][te], p[te])

    grp = np.where(tx["fraud_type"] != "normal", tx["fraud_type"],
                   np.where(tx["scenario_id"].fillna("").isin(["LEGIT_LARGE", "LEGIT_BURST"]), tx["scenario_id"], "ordinary legit"))
    tt = pd.DataFrame({"grp": grp[te], **{k: flags[k][te] for k in flags}})
    by_group = tt.groupby("grp").agg(n=("Fusion", "size"), **{k: (k, "mean") for k in flags}).round(3)

    names = ["Behaviour", "Message", "Network"]
    contrib = X * lr.coef_[0]
    split = np.array(["train"] * len(tx), dtype=object)
    split[va], split[te] = "validation", "test"
    decision = np.where(shown >= 70, "BLOCK_TRANSACTION", np.where(shown >= 40, "MANUAL_REVIEW", "APPROVE"))
    report = pd.DataFrame({
        "transaction_id": tx["transaction_id"], "amount": tx["amount"], "true_fraud_label": y,
        "fraud_type": tx["fraud_type"], "split": split,
        "transaction_risk_score": (B / 100).round(4), "nlp_risk_score": (M / 100).round(4),
        "graph_risk_score": (N / 100).round(4), "unified_risk_score": (shown / 100).round(4),
        "fusion_probability": p.round(4), "behaviour_contrib": contrib[:, 0].round(3),
        "message_contrib": contrib[:, 1].round(3), "network_contrib": contrib[:, 2].round(3),
        "system_decision": decision})
    report.to_csv(EVAL / "evidence_fusion_report.csv", index=False)

    out = {"test_size": int(len(te)), "test_fraud": int(y[te].sum()), "fusion_threshold_probability": round(t_fusion, 3),
           "single_signal_thresholds": {k: round(v, 1) for k, v in thr.items()},
           "weights": {n: round(float(c), 3) for n, c in zip(names, lr.coef_[0])},
           "intercept": round(float(lr.intercept_[0]), 3), "test_metrics": results,
           "flag_rate_by_group_on_test": by_group.reset_index().to_dict(orient="records"),
           "notes": ["Weights fitted on train, threshold chosen on validation, metrics on test.",
                     "Message scores are out-of-family. Messages are linked to payments by the synthetic "
                     "generator, so fusion results are optimistic compared with real data."]}
    (EVAL / "fusion_metrics.json").write_text(json.dumps(out, indent=2))

    print(f"\nFusion weights (logit scale): " + ", ".join(f"{n} {c:+.2f}" for n, c in zip(names, lr.coef_[0])))
    print(f"Threshold chosen on validation: p >= {t_fusion:.2f}   (shown as score 40)")
    print(f"\nTEST SET ({len(te)} payments, {int(y[te].sum())} fraud)")
    for k, v in results.items():
        print(f"  {k:<15} P {v['precision']:.3f}  R {v['recall']:.3f}  F1 {v['f1_score']:.3f}  "
              f"PR-AUC {v['pr_auc']:.3f}  FPR {v['false_positive_rate']:.3f}")
    print("\nShare of payments flagged, by group (test set):")
    print(by_group.to_string())
    print("\nDecisions across all payments:\n" + report["system_decision"].value_counts().to_string())
    print("\nSaved evaluation/evidence_fusion_report.csv and fusion_metrics.json")


if __name__ == "__main__":
    main()