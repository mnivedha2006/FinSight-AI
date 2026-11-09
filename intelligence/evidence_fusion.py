import json
from pathlib import Path
import joblib
import pandas as pd

# ==================================================
# 1. CONFIGURATION & PATHS
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = BASE_DIR / "data" / "processed"
SYNTH_DIR = BASE_DIR / "data" / "synthetic"
MODEL_DIR = BASE_DIR / "models"
EVALUATION_DIR = BASE_DIR / "evaluation"


def fuse_evidence():
  print("Loading datasets and models for evidence fusion...")

  # Load scored transactions and messages
  tx_df = pd.read_csv(SYNTH_DIR / "transactions.csv")
  msg_df = pd.read_csv(SYNTH_DIR / "messages.csv")
  features_df = pd.read_csv(PROC_DIR / "transaction_features.csv")

  # Load models if needed for live scoring lookup
  # For evaluation, we combine our pre-generated test results & features
  print("Synthesizing multi-layered risk scores...")

  fusion_results = []

  for idx, row in tx_df.head(100).iterrows():  # Processing sample for fusion report
    txn_id = row["transaction_id"]
    amount = row["amount"]
    is_fraud = row["is_fraud"]

    # 1. Transaction Anomaly Component (Mocking score based on amount/velocity flags)
    txn_risk = 1.0 if amount > 35000 or row.get("is_high_risk", 0) else 0.1

    # 2. NLP Message Risk Component (Check if linked message exists and is scam)
    linked_msg = msg_df[msg_df["transaction_id"] == txn_id]
    nlp_risk = 1.0 if not linked_msg.empty and linked_msg.iloc[0]["label"] == 1 else 0.0

    # 3. Graph Risk Component (Receiver risk based on degree)
    graph_risk = 0.8 if row["receiver_id"] in tx_df["receiver_id"].value_counts().head(10).index else 0.2

    # Weighted Evidence Fusion Formula
    # 40% Transaction Anomaly + 40% NLP Scam Score + 20% Graph Network Risk
    final_risk_score = (0.40 * txn_risk) + (0.40 * nlp_risk) + (0.20 * graph_risk)

    if final_risk_score >= 0.70:
      decision = "BLOCK_TRANSACTION"
    elif final_risk_score >= 0.40:
      decision = "MANUAL_REVIEW"
    else:
      decision = "APPROVE"

    fusion_results.append({
        "transaction_id": txn_id,
        "amount": amount,
        "true_fraud_label": is_fraud,
        "transaction_risk_score": round(txn_risk, 2),
        "nlp_risk_score": round(nlp_risk, 2),
        "graph_risk_score": round(graph_risk, 2),
        "unified_risk_score": round(final_risk_score, 4),
        "system_decision": decision,
    })

  fusion_df = pd.DataFrame(fusion_results)
  print(f"\nFusion complete! Evaluated {len(fusion_df)} transactions.")
  print(fusion_df[["transaction_id", "unified_risk_score", "system_decision"]].head(10))

  # Save fusion evaluation report
  output_path = EVALUATION_DIR / "evidence_fusion_report.csv"
  fusion_df.to_csv(output_path, index=False)
  print(f"\nEvidence fusion report saved to {output_path}")


if __name__ == "__main__":
  fuse_evidence()