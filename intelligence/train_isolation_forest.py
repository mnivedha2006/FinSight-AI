import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix


def train_isolation_forest():
  BASE_DIR = Path(__file__).resolve().parents[1]
  SYNTH_DIR = BASE_DIR / "data" / "synthetic"
  PROC_DIR = BASE_DIR / "data" / "processed"

  print("Loading engineered features and raw transactions...")
  features_df = pd.read_csv(PROC_DIR / "transaction_features.csv")
  raw_tx = pd.read_csv(SYNTH_DIR / "transactions.csv")

  # Merge ground truth labels (is_fraud) for evaluation ONLY (not used for training!)
  df = features_df.merge(
      raw_tx[["transaction_id", "is_fraud", "fraud_type"]],
      on="transaction_id",
      how="left",
  )

  # Select numerical feature columns for unsupervised training (strictly no labels!)
  feature_cols = [
      "amount_log",
      "high_amount_flag",
      "night_transaction_flag",
      "balance_depletion_ratio",
      "amount_vs_account_average",
      "account_age_risk",
      "beneficiary_age_risk",
      "txn_velocity_1h",
      "txn_velocity_24h",
      "is_new_beneficiary",
  ]

  X = df[feature_cols].fillna(0)

  print(f"Training Isolation Forest on shape {X.shape}...")
  # contamination=0.05 matches our ~5% expected fraud rate (250 out of 5000)
  model = IsolationForest(
      n_estimators=100, contamination=0.05, random_state=42
  )

  # IsolationForest returns 1 for inliers (normal) and -1 for outliers (fraud)
  preds = model.fit_predict(X)

  # Convert predictions: -1 (anomaly/fraud) -> 1, 1 (normal) -> 0
  df["predicted_fraud"] = np.where(preds == -1, 1, 0)

  print("\n=== EVALUATION AGAINST GROUND TRUTH ===")
  print(confusion_matrix(df["is_fraud"], df["predicted_fraud"]))
  print("\nClassification Report:")
  print(classification_report(df["is_fraud"], df["predicted_fraud"]))

  # Save predictions
  output_path = PROC_DIR / "transaction_scored_isolation_forest.csv"
  df.to_csv(output_path, index=False)
  print(f"\nScored transactions saved to {output_path}")


if __name__ == "__main__":
  train_isolation_forest()