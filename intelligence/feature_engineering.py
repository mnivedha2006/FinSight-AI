import numpy as np
import pandas as pd
from pathlib import Path


def engineer_transaction_features():
  BASE_DIR = Path(__file__).resolve().parents[1]
  SYNTH_DIR = BASE_DIR / "data" / "synthetic"
  PROC_DIR = BASE_DIR / "data" / "processed"
  PROC_DIR.mkdir(parents=True, exist_ok=True)

  print("Loading raw synthetic data for feature engineering...")
  transactions = pd.read_csv(SYNTH_DIR / "transactions.csv")
  accounts = pd.read_csv(SYNTH_DIR / "accounts.csv")

  df = transactions.copy()
  df["timestamp"] = pd.to_datetime(df["timestamp"])

  print("Engineering numerical and behavioral features...")

  # 1. Log amount to handle skewness
  df["amount_log"] = np.log1p(df["amount"])

  # 2. High amount flag (> 10,000 units)
  df["high_amount_flag"] = (df["amount"] > 10000).astype(int)

  # 3. Night transaction indicator (between 11 PM and 5 AM)
  df["hour"] = df["timestamp"].dt.hour
  df["night_transaction_flag"] = (
      (df["hour"] >= 23) | (df["hour"] <= 5)
  ).astype(int)

  # 4. Balance depletion ratio (amount spent vs sender balance)
  df["balance_depletion_ratio"] = df["amount"] / (df["sender_balance"] + 1e-5)

  # 5. Merge account historical averages to compute amount vs sender average deviation
  df = df.merge(
      accounts[["account_id", "average_transaction_amount"]],
      left_on="sender_id",
      right_on="account_id",
      how="left",
  )
  df.drop(columns=["account_id"], inplace=True)

  df["amount_vs_account_average"] = (
      df["amount"] / df["average_transaction_amount"]
  )

  # 6. Account age risk (newly created accounts < 60 days)
  df["account_age_risk"] = (df["account_age_days"] < 60).astype(int)

  # 7. Beneficiary age risk (recently added beneficiary < 15 days)
  df["beneficiary_age_risk"] = (df["beneficiary_age_days"] < 15).astype(int)

  # Keep columns clean: exclude target labels from features!
  feature_columns = [
      "transaction_id",
      "timestamp",
      "sender_id",
      "receiver_id",
      "amount",
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

  feature_df = df[feature_columns]

  output_path = PROC_DIR / "transaction_features.csv"
  feature_df.to_csv(output_path, index=False)
  print(f"Feature engineering successful! Saved to {output_path}")
  print(f"Total engineered features shape: {feature_df.shape}")


if __name__ == "__main__":
  engineer_transaction_features()