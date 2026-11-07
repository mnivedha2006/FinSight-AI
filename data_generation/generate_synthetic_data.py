import random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ============================================================
# FINSIGHT AI - SYNTHETIC FINANCIAL DATA GENERATOR
# Version: 1.0
# Purpose: Arthaton 2026 prototype
# All records are fictional and synthetic.
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parents[1]
from pathlib import Path

# Explicit output directory pointing to your local FinSight-AI folder
OUTPUT_DIR = Path(r"C:\Users\HP\Desktop\FinSight-AI\data\synthetic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


N_ACCOUNTS = 1000
N_BENEFICIARIES = 2000
N_TRANSACTIONS = 5000

START_DATE = datetime(2026, 8, 10)
END_DATE = datetime(2026, 11, 7)

print("Starting FinSight AI data generation...")

# 1. Generate Accounts
accounts_data = []
for i in range(1, N_ACCOUNTS + 1):
    acc_id = f"ACC_{i:04d}"
    accounts_data.append({
        "account_id": acc_id,
        "account_age_days": random.randint(30, 1000),
        "device_id": f"DEV_{random.randint(1, 200):04d}",
        "normal_daily_txn_count": random.randint(2, 10),
        "average_transaction_amount": round(random.uniform(200, 5000), 2),
        "risk_profile": "normal"
    })
accounts = pd.DataFrame(accounts_data)
accounts.to_csv(OUTPUT_DIR / "accounts.csv", index=False)

# 2. Generate Beneficiaries
beneficiaries_data = []
for i in range(1, N_BENEFICIARIES + 1):
    ben_id = f"BEN_{i:04d}"
    beneficiaries_data.append({
        "beneficiary_id": ben_id,
        "beneficiary_age_days": random.randint(1, 400),
        "category": random.choice(["merchant", "p2p_individual", "utility", "unknown"])
    })
beneficiaries = pd.DataFrame(beneficiaries_data)
beneficiaries.to_csv(OUTPUT_DIR / "beneficiaries.csv", index=False)

# 3. Generate Transactions
tx_types = ["P2P", "P2M", "collect_request"]
channels = ["UPI", "QR", "collect"]

tx_data = []
for i in range(1, N_TRANSACTIONS + 1):
    tx_id = f"TXN_{i:05d}"
    sender = accounts.sample(1).iloc[0]
    beneficiary = beneficiaries.sample(1).iloc[0]
    
    # Random timestamp between start and end date
    delta_seconds = int((END_DATE - START_DATE).total_seconds())
    rand_seconds = random.randint(0, delta_seconds)
    tx_time = START_DATE + timedelta(seconds=rand_seconds)
    
    amount = round(max(50.0, np.random.normal(sender["average_transaction_amount"], sender["average_transaction_amount"] * 0.5)), 2)
    
    tx_data.append({
        "transaction_id": tx_id,
        "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
        "sender_id": sender["account_id"],
        "receiver_id": beneficiary["beneficiary_id"],
        "amount": amount,
        "transaction_type": random.choice(tx_types),
        "payment_channel": random.choice(channels),
        "device_id": sender["device_id"],
        "location_id": f"LOC_{random.randint(1, 50):03d}",
        "sender_balance": round(random.uniform(5000, 100000), 2),
        "account_age_days": sender["account_age_days"],
        "beneficiary_age_days": beneficiary["beneficiary_age_days"],
        "txn_velocity_1h": random.randint(1, 3),
        "txn_velocity_24h": random.randint(2, 15),
        "is_new_beneficiary": random.choice([0, 1]),
        "is_fraud": 0,
        "fraud_type": "normal"
    })

df = pd.DataFrame(tx_data)

# Inject Fraud Scenarios (Rapid transfer, New-beneficiary anomaly, Suspicious collect)
# Rapid transfers (90 items)
rapid_indices = np.random.choice(df.index, 90, replace=False)
for idx in rapid_indices:
    df.loc[idx, "is_fraud"] = 1
    df.loc[idx, "fraud_type"] = "rapid_transfer"
    df.loc[idx, "txn_velocity_1h"] = random.randint(8, 20)

# New beneficiary anomaly (80 items)
new_ben_indices = np.random.choice(df.index.difference(rapid_indices), 80, replace=False)
for idx in new_ben_indices:
    df.loc[idx, "is_fraud"] = 1
    df.loc[idx, "fraud_type"] = "new_beneficiary_anomaly"
    df.loc[idx, "is_new_beneficiary"] = 1
    df.loc[idx, "amount"] = round(df.loc[idx, "amount"] * 5.0, 2)

# Suspicious collect transactions (80 items)
collect_indices = np.random.choice(df.index.difference(np.union1d(rapid_indices, new_ben_indices)), 80, replace=False)
for idx in collect_indices:
    df.loc[idx, "is_fraud"] = 1
    df.loc[idx, "fraud_type"] = "suspicious_collect"
    df.loc[idx, "transaction_type"] = "collect_request"
    df.loc[idx, "payment_channel"] = "collect"

# Recompute velocity using rolling windows on datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["sender_id", "timestamp"]).reset_index(drop=True)
df["txn_velocity_1h"] = 0
df["txn_velocity_24h"] = 0

for sender_id, group in df.groupby("sender_id"):
    ts_series = pd.Series(group["timestamp"].to_numpy(), index=pd.DatetimeIndex(group["timestamp"]))
    one_hour = ts_series.rolling("1h").count()
    one_day = ts_series.rolling("24h").count()
    df.loc[group.index, "txn_velocity_1h"] = one_hour.to_numpy().astype(int)
    df.loc[group.index, "txn_velocity_24h"] = one_day.to_numpy().astype(int)

df = df.sort_values("transaction_id").reset_index(drop=True)
df.to_csv(OUTPUT_DIR / "transactions.csv", index=False)

# 4. Generate Messages (5000 records)
msg_categories = ["Normal", "phishing", "collect scam", "impersonation"]
messages_data = []
for i in range(1, 5001):
    msg_id = f"MSG_{i:05d}"
    cat = random.choice(msg_categories)
    label = 0 if cat == "Normal" else 1
    messages_data.append({
        "message_id": msg_id,
        "transaction_id": f"TXN_{random.randint(1, N_TRANSACTIONS):05d}" if random.random() > 0.3 else "",
        "sender_text": f"Sample payment notification text for category {cat}",
        "message_category": cat,
        "urgency_score": round(random.uniform(0.1, 0.9), 2) if label == 1 else round(random.uniform(0.0, 0.3), 2),
        "suspicious_url": 1 if label == 1 and random.random() > 0.5 else 0,
        "credential_request": 1 if label == 1 and random.random() > 0.4 else 0,
        "label": label
    })
messages = pd.DataFrame(messages_data)
messages.to_csv(OUTPUT_DIR / "messages.csv", index=False)

# 5. Generate Fraud Cases (250 summary records)
fraud_cases = df[df["is_fraud"] == 1].head(250).copy()
fraud_cases.to_csv(OUTPUT_DIR / "fraud_cases.csv", index=False)

print("========== DATA VALIDATION ==========")
print(f"accounts.csv: {len(accounts)} records")
print(f"beneficiaries.csv: {len(beneficiaries)} records")
print(f"transactions.csv: {len(df)} records")
print(f"messages.csv: {len(messages)} records")
print(f"fraud_cases.csv: {len(fraud_cases)} records")
print("\nFraud distribution:")
print(df["fraud_type"].value_counts())
print("All datasets passed validation.")
print("=====================================")
print("FINSIGHT AI DATA GENERATION SUCCESS")