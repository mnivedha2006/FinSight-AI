import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
SYNTH_DIR = BASE_DIR / "data" / "synthetic"
PROC_DIR = BASE_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
accounts = pd.read_csv(SYNTH_DIR / "accounts.csv")
beneficiaries = pd.read_csv(SYNTH_DIR / "beneficiaries.csv")
transactions = pd.read_csv(SYNTH_DIR / "transactions.csv")
messages = pd.read_csv(SYNTH_DIR / "messages.csv")
fraud_cases = pd.read_csv(SYNTH_DIR / "fraud_cases.csv")

report_lines = []
report_lines.append("FINSIGHT AI - DATA QUALITY REPORT")
report_lines.append("=" * 40)

# 1. Record counts
report_lines.append(f"\n1. RECORD COUNTS:")
report_lines.append(f"Accounts: {len(accounts)}")
report_lines.append(f"Beneficiaries: {len(beneficiaries)}")
report_lines.append(f"Transactions: {len(transactions)}")
report_lines.append(f"Messages: {len(messages)}")
report_lines.append(f"Fraud Cases: {len(fraud_cases)}")

# 2. Duplicate checks
dup_tx = transactions["transaction_id"].duplicated().sum()
dup_acc = accounts["account_id"].duplicated().sum()
report_lines.append(f"\n2. DUPLICATES:")
report_lines.append(f"Duplicate transaction IDs: {dup_tx}")
report_lines.append(f"Duplicate account IDs: {dup_acc}")

# 3. Missing values
report_lines.append(f"\n3. MISSING VALUES (Null counts):")
report_lines.append(str(transactions.isnull().sum()))

# 4. Referential integrity checks
invalid_senders = ~transactions["sender_id"].isin(accounts["account_id"]).sum()
invalid_ben = ~transactions["receiver_id"].isin(beneficiaries["beneficiary_id"]).sum()
report_lines.append(f"\n4. REFERENTIAL INTEGRITY:")
report_lines.append(f"Transactions with invalid sender IDs: {invalid_senders}")
report_lines.append(f"Transactions with invalid beneficiary IDs: {invalid_ben}")

# Write report
report_path = PROC_DIR / "data_quality_report.txt"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

print(f"Data validation complete! Quality report saved to {report_path}")