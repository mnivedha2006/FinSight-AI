import pandas as pd
from pathlib import Path

# Explicit path pointing directly to your project's synthetic data folder
folder = Path(r"C:\Users\HP\Desktop\FinSight-AI\data\synthetic")

transactions = pd.read_csv(folder / "transactions.csv")
messages = pd.read_csv(folder / "messages.csv")
accounts = pd.read_csv(folder / "accounts.csv")

print("\nFINSIGHT AI - DATA INSPECTION")

print("\n1. TRANSACTION SAMPLE")
print(transactions.head(5).to_string(index=False))

print("\n2. FRAUD DISTRIBUTION")
print(transactions["fraud_type"].value_counts())

print("\n3. TRANSACTION AMOUNT SUMMARY")
print(transactions.groupby("fraud_type")["amount"].describe())

print("\n4. MESSAGE CLASSIFICATION")
print(messages["message_category"].value_counts())

print("\n5. ACCOUNT SAMPLE")
print(accounts.head(3))

print("\n6. DATA QUALITY")
print("Transaction nulls:", transactions.isnull().sum().sum())
print("Duplicate transactions:", transactions["transaction_id"].duplicated().sum())

print("\nInspection completed.")