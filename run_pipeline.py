"""Rebuild everything in the right order.   Run from project root:  python run_pipeline.py

Stops at the first step that fails, so you never train on half-updated data.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("Transactions, accounts, beneficiaries", "data_generation/generate_transactions_v2.py"),
    ("Messages (linked to payments)", "data_generation/generate_messages.py"),
    ("Data validation", "data_generation/validate_data.py"),
    ("Transaction features", "intelligence/feature_engineering.py"),
    ("Message features", "intelligence/message_feature_engineering.py"),
    ("Behaviour model (Isolation Forest)", "models/train_anomaly_model.py"),
    ("Message model (TF-IDF + LogReg)", "models/train_nlp_model.py"),
    ("Network graph summary", "graph/build_transaction_graph.py"),
    ("Risk fusion (real, all payments)", "intelligence/risk_fusion.py"),
    ("Evaluation report", "evaluation/make_report.py"),
]
for i, (label, script) in enumerate(STEPS, 1):
    if not (ROOT / script).exists():
        print(f"[{i}/{len(STEPS)}] SKIPPED {label}: {script} not found")
        continue
    print(f"\n[{i}/{len(STEPS)}] {label}\n" + "-" * 60, flush=True)
    if subprocess.run([sys.executable, script], cwd=ROOT).returncode != 0:
        sys.exit(f"\nStopped: {script} failed. Fix it, then run again.")
print("\nPipeline finished.")