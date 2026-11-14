"""Data integrity checks. Run from the project root:  python -m pytest -q"""
from pathlib import Path

import pandas as pd
import pytest

SYN = Path(__file__).resolve().parents[1] / "data" / "synthetic"


def load(name):
    p = SYN / name
    if not p.exists():
        pytest.skip(f"{name} not found; run python run_pipeline.py first")
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def tx():
    return load("transactions.csv")


def test_transaction_ids_are_unique(tx):
    assert tx["transaction_id"].is_unique


def test_every_sender_is_a_known_account(tx):
    acc = load("accounts.csv")
    assert tx["sender_id"].isin(acc["account_id"]).all()


def test_every_receiver_is_a_known_beneficiary(tx):
    ben = load("beneficiaries.csv")
    assert tx["receiver_id"].isin(ben["beneficiary_id"]).all()


def test_amounts_positive_and_timestamps_parse(tx):
    assert (tx["amount"] > 0).all()
    assert pd.to_datetime(tx["timestamp"], errors="coerce").notna().all()


def test_fraud_labels_are_consistent(tx):
    assert set(tx["is_fraud"].unique()) <= {0, 1}
    assert ((tx["is_fraud"] == 1) == (tx["fraud_type"] != "normal")).all()
    assert 0.03 <= tx["is_fraud"].mean() <= 0.08          # rare, like real fraud


def test_hard_negatives_exist_and_are_legitimate(tx):
    hard = tx[tx["scenario_id"].isin(["LEGIT_LARGE", "LEGIT_BURST"])]
    assert len(hard) > 0 and (hard["is_fraud"] == 0).all()


def test_messages_link_to_real_transactions_and_have_labels(tx):
    msgs = load("messages.csv")
    linked = msgs["transaction_id"].dropna().astype(str)
    linked = linked[linked != ""]
    assert linked.isin(tx["transaction_id"]).all()
    assert set(msgs["label"].unique()) <= {0, 1}
    assert msgs["template_id"].nunique() >= 30             # varied, not one sentence per class
    assert msgs["language"].nunique() >= 4