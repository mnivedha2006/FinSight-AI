"""Model, fusion, counterfactual and explainer checks."""
import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

from intelligence.message_rules import find_signals, normalise, rule_score_text

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ["fraud", "scenario", "label", "id", "timestamp", "sender", "receiver"]

SCAM = {
    "en": "Your bank account will be blocked today. Verify immediately at http://sbi-verify.xyz and enter your OTP.",
    "ta": "ungal account udane block aagividum. KYC update seiya click pannunga http://bit.ly/kyc-now, OTP anuppunga.",
    "hi": "आपका खाता तुरंत बंद हो जाएगा। केवाईसी अपडेट के लिए http://verify-bank.top पर क्लिक करें और OTP भेजें।",
}
GENUINE = "Rs 450.00 debited from A/c XX1234 on 09-Nov to Swiggy via UPI. Never share your OTP with anyone."


def need(path):
    p = ROOT / path
    if not p.exists():
        pytest.skip(f"{path} not found; run python run_pipeline.py first")
    return p


# ------------------------------------------------------------------ message rules
@pytest.mark.parametrize("lang", list(SCAM))
def test_scam_scores_higher_than_genuine_alert(lang):
    assert rule_score_text(SCAM[lang]) > rule_score_text(GENUINE)


def test_safety_warning_is_recognised():
    assert find_signals(GENUINE)["Safety warning"] is True


def test_normaliser_keeps_hindi_and_tamil_characters():
    assert "खाता" in normalise("आपका खाता तुरंत बंद")
    assert "உங்கள்" in normalise("உங்கள் கணக்கு இன்று")


def test_normaliser_masks_urls_and_numbers():
    out = normalise("Pay Rs 5000 at http://hdfc-verify.xyz now")
    assert "urltoken" in out and "numtoken" in out and "5000" not in out


def test_message_classifier_ranks_scam_above_genuine():
    model = joblib.load(need("models/message_classifier.joblib"))["pipeline"]
    scores = model.predict_proba([SCAM["en"], SCAM["ta"], SCAM["hi"], GENUINE])[:, 1]
    assert scores[:3].mean() > scores[3]


# -------------------------------------------------------------- leakage guards
def test_behaviour_model_never_sees_labels_or_identifiers():
    art = joblib.load(need("models/fraud_detector.joblib"))
    bad = [c for c in art["feature_columns"] if any(w in c.lower() for w in FORBIDDEN)]
    assert not bad, f"label or identifier columns used as features: {bad}"


def test_fusion_uses_exactly_three_signals():
    fus = joblib.load(need("models/fusion_model.joblib"))
    assert list(fus["features"]) == ["Behaviour", "Message", "Network"]
    assert (np.asarray(fus["coef"]) > 0).all()             # more evidence never lowers the score


# ------------------------------------------------------------ fusion report
@pytest.fixture(scope="module")
def report():
    return pd.read_csv(need("evaluation/evidence_fusion_report.csv"))


def test_report_covers_every_payment(report):
    tx = pd.read_csv(need("data/synthetic/transactions.csv"))
    assert len(report) == len(tx) and report["transaction_id"].is_unique


def test_scores_are_in_range_and_decisions_match_thresholds(report):
    for c in ["transaction_risk_score", "nlp_risk_score", "graph_risk_score", "unified_risk_score"]:
        assert report[c].between(0, 1).all(), c
    s = report["unified_risk_score"] * 100
    expect = np.where(s >= 70, "BLOCK_TRANSACTION", np.where(s >= 40, "MANUAL_REVIEW", "APPROVE"))
    assert (report["system_decision"].values == expect).mean() > 0.999   # 4-decimal rounding at the edges


def test_fusion_beats_each_single_signal_on_pr_auc():
    m = json.loads(need("evaluation/fusion_metrics.json").read_text())["test_metrics"]
    singles = [v["pr_auc"] for k, v in m.items() if k != "Fusion"]
    assert m["Fusion"]["pr_auc"] >= max(singles) - 0.01


# -------------------------------------------------------------- counterfactual
@pytest.fixture(scope="module")
def engine():
    need("models/fusion_model.joblib")
    from intelligence.counterfactual import CounterfactualEngine
    return CounterfactualEngine(ROOT)


def test_unchanged_payment_reproduces_the_original_score(engine):
    sample = engine.report.sample(50, random_state=0)
    for tid, row in sample.iterrows():
        assert abs(engine.simulate(tid)["before"]["score"] - row["unified_risk_score"] * 100) < 0.25


def test_removing_the_message_never_raises_the_score(engine):
    rep = engine.report
    for tid in rep[rep["nlp_risk_score"] > 0].index[:30]:
        r = engine.simulate(tid, drop_message=True)
        assert r["after"]["score"] <= r["before"]["score"] + 1e-6


def test_clearing_network_flags_never_raises_the_score(engine):
    rep = engine.report
    for tid in rep[(rep["net_ring"] == 1) | (rep["net_young"] == 1)].index[:30]:
        r = engine.simulate(tid, established=True, clear_device=True)
        assert r["after"]["score"] <= r["before"]["score"] + 1e-6


def test_no_changes_means_no_listed_changes(engine):
    assert engine.simulate(engine.report.index[0])["changes"] == []


# ------------------------------------------------------------------ explainer
@pytest.fixture(scope="module")
def explainer():
    need("evaluation/evidence_fusion_report.csv")
    from intelligence.explainer import Explainer
    return Explainer(ROOT)


def _reply(text):
    m = MagicMock()
    m.__enter__ = lambda s: s
    m.__exit__ = lambda *a: False
    m.read = lambda *a, **k: json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}}]}).encode()
    return m


def test_without_a_key_the_rule_based_summary_is_used(explainer):
    r = explainer.explain(explainer.report.index[0], "English", None)
    assert r["source"] == "Rule-based" and "risk" in r["text"].lower()


def test_invented_numbers_are_rejected(explainer):
    tid = explainer.report.index[0]
    ev = explainer.evidence(tid)
    ok, bad = explainer.grounded(f"{tid} scored {ev['risk_score']}. Also 4711 victims were hurt.", ev)
    assert not ok and 4711.0 in bad


def test_numbers_from_the_evidence_are_accepted(explainer):
    tid = explainer.report.index[0]
    ev = explainer.evidence(tid)
    assert explainer.grounded(f"{tid} has a risk score of {ev['risk_score']} out of 100.", ev)[0]


def test_gemini_reply_with_made_up_number_falls_back(explainer):
    tid = explainer.report.index[0]
    with patch("urllib.request.urlopen", return_value=_reply(f"{tid} affected 9999 people.")):
        r = explainer.explain(tid, "English", "KEY")
    assert r["source"] == "Rule-based"


@pytest.mark.parametrize("failure", [urllib.error.HTTPError("u", 429, "x", {}, None), TimeoutError()])
def test_api_failures_never_crash(explainer, failure):
    with patch("urllib.request.urlopen", side_effect=failure):
        r = explainer.explain(explainer.report.index[0], "Tamil", "KEY")
    assert r["source"] == "Rule-based" and r["text"]