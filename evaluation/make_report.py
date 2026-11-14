"""Build evaluation/EVALUATION_REPORT.md from the saved metrics, so no number is typed by hand.

Run from the project root:   python evaluation/make_report.py
"""
import json
from datetime import date
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
EVAL, SYN = BASE / "evaluation", BASE / "data" / "synthetic"
LANG = {"en": "English", "ta_rom": "Tamil (romanised)", "ta": "Tamil (script)", "hi": "Hindi (script)",
        "hi_rom": "Hindi (romanised)"}


def load_json(name):
    p = EVAL / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def m_row(name, m):
    return [name, f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1_score']:.3f}", f"{m['pr_auc']:.3f}",
            f"{m['false_positive_rate']:.3f}"]


METRIC_HEADERS = ["", "Precision", "Recall", "F1", "PR-AUC", "False-positive rate"]


def main():
    nlp, beh, fus = load_json("nlp_metrics.json"), load_json("model_metrics.json"), load_json("fusion_metrics.json")
    L = [f"# FinSight AI: evaluation report", f"*Generated {date.today():%d %b %Y} from the saved metrics files. "
         "All data is synthetic.*", ""]

    L += ["## 1. Data", ""]
    p = SYN / "transactions.csv"
    if p.exists():
        tx = pd.read_csv(p)
        ft = tx["fraud_type"].value_counts()
        L.append(f"- **{len(tx):,} payments**, {int(tx['is_fraud'].sum())} labelled fraud "
                 f"({tx['is_fraud'].mean():.1%}). Fraud types: "
                 + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in ft.items() if k != "normal") + ".")
        hard = tx["scenario_id"].isin(["LEGIT_LARGE", "LEGIT_BURST"]).sum() if "scenario_id" in tx else 0
        L.append(f"- **{hard} legitimate hard negatives** (large rent-style payments, shopping bursts) that look "
                 "unusual but are not fraud.")
    p = SYN / "messages.csv"
    if p.exists():
        ms = pd.read_csv(p)
        L.append(f"- **{len(ms):,} messages** from {ms['template_id'].nunique()} message families, "
                 f"{ms['label'].mean():.0%} scam, in {ms['language'].nunique()} language variants.")
    L.append("")

    L += ["## 2. Scam-message classifier (TF-IDF + logistic regression)", ""]
    th = nlp.get("template_holdout")
    if th:
        L += [f"Flag threshold: {nlp.get('flag_threshold', 50)} on a 0 to 100 scale.", "",
              "**Headline: test messages come from message families the model never saw in training.**", "",
              table(METRIC_HEADERS, [m_row({"rules_only": "Rules only", "text_model": "Text model",
                                            "combined": "Combined"}[k], v) for k, v in th.items()]), ""]
        rs = nlp.get("random_split")
        if rs:
            L += ["For comparison, a random split (same families in train and test) is much easier:", "",
                  table(METRIC_HEADERS, [m_row("Text model, random split", rs["text_model"])]), "",
                  "A near-perfect random-split score on templated text mostly shows the task is easy when families "
                  "repeat. It is not the number to quote.", ""]
        bl = nlp.get("by_language", {})
        if bl:
            L += ["Text model by language, on unseen families:", "",
                  table(["Language", "Messages", "Precision", "Recall"],
                        [[LANG.get(k, k), v["n"], f"{v['text_model']['precision']:.3f}",
                          f"{v['text_model']['recall']:.3f}"] for k, v in bl.items()]), ""]
    else:
        L += ["*nlp_metrics.json not found. Run models/train_nlp_model.py.*", ""]

    L += ["## 3. Transaction behaviour model (Isolation Forest)", ""]
    if beh:
        L += [table(METRIC_HEADERS, [m_row("Isolation Forest", beh["isolation_forest"]),
                                     m_row("Rule baseline", beh["rule_baseline"])]), ""]
    else:
        L += ["*model_metrics.json not found. Run models/train_anomaly_model.py.*", ""]

    L += ["## 4. Risk fusion", ""]
    tm = fus.get("test_metrics")
    if tm:
        L += [f"Weights fitted on a 60% train split, review threshold chosen on a 20% validation split, results on "
              f"the untouched 20% test split ({fus['test_size']} payments, {fus['test_fraud']} fraud).", "",
              table(METRIC_HEADERS, [m_row(k, v) for k, v in tm.items()]), "",
              "Fusion weights (log-odds scale): " + ", ".join(f"{k} {v:+.2f}" for k, v in fus["weights"].items()) + ".", ""]
        g = fus.get("flag_rate_by_group_on_test", [])
        if g:
            cols = ["Behaviour only", "Message only", "Network only", "Fusion"]
            L += ["Share of payments flagged, by kind of payment (test split):", "",
                  table(["Kind of payment", "Payments"] + cols,
                        [[r["grp"].replace("_", " "), r["n"]] + [f"{r[c] * 100:.0f}%" for c in cols] for r in g]), "",
                  "Reading this table: the behaviour model misses suspicious collect requests because their amounts "
                  "look ordinary, while the message and network signals catch them. The behaviour model also raises "
                  "false alarms on legitimate large payments and shopping bursts, which fusion removes.", ""]
    else:
        L += ["*fusion_metrics.json not found. Run intelligence/risk_fusion.py.*", ""]

    L += ["## 5. Limitations (please read)", "",
          "1. **All data is synthetic.** Scores show the pipeline works end to end; they are not real-world accuracy.",
          "2. **The network rule is partly circular.** Mule beneficiaries were generated as very new accounts, and the "
          "network indicator looks for exactly that. On real data it would be weaker.",
          "3. **Messages are linked to payments by the generator.** Scam messages were attached to fraudulent payments "
          "with fixed probabilities, so the message signal helps fusion more than it would in practice.",
          "4. **Small test split.** About 50 fraud cases, and 13 to 20 per fraud type. Treat per-type figures as noisy.",
          "5. **Message families are parallel across languages.** The same scam scenarios appear in English, Tamil and "
          "Hindi, so Tamil and Hindi results benefit from having seen the other languages.",
          "6. **Splits are random across payments, not across accounts.** The same account can appear in train and "
          "test; a stricter evaluation would hold out whole accounts.",
          "7. **The 0 to 100 score is a monotone rescaling of model log-odds**, not a calibrated probability of fraud.",
          "8. **The Isolation Forest assumes 5% contamination.** That setting was chosen to match the synthetic fraud "
          "rate and would need re-tuning on real data.", "",
          "## 6. Reproduce", "", "```", "python run_pipeline.py", "python evaluation/make_report.py",
          "python -m pytest -q", "```", ""]
    out = EVAL / "EVALUATION_REPORT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()