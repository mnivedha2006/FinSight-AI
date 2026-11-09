"""Train and honestly evaluate the scam-message classifier.

Run from the project root:   python models/train_nlp_model.py

Three views of performance are reported, all on held-out data:
  * template holdout  - test messages come from message FAMILIES the model never saw in training
                        (the headline number: closest we can get to "new scams" with synthetic data)
  * random split      - easier, in-distribution; shown only for comparison
  * by language       - so Tamil and Hindi are not hidden behind the English average
The saved model is refit on ALL messages after evaluation.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, confusion_matrix,  # noqa: E402
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from intelligence.message_rules import normalise, rule_score_text  # noqa: E402

warnings.filterwarnings("ignore")
DATA = BASE_DIR / "data" / "synthetic" / "messages.csv"
MODEL_DIR, EVAL_DIR = BASE_DIR / "models", BASE_DIR / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)
ML_W, RULE_W, THRESHOLD = 0.5, 0.5, 50  # same weights as the dashboard; flag at 50 on a 0-100 scale


def build_pipeline():
    vec = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalise, tokenizer=str.split, token_pattern=None,
                                 ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=normalise, analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 sublinear_tf=True)),
    ])
    return Pipeline([("vec", vec), ("clf", LogisticRegression(class_weight="balanced", C=3, max_iter=3000))])


def metrics(y, score):
    y, score = np.asarray(y), np.asarray(score)
    pred = (score >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y, pred, zero_division=0)), 4),
            "pr_auc": round(float(average_precision_score(y, score)), 4),
            "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4)}


def three_way(pipe, texts, y):
    ml = pipe.predict_proba(texts)[:, 1] * 100
    rule = np.array([rule_score_text(t) for t in texts], dtype=float)
    comb = ML_W * ml + RULE_W * rule
    return ml, rule, comb, {"rules_only": metrics(y, rule), "text_model": metrics(y, ml),
                            "combined": metrics(y, comb)}


def main():
    df = pd.read_csv(DATA)
    need = {"sender_text", "label", "template_id", "language"}
    if not need.issubset(df.columns):
        sys.exit(f"messages.csv needs columns {sorted(need)}. Run data_generation/generate_messages.py first.")
    X, y, groups = df["sender_text"].astype(str), df["label"].astype(int), df["template_id"]
    print(f"{len(df)} messages | {y.mean():.1%} scam | {groups.nunique()} message families")

    # 1) template holdout (headline)
    # pick a split where every language, with both classes, appears in the held-out families
    for seed in range(42, 342):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, y, groups))
        ok = y.iloc[te].nunique() == 2 and y.iloc[tr].nunique() == 2
        per_lang = df.iloc[te].groupby("language")["label"].nunique()
        if ok and len(per_lang) == df["language"].nunique() and (per_lang == 2).all():
            break
    pipe = build_pipeline().fit(X.iloc[tr], y.iloc[tr])
    ml, rule, comb, holdout = three_way(pipe, X.iloc[te], y.iloc[te])
    print(f"Held-out families: {groups.iloc[te].nunique()} of {groups.nunique()}")

    # 2) random split (easy, for comparison)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    _, _, _, random_split = three_way(build_pipeline().fit(Xtr, ytr), Xte, yte)

    # 3) by language, on the holdout set
    by_lang = {}
    te_df = df.iloc[te].assign(ml=ml, comb=comb)
    for lang, g in te_df.groupby("language"):
        if g["label"].nunique() == 2:
            by_lang[lang] = {"n": int(len(g)), "text_model": metrics(g["label"], g["ml"]),
                             "combined": metrics(g["label"], g["comb"])}

    # save outputs
    final = build_pipeline().fit(X, y)
    joblib.dump({"pipeline": final, "trained_on": len(df), "note": "refit on all messages after evaluation"},
                MODEL_DIR / "message_classifier.joblib")
    out = {**holdout["text_model"], "headline": "text model on held-out message families",
           "template_holdout": holdout, "random_split": random_split, "by_language": by_lang,
           "n_messages": int(len(df)), "n_families": int(groups.nunique()), "flag_threshold": THRESHOLD}
    (EVAL_DIR / "nlp_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({"message": X.iloc[te].values, "true_label": y.iloc[te].values,
                  "language": df["language"].iloc[te].values, "template_id": groups.iloc[te].values,
                  "ml_score": ml.round(1), "rule_score": rule, "combined_score": comb.round(1)}
                 ).to_csv(EVAL_DIR / "nlp_test_predictions.csv", index=False, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y.iloc[te], (ml >= THRESHOLD).astype(int), ax=ax, cmap="Blues",
                                            colorbar=False)
    ax.set_title("Text model on unseen message families")
    fig.savefig(EVAL_DIR / "nlp_confusion_matrix.png", bbox_inches="tight", dpi=130)

    def row(name, m):
        return (f"  {name:<12} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1_score']:.3f}  "
                f"PR-AUC {m['pr_auc']:.3f}  FPR {m['false_positive_rate']:.3f}")
    print("\nTEMPLATE HOLDOUT (headline, unseen message families)")
    for k, v in holdout.items():
        print(row(k, v))
    print("\nRANDOM SPLIT (easier, same families in train and test)")
    for k, v in random_split.items():
        print(row(k, v))
    print("\nBY LANGUAGE (text model, holdout)")
    for k, v in by_lang.items():
        print(f"  {k:<8} n={v['n']:<4} recall {v['text_model']['recall']:.3f}  precision {v['text_model']['precision']:.3f}")
    print("\nSaved model, nlp_metrics.json, nlp_test_predictions.csv, nlp_confusion_matrix.png")


if __name__ == "__main__":
    main()