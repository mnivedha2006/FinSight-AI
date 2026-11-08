import json
from pathlib import Path
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "transaction_features.csv"
RAW_TX_PATH = BASE_DIR / "data" / "synthetic" / "transactions.csv"
MODEL_DIR = BASE_DIR / "models"
EVALUATION_DIR = BASE_DIR / "evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


def select_features(df):
  """Selects numerical model features while strictly excluding identifiers,

  timestamps, and any column containing 'fraud' to prevent data leakage.
  """
  exclude_keywords = ["id", "timestamp", "fraud", "scenario", "receiver", "sender"]
  feature_cols = []
  for col in df.columns:
    # Exclude if any keyword matches
    if any(exc in col.lower() for exc in exclude_keywords):
      continue
    # Keep numeric features only
    if pd.api.types.is_numeric_dtype(df[col]):
      feature_cols.append(col)
  return feature_cols


def train_and_evaluate():
  print("Loading processed features...")
  if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed features not found at {DATA_PATH}. Complete Day 2 first!"
    )

  df = pd.read_csv(DATA_PATH)

  # Ensure is_fraud target label is available for evaluation
  if "is_fraud" not in df.columns:
    print(
        "is_fraud not found in processed features. Merging from raw"
        " transactions..."
    )
    if RAW_TX_PATH.exists():
      raw_tx = pd.read_csv(RAW_TX_PATH)
      if "transaction_id" in df.columns and "transaction_id" in raw_tx.columns:
        df = df.merge(
            raw_tx[["transaction_id", "is_fraud"]],
            on="transaction_id",
            how="left",
        )
      else:
        df["is_fraud"] = raw_tx["is_fraud"].values
    else:
      raise ValueError("Target label 'is_fraud' could not be located.")

  # Feature selection with leakage protection
  feature_cols = select_features(df)
  print(f"Selected model features ({len(feature_cols)}): {feature_cols}")

  X = df[feature_cols]
  y = df["is_fraud"].fillna(0).astype(int)

  # Stratified train/test split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  # --------------------------------------------------
  # 2. ISOLATION FOREST MODEL
  # --------------------------------------------------
  print("\nTraining Isolation Forest anomaly detection model...")
  pipeline = Pipeline(
      [
          ("imputer", SimpleImputer(strategy="median")),
          (
              "model",
              IsolationForest(
                  n_estimators=200, contamination=0.05, random_state=42
              ),
          ),
      ]
  )

  # Fit model on training features (unsupervised: y_train is not used for fitting)
  pipeline.fit(X_train)

  # Predictions: -1 for outliers/fraud, 1 for normal
  test_preds_raw = pipeline.predict(X_test)
  y_pred_if = np.where(test_preds_raw == -1, 1, 0)

  # Anomaly scores for PR-AUC calculation
  test_scores_raw = pipeline.named_steps["model"].decision_function(
      pipeline.named_steps["imputer"].transform(X_test)
  )
  if_scores = -test_scores_raw  # Higher score = more anomalous

  # --------------------------------------------------
  # 3. RULE-BASED BASELINE
  # --------------------------------------------------
  print("Evaluating rule-based baseline...")
  rule_preds = np.zeros(len(X_test), dtype=int)
  if "txn_velocity_1h" in X_test.columns:
    rule_preds |= (X_test["txn_velocity_1h"] >= 3).values
  if "is_new_beneficiary" in X_test.columns:
    rule_preds |= (X_test["is_new_beneficiary"] == 1).values
  if "balance_depletion_ratio" in X_test.columns:
    rule_preds |= (X_test["balance_depletion_ratio"] > 0.8).values

  # --------------------------------------------------
  # 4. METRICS & EVALUATION
  # --------------------------------------------------
  def evaluate_metrics(y_true, y_pred, scores=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = (
        average_precision_score(y_true, scores) if scores is not None else 0.0
    )
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "pr_auc": float(pr_auc),
        "false_positive_rate": float(fpr),
    }

  if_metrics = evaluate_metrics(y_test, y_pred_if, if_scores)
  rule_metrics = evaluate_metrics(y_test, rule_preds)

  print("\n=== EVALUATION RESULTS ===")
  print("Isolation Forest:", json.dumps(if_metrics, indent=2))
  print("Rule Baseline:", json.dumps(rule_metrics, indent=2))

  # --------------------------------------------------
  # 5. SAVE ARTIFACTS
  # --------------------------------------------------
  model_package = {
      "pipeline": pipeline,
      "feature_columns": feature_cols,
      "model_type": "IsolationForest",
  }
  model_path = MODEL_DIR / "fraud_detector.joblib"
  joblib.dump(model_package, model_path)
  print(f"\nModel saved to {model_path}")

  metrics_output = {
      "isolation_forest": if_metrics,
      "rule_baseline": rule_metrics,
  }
  metrics_path = EVALUATION_DIR / "model_metrics.json"
  with open(metrics_path, "w") as f:
    json.dump(metrics_output, f, indent=2)
  print(f"Metrics saved to {metrics_path}")

  # Save test predictions
  original = pd.read_csv(DATA_PATH)
  results = X_test.copy()
  results["true_label"] = y_test.values
  results["if_prediction"] = y_pred_if
  results["rule_prediction"] = rule_preds
  results["if_score"] = if_scores

  if "transaction_id" in original.columns:
    results["transaction_id"] = original.loc[X_test.index, "transaction_id"]

  pred_path = EVALUATION_DIR / "test_predictions.csv"
  results.to_csv(pred_path, index=False)
  print(f"Test predictions saved to {pred_path}")

  # Confusion matrix plot
  fig, ax = plt.subplots(figsize=(6, 6))
  ConfusionMatrixDisplay.from_predictions(
      y_test, y_pred_if, ax=ax, cmap="Blues", colorbar=False
  )
  plt.title("Isolation Forest Confusion Matrix")
  cm_path = EVALUATION_DIR / "confusion_matrix.png"
  plt.savefig(cm_path, bbox_inches="tight")
  plt.close()
  print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
  train_and_evaluate()