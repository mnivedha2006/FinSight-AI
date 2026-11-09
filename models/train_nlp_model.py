import json
from pathlib import Path
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import FeatureUnion, Pipeline

warnings.filterwarnings("ignore")

# ==================================================
# 1. CONFIGURATION & PATHS
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "synthetic" / "messages.csv"
MODEL_DIR = BASE_DIR / "models"
EVALUATION_DIR = BASE_DIR / "evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


def detect_indicators(text):
  """Rule-based heuristic detector for scam indicators in message text."""
  text_lower = str(text).lower()

  urgency_keywords = ["urgent", "immediate", "blocked", "suspended", "expire", "today"]
  cred_keywords = ["otp", "pin", "password", "cvv", "creds", "credential"]

  has_urgency = any(kw in text_lower for kw in urgency_keywords)
  has_cred = any(kw in text_lower for kw in cred_keywords)
  has_url = "http" in text_lower or "bit.ly" in text_lower or "verify-" in text_lower

  return {
      "urgency": int(has_urgency),
      "credential_request": int(has_cred),
      "suspicious_url": int(has_url),
  }


def calculate_rule_score(indicators):
  """Computes a heuristic rule-based risk score (0-100 scale)."""
  score = 0
  if indicators["urgency"]:
    score += 30
  if indicators["credential_request"]:
    score += 40
  if indicators["suspicious_url"]:
    score += 30
  return score


def train_nlp_classifier():
  print("Loading message dataset...")
  if not DATA_PATH.exists():
    raise FileNotFoundError(f"Message dataset not found at {DATA_PATH}.")

  df = pd.read_csv(DATA_PATH)
  print(f"Dataset shape: {df.shape}")
  print(f"Columns available: {df.columns.tolist()}")

  # Automatically detect text and label columns
  text_col = None
  for col in ["sender_text", "message_text", "text", "content", "message"]:
    if col in df.columns:
      text_col = col
      break

  label_col = None
  for col in ["is_scam", "label", "target", "fraud_label"]:
    if col in df.columns:
      label_col = col
      break

  if text_col is None or label_col is None:
    raise ValueError(
        f"Could not auto-detect text or label columns. Found columns:"
        f" {df.columns.tolist()}"
    )

  print(f"Using text column: '{text_col}' | Using label column: '{label_col}'")

  X = df[text_col].astype(str)
  y = df[label_col].fillna(0).astype(int)

  # Train/test split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  print(f"Training set: {len(X_train)} messages | Test set: {len(X_test)} messages")

  # --------------------------------------------------
  # 2. FEATURE UNION (Word + Character TF-IDF)
  # --------------------------------------------------
  print("Building TF-IDF vectorizers (Word + Character n-grams)...")
  vectorizer = FeatureUnion([
      (
          "word_tfidf",
          TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english"),
      ),
      (
          "char_tfidf",
          TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=5000),
      ),
  ])

  # --------------------------------------------------
  # 3. LOGISTIC REGRESSION MODEL PIPELINE
  # --------------------------------------------------
  print("Training Logistic Regression scam classifier...")
  pipeline = Pipeline([
      ("vectorizer", vectorizer),
      (
          "classifier",
          LogisticRegression(class_weight="balanced", C=1.0, random_state=42),
      ),
  ])

  pipeline.fit(X_train, y_train)

  # Predictions
  y_pred = pipeline.predict(X_test)
  y_proba = pipeline.predict_proba(X_test)[:, 1]

  # Metrics evaluation
  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred, zero_division=0)
  rec = recall_score(y_test, y_pred, zero_division=0)
  f1 = f1_score(y_test, y_pred, zero_division=0)
  pr_auc = average_precision_score(y_test, y_proba)
  cm = confusion_matrix(y_test, y_pred)
  tn, fp, fn, tp = cm.ravel()
  fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

  metrics = {
      "accuracy": float(acc),
      "precision": float(prec),
      "recall": float(rec),
      "f1_score": float(f1),
      "pr_auc": float(pr_auc),
      "false_positive_rate": float(fpr),
  }

  print("\n=== NLP MODEL EVALUATION RESULTS ===")
  print(json.dumps(metrics, indent=2))
  print("\nClassification Report:")
  print(classification_report(y_test, y_pred))

  # --------------------------------------------------
  # 4. SAVE ARTIFACTS
  # --------------------------------------------------
  model_path = MODEL_DIR / "message_classifier.joblib"
  joblib.dump({"pipeline": pipeline}, model_path)
  print(f"\nMessage classifier saved to {model_path}")

  metrics_path = EVALUATION_DIR / "nlp_metrics.json"
  with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
  print(f"NLP metrics saved to {metrics_path}")

  # Save test predictions with rule indicators
  results_df = pd.DataFrame({
      "message": X_test.values,
      "true_label": y_test.values,
      "predicted_label": y_pred,
      "scam_probability": y_proba,
  })
  
  # Add rule scores
  rule_scores = []
  for msg in X_test.values:
    inds = detect_indicators(msg)
    rule_scores.append(calculate_rule_score(inds))
  results_df["rule_score"] = rule_scores

  pred_path = EVALUATION_DIR / "nlp_test_predictions.csv"
  results_df.to_csv(pred_path, index=False)
  print(f"NLP test predictions saved to {pred_path}")

  # Confusion matrix plot
  fig, ax = plt.subplots(figsize=(6, 6))
  ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Purples", colorbar=False)
  plt.title("NLP Scam Classifier Confusion Matrix")
  cm_path = EVALUATION_DIR / "nlp_confusion_matrix.png"
  plt.savefig(cm_path, bbox_inches="tight")
  plt.close()
  print(f"NLP confusion matrix saved to {cm_path}")


if __name__ == "__main__":
  train_nlp_classifier()