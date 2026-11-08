from pathlib import Path
import pandas as pd


def engineer_message_features():
  BASE_DIR = Path(__file__).resolve().parents[1]
  SYNTH_DIR = BASE_DIR / "data" / "synthetic"
  PROC_DIR = BASE_DIR / "data" / "processed"
  PROC_DIR.mkdir(parents=True, exist_ok=True)

  print("Loading raw message data...")
  messages = pd.read_csv(SYNTH_DIR / "messages.csv")

  df = messages.copy()

  # Automatically detect the message text column name
  text_col = None
  for col in ["sender_text", "message_text", "text", "content", "message"]:
    if col in df.columns:
      text_col = col
      break

  if text_col is None:
    raise ValueError(
        f"Could not find message text column. Available columns:"
        f" {df.columns.tolist()}"
    )

  print(f"Using '{text_col}' as the message text column.")
  print("Engineering NLP text features...")

  # 1. Message length
  df["message_length"] = df[text_col].astype(str).str.len()

  # 2. Urgency keyword flag
  urgency_keywords = ["urgent", "immediate", "blocked", "suspended", "expire"]
  df["urgency_flag"] = (
      df[text_col]
      .astype(str)
      .str.lower()
      .str.contains("|".join(urgency_keywords))
      .astype(int)
  )

  # 3. OTP / PIN / Password request flag
  credential_keywords = ["otp", "pin", "password", "cvv", "creds"]
  df["credential_request_flag"] = (
      df[text_col]
      .astype(str)
      .str.lower()
      .str.contains("|".join(credential_keywords))
      .astype(int)
  )

  # 4. Suspicious URL flag
  df["suspicious_url_flag"] = (
      df[text_col]
      .astype(str)
      .str.contains("http|bit.ly|verify-", case=False)
      .astype(int)
  )

  output_path = PROC_DIR / "message_features.csv"
  df.to_csv(output_path, index=False)
  print(f"Message feature engineering successful! Saved to {output_path}")
  print(f"Total message features shape: {df.shape}")


if __name__ == "__main__":
  engineer_message_features()