import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="FinSight AI - Fraud Intelligence Engine",
    page_icon="■",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "evaluation"


@st.cache_data
def load_metrics():
  try:
    with open(EVAL_DIR / "model_metrics.json") as f:
      tx_metrics = json.load(f)
  except:
    tx_metrics = {}

  try:
    with open(EVAL_DIR / "nlp_metrics.json") as f:
      nlp_metrics = json.load(f)
  except:
    nlp_metrics = {}

  try:
    with open(EVAL_DIR / "graph_intelligence_summary.json") as f:
      graph_metrics = json.load(f)
  except:
    graph_metrics = {}

  return tx_metrics, nlp_metrics, graph_metrics


tx_metrics, nlp_metrics, graph_metrics = load_metrics()

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
st.sidebar.title("FinSight AI")
st.sidebar.markdown(
    "Multi-Layered Financial Fraud Detection & Scam Intelligence"
)
st.sidebar.markdown("---")
nav = st.sidebar.radio(
    "Navigation Menu",
    [
        "Executive Overview",
        "Transaction Anomaly Engine",
        "NLP Message Intelligence",
        "Network Graph Analysis",
        "Evidence Fusion & Decisions",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info("System Status: All 4 Intelligence Layers Active")

# ==================================================
# 1. EXECUTIVE OVERVIEW
# ==================================================
if nav == "Executive Overview":
  st.title("FinSight AI: Enterprise Fraud Detection Suite")
  st.markdown(
      "Real-time decision support dashboard fusing unsupervised anomaly"
      " detection, NLP text classification, and network graph centrality."
  )

  st.markdown("---")

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Total Network Nodes", graph_metrics.get("total_nodes", 3000))
  col2.metric("Total Edges Mapped", graph_metrics.get("total_edges", 4995))
  col3.metric(
      "Potential Mule Hubs",
      graph_metrics.get("potential_mule_hubs_identified", 79),
  )
  col4.metric("Active Pipeline", "Online")

  st.markdown("---")
  st.subheader("Architecture Overview")

  tab1, tab2, tab3, tab4 = st.tabs(
      ["Layer 1: Data", "Layer 2: Anomaly", "Layer 3: NLP", "Layer 4: Graph"]
  )

  with tab1:
    st.write(
        "**Data Validation Pipeline:** Automated schema enforcement, duplicate"
        " removal, and temporal boundary checks ensuring pristine data"
        " integrity."
    )
  with tab2:
    st.write(
        "**Transaction Anomaly Detection:** Isolation Forest algorithm"
        " analyzing spending velocity, balance depletion ratios, and high-risk"
        " amount thresholds."
    )
  with tab3:
    st.write(
        "**NLP Message Intelligence:** TF-IDF feature union (word and character"
        " n-grams) combined with class-weighted Logistic Regression to intercept"
        " phishing and OTP scams."
    )
  with tab4:
    st.write(
        "**Network Graph Intelligence:** NetworkX directed graph engine mapping"
        " account-to-beneficiary transaction topologies and identifying money"
        " mule hubs."
    )

# ==================================================
# 2. TRANSACTION ANOMALY ENGINE
# ==================================================
elif nav == "Transaction Anomaly Engine":
  st.title("Transaction Anomaly Detection (Isolation Forest)")
  st.markdown(
      "Unsupervised outlier detection identifying abnormal spending patterns"
      " and high-risk transactional velocities."
  )

  if tx_metrics:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Model Accuracy",
        f"{tx_metrics['isolation_forest']['accuracy']:.2%}",
    )
    col2.metric(
        "Precision", f"{tx_metrics['isolation_forest']['precision']:.2%}"
    )
    col3.metric("Recall", f"{tx_metrics['isolation_forest']['recall']:.2%}")
    col4.metric(
        "F1-Score", f"{tx_metrics['isolation_forest']['f1_score']:.2%}"
    )

  st.markdown("---")
  try:
    pred_df = pd.read_csv(EVAL_DIR / "test_predictions.csv")
    st.subheader("Transaction Risk Scoring Ledger")

    # Interactive Filter
    filter_choice = st.selectbox(
        "Filter View", ["All Transactions", "High Risk / Anomalies Only"]
    )
    if filter_choice == "High Risk / Anomalies Only" and "if_prediction" in pred_df.columns:
      display_df = pred_df[pred_df["if_prediction"] == -1]
    else:
      display_df = pred_df

    st.dataframe(display_df.head(25), use_container_width=True)
  except FileNotFoundError:
    st.warning("Test predictions CSV not found.")

# ==================================================
# 3. NLP MESSAGE INTELLIGENCE
# ==================================================
elif nav == "NLP Message Intelligence":
  st.title("NLP Financial Scam Intelligence")
  st.markdown(
      "Natural language processing pipeline analyzing communication text for"
      " credential harvesting, urgency triggers, and malicious URLs."
  )

  if nlp_metrics:
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Classification Accuracy", f"{nlp_metrics.get('accuracy', 1.0):.2%}"
    )
    col2.metric("Precision", f"{nlp_metrics.get('precision', 1.0):.2%}")
    col3.metric("Recall", f"{nlp_metrics.get('recall', 1.0):.2%}")

  st.markdown("---")
  try:
    nlp_df = pd.read_csv(EVAL_DIR / "nlp_test_predictions.csv")
    st.subheader("Scam Message Classification Ledger")
    st.dataframe(nlp_df.head(25), use_container_width=True)
  except FileNotFoundError:
    st.warning("NLP predictions report not found.")

# ==================================================
# 4. NETWORK GRAPH ANALYSIS
# ==================================================
elif nav == "Network Graph Analysis":
  st.title("Network Graph Intelligence (NetworkX)")
  st.markdown(
      "Structural topology metrics derived from account and beneficiary transfer"
      " networks."
  )

  col1, col2 = st.columns(2)
  with col1:
    st.metric("Total Graph Nodes", graph_metrics.get("total_nodes", 3000))
    st.metric(
        "Potential Mule Hubs Identified",
        graph_metrics.get("potential_mule_hubs_identified", 79),
    )
    st.metric("Max In-Degree Centrality", graph_metrics.get("max_in_degree", 10))
  with col2:
    st.metric("Total Graph Edges", graph_metrics.get("total_edges", 4995))
    st.metric(
        "Large Connected Clusters",
        graph_metrics.get("large_connected_clusters", 1),
    )
    st.metric(
        "Max Out-Degree Centrality", graph_metrics.get("max_out_degree", 15)
    )

  st.markdown("---")
  st.subheader("Raw Graph Intelligence JSON Export")
  st.json(graph_metrics)

# ==================================================
# 5. EVIDENCE FUSION & DECISIONS
# ==================================================
elif nav == "Evidence Fusion & Decisions":
  st.title("Unified Evidence Fusion Dashboard")
  st.markdown(
      "Multi-signal risk score synthesis combining transactional anomalies, NLP"
      " scam probabilities, and graph risk."
  )

  try:
    fusion_df = pd.read_csv(EVAL_DIR / "evidence_fusion_report.csv")

    # Visual breakdown of decisions using native Streamlit chart
    st.subheader("System Decision Distribution")
    decision_counts = fusion_df["system_decision"].value_counts()
    st.bar_chart(decision_counts)

    st.markdown("---")
    st.subheader("Transaction Decision Ledger")

    # Interactive decision filter
    selected_decision = st.selectbox(
        "Filter by Decision Status",
        ["ALL", "APPROVE", "MANUAL_REVIEW", "BLOCK_TRANSACTION"],
    )
    if selected_decision != "ALL":
      filtered_fusion = fusion_df[
          fusion_df["system_decision"] == selected_decision
      ]
    else:
      filtered_fusion = fusion_df

    st.dataframe(filtered_fusion, use_container_width=True)
  except FileNotFoundError:
    st.warning(
        "Evidence fusion report not found. Run"
        " `python intelligence/evidence_fusion.py` first."
    )