import json
from pathlib import Path
import networkx as nx
import pandas as pd

# ==================================================
# 1. CONFIGURATION & PATHS
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]  # Points to FinSight-AI root
SYNTH_DIR = BASE_DIR / "data" / "synthetic"
EVALUATION_DIR = BASE_DIR / "evaluation"
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


def build_and_analyze_graph():
  print("Loading synthetic datasets for graph analysis...")
  transactions = pd.read_csv(SYNTH_DIR / "transactions.csv")
  accounts = pd.read_csv(SYNTH_DIR / "accounts.csv")
  beneficiaries = pd.read_csv(SYNTH_DIR / "beneficiaries.csv")

  print(f"Loaded {len(transactions)} transactions for network mapping.")

  # Initialize Directed Graph
  G = nx.DiGraph()

  # Add nodes with attributes if available
  for _, row in accounts.iterrows():
    G.add_node(
        row["account_id"],
        node_type="account",
        risk_score=row.get("risk_score", 0),
    )

  for _, row in beneficiaries.iterrows():
    G.add_node(
        row["beneficiary_id"],
        node_type="beneficiary",
        bank=row.get("bank_name", "Unknown"),
    )

  # Add edges from transactions
  print("Constructing transaction edges...")
  for _, row in transactions.iterrows():
    sender = row["sender_id"]
    receiver = row["receiver_id"]
    amount = row["amount"]
    is_fraud = row.get("is_fraud", 0)

    # Add edge with attributes
    G.add_edge(
        sender,
        receiver,
        weight=amount,
        is_fraud=is_fraud,
        txn_id=row["transaction_id"],
    )

  print(
      f"Graph built successfully! Total Nodes: {G.number_of_nodes()}, Total"
      f" Edges: {G.number_of_edges()}"
  )

  # --------------------------------------------------
  # 2. COMPUTE NETWORK METRICS
  # --------------------------------------------------
  print("Calculating graph centrality and risk indicators...")

  # In-degree (how many unique senders pay to a beneficiary or account)
  in_degrees = dict(G.in_degree())
  # Out-degree (how many unique transfers an account sends)
  out_degrees = dict(G.out_degree())

  # Find high in-degree nodes (potential money mule hubs)
  mule_candidates = [
      node
      for node, deg in in_degrees.items()
      if deg > 5 and G.nodes[node].get("node_type") == "beneficiary"
  ]

  # Weakly connected components (fraud rings / isolated transaction groups)
  components = list(nx.weakly_connected_components(G))
  fraud_rings = [list(comp) for comp in components if len(comp) > 10]

  graph_summary = {
      "total_nodes": G.number_of_nodes(),
      "total_edges": G.number_of_edges(),
      "potential_mule_hubs_identified": len(mule_candidates),
      "large_connected_clusters": len(fraud_rings),
      "max_in_degree": max(in_degrees.values()) if in_degrees else 0,
      "max_out_degree": max(out_degrees.values()) if out_degrees else 0,
  }

  print("\n=== GRAPH INTELLIGENCE SUMMARY ===")
  print(json.dumps(graph_summary, indent=2))

  # Save summary
  summary_path = EVALUATION_DIR / "graph_intelligence_summary.json"
  with open(summary_path, "w") as f:
    json.dump(graph_summary, f, indent=2)
  print(f"Graph intelligence summary saved to {summary_path}")


if __name__ == "__main__":
  build_and_analyze_graph()