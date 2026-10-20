"""
graph/graph_builder.py
Builds a heterogeneous PyTorch Geometric graph from the IEEE-CIS dataset.

Node types:
  - transaction  (each row is a node; labeled with isFraud)
  - card         (unique card1 values)
  - merchant     (unique ProductCD values)
  - email_domain (unique P_emaildomain values)

Edge types:
  - transaction → card         (via card1)
  - transaction → merchant     (via ProductCD)
  - transaction → email_domain (via P_emaildomain)
  + reverse edges for all of the above

Saves the built graph to models/graph_data.pt
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger("graph_builder")


def load_raw_data() -> pd.DataFrame:
    """Load and merge the IEEE-CIS transaction and identity CSVs."""
    log.info("Loading transaction CSV...")

    # Explicit dtypes cut memory usage by ~60%
    txn_dtypes = {
        'TransactionID': 'int32',
        'isFraud': 'int8',
        'TransactionDT': 'int32',
        'TransactionAmt': 'float32',
        'card1': 'float32', 'card2': 'float32',
        'card3': 'float32', 'card5': 'float32',
        'addr1': 'float32', 'addr2': 'float32',
        'dist1': 'float32', 'dist2': 'float32',
    }
    # Add float32 for all C, D, V columns
    for i in range(1, 15):
        txn_dtypes[f'C{i}'] = 'float32'
    for i in range(1, 16):
        txn_dtypes[f'D{i}'] = 'float32'
    for i in range(1, 340):
        txn_dtypes[f'V{i}'] = 'float32'

    txn = pd.read_csv(config.TRANSACTION_CSV, dtype=txn_dtypes)
    log.info("Loaded %d transactions", len(txn))

    if config.IDENTITY_CSV.exists():
        identity = pd.read_csv(config.IDENTITY_CSV)
        df = txn.merge(identity, on=config.TRANSACTION_ID_COL, how="left")
        log.info("Merged with identity — shape: %s", df.shape)
        del txn, identity
    else:
        df = txn
        del txn

    import gc
    gc.collect()

    # Use 100K rows — enough to train a solid GNN, fits in memory
    log.info("Sampling 100K transactions for graph building...")
    fraud = df[df['isFraud'] == 1]
    legit = df[df['isFraud'] == 0].sample(
        n=min(90000, len(df[df['isFraud'] == 0])),
        random_state=42
    )
    df = pd.concat([fraud, legit]).sample(frac=1, random_state=42).reset_index(drop=True)
    log.info("Final dataset: %d rows (fraud: %d, legit: %d)",
        len(df), len(fraud), len(legit))
    del fraud, legit
    gc.collect()

    return df


def build_node_maps(df: pd.DataFrame) -> dict:
    """
    Build integer-index maps for each non-transaction node type.

    Returns:
        dict with keys 'card', 'merchant', 'email_domain',
        each mapping original value → integer index.
    """
    card_vals = df["card1"].fillna(-1).astype(int).unique()
    merchant_vals = df["ProductCD"].fillna("UNKNOWN").unique()
    email_vals = df["P_emaildomain"].fillna("UNKNOWN").unique()

    node_maps = {
        "card": {v: i for i, v in enumerate(card_vals)},
        "merchant": {v: i for i, v in enumerate(merchant_vals)},
        "email_domain": {v: i for i, v in enumerate(email_vals)},
    }
    log.info(
        "Node maps — cards: %d, merchants: %d, email_domains: %d",
        len(node_maps["card"]),
        len(node_maps["merchant"]),
        len(node_maps["email_domain"]),
    )
    return node_maps


def build_transaction_features(df: pd.DataFrame) -> tuple:
    """
    Extract and normalize transaction node features.

    Returns:
        (feature_tensor, scaler) where feature_tensor has shape [N, F]
    """
    feature_cols = [c for c in config.GRAPH_NODE_FEATURES if c in df.columns]
    log.info("Using %d transaction node features: %s", len(feature_cols), feature_cols)

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return torch.tensor(X_scaled, dtype=torch.float), scaler, feature_cols


def build_secondary_node_features(df: pd.DataFrame, node_maps: dict) -> dict:
    """
    Build single-feature tensors for card, merchant, email_domain nodes.
    Feature = degree centrality (normalized count of connected transactions).

    Returns:
        dict mapping node_type → feature tensor of shape [num_nodes, 1]
    """
    node_features = {}

    for node_type, col, fill_val in [
        ("card", "card1", -1),
        ("merchant", "ProductCD", "UNKNOWN"),
        ("email_domain", "P_emaildomain", "UNKNOWN"),
    ]:
        mapping = node_maps[node_type]
        n_nodes = len(mapping)

        if col == "card1":
            counts = df[col].fillna(fill_val).astype(int).value_counts()
        else:
            counts = df[col].fillna(fill_val).value_counts()

        degree = np.zeros(n_nodes, dtype=np.float32)
        total = len(df)
        for val, cnt in counts.items():
            if val in mapping:
                degree[mapping[val]] = cnt / total

        node_features[node_type] = torch.tensor(degree, dtype=torch.float).unsqueeze(1)
        log.info("Node features for '%s': shape %s", node_type, node_features[node_type].shape)

    return node_features


def build_edges(df: pd.DataFrame, node_maps: dict) -> dict:
    """
    Build edge index tensors for each edge type.

    Returns:
        dict mapping (src_type, rel, dst_type) → edge_index tensor [2, E]
    """
    edges = {}
    n_txn = len(df)
    txn_ids = torch.arange(n_txn, dtype=torch.long)

    # transaction → card
    card_idx = df["card1"].fillna(-1).astype(int).map(
        lambda v: node_maps["card"].get(v, 0)
    ).values
    edges[("transaction", "uses_card", "card")] = torch.stack([
        txn_ids, torch.tensor(card_idx, dtype=torch.long)
    ])

    # transaction → merchant
    merchant_idx = df["ProductCD"].fillna("UNKNOWN").map(
        lambda v: node_maps["merchant"].get(v, 0)
    ).values
    edges[("transaction", "at_merchant", "merchant")] = torch.stack([
        txn_ids, torch.tensor(merchant_idx, dtype=torch.long)
    ])

    # transaction → email_domain
    email_idx = df["P_emaildomain"].fillna("UNKNOWN").map(
        lambda v: node_maps["email_domain"].get(v, 0)
    ).values
    edges[("transaction", "uses_email", "email_domain")] = torch.stack([
        txn_ids, torch.tensor(email_idx, dtype=torch.long)
    ])

    # Reverse edges
    reverse = {}
    for (src, rel, dst), edge_idx in edges.items():
        reverse[(dst, f"rev_{rel}", src)] = edge_idx.flip(0)
    edges.update(reverse)

    for etype, eidx in edges.items():
        log.info("Edges %s: %d edges", etype, eidx.shape[1])

    return edges


def build_graph(df: pd.DataFrame) -> HeteroData:
    """
    Build the full HeteroData graph object.

    Args:
        df: Merged IEEE-CIS DataFrame

    Returns:
        PyG HeteroData object with node features, edge indices, and labels
    """
    log.info("Building heterogeneous graph from %d transactions...", len(df))
    data = HeteroData()

    node_maps = build_node_maps(df)
    txn_features, scaler, feature_cols = build_transaction_features(df)
    secondary_features = build_secondary_node_features(df, node_maps)
    edges = build_edges(df, node_maps)

    # Assign node features
    data["transaction"].x = txn_features
    data["transaction"].y = torch.tensor(
        df[config.TARGET_COLUMN].fillna(0).values, dtype=torch.long
    )
    data["transaction"].transaction_ids = torch.tensor(
        df[config.TRANSACTION_ID_COL].values, dtype=torch.long
    )
    # TransactionDT (seconds since a reference point) — used for a
    # chronological train/val/test split instead of a random one, so
    # the model is evaluated the way it would actually be used: never
    # trained on anything that happened after what it's tested on.
    data["transaction"].transaction_dt = torch.tensor(
        df["TransactionDT"].fillna(0).values, dtype=torch.long
    )

    for node_type, feat in secondary_features.items():
        data[node_type].x = feat

    # Assign edges
    for (src, rel, dst), edge_idx in edges.items():
        data[src, rel, dst].edge_index = edge_idx

    # Print stats
    fraud_count = data["transaction"].y.sum().item()
    total = data["transaction"].y.shape[0]
    log.info("=" * 50)
    log.info("Graph statistics:")
    log.info("  Transaction nodes: %d", data["transaction"].x.shape[0])
    log.info("  Card nodes:        %d", data["card"].x.shape[0])
    log.info("  Merchant nodes:    %d", data["merchant"].x.shape[0])
    log.info("  Email domain nodes:%d", data["email_domain"].x.shape[0])
    log.info("  Fraud transactions: %d / %d (%.2f%%)", fraud_count, total, 100 * fraud_count / total)
    log.info("=" * 50)

    return data, node_maps, scaler, feature_cols


def save_artifacts(data: HeteroData, node_maps: dict, scaler, feature_cols: list) -> None:
    """Save graph, node maps, and scaler to disk."""
    torch.save(data, config.GRAPH_DATA_PATH)
    log.info("Graph saved to %s", config.GRAPH_DATA_PATH)

    with open(config.NODE_MAPS_PATH, "wb") as f:
        pickle.dump({"node_maps": node_maps, "feature_cols": feature_cols}, f)
    log.info("Node maps saved to %s", config.NODE_MAPS_PATH)

    import joblib
    joblib.dump(scaler, config.SCALER_PATH)
    log.info("Scaler saved to %s", config.SCALER_PATH)


def load_graph() -> tuple:
    """Load the pre-built graph and associated artifacts from disk."""
    if not config.GRAPH_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Graph not found at {config.GRAPH_DATA_PATH}. "
            "Run graph/graph_builder.py first."
        )
    # weights_only=False is required here: HeteroData is a custom class,
    # not a plain tensor, and torch>=2.6 defaults weights_only=True which
    # rejects it. Safe because this is our own locally-generated artifact,
    # not an externally-sourced checkpoint.
    data = torch.load(config.GRAPH_DATA_PATH, weights_only=False)

    with open(config.NODE_MAPS_PATH, "rb") as f:
        artifacts = pickle.load(f)

    import joblib
    scaler = joblib.load(config.SCALER_PATH)

    return data, artifacts["node_maps"], scaler, artifacts["feature_cols"]


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    df = load_raw_data()
    data, node_maps, scaler, feature_cols = build_graph(df)
    save_artifacts(data, node_maps, scaler, feature_cols)
    log.info("Graph building complete.")
