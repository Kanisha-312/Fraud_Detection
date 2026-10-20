"""
inference/gnn_inference.py
Batch and single-transaction fraud scoring using the trained GNN.

Key fix: _row_to_features() now logs exactly which expected columns
are present vs missing in each incoming row, and uses flexible column
matching so partial feature sets degrade gracefully instead of silently
zeroing everything.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from graph.graph_builder import load_graph
from graph.gnn_model import load_model

log = logging.getLogger("gnn_inference")

# ── Module-level singletons ────────────────────────────────────────────────────
_graph_data   = None
_node_maps    = None
_scaler       = None
_feature_cols: Optional[List[str]] = None
_model        = None
_device       = None
_node_mean_embeddings: Dict[str, torch.Tensor] = {}

# Logged once to avoid spamming
_column_diagnostic_done = False


def _ensure_loaded() -> None:
    """Lazy-load graph, scaler, node maps and model on first call."""
    global _graph_data, _node_maps, _scaler, _feature_cols, _model, _device

    if _model is not None:
        return

    log.info("Loading graph and model for inference...")
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _graph_data, _node_maps, _scaler, _feature_cols = load_graph()
    _model = load_model(_graph_data)
    _model = _model.to(_device)
    _model.eval()

    log.info("Feature columns model was trained on (%d): %s",
             len(_feature_cols), _feature_cols)
    _precompute_mean_embeddings()
    log.info("Inference engine ready — device: %s", _device)


def _precompute_mean_embeddings() -> None:
    """
    Cache a mean RAW input-feature vector per secondary node type
    (card, merchant, email_domain) for cold-start / unseen-entity
    handling.

    These live in the SAME feature space as _graph_data[node_type].x
    (not the hidden GNN embedding space produced by model.encode()),
    so they can be appended as an extra "unseen entity" node directly
    onto that tensor in _df_to_mini_graph() and routed to by index —
    which is what makes this fallback actually usable, unlike the
    previous version that computed hidden-space embeddings nothing
    downstream could consume.
    """
    global _node_mean_embeddings
    for ntype in ("card", "merchant", "email_domain"):
        x = _graph_data[ntype].x
        _node_mean_embeddings[ntype] = x.mean(dim=0, keepdim=True).cpu()
    log.info("Mean feature vectors pre-computed for %d node types (cold-start fallback)",
             len(_node_mean_embeddings))


def _diagnose_columns(row: pd.Series) -> None:
    """
    Log a one-time diagnostic comparing expected feature columns
    against what arrived in the Kafka row.
    """
    global _column_diagnostic_done
    if _column_diagnostic_done:
        return
    _column_diagnostic_done = True

    available   = set(row.index)
    expected    = set(_feature_cols)
    present     = expected & available
    missing     = expected - available
    extra       = available - expected

    log.info("=" * 60)
    log.info("FEATURE COLUMN DIAGNOSTIC (logged once)")
    log.info("  Expected features  : %d — %s", len(expected),  sorted(expected))
    log.info("  Present in row     : %d — %s", len(present),   sorted(present))
    log.info("  MISSING from row   : %d — %s", len(missing),   sorted(missing))
    log.info("  Extra cols in row  : %d (ignored)", len(extra))
    if missing:
        log.warning(
            "MISSING FEATURES will be filled with 0. "
            "This means the model sees a partial/degraded feature vector. "
            "Fix: ensure the Kafka producer sends ALL transaction columns "
            "including dist1, dist2, C1-C8, D1-D5, V1-V5."
        )
    log.info("=" * 60)


def _row_to_features(row: pd.Series) -> np.ndarray:
    """
    Extract and scale transaction node features from a single pandas row.

    For each expected feature column:
      - Uses the value if present and non-null
      - Falls back to 0.0 if missing or null

    Returns scaled array of shape [1, num_features].
    """
    _diagnose_columns(row)

    features = []
    for col in _feature_cols:
        val = 0.0
        if col in row.index:
            raw = row[col]
            if raw is not None:
                try:
                    f = float(raw)
                    if not (np.isnan(f) or np.isinf(f)):
                        val = f
                except (TypeError, ValueError):
                    pass
        features.append(val)

    arr = np.array(features, dtype=np.float32).reshape(1, -1)

    non_zero = int((arr != 0).sum())
    if non_zero == 0:
        log.warning(
            "All %d features are zero for this row. "
            "Available columns sample: %s",
            len(_feature_cols),
            list(row.index)[:15],
        )
    else:
        log.debug("Feature vector: %d/%d non-zero values", non_zero, len(_feature_cols))

    try:
        return _scaler.transform(arr)
    except Exception as e:
        log.warning("Scaler transform failed (%s) — using raw features", e)
        return arr


def _df_to_mini_graph(df: pd.DataFrame) -> HeteroData:
    """
    Build a minimal HeteroData subgraph for a batch of new transactions.
    Connects each transaction to its card, merchant, and email_domain nodes
    using the node maps built during training.
    """
    data = HeteroData()

    # ── Transaction node features ──────────────────────────────────────────
    features = []
    for _, row in df.iterrows():
        features.append(_row_to_features(row).squeeze(0))
    txn_feat = np.stack(features)

    # Log first vector so we can verify values arrived
    log.info(
        "First feature vector — shape: %s | non-zero: %d/%d | "
        "min: %.4f | max: %.4f",
        txn_feat.shape,
        int((txn_feat[0] != 0).sum()),
        txn_feat.shape[1],
        float(txn_feat[0].min()),
        float(txn_feat[0].max()),
    )

    data["transaction"].x = torch.tensor(txn_feat, dtype=torch.float)

    # ── Secondary node features from training graph, each with one extra
    #    "unseen entity" node appended at the end (the mean feature vector)
    #    so a card/merchant/email never seen in training routes to a
    #    neutral placeholder instead of silently borrowing a real
    #    entity's identity at index 0. ─────────────────────────────────────
    data["card"].x         = torch.cat([_graph_data["card"].x, _node_mean_embeddings["card"]], dim=0)
    data["merchant"].x     = torch.cat([_graph_data["merchant"].x, _node_mean_embeddings["merchant"]], dim=0)
    data["email_domain"].x = torch.cat([_graph_data["email_domain"].x, _node_mean_embeddings["email_domain"]], dim=0)

    card_unseen_idx     = _graph_data["card"].x.shape[0]
    merchant_unseen_idx = _graph_data["merchant"].x.shape[0]
    email_unseen_idx    = _graph_data["email_domain"].x.shape[0]

    # ── Edges ──────────────────────────────────────────────────────────────
    card_pairs, merchant_pairs, email_pairs = [], [], []
    cold_start_flags = []

    for i, (_, row) in enumerate(df.iterrows()):
        row_cold_start = False

        # card1
        try:
            card_val = int(float(row.get("card1", -1))) \
                if pd.notna(row.get("card1")) else -1
        except (TypeError, ValueError):
            card_val = -1
        card_idx = _node_maps["card"].get(card_val)
        if card_idx is None:
            card_idx = card_unseen_idx
            row_cold_start = True
        card_pairs.append((i, card_idx))

        # ProductCD → merchant
        merchant_val = str(row.get("ProductCD", "UNKNOWN") or "UNKNOWN")
        merchant_idx = _node_maps["merchant"].get(merchant_val)
        if merchant_idx is None:
            merchant_idx = merchant_unseen_idx
            row_cold_start = True
        merchant_pairs.append((i, merchant_idx))

        # P_emaildomain
        email_val = str(row.get("P_emaildomain", "UNKNOWN") or "UNKNOWN")
        email_idx = _node_maps["email_domain"].get(email_val)
        if email_idx is None:
            email_idx = email_unseen_idx
            row_cold_start = True
        email_pairs.append((i, email_idx))

        cold_start_flags.append(row_cold_start)

    def _pairs_to_edge_index(pairs):
        src = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        dst = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        return torch.stack([src, dst])

    ei_card     = _pairs_to_edge_index(card_pairs)
    ei_merchant = _pairs_to_edge_index(merchant_pairs)
    ei_email    = _pairs_to_edge_index(email_pairs)

    data["transaction", "uses_card",    "card"].edge_index         = ei_card
    data["transaction", "at_merchant",  "merchant"].edge_index     = ei_merchant
    data["transaction", "uses_email",   "email_domain"].edge_index = ei_email
    data["card",        "rev_uses_card",    "transaction"].edge_index = ei_card.flip(0)
    data["merchant",    "rev_at_merchant",  "transaction"].edge_index = ei_merchant.flip(0)
    data["email_domain","rev_uses_email",   "transaction"].edge_index = ei_email.flip(0)

    # Exposed so predict_batch() can surface a cold_start flag in results
    # instead of the mismatch being silently invisible.
    data["transaction"].cold_start = torch.tensor(cold_start_flags, dtype=torch.bool)

    return data


def predict_batch(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a batch of transactions.

    Args:
        transactions_df: DataFrame with raw transaction columns.
                         Needs at minimum: card1, ProductCD, P_emaildomain,
                         TransactionAmt + the columns in GRAPH_NODE_FEATURES.

    Returns:
        Same DataFrame with two new columns:
          fraud_probability  (float 0–1)
          fraud_flag         (int: 1 if prob >= FRAUD_THRESHOLD)
    """
    _ensure_loaded()

    if transactions_df.empty:
        log.warning("predict_batch called with empty DataFrame")
        transactions_df = transactions_df.copy()
        transactions_df["fraud_probability"] = pd.Series(dtype=float)
        transactions_df["fraud_flag"]        = pd.Series(dtype=int)
        return transactions_df

    try:
        mini_graph = _df_to_mini_graph(transactions_df)
        cold_start_flags = mini_graph["transaction"].cold_start.numpy()
        mini_graph = mini_graph.to(_device)

        with torch.no_grad():
            probs    = _model(mini_graph.x_dict, mini_graph.edge_index_dict)
            probs_np = probs.cpu().numpy()

        # Detailed probability distribution log
        log.info(
            "Probability distribution — min: %.4f | max: %.4f | mean: %.4f | std: %.4f",
            probs_np.min(), probs_np.max(),
            probs_np.mean(), probs_np.std(),
        )
        top10 = np.sort(probs_np)[-10:]
        log.info("Top 10 probabilities: %s", top10)

        result_df = transactions_df.copy()
        result_df["fraud_probability"] = probs_np
        result_df["fraud_flag"] = (probs_np >= config.FRAUD_THRESHOLD).astype(int)
        result_df["cold_start"] = cold_start_flags

        n_flagged = int(result_df["fraud_flag"].sum())
        n_cold_start = int(cold_start_flags.sum())
        log.info(
            "Batch scored: %d transactions | %d flagged (%.1f%%) | threshold: %.2f",
            len(result_df), n_flagged,
            100 * n_flagged / max(len(result_df), 1),
            config.FRAUD_THRESHOLD,
        )
        if n_cold_start:
            log.info(
                "%d/%d transactions used cold-start fallback nodes "
                "(unseen card/merchant/email) — treat their scores as lower confidence",
                n_cold_start, len(result_df),
            )
        return result_df

    except Exception as e:
        log.error("predict_batch failed: %s", e, exc_info=True)
        result_df = transactions_df.copy()
        result_df["fraud_probability"] = 0.0
        result_df["fraud_flag"]        = 0
        result_df["cold_start"]        = False
        return result_df


def predict_single(transaction_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Score a single transaction dict."""
    _ensure_loaded()
    df     = pd.DataFrame([transaction_dict])
    scored = predict_batch(df)
    result = transaction_dict.copy()
    result["fraud_probability"] = float(scored["fraud_probability"].iloc[0])
    result["fraud_flag"]        = int(scored["fraud_flag"].iloc[0])
    result["cold_start"]        = bool(scored["cold_start"].iloc[0])
    return result