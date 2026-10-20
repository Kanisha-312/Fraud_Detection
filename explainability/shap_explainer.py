"""
explainability/shap_explainer.py

Explains fraud predictions using two methods:
  1. Graph-level: edge importance via local subgraph analysis
  2. Feature-level: gradient-based attribution (replaces KernelSHAP)

Root-cause fixes applied:
  - KernelSHAP removed: it requires transaction_idx to be valid in the
    training graph, which is never true for streaming transactions.
    Replaced with input-gradient attribution which works on any feature
    vector without needing a graph index.
  - feature_arr now sanitizes nulls from Spark schema (None, NaN, inf)
    before building the feature vector, fixing the all-zeros input bug.
  - fallback_feature_importance now uses RAW (unscaled) features so that
    the weight multiplication produces meaningful magnitudes.
  - amount field extraction moved to pipeline so it reads directly from
    the raw Kafka row before Spark nulls it out.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

log = logging.getLogger("shap_explainer")

class FraudExplainer:
    """Explains fraud predictions for streaming transactions."""
    
    def __init__(self):
        self._model       = None
        self._graph_data  = None
        self._node_maps   = None
        self._scaler      = None
        self._feature_cols: Optional[List[str]] = None
        self._device      = None

    def _ensure_loaded(self) -> None:
        """Lazy-load model and graph artifacts on first call."""
        if self._model is not None:
            return
            
        from graph.graph_builder import load_graph
        from graph.gnn_model import load_model

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._graph_data, self._node_maps, self._scaler, self._feature_cols = load_graph()
        self._model = load_model(self._graph_data)
        self._model = self._model.to(self._device)
        self._model.eval()
        
        log.info("FraudExplainer loaded — device: %s, features: %d", 
                 self._device, len(self._feature_cols))

    def _get_feature_names(self) -> List[str]:
        return self._feature_cols or list(config.GRAPH_NODE_FEATURES)

    # ── Feature extraction ─────────────────────────────────────────────────

    def _extract_raw_features(self, transaction_data: Dict) -> np.ndarray:
        """
        Extract raw (unscaled) feature values from a transaction dict.
        Handles all the ways Spark/JSON can deliver nulls:
          None, float('nan'), 'nan', '', 'null', 'None'
        Falls back to 0.0 for any missing or unparseable value.
        Returns array of shape [1, num_features].
        """
        _NULL_STRINGS = {"nan", "null", "none", ""}
        features = []
        
        for col in self._get_feature_names():
            val = 0.0
            raw = transaction_data.get(col)
            if raw is not None:
                try:
                    f = float(raw)
                    if not (np.isnan(f) or np.isinf(f)):
                        val = f
                except (TypeError, ValueError):
                    if str(raw).strip().lower() not in _NULL_STRINGS:
                        log.debug("Cannot parse feature %s=%r as float", col, raw)
            features.append(val)
            
        arr = np.array(features, dtype=np.float32).reshape(1, -1)
        non_zero = int((arr != 0).sum())
        log.debug("Raw features: %d/%d non-zero", non_zero, len(features))
        return arr

    def _scale_features(self, raw_arr: np.ndarray) -> np.ndarray:
        """Scale raw features using the training scaler."""
        try:
            return self._scaler.transform(raw_arr)
        except Exception as e:
            log.warning("Scaler failed (%s) — using raw features", e)
            return raw_arr

    # ── Gradient-based feature attribution ────────────────────────────────

    def _gradient_attribution(
        self,
        raw_features: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute input-gradient × input attribution for feature importance.
        This method:
          1. Builds a mini single-transaction graph
          2. Runs a forward pass with gradients enabled
          3. Computes grad * input as the attribution signal
          4. Returns per-feature importance scores
        Unlike KernelSHAP, this works correctly for any transaction
        regardless of whether it appears in the training graph.
        """
        from inference.gnn_inference import _df_to_mini_graph, _ensure_loaded as _inf_load
        _inf_load()

        feature_names = self._get_feature_names()
        scaled = self._scale_features(raw_features)

        # Build a dummy row dict for the mini-graph builder
        dummy_row = {name: float(raw_features[0, i]) 
                     for i, name in enumerate(feature_names)}

        try:
            # Build mini graph with gradient tracking on transaction features
            mini_df = pd.DataFrame([dummy_row])
            mini_graph = _df_to_mini_graph(mini_df)
            mini_graph = mini_graph.to(self._device)

            # Enable gradient on transaction node features
            txn_feat = mini_graph["transaction"].x.detach().requires_grad_(True)
            mini_graph["transaction"].x = txn_feat

            # Forward pass
            prob = self._model(mini_graph.x_dict, mini_graph.edge_index_dict)
            prob[0].backward()

            # grad × input attribution
            grads = txn_feat.grad[0].cpu().numpy()          # shape [F]
            inputs = scaled[0]                               # shape [F]
            attribution = grads * inputs                     # element-wise

            n = min(len(feature_names), len(attribution))
            result = {feature_names[i]: float(attribution[i]) for i in range(n)}
            result = dict(sorted(result.items(), key=lambda x: abs(x[1]), reverse=True))
            
            non_zero = sum(1 for v in result.values() if abs(v) > 1e-6)
            log.info("Gradient attribution: %d/%d non-zero features", non_zero, len(result))
            return result

        except Exception as e:
            log.warning("Gradient attribution failed (%s) — using weight proxy", e)
            return self._weight_proxy_attribution(raw_features)

    def _weight_proxy_attribution(
        self,
        raw_features: np.ndarray,
    ) -> Dict[str, float]:
        """
        Fallback: use raw feature value × first-layer weight magnitude.
        Uses RAW (unscaled) features so values are not near-zero.
        """
        feature_names = self._get_feature_names()
        features_flat = raw_features.flatten()[:len(feature_names)]
        
        try:
            first_proj = list(self._model.input_proj["transaction"].parameters())[0]
            weights = first_proj.detach().cpu().numpy()        # [hidden, F]
            weight_mag = np.abs(weights).mean(axis=0)          # [F]
            importance = np.abs(features_flat) * weight_mag[:len(features_flat)]
        except Exception:
            importance = np.abs(features_flat)

        n = min(len(feature_names), len(importance))
        result = {feature_names[i]: float(importance[i]) for i in range(n)}
        
        # Normalize so largest = 1.0 (makes display meaningful)
        max_val = max(abs(v) for v in result.values()) if result else 1.0
        if max_val > 1e-8:
            result = {k: v / max_val for k, v in result.items()}
            
        return dict(sorted(result.items(), key=lambda x: abs(x[1]), reverse=True))

    # ── Graph-level explanation ────────────────────────────────────────────

    def _graph_explanation(self, transaction_data: Dict) -> List[Dict]:
        """
        Explain which graph connections (card, merchant, email domain)
        are associated with high fraud rates in the training data.
        Works for any transaction — does not require a graph index.
        """
        connections = []
        card_val = transaction_data.get("card1")
        merchant_val = str(transaction_data.get("ProductCD") or "UNKNOWN")
        email_val = str(transaction_data.get("P_emaildomain") or "UNKNOWN")

        checks = [
            ("card", "card1", card_val, "uses_card"),
            ("merchant", "ProductCD", merchant_val, "at_merchant"),
            ("email_domain", "P_emaildomain", email_val, "uses_email"),
        ]

        for node_type, col, val, rel in checks:
            try:
                if val is None:
                    continue

                if node_type == "card":
                    node_idx = self._node_maps[node_type].get(int(float(val)), None)
                else:
                    node_idx = self._node_maps[node_type].get(str(val), None)

                if node_idx is None:
                    connections.append({
                        "connected_to": str(val),
                        "edge_type": rel,
                        "dst_type": node_type,
                        "importance": 0.0,
                        "reason": "First-time entity — no history in training data",
                    })
                    continue

                # Find all transactions connected to this node
                fwd_key = None
                for k in self._graph_data.edge_index_dict:
                    if k[0] == "transaction" and k[2] == node_type:
                        fwd_key = k
                        break

                if fwd_key is None:
                    continue

                edge_idx = self._graph_data.edge_index_dict[fwd_key]
                
                # Transactions connected to this node
                conn_mask = edge_idx[1] == node_idx
                conn_txn_ids = edge_idx[0][conn_mask]

                if len(conn_txn_ids) == 0:
                    fraud_rate = 0.0
                    count = 0
                else:
                    labels = self._graph_data["transaction"].y[conn_txn_ids].float()
                    fraud_rate = labels.mean().item()
                    count = len(conn_txn_ids)

                importance = round(fraud_rate, 4)
                connections.append({
                    "connected_to": str(val),
                    "edge_type": rel,
                    "dst_type": node_type,
                    "importance": importance,
                    "reason": (
                        f"{node_type.replace('_', ' ').title()} has "
                        f"{fraud_rate:.1%} fraud rate across {count} transactions"
                    ),
                })
            except Exception as e:
                log.debug("Graph explanation error for %s=%s: %s", node_type, val, e)

        connections.sort(key=lambda x: x["importance"], reverse=True)
        return connections

    # ── Main entry point ───────────────────────────────────────────────────

    def explain_transaction(
        self,
        transaction_id: Any,
        fraud_probability: float,
        transaction_data: Optional[Dict] = None,
        transaction_idx: Optional[int] = None,
    ) -> Dict:
        """
        Produce a full explanation for a flagged transaction.
        Args:
            transaction_id:    TransactionID value
            fraud_probability: Model output probability (0-1)
            transaction_data:  Raw row dict from Kafka/Spark
            transaction_idx:   Ignored (kept for API compatibility)
        Returns:
            Structured explanation dict with top_features, 
            suspicious_connections, and human_readable_summary.
        """
        self._ensure_loaded()
        top_features = []
        suspicious_connections = []

        if transaction_data is not None:
            # Extract raw features (handles all null variants from Spark)
            raw_arr = self._extract_raw_features(transaction_data)
            non_zero = int((raw_arr != 0).sum())
            log.info("Explaining txn %s — raw features: %d/%d non-zero", 
                     transaction_id, non_zero, raw_arr.shape[1])

            # Feature attribution
            attr_dict = self._gradient_attribution(raw_arr)

            # Format top 10 features
            for feat_name, score in list(attr_dict.items())[:10]:
                top_features.append({
                    "feature": feat_name,
                    "attribution_value": round(float(score), 4),
                    "direction": "increases_risk" if score > 0 else "decreases_risk",
                })

            # Graph explanation
            suspicious_connections = self._graph_explanation(transaction_data)

        # Human-readable summary
        summary = self._build_summary(
            transaction_id, fraud_probability, top_features, suspicious_connections
        )

        return {
            "transaction_id": transaction_id,
            "fraud_probability": round(fraud_probability, 4),
            "top_features": top_features,
            "suspicious_connections": suspicious_connections,
            "human_readable_summary": summary,
        }

    def _build_summary(
        self,
        transaction_id: Any,
        fraud_probability: float,
        top_features: List[Dict],
        suspicious_connections: List[Dict],
    ) -> str:
        """Build a human-readable explanation string."""
        parts = [
            f"Transaction {transaction_id} flagged with "
            f"{fraud_probability:.1%} fraud probability."
        ]

        # Only mention a feature if its attribution is non-trivial
        if top_features:
            top = top_features[0]
            if abs(top["attribution_value"]) > 1e-4:
                direction = "elevated" if top["direction"] == "increases_risk" else "low"
                parts.append(
                    f"Strongest signal: {top['feature']} "
                    f"(attribution: {top['attribution_value']:+.4f}), "
                    f"which is {direction} relative to training baseline."
                )

        # Mention highest-risk connection
        risky = [c for c in suspicious_connections if c.get("importance", 0) > 0.1]
        if risky:
            c = risky[0]
            parts.append(
                f"Connected to {c['connected_to']} ({c['dst_type']}) — {c['reason']}."
            )

        if fraud_probability > 0.8:
            parts.append("Recommendation: Block immediately.")
        elif fraud_probability > float(config.FRAUD_THRESHOLD):
            parts.append("Recommendation: Flag for manual review.")

        return " ".join(parts)


# ── Module-level singleton ─────────────────────────────────────────────────────

_explainer_instance: Optional[FraudExplainer] = None

def get_explainer() -> FraudExplainer:
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = FraudExplainer()
        _explainer_instance._ensure_loaded()
    return _explainer_instance

def explain_transaction(
    transaction_id: Any,
    fraud_probability: float,
    transaction_data: Optional[Dict] = None,
    transaction_idx: Optional[int] = None,
) -> Dict:
    """Module-level entry point for the streaming pipeline."""
    return get_explainer().explain_transaction(
        transaction_id, fraud_probability, transaction_data, transaction_idx
    )