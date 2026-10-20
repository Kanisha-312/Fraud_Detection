"""
graph/gnn_model.py
HeteroGraphSAGE model for fraud detection using PyTorch Geometric.

Architecture:
  - HeteroConv wrapping SAGEConv for each edge type
  - 3 message-passing layers with ReLU + Dropout
  - Linear classifier on transaction node embeddings
  - get_embeddings() returns penultimate representations; not currently
    consumed by any explainer, but the natural entry point if
    torch_geometric.explain.GNNExplainer is integrated later
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger("gnn_model")


class HeteroFraudGNN(nn.Module):
    """
    Heterogeneous Graph SAGE Network for transaction fraud detection.

    Takes a heterogeneous graph with transaction, card, merchant, and
    email_domain nodes and classifies each transaction as fraud (1) or
    legitimate (0).

    Args:
        metadata: Tuple of (node_types, edge_types) from the HeteroData object
        in_channels_dict: Dict mapping node_type → input feature dimension
        hidden_channels: Hidden layer dimension (default: 128)
        num_layers: Number of message-passing layers (default: 3)
        dropout: Dropout probability (default: 0.3)
    """

    def __init__(
        self,
        metadata: Tuple,
        in_channels_dict: Dict[str, int],
        hidden_channels: int = config.GNN_HIDDEN_CHANNELS,
        num_layers: int = config.GNN_NUM_LAYERS,
        dropout: float = config.GNN_DROPOUT,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        node_types, edge_types = metadata

        # Input projections — map each node type to the same hidden dim
        self.input_proj = nn.ModuleDict()
        for node_type in node_types:
            in_dim = in_channels_dict.get(node_type, 1)
            self.input_proj[node_type] = Linear(in_dim, hidden_channels)

        # Message passing layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                src_type, rel, dst_type = edge_type
                conv_dict[edge_type] = SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                    normalize=True,
                    project=True,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))

        # Batch norms per layer per node type
        self.batch_norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.BatchNorm1d(hidden_channels)
                for node_type in node_types
            })
            for _ in range(num_layers)
        ])

        # Classification head on transaction embeddings
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

        log.info(
            "HeteroFraudGNN created — layers: %d, hidden: %d, dropout: %.2f",
            num_layers, hidden_channels, dropout,
        )

    def encode(self, x_dict: Dict[str, Tensor], edge_index_dict: Dict) -> Dict[str, Tensor]:
        """
        Run graph convolutions and return node embeddings for all node types.

        Args:
            x_dict: Dict mapping node_type → feature tensor [N, F]
            edge_index_dict: Dict mapping edge_type → edge_index [2, E]

        Returns:
            Dict mapping node_type → embedding tensor [N, hidden_channels]
        """
        # Project inputs to hidden dim
        h_dict = {
            node_type: F.relu(self.input_proj[node_type](x))
            for node_type, x in x_dict.items()
        }

        # Message passing
        for i, conv in enumerate(self.convs):
            h_dict_new = conv(h_dict, edge_index_dict)
            # Apply batch norm + residual + dropout
            for node_type in h_dict_new:
                if h_dict_new[node_type] is not None:
                    h = self.batch_norms[i][node_type](h_dict_new[node_type])
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    # Residual connection
                    if node_type in h_dict and h_dict[node_type].shape == h.shape:
                        h = h + h_dict[node_type]
                    h_dict_new[node_type] = h
            h_dict = h_dict_new

        return h_dict

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict,
    ) -> Tensor:
        """
        Full forward pass — returns fraud probabilities for transaction nodes.

        Args:
            x_dict: Node feature dict
            edge_index_dict: Edge index dict

        Returns:
            Fraud probability tensor of shape [N_transactions]
        """
        h_dict = self.encode(x_dict, edge_index_dict)
        txn_emb = h_dict["transaction"]           # [N_txn, hidden]
        logits = self.classifier(txn_emb).squeeze(-1)  # [N_txn]
        return torch.sigmoid(logits)

    def get_embeddings(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict,
    ) -> Tensor:
        """
        Return penultimate transaction node embeddings (pre-classifier).
        Not currently consumed by any explainer — the gradient-attribution
        explainer in explainability/shap_explainer.py operates on raw
        input features via autograd, not on this method's output.

        Returns:
            Tensor of shape [N_transactions, hidden_channels]
        """
        h_dict = self.encode(x_dict, edge_index_dict)
        return h_dict["transaction"]


def build_model(
    data: HeteroData,
    hidden_channels: int = config.GNN_HIDDEN_CHANNELS,
    num_layers: int = config.GNN_NUM_LAYERS,
    dropout: float = config.GNN_DROPOUT,
) -> "HeteroFraudGNN":
    """
    Instantiate the GNN from a HeteroData object.

    Args:
        data: PyG HeteroData with node features assigned
        hidden_channels, num_layers, dropout: override the config.py
            defaults — used by the ablation sweep in training/ablation.py

    Returns:
        Initialized HeteroFraudGNN model
    """
    in_channels_dict = {
        node_type: data[node_type].x.shape[1]
        for node_type in data.node_types
        if hasattr(data[node_type], "x") and data[node_type].x is not None
    }
    model = HeteroFraudGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels_dict,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %s", f"{total_params:,}")
    return model


def save_model(model: "HeteroFraudGNN", path=None) -> None:
    """Save model state dict to disk."""
    path = path or config.GNN_MODEL_PATH
    torch.save(model.state_dict(), path)
    log.info("Model saved to %s", path)


def load_model(data: HeteroData, path=None) -> "HeteroFraudGNN":
    """Load model from saved state dict."""
    path = path or config.GNN_MODEL_PATH
    model = build_model(data)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    log.info("Model loaded from %s", path)
    return model
