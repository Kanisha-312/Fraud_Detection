"""
training/train_gnn.py
Trains the HeteroFraudGNN on the pre-built graph.

Features:
  - Stratified train/val/test split (70/15/15)
  - Weighted cross-entropy for class imbalance
  - AUC-ROC, precision, recall, F1 tracking per epoch
  - Best model checkpoint saved on val AUC
  - Training curves saved as PNG
"""

import logging
import gc
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from graph.graph_builder import load_graph
from graph.gnn_model import build_model, save_model

log = logging.getLogger("train_gnn")


def create_masks(data: HeteroData) -> tuple:
    """
    Create stratified train/val/test masks on transaction nodes,
    using a RANDOM split (ignores chronology).

    Kept for comparison purposes — see create_temporal_masks() for the
    split actually used going forward. This one is transductive in a
    second sense beyond message passing: it lets the model be evaluated
    on transactions that happened before ones it trained on, which a
    live system could never do.

    Returns:
        (train_mask, val_mask, test_mask) — boolean tensors of shape [N_transactions]
    """
    n = data["transaction"].y.shape[0]
    labels = data["transaction"].y.numpy()
    indices = np.arange(n)

    # Stratified split: 70% train, 30% temp
    train_idx, temp_idx = train_test_split(
        indices, test_size=(1 - config.GNN_TRAIN_RATIO),
        stratify=labels, random_state=42
    )
    # Split temp into val and test (50/50 of the 30%)
    val_ratio_of_temp = config.GNN_VAL_RATIO / (config.GNN_VAL_RATIO + config.GNN_TEST_RATIO)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_ratio_of_temp),
        stratify=labels[temp_idx], random_state=42
    )

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    log.info(
        "Random split — train: %d, val: %d, test: %d",
        train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item()
    )
    fraud_in_train = data["transaction"].y[train_mask].sum().item()
    log.info("Fraud in train set: %d / %d", fraud_in_train, train_mask.sum().item())

    return train_mask, val_mask, test_mask


def create_temporal_masks(data: HeteroData) -> tuple:
    """
    Create train/val/test masks by TransactionDT order — the split
    actually used for training and evaluation.

    Why: a random stratified split lets the model be evaluated on
    transactions that occurred chronologically BEFORE ones it trained
    on. A live system can never do that — it only ever sees the past.
    Sorting by TransactionDT and taking the earliest 70% as train, the
    next 15% as val, and the latest 15% as test makes the evaluation
    match how the model is actually used.

    This does not by itself remove every form of leakage — message
    passing still lets a train-period transaction's embedding be
    influenced by which card/merchant/email node it's connected to,
    and that node's OTHER edges include future transactions unless the
    node features themselves are also computed train-only (see
    rebuild_secondary_features_temporal(), called separately after
    this in main()).

    Returns:
        (train_mask, val_mask, test_mask) — boolean tensors of shape [N_transactions]
    """
    dt = data["transaction"].transaction_dt.numpy()
    n = len(dt)
    order = np.argsort(dt, kind="stable")

    n_train = int(n * config.GNN_TRAIN_RATIO)
    n_val = int(n * config.GNN_VAL_RATIO)

    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    log.info(
        "Temporal split — train: %d (t<=%d), val: %d (t<=%d), test: %d (t<=%d)",
        len(train_idx), int(dt[train_idx[-1]]),
        len(val_idx), int(dt[val_idx[-1]]),
        len(test_idx), int(dt[test_idx[-1]]),
    )
    log.info(
        "Fraud — train: %d/%d (%.2f%%), val: %d/%d (%.2f%%), test: %d/%d (%.2f%%)",
        data["transaction"].y[train_mask].sum().item(), len(train_idx),
        100 * data["transaction"].y[train_mask].float().mean().item(),
        data["transaction"].y[val_mask].sum().item(), len(val_idx),
        100 * data["transaction"].y[val_mask].float().mean().item(),
        data["transaction"].y[test_mask].sum().item(), len(test_idx),
        100 * data["transaction"].y[test_mask].float().mean().item(),
    )

    return train_mask, val_mask, test_mask


def rebuild_secondary_features_temporal(data: HeteroData, train_mask: torch.Tensor) -> None:
    """
    Recompute card/merchant/email_domain node features (degree
    centrality) using ONLY train-split transactions, in place.

    Why this matters: graph_builder.py originally computes these
    features as "fraction of ALL transactions touching this entity" —
    including val/test transactions. That means a card's popularity
    feature already encodes information about the future test set
    before training even starts, regardless of which transactions are
    masked out of the loss. This closes that channel by only counting
    edges from train-period transactions.

    Call this once, right after create_temporal_masks(), before the
    model is built or trained.
    """
    edge_types = {
        "card": ("transaction", "uses_card", "card"),
        "merchant": ("transaction", "at_merchant", "merchant"),
        "email_domain": ("transaction", "uses_email", "email_domain"),
    }
    n_train = int(train_mask.sum().item())
    for ntype, etype in edge_types.items():
        src, dst = data[etype].edge_index
        train_edges = train_mask[src]
        dst_train_only = dst[train_edges]
        n_nodes = data[ntype].x.shape[0]
        counts = torch.bincount(dst_train_only, minlength=n_nodes).float()
        data[ntype].x = (counts / max(n_train, 1)).unsqueeze(1)
    log.info(
        "Recomputed card/merchant/email_domain features from train-only "
        "degree (%d train transactions) — val/test signal removed from "
        "secondary node features",
        n_train,
    )


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute fraud/non-fraud class weights to handle imbalance.
    Weight for fraud class = (1 - fraud_rate) / fraud_rate
    """
    fraud_rate = labels.float().mean().item()
    fraud_weight = (1 - fraud_rate) / max(fraud_rate, 1e-6)
    log.info("Fraud rate: %.4f — applying fraud class weight: %.2f", fraud_rate, fraud_weight)
    pos_weight = torch.tensor([fraud_weight], dtype=torch.float)
    return pos_weight


def evaluate(
    model: nn.Module,
    data: HeteroData,
    mask: torch.Tensor,
    device: torch.device,
    threshold: float = config.FRAUD_THRESHOLD,
) -> dict:
    """
    Evaluate the model on a given mask split.

    Returns:
        dict with loss, auc, precision, recall, f1
    """
    model.eval()
    with torch.no_grad():
        probs = model(data.x_dict, data.edge_index_dict)
        labels = data["transaction"].y.float().to(device)

        # Use raw probs (already sigmoided by the model's forward()) for metrics.
        # Note: BCEWithLogitsLoss would need raw logits, which forward() doesn't
        # expose, so loss below is plain BCELoss on the sigmoided output instead.
        probs_mask = probs[mask].cpu().numpy()
        labels_mask = labels[mask].cpu().numpy()

        preds = (probs_mask >= threshold).astype(int)
        try:
            auc = roc_auc_score(labels_mask, probs_mask)
        except ValueError:
            auc = 0.5

        precision = precision_score(labels_mask, preds, zero_division=0)
        recall = recall_score(labels_mask, preds, zero_division=0)
        f1 = f1_score(labels_mask, preds, zero_division=0)

        loss = float(nn.BCELoss()(
            torch.tensor(probs_mask), torch.tensor(labels_mask)
        ))

    return {
        "loss": loss,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "probs": probs_mask,
        "labels": labels_mask,
        "preds": preds,
    }


def select_cost_based_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    fn_cost: float,
    fp_cost: float,
) -> dict:
    """
    Pick a decision threshold by minimizing expected cost, instead of
    using a fixed threshold with no stated rationale.

    expected_cost(t) = fn_cost * (# false negatives at t)
                      + fp_cost * (# false positives at t)

    fn_cost / fp_cost must be stated explicitly — there's no
    threshold-free way to pick one, and pretending otherwise (a bare
    "0.3") is a guess dressed up as a decision. The ratio used here
    (see call site) is a stated assumption for this project, not a
    number Lloyds gave us — call this out explicitly when presenting it.

    Returns a dict with the chosen threshold and the precision/recall/
    cost at that threshold, plus the full curve for plotting.
    """
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    precisions, recalls = precisions[:-1], recalls[:-1]

    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())

    best = None
    for p, r, t in zip(precisions, recalls, thresholds):
        tp = r * n_pos
        fn = n_pos - tp
        fp = (tp / p - tp) if p > 0 else n_neg
        cost = fn_cost * fn + fp_cost * fp
        if best is None or cost < best["cost"]:
            best = {
                "threshold": float(t), "precision": float(p),
                "recall": float(r), "cost": float(cost),
                "expected_fn": float(fn), "expected_fp": float(fp),
            }

    best["fn_cost"] = fn_cost
    best["fp_cost"] = fp_cost
    best["curve"] = {
        "precision": precisions.tolist(),
        "recall": recalls.tolist(),
        "thresholds": thresholds.tolist(),
    }
    return best


def flag_confidence(probs: np.ndarray, threshold: float, band: float = 0.05) -> np.ndarray:
    """
    Flag predictions within `band` of the decision threshold as low
    confidence — the model's probability output alone doesn't
    distinguish "confidently fraud" from "barely over the line", and a
    single threshold treats them identically. Callers combine this with
    the separate cold_start flag (inference/gnn_inference.py) to decide
    what should route to manual review instead of an automatic decision.
    """
    return np.abs(probs - threshold) <= band


def plot_training_curves(history: dict, save_path: Path) -> None:
    """Plot and save loss + AUC training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3")
    ax.plot(epochs, history["val_loss"], label="Val Loss", color="#FF5722")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["val_auc"], label="Val AUC-ROC", color="#4CAF50")
    ax.axhline(y=max(history["val_auc"]), color="gray", linestyle="--",
               label=f"Best AUC: {max(history['val_auc']):.4f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Validation AUC-ROC")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Training curves saved to %s", save_path)


def train(
    model: nn.Module,
    data: HeteroData,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Main training loop.

    Returns:
        history dict with per-epoch metrics
    """
    model = model.to(device)

    # Move graph data to device
    data = data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.GNN_LEARNING_RATE,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )

    # Weighted loss for imbalance — applied manually via per-sample weights
    # below (nn.BCELoss(reduction="none") is instantiated inline where it's
    # actually used, so no unused criterion object is kept here).
    pos_weight = compute_class_weights(
        data["transaction"].y[train_mask].cpu()
    ).to(device)

    history = {
        "train_loss": [], "val_loss": [],
        "val_auc": [], "val_precision": [], "val_recall": [], "val_f1": [],
    }
    best_val_auc = 0.0
    best_epoch = 0

    log.info("Starting training for %d epochs...", config.GNN_EPOCHS)
    log.info("Device: %s", device)

    for epoch in range(1, config.GNN_EPOCHS + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        probs = model(data.x_dict, data.edge_index_dict)
        train_probs = probs[train_mask]
        train_labels = data["transaction"].y[train_mask].float()

        # Apply class weighting manually via sample weights
        sample_weights = torch.where(
            train_labels == 1, pos_weight.squeeze(), torch.ones_like(train_labels)
        )
        loss = (nn.BCELoss(reduction="none")(train_probs, train_labels) * sample_weights).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss_value = loss.item()
        # Release the full-graph forward pass's intermediate tensors
        # (activations for every message-passing layer, retained by
        # autograd for backward()) before the next full-graph forward
        # pass in evaluate() below allocates its own set. Full-batch
        # training on this graph size peaks close to a few GB; without
        # this the CPU allocator's cache can be the difference between
        # fitting in memory and an OOM kill on a constrained machine.
        del probs, train_probs, train_labels, sample_weights, loss
        gc.collect()

        # Validate
        val_metrics = evaluate(model, data, val_mask, device)
        scheduler.step(val_metrics["auc"])

        history["train_loss"].append(train_loss_value)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])

        elapsed = time.time() - t0
        log.info(
            "Epoch %3d/%d | Train Loss: %.4f | Val Loss: %.4f | "
            "AUC: %.4f | P: %.3f | R: %.3f | F1: %.3f | %.1fs",
            epoch, config.GNN_EPOCHS,
            train_loss_value, val_metrics["loss"],
            val_metrics["auc"], val_metrics["precision"],
            val_metrics["recall"], val_metrics["f1"],
            elapsed,
        )
        del val_metrics
        gc.collect()

        # Save best model
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            save_model(model)
            log.info("  ✓ New best model saved (AUC=%.4f)", best_val_auc)

    log.info(
        "Training complete. Best epoch: %d, Best val AUC: %.4f",
        best_epoch, best_val_auc
    )
    return history


def final_evaluation(
    model: nn.Module,
    data: HeteroData,
    test_mask: torch.Tensor,
    device: torch.device,
) -> None:
    """Run final evaluation on the test set and print detailed report."""
    from graph.gnn_model import load_model
    # Reload best checkpoint
    best_model = load_model(data.cpu())
    best_model = best_model.to(device)
    data = data.to(device)

    test_metrics = evaluate(best_model, data, test_mask, device)

    log.info("=" * 60)
    log.info("FINAL TEST SET RESULTS (fixed threshold = %.2f)", config.FRAUD_THRESHOLD)
    log.info("  AUC-ROC:   %.4f", test_metrics["auc"])
    log.info("  Precision: %.4f", test_metrics["precision"])
    log.info("  Recall:    %.4f", test_metrics["recall"])
    log.info("  F1:        %.4f", test_metrics["f1"])
    log.info("=" * 60)

    cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
    log.info("Confusion Matrix:\n%s", cm)
    log.info("\n%s", classification_report(
        test_metrics["labels"], test_metrics["preds"],
        target_names=["Legitimate", "Fraud"]
    ))

    # Cost-based threshold selection — stated assumption: a missed
    # fraud (false negative) costs 25x a false-alarm manual review
    # (false positive). This ratio is an illustrative business
    # assumption for this project, NOT a number supplied by a bank —
    # say so explicitly if asked. A real deployment would replace it
    # with Lloyds' own cost figures.
    FN_COST, FP_COST = 25.0, 1.0
    cost_result = select_cost_based_threshold(
        test_metrics["labels"], test_metrics["probs"], FN_COST, FP_COST
    )
    log.info("=" * 60)
    log.info(
        "COST-BASED THRESHOLD (FN cost=%.0f : FP cost=%.0f, i.e. missing "
        "fraud assumed %.0fx worse than a false alarm)",
        FN_COST, FP_COST, FN_COST / FP_COST,
    )
    log.info(
        "  Threshold: %.3f (vs fixed %.2f) | Precision: %.4f | Recall: %.4f",
        cost_result["threshold"], config.FRAUD_THRESHOLD,
        cost_result["precision"], cost_result["recall"],
    )
    log.info("=" * 60)

    # Save PR curve plot
    curve = cost_result["curve"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(curve["recall"], curve["precision"], color="#2196F3")
    ax.scatter([cost_result["recall"]], [cost_result["precision"]],
               color="#E53935", zorder=5,
               label=f"cost-optimal (t={cost_result['threshold']:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (test set)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    pr_curve_path = config.MODEL_DIR / "precision_recall_curve.png"
    plt.savefig(pr_curve_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("PR curve saved to %s", pr_curve_path)

    # Save test results for dashboard
    import json
    results = {
        "test_auc": test_metrics["auc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "confusion_matrix": cm.tolist(),
        "fixed_threshold": config.FRAUD_THRESHOLD,
        "cost_based_threshold": {
            "threshold": cost_result["threshold"],
            "precision": cost_result["precision"],
            "recall": cost_result["recall"],
            "fn_cost_assumption": FN_COST,
            "fp_cost_assumption": FP_COST,
            "note": (
                "FN:FP cost ratio is an illustrative assumption for this "
                "project, not a figure supplied by a bank. Replace with "
                "real cost figures before using this threshold operationally."
            ),
        },
        "split_method": "temporal (TransactionDT-ordered, earliest 70/15/15)",
    }
    results_path = config.MODEL_DIR / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Test results saved to %s", results_path)


def main() -> None:
    """Entry point for training."""
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    log.info("Loading graph data from %s", config.GRAPH_DATA_PATH)
    data, node_maps, scaler, feature_cols = load_graph()
    log.info("Graph loaded successfully")

    train_mask, val_mask, test_mask = create_temporal_masks(data)
    rebuild_secondary_features_temporal(data, train_mask)

    model = build_model(data)
    log.info("Model built successfully")

    history = train(model, data, train_mask, val_mask, device)

    plot_training_curves(history, config.TRAINING_CURVES_PATH)

    # Reload data to CPU for evaluation — re-apply the same temporal
    # split and leakage-safe features (load_graph() returns the raw
    # graph without them).
    data, _, _, _ = load_graph()
    train_mask, _, test_mask = create_temporal_masks(data)
    rebuild_secondary_features_temporal(data, train_mask)
    final_evaluation(model, data, test_mask, device)


if __name__ == "__main__":
    main()
