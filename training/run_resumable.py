"""
training/run_resumable.py
Runs the real training loop (temporal split + leakage-safe features)
with checkpointing, so a long run can be safely interrupted and resumed
— useful on constrained/shared hardware where a single process might
not get an uninterrupted multi-minute window. Reuses the exact same
functions as train_gnn.py; this only adds checkpoint/resume around them.

Usage: python3 -m training.run_resumable [--epochs N]
Re-running the same command resumes from the last checkpoint instead
of starting over.
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from graph.graph_builder import load_graph
from graph.gnn_model import build_model, save_model
from training.train_gnn import (
    create_temporal_masks, rebuild_secondary_features_temporal,
    compute_class_weights, evaluate, plot_training_curves,
    final_evaluation,
)

log = logging.getLogger("run_resumable")

CHECKPOINT_PATH = config.MODEL_DIR / "training_checkpoint.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.GNN_EPOCHS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    device = torch.device("cpu")

    data, _, _, _ = load_graph()
    train_mask, val_mask, test_mask = create_temporal_masks(data)
    rebuild_secondary_features_temporal(data, train_mask)

    model = build_model(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.GNN_LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    pos_weight = compute_class_weights(data["transaction"].y[train_mask]).to(device)

    start_epoch = 1
    best_val_auc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_precision": [], "val_recall": [], "val_f1": []}

    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_auc = ckpt["best_val_auc"]
        history = ckpt["history"]
        log.info("Resumed from checkpoint at epoch %d (best val AUC so far: %.4f)",
                 ckpt["epoch"], best_val_auc)

    if start_epoch > args.epochs:
        log.info("Already completed %d epochs (target: %d) — nothing to do.", start_epoch - 1, args.epochs)
    else:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            model.train()
            optimizer.zero_grad()

            probs = model(data.x_dict, data.edge_index_dict)
            train_probs = probs[train_mask]
            train_labels = data["transaction"].y[train_mask].float()
            sample_weights = torch.where(
                train_labels == 1, pos_weight.squeeze(), torch.ones_like(train_labels)
            )
            loss = (nn.BCELoss(reduction="none")(train_probs, train_labels) * sample_weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_value = loss.item()
            del probs, train_probs, train_labels, sample_weights, loss
            gc.collect()

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
                "Epoch %3d/%d | Train Loss: %.4f | Val Loss: %.4f | AUC: %.4f | P: %.3f | R: %.3f | F1: %.3f | %.1fs",
                epoch, args.epochs, train_loss_value, val_metrics["loss"],
                val_metrics["auc"], val_metrics["precision"], val_metrics["recall"], val_metrics["f1"], elapsed,
            )

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                save_model(model)
                log.info("  ✓ New best model saved (AUC=%.4f)", best_val_auc)

            del val_metrics
            gc.collect()

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_auc": best_val_auc,
                "history": history,
            }, CHECKPOINT_PATH)

        log.info("Training complete through epoch %d. Best val AUC: %.4f", args.epochs, best_val_auc)
        plot_training_curves(history, config.TRAINING_CURVES_PATH)

    # Final test-set evaluation + cost-based threshold + PR curve,
    # reusing train_gnn.py's final_evaluation exactly.
    data, _, _, _ = load_graph()
    train_mask, _, test_mask = create_temporal_masks(data)
    rebuild_secondary_features_temporal(data, train_mask)
    final_evaluation(model, data, test_mask, device)

    CHECKPOINT_PATH.unlink(missing_ok=True)
    log.info("Checkpoint cleaned up — training run complete.")


if __name__ == "__main__":
    main()
