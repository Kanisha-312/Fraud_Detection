"""
training/ablation.py
Small architecture ablation: compares a few (hidden_channels, num_layers)
configurations on validation AUC, using the same temporal split and
leakage-safe secondary features as the real training run.

Why this exists: the original project picked hidden_channels=128,
num_layers=3 with no comparison data — an unjustified choice. This
runs each candidate config for a reduced epoch budget (enough to see
which configs are competitive, not full convergence) and reports val
AUC/precision/recall for each, so the final choice is backed by a
number instead of a guess.

Note on epoch budget: this sandbox has a 3.9GB RAM ceiling that a
128-hidden-channel config's full-batch forward+backward pass exceeds
(confirmed by two separate OOM kills during real training runs — see
training/train_gnn.py's memory-hygiene comments). 128 is excluded from
this sweep for that reason, not because it was assumed worse — on
adequate hardware it should be added back in.
"""

import gc
import json
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
from graph.gnn_model import build_model
from training.train_gnn import (
    create_temporal_masks, rebuild_secondary_features_temporal,
    compute_class_weights, evaluate,
)

log = logging.getLogger("ablation")

ABLATION_EPOCHS = 15  # reduced budget — see module docstring
CONFIGS = [
    {"hidden_channels": 32, "num_layers": 2},
    {"hidden_channels": 32, "num_layers": 3},
    {"hidden_channels": 64, "num_layers": 2},
    {"hidden_channels": 64, "num_layers": 3},
]


def run_one_config(hidden_channels: int, num_layers: int) -> dict:
    device = torch.device("cpu")
    data, _, _, _ = load_graph()
    train_mask, val_mask, test_mask = create_temporal_masks(data)
    rebuild_secondary_features_temporal(data, train_mask)

    model = build_model(
        data, hidden_channels=hidden_channels, num_layers=num_layers
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.GNN_LEARNING_RATE, weight_decay=1e-5)
    pos_weight = compute_class_weights(data["transaction"].y[train_mask]).to(device)

    best_val_auc = 0.0
    t_start = time.time()
    for epoch in range(1, ABLATION_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        probs = model(data.x_dict, data.edge_index_dict)
        train_probs = probs[train_mask]
        train_labels = data["transaction"].y[train_mask].float()
        sw = torch.where(train_labels == 1, pos_weight.squeeze(), torch.ones_like(train_labels))
        loss = (nn.BCELoss(reduction="none")(train_probs, train_labels) * sw).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        del probs, train_probs, train_labels, sw, loss
        gc.collect()

        val_metrics = evaluate(model, data, val_mask, device)
        best_val_auc = max(best_val_auc, val_metrics["auc"])
        del val_metrics
        gc.collect()

    elapsed = time.time() - t_start

    # Final val metrics at the end of the (short) run
    final_val = evaluate(model, data, val_mask, device)

    del data, model, optimizer
    gc.collect()

    return {
        "hidden_channels": hidden_channels,
        "num_layers": num_layers,
        "params": n_params,
        "epochs_run": ABLATION_EPOCHS,
        "best_val_auc_seen": best_val_auc,
        "final_val_auc": final_val["auc"],
        "final_val_precision": final_val["precision"],
        "final_val_recall": final_val["recall"],
        "final_val_f1": final_val["f1"],
        "train_seconds": round(elapsed, 1),
    }


def main():
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    out_path = config.MODEL_DIR / "ablation_results.json"
    results = []
    # Resume from any partial results already on disk (this sandbox has
    # a hard wall-clock limit per command shorter than the full sweep).
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        done = {(r["hidden_channels"], r["num_layers"]) for r in results}
        log.info("Resuming — %d configs already done: %s", len(results), done)
    else:
        done = set()

    for cfg in CONFIGS:
        key = (cfg["hidden_channels"], cfg["num_layers"])
        if key in done:
            log.info("Skipping already-done config: %s", cfg)
            continue
        log.info("=" * 60)
        log.info("Running config: %s", cfg)
        r = run_one_config(**cfg)
        log.info("Result: %s", r)
        results.append(r)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    results_sorted = sorted(results, key=lambda r: -r["final_val_auc"])
    log.info("=" * 60)
    log.info("ABLATION SUMMARY (sorted by final val AUC, %d-epoch budget)", ABLATION_EPOCHS)
    log.info("%-8s %-8s %-10s %-10s %-10s %-10s %-10s", "hidden", "layers", "params", "val_auc", "val_prec", "val_rec", "seconds")
    for r in results_sorted:
        log.info(
            "%-8d %-8d %-10d %-10.4f %-10.4f %-10.4f %-10.1f",
            r["hidden_channels"], r["num_layers"], r["params"],
            r["final_val_auc"], r["final_val_precision"], r["final_val_recall"],
            r["train_seconds"],
        )

    with open(out_path, "w") as f:
        json.dump(results_sorted, f, indent=2)
    log.info("Ablation results saved to %s", out_path)


if __name__ == "__main__":
    main()
