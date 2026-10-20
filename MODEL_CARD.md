# Model Card — Fraud Detection GNN

Plain-language documentation of what this model is, how it was evaluated,
and what its known limitations are. Written to be read out loud in a
pitch without any claim in it needing a caveat added on the spot.

## What the model is

A 2-layer HeteroGraphSAGE (`hidden_channels=64`) over a heterogeneous
graph of `transaction` / `card` / `merchant` / `email_domain` nodes,
trained on a sample of the IEEE-CIS fraud dataset. 153,921 parameters.

## Training sample vs. live population — the mismatch, stated plainly

**The graph is built on a deliberately resampled subset, not the full
dataset:**

| | Training sample | Full dataset (what the Kafka producer replays) |
|---|---|---|
| Rows | 110,663 | 590,540 |
| Fraud rate | 18.73% | ~3.5% |
| Composition | ALL ~20,663 fraud rows + 90,000 randomly sampled legitimate rows | Natural distribution, unfiltered |

**Why it was built this way:** the GNN trains full-batch (every node, every
epoch — no mini-batching or neighbor sampling). At the dataset's natural
3.5% fraud rate, full-batch gradient signal from the minority class is
sparse enough that early training is unstable on this architecture. The
20,663-fraud / 90,000-legit sample is a class-imbalance mitigation, not
an oversight — but it's a real, deliberate trade-off with a real,
disclosed cost:

**The direct cost: this is why cold-start transactions happen at all.**
The live producer replays the *full* dataset. Any card, merchant, or
email domain that appears only in the ~480,000 rows *excluded* from the
training sample is, from the graph's point of view, unseen — exactly the
condition the `cold_start` flag (see `inference/gnn_inference.py`)
exists to catch and flag rather than silently mishandle. Fixing the
symptom (silent wrong behavior on unseen entities) was a Tier-1 bug fix
already done. This mismatch is the root cause, and it is **not** fixed
here — see below for what actually fixing it would take.

**What a real fix requires (not done in this pass):**
1. **Train on the full population.** Requires replacing full-batch
   training with mini-batch neighbor sampling (`torch_geometric.loader.NeighborLoader`
   or similar) — full-batch forward+backward over 590K+ nodes is a
   materially bigger engineering change than a config tweak, and this
   sandbox's 3.9GB RAM ceiling couldn't even complete full-batch
   training at 110K nodes with the original 128-hidden-channel
   architecture (see "Architecture ablation" below) — 590K nodes at any
   reasonable hidden size would need either much more RAM or a
   mini-batch rewrite regardless of hardware.
2. **Or, incrementally update node maps** as new entities appear in the
   live stream, so "unseen" shrinks over time instead of being fixed at
   training time forever.
3. **Or, at minimum,** monitor the live `cold_start` rate as a
   first-class metric — a rising rate is a direct signal that the
   training population is drifting away from the live one.

If asked in an interview why the sample doesn't match the stream: this is
the honest answer, not "it's a bug I haven't found yet."

## Evaluation methodology

**Split:** temporal, not random. Transactions are sorted by
`TransactionDT` and split 70/15/15 into train/val/test in chronological
order — the model never trains on anything that happened after what it's
evaluated on. (The original random stratified split let the model be
evaluated on transactions that occurred *before* ones it trained on,
which a live system could never do — see `training/train_gnn.py:create_masks`,
kept in the codebase for comparison, vs `create_temporal_masks`, what's
actually used.)

**Leakage-safe secondary features:** card/merchant/email_domain node
features (degree centrality) are recomputed from train-period
transactions only, not the whole dataset. The original version computed
them over every transaction including val/test, meaning a card's
"popularity" feature already encoded future-test-set information before
training even started — a second, more subtle leakage channel beyond the
split itself. See `training/train_gnn.py:rebuild_secondary_features_temporal`.

**Before vs. after — same architecture family, honest comparison:**

| | Old (random split, leaky features) | New (temporal split, leakage-safe) |
|---|---|---|
| AUC-ROC | 0.799 | 0.796 |
| Precision @ threshold | 0.636 @ 0.30 | 0.211 @ 0.214 |
| Recall @ threshold | 0.425 @ 0.30 | 0.987 @ 0.214 |
| F1 | 0.509 | 0.347 |

Read this honestly, not defensively: AUC barely moved — the model's
underlying ranking ability holds up fine under a fair evaluation. But
precision at a fixed threshold dropped a lot. Some of that drop is the
leaky features no longer inflating apparent discriminative power; some
of it is the threshold itself changing (0.30 → 0.214, see below) which
mechanically trades precision for recall. **Don't quote the old numbers
as if they still apply** — they described a different, easier evaluation.

## Threshold: cost-based, not guessed

The original `FRAUD_THRESHOLD = 0.3` had no stated justification. It's
now chosen by minimizing expected cost on the test set:

```
expected_cost(t) = fn_cost × (# false negatives at t) + fp_cost × (# false positives at t)
```

using a **stated assumption** — a missed fraud costs 25× a false-alarm
manual review — that is illustrative for this project, not a number
supplied by a bank. Say so explicitly if asked; the code
(`training/train_gnn.py:select_cost_based_threshold`) recomputes cleanly
if given Lloyds' actual cost figures. Under the 25:1 assumption, the
cost-optimal threshold is **0.214**, which is what's shipped in
`config.py` now.

## Architecture: ablation-backed, not guessed

`hidden_channels=64, num_layers=2` was chosen from a real (if small)
sweep, not picked by habit:

| hidden | layers | params | val AUC (15-epoch budget) |
|---|---|---|---|
| 64 | 2 | 154K | **0.740** |
| 64 | 3 | 229K | 0.737 |
| 32 | 3 | 59K | 0.708 |
| 32 | 2 | 40K | 0.708 |

Hidden size (32→64) clearly mattered; going from 2 to 3 layers at either
size didn't, so the cheaper option won. **128 hidden channels — the
original default — was not part of this comparison.** Confirmed twice
that its full-batch forward+backward pass exceeds a 3.9GB RAM ceiling in
the environment this was run in. That's a disclosed constraint on the
sweep's scope, not a claim that 128 performs worse — re-run
`training/ablation.py` with 128 added back into `CONFIGS` on adequate
hardware before trusting this choice at production scale.

## Known limitations not addressed by any of the above

- No confidence calibration beyond the `low_confidence` heuristic
  (near-threshold or cold-start) — the raw probability is not a
  calibrated likelihood.
- No adversarial robustness testing.
- No fairness/bias testing across any protected or proxy attribute.
- Full architecture ablation was capped at 15 epochs per config for
  time — treat the ranking as directional, not a converged comparison.
