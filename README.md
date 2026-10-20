# 🔍 Real-Time Fraud Detection Pipeline

> **Kafka → Spark Structured Streaming → Graph Neural Network → Gradient-Attribution Explainability**

A production-style, end-to-end real-time fraud detection system. Streams live transactions through a heterogeneous **Graph Neural Network (HeteroGraphSAGE)** trained on the IEEE-CIS dataset, with per-decision gradient-attribution explainability and a live Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Apache Kafka](https://img.shields.io/badge/Apache%20Kafka-3.x-231F20?logo=apachekafka)](https://kafka.apache.org/)
[![PySpark](https://img.shields.io/badge/PySpark-3.5.0-E25A1C?logo=apachespark)](https://spark.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://www.docker.com/)

---

##  Architecture

```
Live Transactions (IEEE-CIS CSV replay)
         │
         ▼
  Kafka Topic: transactions-raw
         │
         ▼
  Spark Structured Streaming
  (velocity features, windowed aggregations)
         │
         ▼
  HeteroGraphSAGE GNN (PyTorch Geometric)
  (account → merchant → email graph, fraud scoring)
         │
         ├── fraud_flag=1 ──→ Gradient Attribution + Graph Connections
         │                    (per-transaction explanation)
         ▼
  Kafka Topic: fraud-results
  + JSON file sink (outputs/results/)
         │
         ▼
  Streamlit Dashboard
  (live monitor · investigation · model perf · graph explorer)
```

---

##  Model Performance

| Metric | Score |
|---|---|
| **AUC-ROC** | **0.796** |
| Precision | 0.211 |
| Recall | 0.987 |
| F1 Score | 0.347 |

Evaluated on a temporal (chronological) test split — see [MODEL_CARD.md](MODEL_CARD.md)
for the full methodology, an honest before/after comparison against the
original (leakier) evaluation, why precision looks lower here than an
earlier version of this table claimed, and how the decision threshold
was chosen.

Trained on a 110,663-row sample (all ~20,663 fraud rows + 90,000
randomly sampled legitimate rows) from the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
dataset (590,540 rows total) — not the full dataset. See MODEL_CARD.md
for why, and what that means for transactions on cards/merchants outside
the sample.

---

##  Dashboard Preview

| Page | What it shows |
|---|---|
| **Live Monitor** | Real-time fraud rate, transaction feed, auto-refreshes every 3s |
| **Fraud Investigation** | Gradient-attribution explanations + graph connections for any flagged transaction |
| **Model Performance** | Training curves, confusion matrix, feature importance |
| **Graph Explorer** | Subgraph visualization for any card ID |

>  See `fraud-detection-pipeline-demo.mp4` for a full walkthrough.

---

##  System Requirements

| | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 8+ cores |
| Disk | 15 GB free | 20 GB |
| OS | Windows 10/11 + Docker Desktop | Linux / macOS |
| Python | 3.10 | 3.10 |

---

##  Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/fraud-detection-pipeline.git
cd fraud-detection-pipeline
```

### 2. Install Docker Desktop

Download from [docker.com](https://www.docker.com/products/docker-desktop/).  
On Windows, **enable WSL 2** when prompted.

### 3. Get the dataset

Download the IEEE-CIS Fraud Detection dataset from Kaggle:  
👉 https://www.kaggle.com/competitions/ieee-fraud-detection/data

Place the files in the `data/` folder:
```
data/
├── train_transaction.csv
└── train_identity.csv
```

### 4. Create a Python virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

If PyTorch Geometric fails, install manually:
```bash
pip install torch==2.1.0
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

---

##  Run Order

Open a **separate terminal** for each step.

#### Step 1 - Start infrastructure
```bash
docker-compose up -d
```
Wait ~30 seconds, then verify:
- Spark UI → http://localhost:8080
- Kafka → `docker ps` (all containers should be `healthy`)

#### Step 2 - Build the transaction graph *(one-time)*
```bash
python graph/graph_builder.py
```
Loads the IEEE-CIS CSVs, builds the heterogeneous account-merchant-email graph, and saves to `models/graph_data.pt`. Takes 2–5 minutes.

#### Step 3 - Train the GNN *(one-time)*
```bash
python training/train_gnn.py
```
Trains HeteroGraphSAGE for 50 epochs. Saves:
- `models/gnn_model.pt` - best checkpoint
- `models/training_curves.png` - loss + AUC plots
- `models/test_results.json` - final test metrics

Takes 10–30 minutes on CPU (faster with GPU).

#### Step 4 - Start the Kafka producer
```bash
python producer/kafka_producer.py --mode fast       # for testing
python producer/kafka_producer.py --mode realistic  # for demo
python producer/kafka_producer.py --mode burst      # for stress testing
```
Expected output: `Produced 1000 / 590540 transactions`

#### Step 5 - Start the streaming pipeline
```bash
python pipeline/streaming_pipeline.py
```
Expected output: `Batch 0 done - 47 txns | 2 flagged | 94 txns/sec`

Results are written to `outputs/results/` as JSONL files.

#### Step 6 - Launch the dashboard
```bash
streamlit run dashboard/app.py
```
Open browser at: http://localhost:8501

---

##  Stopping Everything

```bash
# Ctrl+C in the pipeline and producer terminals, then:
docker-compose down
```

---

##  Project Structure

```
fraud-detection-pipeline/
├── config.py                        # All settings and hyperparameters
├── requirements.txt
├── docker-compose.yml               # Kafka + Zookeeper + Spark
├── Dockerfile
│
├── data/                            # ⚠️ Not committed - download from Kaggle
│   ├── train_transaction.csv
│   └── train_identity.csv
│
├── producer/
│   └── kafka_producer.py            # CSV replay → Kafka
│
├── spark/
│   └── feature_engineering.py       # Structured Streaming + velocity features
│
├── graph/
│   ├── graph_builder.py             # Build HeteroData graph
│   └── gnn_model.py                 # HeteroGraphSAGE model definition
│
├── training/
│   └── train_gnn.py                 # Training loop + evaluation
│
├── inference/
│   └── gnn_inference.py             # Batch + single-transaction scoring
│
├── explainability/
│   └── shap_explainer.py            # Gradient attribution + graph-connection explanation
│                                     # (module filename kept for history; no SHAP/GNNExplainer is used)
│
├── pipeline/
│   └── streaming_pipeline.py        # End-to-end orchestrator
│
├── dashboard/
│   └── app.py                       # Streamlit dashboard (4 pages)
│
├── models/                          # Saved after training (not committed)
│   ├── gnn_model.pt
│   ├── graph_data.pt
│   ├── scaler.pkl
│   ├── node_maps.pkl
│   └── training_curves.png
│
└── outputs/
    └── results/                     # JSONL output from streaming pipeline
```

---

##  Troubleshooting

| Error | Fix |
|---|---|
| `No brokers available` | Kafka isn't ready. Wait 30s after `docker-compose up` and retry. |
| `Graph not found` | Run `python graph/graph_builder.py` before the pipeline. |
| `CUDA out of memory` | Training uses CPU by default. If using GPU, set `SPARK_MASTER = "local[*]"` in `config.py`. |
| `train_transaction.csv not found` | Download from Kaggle (Step 3 above). |
| `Dashboard shows no data` | The `outputs/results/` directory is empty - run Steps 4 and 5 first. |
| High memory usage | In Docker Desktop → Settings → Resources, set memory to 6 GB minimum. |

---

##  Tech Stack

| Component | Tool | Why |
|---|---|---|
| Event streaming | Apache Kafka | Durable, replayable, high-throughput event log |
| Stream processing | PySpark Structured Streaming | Stateful windowed velocity features at scale |
| Graph ML | PyTorch Geometric (GraphSAGE) | Captures fraud ring structure across accounts |
| Explainability | Gradient (input × grad) attribution + graph-connection analysis | Regulatory compliance (GDPR Art. 22, SR 11-7) — see note below |
| Dashboard | Streamlit | Rapid, Python-native UI |
| Containerization | Docker Compose | Reproducible local infrastructure |
| Dataset | IEEE-CIS Fraud Detection | Real-world finance data (~590K transactions) |

---

##  Design Decisions

**Why Kafka over a simple queue?**  
Kafka's durable offset tracking allows any consumer to replay the full transaction log from any point - critical for fraud auditing and pipeline recovery.

**Why GNN over XGBoost?**  
Fraud often involves rings of connected accounts sharing merchants, IPs, and email domains. A heterogeneous GNN learns these latent structural patterns that hand-engineered features miss entirely.

**Why gradient attribution instead of SHAP?**  
Financial institutions are subject to GDPR Article 22 (right to explanation) and SR 11-7 model risk governance, which require per-transaction, per-feature attribution. KernelSHAP was tried first but needs a valid index into the training graph, which streaming transactions never have. Input-gradient × input attribution gives per-feature scores for any transaction — seen or unseen — without that requirement. It is a real trade-off: gradient attribution is faster and index-free, but SHAP's game-theoretic guarantees (and `torch_geometric.explain.GNNExplainer` for the graph structure itself) are the stronger long-term choice for a compliance-grade explanation and are the next step before production.

**Path to production:**  
Replace the CSV Kafka producer with a real card-network feed → swap `local[*]` Spark with a cluster → add PSI-based model drift monitoring → implement an automated retraining pipeline.

---


