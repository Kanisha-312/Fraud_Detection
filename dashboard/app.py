"""
dashboard/app.py
Streamlit dashboard for the Real-Time Fraud Detection Pipeline.

Pages:
  1. Live Monitor    — real-time metrics, fraud rate chart, transaction feed
  2. Investigation   — gradient-attribution explanations, graph connections for flagged txns
  3. Model Performance — training curves, confusion matrix, feature importance
  4. Graph Explorer  — interactive subgraph visualization per card ID

Run:
    streamlit run dashboard/app.py
"""

import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
log = logging.getLogger("dashboard")

# ─────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Data loading (cached with TTL)
# ─────────────────────────────────────────────

@st.cache_data(ttl=3)
def load_results() -> pd.DataFrame:
    """Load all JSONL result files from outputs/results directory."""
    rows = []
    for fpath in sorted(config.RESULTS_DIR.glob("*.jsonl")):
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass

    if not rows:
        return pd.DataFrame(columns=[
            "transaction_id", "fraud_probability", "fraud_flag",
            "amount", "event_time", "processed_at", "batch_id", "explanation",
        ])

    df = pd.DataFrame(rows)
    if "processed_at" in df.columns:
        df["processed_at"] = pd.to_datetime(df["processed_at"], errors="coerce")
        return df.sort_values("processed_at", ascending=False)
    return df


@st.cache_data(ttl=60)
def load_test_results() -> Optional[Dict]:
    """Load saved test set evaluation metrics."""
    path = config.MODEL_DIR / "test_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_flagged(df: pd.DataFrame) -> pd.DataFrame:
    """Return only fraud-flagged rows."""
    if df.empty or "fraud_flag" not in df.columns:
        return pd.DataFrame()
    return df[df["fraud_flag"] == 1].copy()


# ─────────────────────────────────────────────
# Page 1 — Live Monitor
# Root cause fix: use st.rerun() instead of
# while True loop, and assign unique keys to
# every widget that gets recreated on refresh.
# ─────────────────────────────────────────────

def page_live_monitor() -> None:
    """Real-time fraud monitoring — refreshes via st.rerun()."""
    st.title("🔴 Live Fraud Monitor")

    # Auto-refresh toggle in sidebar
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh (3s)", value=True, key="live_monitor_auto_refresh"
    )

    df = load_results()

    # ── Metric cards ──────────────────────────────────────
    total = len(df)
    fraud_count = int(df["fraud_flag"].sum()) if "fraud_flag" in df.columns else 0
    fraud_rate = (fraud_count / max(total, 1)) * 100

    if "latency_ms" in df.columns and total > 0:
        avg_latency_ms = pd.to_numeric(df["latency_ms"], errors="coerce").mean()
        if pd.isna(avg_latency_ms):
            avg_latency_ms = 0.0
    else:
        avg_latency_ms = 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Fraud Detected", f"{fraud_count:,}")
    col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    col4.metric("Avg Inference Latency", f"{avg_latency_ms:.1f} ms")

    st.divider()

    # ── Fraud rate over time chart ─────────────────────────
    st.subheader("Fraud Rate Over Time")
    if not df.empty and "processed_at" in df.columns:
        df_time = df.copy()
        df_time["minute"] = df_time["processed_at"].dt.floor("1min")
        time_series = (
            df_time.groupby("minute")
            .agg(total=("fraud_flag", "count"), flagged=("fraud_flag", "sum"))
            .reset_index()
        )
        time_series["rate"] = (
            time_series["flagged"] / time_series["total"].clip(1)
        ) * 100
        time_series = time_series.tail(100)

        fig_rate = go.Figure()
        fig_rate.add_trace(go.Scatter(
            x=time_series["minute"],
            y=time_series["rate"],
            mode="lines+markers",
            name="Fraud Rate %",
            line=dict(color="#E53935", width=2),
            fill="tozeroy",
            fillcolor="rgba(229,57,53,0.1)",
        ))
        fig_rate.update_layout(
            xaxis_title="Time",
            yaxis_title="Fraud Rate (%)",
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        # Unique key prevents DuplicateElementId on re-render
        st.plotly_chart(
            fig_rate,
            use_container_width=True,
            key="live_monitor_fraud_rate_chart",
        )
    else:
        st.info("Waiting for pipeline data... Start the producer and streaming pipeline.")

    # ── Recent transactions table ──────────────────────────
    st.subheader("Recent Transactions")
    if not df.empty:
        display_cols = [c for c in [
            "transaction_id", "amount", "fraud_probability",
            "fraud_flag", "processed_at",
        ] if c in df.columns]
        display_df = df[display_cols].head(20).copy()

        if "fraud_probability" in display_df.columns:
            display_df["fraud_probability"] = display_df["fraud_probability"].map(
                lambda x: f"{float(x):.3f}" if pd.notna(x) else "—"
            )

        def _color_fraud(val):
            try:
                if float(val) >= float(config.FRAUD_THRESHOLD):
                    return "color: #E53935; font-weight: bold"
            except (ValueError, TypeError):
                pass
            return ""

        prob_col = ["fraud_probability"] if "fraud_probability" in display_df.columns else []
        styled = display_df.style.applymap(_color_fraud, subset=prob_col)
        st.dataframe(
            styled,
            use_container_width=True,
            height=400,
        )
    else:
        st.info("No transactions yet.")

    # ── Auto-refresh ──────────────────────────────────────
    if auto_refresh:
        st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')} — refreshing in 3s")
        time.sleep(3)
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────
# Page 2 — Fraud Investigation
# ─────────────────────────────────────────────

def page_investigation() -> None:
    """SHAP explanation and graph visualization for flagged transactions."""
    st.title("🔎 Fraud Investigation")

    df = load_results()
    flagged = get_flagged(df)

    if flagged.empty:
        st.warning("No flagged transactions yet. Run the pipeline to generate fraud alerts.")
        return

    txn_ids = flagged["transaction_id"].astype(str).tolist()
    selected_id = st.selectbox(
        "Select flagged transaction ID:",
        txn_ids,
        key="investigation_txn_selector",
    )

    row = flagged[flagged["transaction_id"].astype(str) == selected_id].iloc[0]
    explanation = row.get("explanation") or {}
    if isinstance(explanation, str):
        try:
            explanation = json.loads(explanation)
        except Exception:
            explanation = {}

    # ── Summary metrics ────────────────────────────────────
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            "Fraud Probability",
            f"{float(row.get('fraud_probability', 0)):.1%}",
        )
        st.metric(
            "Amount",
            f"${float(row.get('amount', 0)):.2f}",
        )
        st.metric(
            "Processed At",
            str(row.get("processed_at", "—"))[:19],
        )
    with col2:
        summary = explanation.get("human_readable_summary", "No explanation available.")
        st.info(f"**Summary:** {summary}")
        if bool(row.get("cold_start", False)):
            st.warning(
                "⚠️ Cold-start transaction — card, merchant, or email domain "
                "was never seen in training and was scored using a neutral "
                "fallback node. Treat this score as lower confidence."
            )
        elif bool(row.get("low_confidence", False)):
            st.warning(
                "⚠️ Near the decision threshold — this score is close enough "
                "to the boundary that small feature noise could flip the "
                "decision. Consider manual review rather than an automatic "
                "call."
            )

    st.divider()

    # ── Feature attribution chart (gradient × input, not SHAP) ─────────────
    st.subheader("Feature Contributions (Gradient Attribution)")
    top_features = explanation.get("top_features", [])

    if top_features:
        feat_df = pd.DataFrame(top_features)
        directions = feat_df.get("direction", pd.Series(
            ["increases_risk"] * len(feat_df)
        ))
        colors = [
            "#E53935" if d == "increases_risk" else "#43A047"
            for d in directions
        ]
        fig_shap = go.Figure(go.Bar(
            x=feat_df["attribution_value"],
            y=feat_df["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in feat_df["attribution_value"]],
            textposition="outside",
        ))
        fig_shap.update_layout(
            xaxis_title="Attribution (grad × input)",
            yaxis_title="Feature",
            height=max(300, len(feat_df) * 30),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(zeroline=True, zerolinewidth=2),
        )
        st.plotly_chart(
            fig_shap,
            use_container_width=True,
            key=f"inv_shap_chart_{selected_id}",
        )
    else:
        st.info("No attribution data available for this transaction.")

    st.divider()

    # ── Suspicious connections network graph ───────────────
    st.subheader("Suspicious Graph Connections")
    connections = explanation.get("suspicious_connections", [])

    if connections:
        node_x, node_y, node_text, node_color = [], [], [], []
        edge_x, edge_y = [], []

        node_x.append(0)
        node_y.append(0)
        node_text.append(f"Txn {selected_id}")
        node_color.append("#E53935")

        n_conns = len(connections)
        for i, conn in enumerate(connections):
            angle = 2 * math.pi * i / max(n_conns, 1)
            cx = math.cos(angle) * 1.5
            cy = math.sin(angle) * 1.5
            node_x.append(cx)
            node_y.append(cy)
            node_text.append(
                f"{conn.get('dst_type', '')}:\n{conn.get('connected_to', '')}"
            )
            importance = float(conn.get("importance", 0.0))
            node_color.append(f"rgba(229,57,53,{min(0.3 + importance, 1.0):.2f})")
            edge_x += [0, cx, None]
            edge_y += [0, cy, None]

        fig_graph = go.Figure()
        fig_graph.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=1, color="#aaa"),
            hoverinfo="none",
        ))
        fig_graph.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            marker=dict(size=20, color=node_color,
                        line=dict(width=1, color="white")),
            text=node_text,
            textposition="top center",
            hoverinfo="text",
        ))
        fig_graph.update_layout(
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(
            fig_graph,
            use_container_width=True,
            key=f"inv_graph_chart_{selected_id}",
        )

        # Connection table — only keep columns that exist
        conn_df = pd.DataFrame(connections)
        keep_cols = [c for c in
                     ["connected_to", "edge_type", "importance", "reason"]
                     if c in conn_df.columns]
        st.dataframe(
            conn_df[keep_cols].head(10),
            use_container_width=True,
        )
    else:
        st.info("No graph connection data available for this transaction.")


# ─────────────────────────────────────────────
# Page 3 — Model Performance
# ─────────────────────────────────────────────

def page_model_performance() -> None:
    """Training curves, confusion matrix, feature importance."""
    st.title("📊 Model Performance")

    # Training curves image
    st.subheader("Training Curves")
    if config.TRAINING_CURVES_PATH.exists():
        st.image(
            str(config.TRAINING_CURVES_PATH),
            use_column_width=True,
        )
    else:
        st.info("Training curves not found. Run `python training/train_gnn.py` first.")

    st.divider()

    # Test set metrics
    test_results = load_test_results()
    if test_results:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test AUC-ROC",   f"{test_results.get('test_auc', 0):.4f}")
        col2.metric("Test Precision", f"{test_results.get('test_precision', 0):.4f}")
        col3.metric("Test Recall",    f"{test_results.get('test_recall', 0):.4f}")
        col4.metric("Test F1",        f"{test_results.get('test_f1', 0):.4f}")

        split_method = test_results.get("split_method")
        cost_info = test_results.get("cost_based_threshold")
        if split_method or cost_info:
            caption_parts = []
            if split_method:
                caption_parts.append(f"Split: {split_method}")
            if cost_info:
                caption_parts.append(
                    f"Threshold {cost_info.get('threshold', 0):.3f} chosen by "
                    f"minimizing cost at an assumed {cost_info.get('fn_cost_assumption', 0):.0f}:"
                    f"{cost_info.get('fp_cost_assumption', 0):.0f} false-negative:false-positive "
                    f"cost ratio (illustrative — not a Lloyds-supplied figure)"
                )
            st.caption(" • ".join(caption_parts))

        st.divider()

        # Confusion matrix heatmap
        st.subheader("Confusion Matrix")
        cm = test_results.get("confusion_matrix")
        if cm:
            cm_df = pd.DataFrame(
                cm,
                index=["Actual: Legit", "Actual: Fraud"],
                columns=["Predicted: Legit", "Predicted: Fraud"],
            )
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale="Reds",
                aspect="auto",
            )
            fig_cm.update_layout(height=350)
            st.plotly_chart(
                fig_cm,
                use_container_width=True,
                key="perf_confusion_matrix_chart",
            )
    else:
        st.info("Test results not found. Run training first.")

    st.divider()

    # Feature importance from accumulated gradient attributions
    st.subheader("Feature Importance (from Gradient Attribution)")
    df = load_results()
    flagged = get_flagged(df)

    if not flagged.empty:
        all_attr: Dict[str, List[float]] = {}
        for _, frow in flagged.iterrows():
            exp = frow.get("explanation") or {}
            if isinstance(exp, str):
                try:
                    exp = json.loads(exp)
                except Exception:
                    continue
            for feat_dict in exp.get("top_features", []):
                feat = feat_dict.get("feature", "")
                val = abs(feat_dict.get("attribution_value", 0.0))
                all_attr.setdefault(feat, []).append(val)

        if all_attr:
            importance_data = [
                {"feature": k, "mean_abs_attribution": float(np.mean(v))}
                for k, v in all_attr.items()
            ]
            importance_df = (
                pd.DataFrame(importance_data)
                .sort_values("mean_abs_attribution", ascending=False)
                .head(20)
            )
            fig_imp = px.bar(
                importance_df,
                x="mean_abs_attribution",
                y="feature",
                orientation="h",
                color="mean_abs_attribution",
                color_continuous_scale="Reds",
                labels={"mean_abs_attribution": "Mean |Attribution|", "feature": "Feature"},
            )
            fig_imp.update_layout(
                height=max(300, len(importance_df) * 25),
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(
                fig_imp,
                use_container_width=True,
                key="perf_feature_importance_chart",
            )
        else:
            st.info("Not enough attribution data yet — run the pipeline to accumulate explanations.")
    else:
        st.info("No flagged transactions yet.")


# ─────────────────────────────────────────────
# Page 4 — Graph Explorer
# ─────────────────────────────────────────────

def page_graph_explorer() -> None:
    """Interactive subgraph for a given card ID."""
    st.title("🕸️ Graph Explorer")
    st.caption("Visualize all transactions connected to a specific card ID")

    card_id_input = st.text_input(
        "Enter card1 ID:",
        value="",
        key="graph_explorer_card_input",
    )

    if not card_id_input.strip():
        st.info("Enter a card ID to explore its transaction graph.")
        return

    try:
        card_id = int(card_id_input.strip())
    except ValueError:
        st.error("Card ID must be an integer.")
        return

    df = load_results()
    if df.empty:
        st.warning("No pipeline results available yet.")
        return

    st.subheader(f"Transactions for card1 = {card_id}")

    if "card1" not in df.columns:
        st.warning(
            "This results file doesn't have card1 recorded — re-run the "
            "pipeline to regenerate results with card1/merchant/email_domain included."
        )
        return

    card_txns = df[
        df["transaction_id"].notna() & (df["card1"] == card_id)
    ].copy()

    if card_txns.empty:
        st.info(f"No transactions found for card1 = {card_id}.")
        return

    sample = card_txns.head(30)

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    edge_x, edge_y = [], []

    # Card node at center
    node_x.append(0); node_y.append(0)
    node_text.append(f"Card {card_id}")
    node_color.append("#1976D2"); node_size.append(25)

    for i, (_, row) in enumerate(sample.iterrows()):
        angle = 2 * math.pi * i / max(len(sample), 1)
        cx = math.cos(angle) * 2.0
        cy = math.sin(angle) * 2.0
        node_x.append(cx); node_y.append(cy)
        is_fraud = row.get("fraud_flag", 0) == 1
        prob = float(row.get("fraud_probability", 0))
        node_text.append(
            f"Txn {row.get('transaction_id', '?')}<br>"
            f"Amount: ${float(row.get('amount', 0)):.2f}<br>"
            f"Fraud prob: {prob:.2f}"
        )
        node_color.append("#E53935" if is_fraud else "#43A047")
        node_size.append(15)
        edge_x += [0, cx, None]
        edge_y += [0, cy, None]

    fig_explorer = go.Figure()
    fig_explorer.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="#ccc"),
        hoverinfo="none",
    ))
    fig_explorer.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color="white"),
        ),
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        hovertext=node_text,
    ))
    for color, label in [
        ("#E53935", "Fraud"),
        ("#43A047", "Legitimate"),
        ("#1976D2", "Card node"),
    ]:
        fig_explorer.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=color),
            name=label,
        ))

    fig_explorer.update_layout(
        showlegend=True,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=550,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    st.plotly_chart(
        fig_explorer,
        use_container_width=True,
        key=f"graph_explorer_network_{card_id}",
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Transactions Shown", len(sample))
    flagged_count = int(sample["fraud_flag"].sum()) \
        if "fraud_flag" in sample.columns else 0
    col2.metric("Flagged", flagged_count)
    avg_amount = sample["amount"].mean() \
        if "amount" in sample.columns else 0.0
    col3.metric("Avg Amount", f"${avg_amount:.2f}")


# ─────────────────────────────────────────────
# Sidebar + routing
# ─────────────────────────────────────────────

def main() -> None:
    """Main entry point with sidebar navigation."""
    with st.sidebar:
        st.title("🛡️ Fraud Detection")
        st.caption("Real-Time GNN Pipeline")
        st.divider()

        page = st.radio(
            "Navigate",
            options=[
                "Live Monitor",
                "Fraud Investigation",
                "Model Performance",
                "Graph Explorer",
            ],
            index=0,
            key="sidebar_page_selector",
        )

        st.divider()
        df_sidebar = load_results()
        st.caption(f"Results dir: `{config.RESULTS_DIR.name}`")
        st.caption(f"Total records: {len(df_sidebar):,}")
        if not df_sidebar.empty and "fraud_flag" in df_sidebar.columns:
            st.caption(f"Fraud flagged: {int(df_sidebar['fraud_flag'].sum()):,}")

    if page == "Live Monitor":
        page_live_monitor()
    elif page == "Fraud Investigation":
        page_investigation()
    elif page == "Model Performance":
        page_model_performance()
    elif page == "Graph Explorer":
        page_graph_explorer()


if __name__ == "__main__":
    main()