"""
app.py — Streamlit Dashboard for FraudLens
-------------------------------------------
Run with: streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime, timedelta
from pipeline import run_investigation

st.set_page_config(
    page_title="FraudLens",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 FraudLens — Fraud Investigation Copilot")
st.caption("Multi-agent AI system for payment anomaly detection and executive reporting")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Investigation Controls")
    query = st.text_input(
        "Investigation Query",
        value="Why was transaction volume high yesterday?"
    )
    run_btn = st.button("🚀 Run Investigation", type="primary")
    st.divider()
    st.caption("FraudLens v1.0 | LangGraph + XGBoost + FAISS + Groq")

# ── Load transaction data for charts ─────────────────────────────────────
@st.cache_data
def load_data():
    db_path = os.path.join(os.path.dirname(__file__), "../data/transactions.db")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT date, COUNT(*) as txn_count,
               SUM(is_fraud) as fraud_count,
               AVG(amount_usd) as avg_amount
        FROM transactions
        GROUP BY date ORDER BY date
    """, conn)
    conn.close()
    return df

df = load_data()

# ── Transaction volume chart ──────────────────────────────────────────────
st.subheader("📊 Transaction Volume (Last 30 Days)")
fig = px.bar(
    df.tail(30), x="date", y="txn_count",
    color="fraud_count",
    color_continuous_scale="Reds",
    labels={"txn_count": "Transactions", "fraud_count": "Fraud Count"},
)
fig.update_layout(height=300, margin=dict(t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

# ── Run investigation ─────────────────────────────────────────────────────
if run_btn:
    with st.spinner("🤖 Agents running investigation..."):
        state = run_investigation(query)

    anomalies   = state["anomalies"]
    raw_data    = state["raw_data"]
    explanation = state["explanation"]

    # ── Alert banner ──────────────────────────────────────────────────────
    alert = anomalies["alert_level"]
    color_map = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
    st.subheader(f"{color_map.get(alert, '⚪')} Alert Level: {alert}")

    # ── Key metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions Yesterday", f"{raw_data['txn_count']:,}")
    col2.metric("Spike Ratio", f"{raw_data['spike_ratio']}x")
    col3.metric("Flagged Transactions", anomalies["flagged_count"])
    col4.metric("Model Confidence", f"{anomalies['confidence_score']:.1%}")

    st.divider()

    # ── SHAP feature importance ───────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🎯 SHAP Feature Importance")
        shap_df = pd.DataFrame(
            list(anomalies["feature_importance"].items()),
            columns=["Feature", "SHAP Value"]
        )
        fig2 = px.bar(
            shap_df, x="SHAP Value", y="Feature",
            orientation="h", color="SHAP Value",
            color_continuous_scale="Reds"
        )
        fig2.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        st.subheader("🏪 Top Suspicious Merchants")
        merchant_df = pd.DataFrame([
            {"Merchant": m, "Flagged": d["flagged_count"], "Avg Score": round(d["avg_score"], 2)}
            for m, d in anomalies["top_risky_merchants"].items()
        ])
        st.dataframe(merchant_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Root cause explanation ────────────────────────────────────────────
    st.subheader("🧠 Root Cause Analysis")
    if state.get("human_review"):
        st.warning(explanation)
    else:
        st.info(explanation)

    # ── Report download ───────────────────────────────────────────────────
    report_path = state.get("report_path", "")
    if report_path and os.path.exists(report_path):
        with open(report_path, "rb") as f:
            st.download_button(
                label="📄 Download Executive PDF Report",
                data=f,
                file_name=os.path.basename(report_path),
                mime="application/pdf"
            )