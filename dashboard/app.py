"""
app.py — FraudLens Streamlit Dashboard
---------------------------------------
Run with:  streamlit run dashboard/app.py
This gives you a visual interface to:
  - Ask fraud investigation questions
  - See all four agents running live
  - View charts and anomaly breakdowns
  - Download the executive PDF report
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import sqlite3
import json
from datetime import datetime, timedelta
from pipeline import run_investigation

st.set_page_config(
    page_title = "FraudLens — Fraud Investigation Copilot",
    page_icon  = "🔍",
    layout     = "wide"
)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#EB001B; margin-bottom:0'>🔍 FraudLens</h1>
<p style='color:#888; font-size:16px; margin-top:4px'>
  Financial Fraud Investigation Copilot — Powered by LangGraph · XGBoost · RAG
</p>
<hr style='border-color:#eee'>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Ask a question")
    query = st.text_area(
        "Investigation query",
        value="Why was transaction volume high yesterday?",
        height=100
    )
    run_btn = st.button("🚀 Run Investigation", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.markdown("""
- 🔗 LangGraph (agent orchestration)
- 🌲 XGBoost + SHAP (anomaly detection)
- 📚 RAG + FAISS (policy retrieval)
- 🐍 Python + SQL + FastAPI
- 🐳 Docker ready
""")

# ── Main area ─────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "../data/transactions.db")

# Show existing data at a glance
conn = sqlite3.connect(DB_PATH)
daily_df = pd.read_sql("""
    SELECT date, COUNT(*) as txn_count, SUM(is_fraud) as fraud_count
    FROM transactions
    GROUP BY date ORDER BY date DESC LIMIT 14
""", conn)
conn.close()

col1, col2, col3, col4 = st.columns(4)
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
yest_row  = daily_df[daily_df["date"] == yesterday]
yest_count = int(yest_row["txn_count"].iloc[0]) if len(yest_row) else 0
baseline   = daily_df[daily_df["date"] != yesterday]["txn_count"].mean()

col1.metric("Yesterday's transactions", f"{yest_count:,}", f"+{yest_count - baseline:.0f} vs avg")
col2.metric("30-day daily average",     f"{baseline:.0f}")
col3.metric("Spike ratio",              f"{yest_count/baseline:.1f}x" if baseline else "N/A")
col4.metric("Total records in DB",      f"{daily_df['txn_count'].sum():,}")

# Volume chart
st.markdown("#### Transaction volume — last 14 days")
chart_df = daily_df.sort_values("date").set_index("date")
st.bar_chart(chart_df["txn_count"])

# Run pipeline
if run_btn:
    st.markdown("---")
    st.markdown("### 🤖 Agent Pipeline Running...")

    progress   = st.progress(0)
    status_box = st.empty()

    with st.spinner("Agent 1: Pulling transaction data..."):
        status_box.info("**Agent 1** — Data Ingestion: querying database...")
        import time; time.sleep(0.3)

    # Actually run the pipeline
    with st.spinner("Running full investigation..."):
        result = run_investigation(query)
    progress.progress(100)

    # ── Results ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Investigation Results")

    raw       = result.get("raw_data", {})
    anomalies = result.get("anomalies", {})
    alert     = anomalies.get("alert_level", "UNKNOWN")

    alert_colors = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
    st.markdown(f"## {alert_colors.get(alert,'⚪')} Alert Level: **{alert}**")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions", f"{raw.get('txn_count',0):,}")
    m2.metric("Spike ratio",  f"{raw.get('spike_ratio',0)}x")
    m3.metric("Flagged",      f"{anomalies.get('flagged_count',0)}")
    m4.metric("Confidence",   f"{anomalies.get('confidence_score',0):.1%}")

    # Feature importance
    st.markdown("#### 🎯 SHAP Feature Importance")
    fi = anomalies.get("feature_importance", {})
    if fi:
        fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "SHAP Importance"])
        st.bar_chart(fi_df.set_index("Feature"))

    # Root cause
    st.markdown("#### 🧠 Root Cause Explanation")
    explanation = result.get("explanation", "")
    if result.get("human_review"):
        st.warning(explanation)
    else:
        st.info(explanation)

    # Agent log
    with st.expander("📋 Agent Pipeline Log"):
        for msg in result.get("messages", []):
            st.markdown(f"**{msg['agent']}**")
            st.markdown(f"> {msg['summary']}")
            st.markdown("---")

    # PDF download
    report_path = result.get("report_path", "")
    if report_path and os.path.exists(report_path):
        with open(report_path, "rb") as f:
            st.download_button(
                label     = "📥 Download Executive PDF Report",
                data      = f,
                file_name = os.path.basename(report_path),
                mime      = "application/pdf",
                type      = "primary"
            )
