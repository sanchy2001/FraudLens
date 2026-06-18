"""
test_pipeline.py
----------------
Unit tests for FraudLens.
Tests each agent individually to make sure they work correctly.

Run with:  python -m pytest test_pipeline.py -v
"""

import pytest
import os
import sys
import sqlite3
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))


# ── Test 1: Database exists and has data ─────────────────────────────────
def test_database_exists():
    """Check the transactions database was created successfully."""
    db_path = os.path.join("data", "transactions.db")
    assert os.path.exists(db_path), "Database file not found — run generate_data.py first"


def test_database_has_transactions():
    """Check the database has 10,000 transactions."""
    db_path = os.path.join("data", "transactions.db")
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    conn.close()
    assert count == 10_000, f"Expected 10,000 transactions, got {count}"


def test_database_has_spike():
    """Check yesterday has significantly more transactions than average."""
    db_path = os.path.join("data", "transactions.db")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT date, COUNT(*) as cnt
        FROM transactions
        GROUP BY date
        ORDER BY date DESC
    """, conn)
    conn.close()

    top_day   = df.iloc[0]["cnt"]   # most recent day (the spike)
    other_avg = df.iloc[1:]["cnt"].mean()
    spike_ratio = top_day / other_avg

    assert spike_ratio > 5, f"Expected spike ratio > 5x, got {spike_ratio:.1f}x"


# ── Test 2: Model exists and works ────────────────────────────────────────
def test_model_exists():
    """Check the trained model file exists."""
    model_path = os.path.join("models", "fraud_model.pkl")
    assert os.path.exists(model_path), "Model not found — run train_model.py first"


def test_model_predicts():
    """Check the model can make predictions on sample data."""
    with open(os.path.join("models", "fraud_model.pkl"), "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    # Fake one transaction: high amount, electronics, Nigeria, web, 2am
    sample = pd.DataFrame([{
        "amount_usd"        : 2500.0,
        "merchant_category" : 1,     # electronics (encoded)
        "country"           : 5,     # NG (encoded)
        "device_type"       : 1,     # web (encoded)
        "hour"              : 2,
    }])

    prob = model.predict_proba(sample)[0][1]
    assert 0 <= prob <= 1, "Model output should be between 0 and 1"
    assert prob > 0.5, f"High-risk transaction should score > 0.5, got {prob:.3f}"


def test_model_roc_auc():
    """Check the model achieves acceptable ROC-AUC on test data."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelEncoder

    db_path = os.path.join("data", "transactions.db")
    conn    = sqlite3.connect(db_path)
    df      = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()

    with open(os.path.join("models", "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)

    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    X = df[["amount_usd", "merchant_category", "country", "device_type", "hour"]]
    y = df["is_fraud"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with open(os.path.join("models", "fraud_model.pkl"), "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    probs = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, probs)

    assert auc > 0.90, f"Expected ROC-AUC > 0.90, got {auc:.4f}"


# ── Test 3: RAG works ─────────────────────────────────────────────────────
def test_rag_index_exists():
    """Check the RAG index was built."""
    index_path = os.path.join("rag", "tfidf_index.pkl")
    assert os.path.exists(index_path), "RAG index not found — run rag_engine.py first"


def test_rag_retrieves_relevant_chunks():
    """Check RAG returns relevant policy text for a fraud query."""
    from rag.rag_engine import get_rag
    rag     = get_rag()
    results = rag.retrieve("electronics merchant category high risk fraud")

    assert len(results) > 0, "RAG returned no results"
    combined = " ".join(results).lower()
    assert "electronics" in combined or "high risk" in combined, \
        "RAG did not return relevant policy chunks"


# ── Test 4: Agent 1 works ─────────────────────────────────────────────────
def test_agent1_runs():
    """Check Agent 1 pulls data and populates state correctly."""
    from agents.agent1_data import run

    state = {"messages": [], "raw_data": {}}
    state = run(state)

    assert "raw_data" in state
    assert state["raw_data"]["txn_count"] > 0
    assert state["raw_data"]["spike_ratio"] > 1
    assert "merchant_breakdown" in state["raw_data"]


# ── Test 5: Full pipeline runs end to end ────────────────────────────────
def test_full_pipeline():
    """Check the entire 4-agent pipeline runs without crashing."""
    from pipeline import run_investigation

    result = run_investigation("Why was transaction volume high yesterday?")

    assert "raw_data"    in result
    assert "anomalies"   in result
    assert "explanation" in result
    assert "report_path" in result
    assert os.path.exists(result["report_path"]), "PDF report was not generated"
    assert len(result["messages"]) == 4, "Expected 4 agent messages"