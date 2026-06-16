"""
agent2_anomaly.py — Anomaly Detection Agent
--------------------------------------------
This agent is like a security guard who looks at each transaction
and scores it: "How suspicious is this? 0 = clean, 1 = definitely fraud."

It uses our trained XGBoost model + SHAP to:
  1. Score every transaction from yesterday
  2. Find which ones are flagged as suspicious
  3. Explain WHY (using SHAP feature importances)
  4. Identify the top risk patterns
"""

import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
import shap
from datetime import datetime, timedelta

DB_PATH    = os.path.join(os.path.dirname(__file__), "../data/transactions.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/fraud_model.pkl")
ENC_PATH   = os.path.join(os.path.dirname(__file__), "../models/encoders.pkl")

FEATURE_COLS = ["amount_usd", "merchant_category", "country", "device_type", "hour"]
THRESHOLD    = 0.4   # transactions above this score are flagged


def run(state: dict) -> dict:
    print("\n[Agent 2] Running anomaly detection with XGBoost + SHAP...")

    # ── Load model and encoders ───────────────────────────────────────────
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model     = bundle["model"]
    explainer = bundle["explainer"]

    with open(ENC_PATH, "rb") as f:
        encoders = pickle.load(f)

    # ── Load yesterday's transactions ─────────────────────────────────────
    yesterday = state["raw_data"]["date"]
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM transactions WHERE date = ?",
        conn, params=[yesterday]
    )
    conn.close()

    if df.empty:
        state["anomalies"] = {"error": "No data for yesterday"}
        return state

    # ── Encode categorical columns (same as training) ─────────────────────
    for col, le in encoders.items():
        # Handle unseen labels gracefully
        known = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
        df[col] = le.transform(df[col])

    X = df[FEATURE_COLS]

    # ── Score every transaction ────────────────────────────────────────────
    fraud_probs = model.predict_proba(X)[:, 1]
    df["fraud_score"] = fraud_probs
    df["flagged"]     = (fraud_probs >= THRESHOLD).astype(int)

    flagged_df = df[df["flagged"] == 1]
    n_flagged  = len(flagged_df)
    print(f"[Agent 2] {n_flagged} transactions flagged out of {len(df):,}")

    # ── SHAP: explain the TOP 200 highest-risk transactions ──────────────
    # Computing SHAP for all 2000 would be slow; top 200 is representative
    top_risky = df.nlargest(200, "fraud_score")
    shap_values = explainer.shap_values(top_risky[FEATURE_COLS])

    # Mean absolute SHAP per feature = global feature importance
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = dict(zip(FEATURE_COLS, mean_shap.round(4).tolist()))
    # Sort descending
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    # ── Top suspicious merchants ──────────────────────────────────────────
    # Decode merchant names back from numbers for readability
    le_merchant = encoders["merchant_category"]
    df["merchant_name"] = le_merchant.inverse_transform(df["merchant_category"])
    flagged_by_merchant = (
        flagged_df.copy()
        .assign(merchant_name=le_merchant.inverse_transform(flagged_df["merchant_category"]))
        .groupby("merchant_name")
        .agg(flagged_count=("flagged", "sum"),
             avg_score=("fraud_score", "mean"))
        .sort_values("flagged_count", ascending=False)
        .head(5)
        .to_dict("index")
    )

    # ── Compute confidence score ──────────────────────────────────────────
    # How sure is the model? High score = confident it found real fraud.
    # We use average fraud probability of flagged transactions as proxy.
    confidence = float(flagged_df["fraud_score"].mean()) if n_flagged > 0 else 0.0
    confidence = round(confidence, 3)

    anomalies = {
        "total_transactions"   : len(df),
        "flagged_count"        : n_flagged,
        "flag_rate_pct"        : round(n_flagged / len(df) * 100, 2),
        "feature_importance"   : feature_importance,
        "top_risky_merchants"  : flagged_by_merchant,
        "avg_fraud_score"      : round(fraud_probs.mean(), 4),
        "max_fraud_score"      : round(fraud_probs.max(), 4),
        "confidence_score"     : confidence,
        "alert_level"          : (
            "CRITICAL" if confidence > 0.80 else
            "HIGH"     if confidence > 0.65 else
            "MEDIUM"   if confidence > 0.50 else
            "LOW"
        ),
    }

    print(f"[Agent 2] Alert level: {anomalies['alert_level']} | Confidence: {confidence:.1%}")

    state["anomalies"] = anomalies
    state["messages"].append({
        "agent": "Agent 2 - Anomaly Detection",
        "summary": (
            f"Flagged {n_flagged} transactions ({anomalies['flag_rate_pct']}%). "
            f"Top driver: {list(feature_importance.keys())[0]}. "
            f"Alert: {anomalies['alert_level']} (confidence {confidence:.1%})"
        )
    })
    return state
