"""
agent1_data.py — Data Ingestion Agent
--------------------------------------
This agent is like a data analyst who opens the database,
runs SQL queries, and hands the findings to the next agent.

It answers: "What actually happened in the data yesterday?"
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/transactions.db")


def run(state: dict) -> dict:
    """
    Pulls yesterday's transaction data and computes:
      - transaction count vs 30-day average  (to find the spike)
      - breakdown by merchant category
      - breakdown by country
      - breakdown by device type
      - breakdown by hour of day
    Returns all findings in state["raw_data"].
    """
    print("\n[Agent 1] Pulling transaction data from database...")

    conn = sqlite3.connect(DB_PATH)
    today     = datetime.now().date()
    month_ago = today - timedelta(days=31)

    # Find the most recent date that has transactions (handles test data date offsets)
    result = conn.execute(
        "SELECT date FROM transactions ORDER BY date DESC LIMIT 1"
    ).fetchone()
    yesterday = pd.Timestamp(result[0]).date() if result else today - timedelta(days=1)

    # ── 1. Daily summary for the past 30 days ────────────────────────────
    daily_df = pd.read_sql("""
        SELECT date,
               COUNT(*)                        AS txn_count,
               ROUND(SUM(amount_usd), 2)       AS total_amount,
               ROUND(AVG(amount_usd), 2)       AS avg_amount,
               SUM(is_fraud)                   AS fraud_count
        FROM transactions
        WHERE date >= ?
        GROUP BY date
        ORDER BY date
    """, conn, params=[str(month_ago)])

    # ── 2. Yesterday's detailed breakdown ────────────────────────────────
    yesterday_df = pd.read_sql("""
        SELECT * FROM transactions WHERE date = ?
    """, conn, params=[str(yesterday)])

    # ── 3. 30-day baseline (exclude yesterday) ────────────────────────────
    baseline_df = daily_df[daily_df["date"] != str(yesterday)]
    baseline_avg = baseline_df["txn_count"].mean() if len(baseline_df) > 0 else 1

    yesterday_row = daily_df[daily_df["date"] == str(yesterday)]
    yesterday_count = int(yesterday_row["txn_count"].iloc[0]) if len(yesterday_row) > 0 else 0
    yesterday_amount = float(yesterday_row["total_amount"].iloc[0]) if len(yesterday_row) > 0 else 0

    spike_ratio = yesterday_count / baseline_avg if baseline_avg > 0 else 1

    # ── 4. Breakdowns ─────────────────────────────────────────────────────
    merchant_breakdown = (
        yesterday_df.groupby("merchant_category")
        .agg(count=("transaction_id", "count"),
             total=("amount_usd", "sum"),
             fraud=("is_fraud", "sum"))
        .sort_values("count", ascending=False)
        .to_dict("index")
    )

    country_breakdown = (
        yesterday_df.groupby("country")
        .agg(count=("transaction_id", "count"),
             fraud=("is_fraud", "sum"))
        .sort_values("count", ascending=False)
        .to_dict("index")
    )

    hour_breakdown = (
        yesterday_df.groupby("hour")["transaction_id"]
        .count()
        .to_dict()
    )

    conn.close()

    raw_data = {
        "date"              : str(yesterday),
        "txn_count"         : yesterday_count,
        "total_amount_usd"  : round(yesterday_amount, 2),
        "baseline_avg"      : round(baseline_avg, 1),
        "spike_ratio"       : round(spike_ratio, 2),
        "merchant_breakdown": merchant_breakdown,
        "country_breakdown" : country_breakdown,
        "hour_breakdown"    : hour_breakdown,
        "fraud_count"       : int(yesterday_df["is_fraud"].sum()),
    }

    print(f"[Agent 1] Done. Yesterday: {yesterday_count:,} txns | Spike ratio: {spike_ratio:.1f}x")

    state["raw_data"] = raw_data
    state["messages"].append({
        "agent": "Agent 1 - Data Ingestion",
        "summary": f"Pulled {yesterday_count:,} transactions for {yesterday}. "
                   f"Spike ratio vs 30-day average: {spike_ratio:.1f}x"
    })
    return state
