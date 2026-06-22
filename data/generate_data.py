import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

NUM_TRANSACTIONS = 100_000
FRAUD_RATE       = 0.08
DB_PATH          = os.path.join(os.path.dirname(__file__), "transactions.db")

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "restaurant", "travel",
    "clothing", "fuel", "pharmacy", "entertainment",
    "online_retail", "atm_withdrawal"
]

COUNTRIES    = ["IN", "US", "GB", "AE", "SG", "NG", "BR", "DE", "CN", "AU"]
DEVICE_TYPES = ["mobile", "web", "pos_terminal", "atm"]


def make_transactions(n, fraud_rate):
    today     = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)

    normal_count = int(n * 0.80)
    spike_count  = n - normal_count

    normal_dates = [
        today - timedelta(days=random.randint(2, 90),
                          hours=random.randint(0, 23),
                          minutes=random.randint(0, 59))
        for _ in range(normal_count)
    ]
    spike_dates = [
        yesterday + timedelta(hours=random.randint(0, 23),
                              minutes=random.randint(0, 59))
        for _ in range(spike_count)
    ]

    all_dates = normal_dates + spike_dates
    random.shuffle(all_dates)

    is_fraud = np.random.choice([0, 1], size=n, p=[1 - fraud_rate, fraud_rate])

    amounts = np.where(
        is_fraud == 1,
        np.random.uniform(200, 5000, n),
        np.random.uniform(5, 500, n)
    ).round(2)

    fraud_merchant_weights  = [0.05, 0.25, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.20, 0.15]
    normal_merchant_weights = [0.20, 0.08, 0.20, 0.10, 0.12, 0.10, 0.08, 0.05, 0.05, 0.02]

    merchants = []
    for fraud in is_fraud:
        w = fraud_merchant_weights if fraud else normal_merchant_weights
        merchants.append(np.random.choice(MERCHANT_CATEGORIES, p=w))

    fraud_country_weights  = [0.05, 0.15, 0.05, 0.15, 0.15, 0.20, 0.10, 0.05, 0.05, 0.05]
    normal_country_weights = [0.40, 0.20, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02]

    countries = []
    for fraud in is_fraud:
        w = fraud_country_weights if fraud else normal_country_weights
        countries.append(np.random.choice(COUNTRIES, p=w))

    devices = np.random.choice(DEVICE_TYPES, size=n, p=[0.40, 0.30, 0.20, 0.10])
    hours   = [d.hour for d in all_dates]

    df = pd.DataFrame({
        "transaction_id"    : [f"TXN{str(i).zfill(6)}" for i in range(n)],
        "timestamp"         : all_dates,
        "date"              : [d.date() for d in all_dates],
        "hour"              : hours,
        "amount_usd"        : amounts,
        "merchant_category" : merchants,
        "country"           : countries,
        "device_type"       : devices,
        "is_fraud"          : is_fraud,
    })
    return df.sort_values("timestamp").reset_index(drop=True)


def save_to_sqlite(df, db_path):
    conn = sqlite3.connect(db_path)
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.execute("""
        CREATE VIEW IF NOT EXISTS daily_summary AS
        SELECT date,
               COUNT(*)                       AS total_transactions,
               SUM(amount_usd)                AS total_amount,
               AVG(amount_usd)                AS avg_amount,
               SUM(is_fraud)                  AS fraud_count,
               ROUND(AVG(is_fraud)*100, 2)   AS fraud_rate_pct
        FROM transactions
        GROUP BY date ORDER BY date
    """)
    conn.commit()
    conn.close()
    print(f"[OK] Saved {len(df):,} transactions to {db_path}")


if __name__ == "__main__":
    print("Generating synthetic transaction data...")
    df = make_transactions(NUM_TRANSACTIONS, FRAUD_RATE)
    save_to_sqlite(df, DB_PATH)
    print(f"[OK] Date range : {df['date'].min()} to {df['date'].max()}")
    print(f"[OK] Fraud count: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"[OK] Yesterday  : {(df['date'] == (datetime.now()-timedelta(days=1)).date()).sum()} transactions (the spike)")