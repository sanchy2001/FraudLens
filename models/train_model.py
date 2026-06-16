import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

DB_PATH    = os.path.join(os.path.dirname(__file__), "../data/transactions.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")
ENC_PATH   = os.path.join(os.path.dirname(__file__), "encoders.pkl")


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()
    return df


def encode_features(df, encoders=None, fit=True):
    cat_cols = ["merchant_category", "country", "device_type"]
    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        for col in cat_cols:
            df[col] = encoders[col].transform(df[col])
    return df, encoders


def build_features(df):
    return df[["amount_usd", "merchant_category", "country", "device_type", "hour"]]


def train():
    print("Loading data...")
    df = load_data()

    df, encoders = encode_features(df, fit=True)
    X = build_features(df)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train):,} samples...")

    scale = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators     = 200,
        max_depth        = 6,
        learning_rate    = 0.05,
        scale_pos_weight = scale,
        eval_metric      = "auc",
        random_state     = 42,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred      = (y_pred_prob >= 0.4).astype(int)

    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"\nROC-AUC : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["normal", "fraud"]))

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "explainer": explainer}, f)

    with open(ENC_PATH, "wb") as f:
        pickle.dump(encoders, f)

    print(f"\n[OK] Model saved  → {MODEL_PATH}")
    print(f"[OK] Encoders saved → {ENC_PATH}")
    return model, explainer, encoders


if __name__ == "__main__":
    train()