"""
main.py — FastAPI REST API for FraudLens
-----------------------------------------
Exposes the LangGraph pipeline as a REST endpoint.
Run with: uvicorn api.main:app --reload
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import run_investigation

app = FastAPI(
    title="FraudLens API",
    description="Multi-agent fraud investigation copilot",
    version="1.0.0"
)


class InvestigationRequest(BaseModel):
    query: str = "Why was transaction volume high yesterday?"


class InvestigationResponse(BaseModel):
    alert_level: str
    confidence_score: float
    flagged_count: int
    total_transactions: int
    spike_ratio: float
    explanation: str
    report_path: str
    human_review: bool


@app.get("/health")
def health():
    return {"status": "ok", "service": "FraudLens"}


@app.post("/investigate", response_model=InvestigationResponse)
def investigate(request: InvestigationRequest):
    state = run_investigation(request.query)
    return InvestigationResponse(
        alert_level        = state["anomalies"]["alert_level"],
        confidence_score   = state["anomalies"]["confidence_score"],
        flagged_count      = state["anomalies"]["flagged_count"],
        total_transactions = state["anomalies"]["total_transactions"],
        spike_ratio        = state["raw_data"]["spike_ratio"],
        explanation        = state["explanation"],
        report_path        = state.get("report_path", ""),
        human_review       = state.get("human_review", False),
    )