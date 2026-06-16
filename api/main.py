"""
main.py — FastAPI REST API
--------------------------
This exposes FraudLens as an API endpoint.
Think of it like a waiter at a restaurant:
  - You place an order (POST /investigate)
  - The kitchen (pipeline) does the work
  - The waiter brings back the result

Other systems (like a Slack bot or internal portal)
can call this API to trigger fraud investigations.

Run with:  uvicorn api.main:app --reload --port 8000
Then visit: http://localhost:8000/docs  (auto-generated docs!)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from pipeline import run_investigation
import uvicorn

app = FastAPI(
    title       = "FraudLens API",
    description = "Financial Fraud Investigation Copilot — LangGraph + XGBoost + RAG",
    version     = "1.0.0"
)


# ── Request / Response models ─────────────────────────────────────────────
class InvestigationRequest(BaseModel):
    query: str = "Why was transaction volume high yesterday?"


class AgentMessage(BaseModel):
    agent  : str
    summary: str


class InvestigationResponse(BaseModel):
    query         : str
    alert_level   : str
    confidence    : float
    spike_ratio   : float
    txn_count     : int
    flagged_count : int
    explanation   : str
    human_review  : bool
    report_path   : Optional[str]
    agent_log     : list[AgentMessage]


# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "FraudLens — Fraud Investigation Copilot",
        "version": "1.0.0",
        "endpoints": {
            "POST /investigate": "Run a fraud investigation",
            "GET  /report/{date}": "Download PDF report for a date",
            "GET  /health": "Health check",
            "GET  /docs": "Swagger UI (auto-generated)"
        }
    }


@app.get("/health")
def health():
    """Simple health check — used by Docker and load balancers."""
    return {"status": "healthy", "service": "fraudlens"}


@app.post("/investigate", response_model=InvestigationResponse)
def investigate(req: InvestigationRequest):
    """
    Run the full 4-agent fraud investigation pipeline.

    Example body:
    {
        "query": "Why was transaction volume high yesterday?"
    }
    """
    try:
        result = run_investigation(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    raw       = result.get("raw_data", {})
    anomalies = result.get("anomalies", {})

    return InvestigationResponse(
        query         = req.query,
        alert_level   = anomalies.get("alert_level", "UNKNOWN"),
        confidence    = anomalies.get("confidence_score", 0.0),
        spike_ratio   = raw.get("spike_ratio", 0.0),
        txn_count     = raw.get("txn_count", 0),
        flagged_count = anomalies.get("flagged_count", 0),
        explanation   = result.get("explanation", ""),
        human_review  = result.get("human_review", False),
        report_path   = result.get("report_path"),
        agent_log     = [AgentMessage(**m) for m in result.get("messages", [])]
    )


@app.get("/report/{date}")
def download_report(date: str):
    """
    Download the PDF investigation report for a given date.
    Example: GET /report/2026-06-14
    """
    report_dir  = os.path.join(os.path.dirname(__file__), "../reports")
    report_path = os.path.join(report_dir, f"fraud_report_{date}.pdf")

    if not os.path.exists(report_path):
        raise HTTPException(
            status_code = 404,
            detail      = f"No report found for {date}. Run /investigate first."
        )

    return FileResponse(
        path             = report_path,
        media_type       = "application/pdf",
        filename         = f"fraud_report_{date}.pdf"
    )


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
