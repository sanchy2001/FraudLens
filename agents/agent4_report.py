"""
agent4_report.py — Executive Report Generator
----------------------------------------------
This agent is like a PA who takes all the findings and
types them up into a clean PDF report that the CEO can read.

It takes:
  - raw_data     (from Agent 1)
  - anomalies    (from Agent 2)
  - explanation  (from Agent 3)

And produces a professional PDF saved to /reports/
"""

import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "../reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────────
MC_RED    = colors.HexColor("#EB001B")   # Mastercard red
MC_ORANGE = colors.HexColor("#F79E1B")   # Mastercard orange
DARK_GRAY = colors.HexColor("#2C2C2A")
MID_GRAY  = colors.HexColor("#5F5E5A")
LIGHT_BG  = colors.HexColor("#F8F8F6")
ALERT_COLORS = {
    "CRITICAL": colors.HexColor("#A32D2D"),
    "HIGH"    : colors.HexColor("#D85A30"),
    "MEDIUM"  : colors.HexColor("#BA7517"),
    "LOW"     : colors.HexColor("#3B6D11"),
}


def _alert_color(level: str) -> colors.Color:
    return ALERT_COLORS.get(level, MID_GRAY)


def run(state: dict) -> dict:
    print("\n[Agent 4] Generating executive PDF report...")

    raw_data    = state["raw_data"]
    anomalies   = state.get("anomalies") or {}
    explanation = state.get("explanation", "No explanation generated.")
    alert_level = anomalies.get("alert_level", "LOW")
    confidence  = anomalies.get("confidence_score", 0)
    human_review = state.get("human_review", False)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    fname = f"fraud_report_{raw_data['date']}.pdf"
    fpath = os.path.join(REPORTS_DIR, fname)

    doc = SimpleDocTemplate(
        fpath,
        pagesize      = A4,
        leftMargin    = 2*cm,
        rightMargin   = 2*cm,
        topMargin     = 2*cm,
        bottomMargin  = 2*cm,
    )

    styles   = getSampleStyleSheet()
    elements = []

    # ── Header ────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "title", fontSize=20, textColor=DARK_GRAY,
        spaceAfter=4, fontName="Helvetica-Bold"
    )
    sub_style = ParagraphStyle(
        "sub", fontSize=11, textColor=MID_GRAY, spaceAfter=2
    )
    body_style = ParagraphStyle(
        "body", fontSize=10, textColor=DARK_GRAY,
        leading=16, spaceAfter=6
    )
    label_style = ParagraphStyle(
        "label", fontSize=9, textColor=MID_GRAY, fontName="Helvetica-Bold"
    )
    value_style = ParagraphStyle(
        "value", fontSize=11, textColor=DARK_GRAY, fontName="Helvetica-Bold"
    )

    elements.append(Paragraph("FraudLens Investigation Report", title_style))
    elements.append(Paragraph("Financial Fraud Investigation Copilot — Automated Analysis", sub_style))
    elements.append(HRFlowable(width="100%", thickness=2, color=MC_RED, spaceAfter=12))

    # ── Alert banner ──────────────────────────────────────────────────────
    ac = _alert_color(alert_level)
    banner_data = [[
        Paragraph(f"ALERT LEVEL: {alert_level}", ParagraphStyle(
            "banner", fontSize=13, textColor=colors.white,
            fontName="Helvetica-Bold", alignment=TA_CENTER
        )),
        Paragraph(f"Confidence: {confidence:.1%}", ParagraphStyle(
            "conf", fontSize=11, textColor=colors.white, alignment=TA_CENTER
        )),
        Paragraph(f"Date: {raw_data['date']}", ParagraphStyle(
            "dt", fontSize=11, textColor=colors.white, alignment=TA_CENTER
        )),
    ]]
    banner = Table(banner_data, colWidths=["40%", "30%", "30%"])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), ac),
        ("PADDING",    (0,0), (-1,-1), 8),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.white),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))
    elements.append(banner)
    elements.append(Spacer(1, 16))

    # ── Key metrics ───────────────────────────────────────────────────────
    elements.append(Paragraph("Key Metrics", ParagraphStyle(
        "section", fontSize=13, textColor=DARK_GRAY,
        fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=8
    )))

    metrics_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Transactions (spike day)",
         f"{raw_data.get('txn_count', 0):,}",
         "30-day daily average",
         f"{raw_data.get('baseline_avg', 0):.0f}"],
        ["Spike ratio",
         f"{raw_data.get('spike_ratio', 0)}x",
         "Total amount (USD)",
         f"${raw_data.get('total_amount_usd', 0):,.0f}"],
        ["Flagged transactions",
         f"{anomalies.get('flagged_count', 0)}",
         "Flag rate",
         f"{anomalies.get('flag_rate_pct', 0)}%"],
        ["Fraud count (known)",
         f"{raw_data.get('fraud_count', 0)}",
         "Max fraud score",
         f"{anomalies.get('max_fraud_score', 0):.3f}"],
    ]
    metrics_table = Table(metrics_data, colWidths=["30%", "20%", "30%", "20%"])
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  DARK_GRAY),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT_BG, colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#D3D1C7")),
        ("PADDING",     (0,0), (-1,-1), 7),
        ("FONTNAME",    (1,1), (1,-1),  "Helvetica-Bold"),
        ("FONTNAME",    (3,1), (3,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,0), (0,-1),  MID_GRAY),
        ("TEXTCOLOR",   (2,0), (2,-1),  MID_GRAY),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 16))

    # ── Feature importance ────────────────────────────────────────────────
    elements.append(Paragraph("Top Fraud-Driving Features (SHAP Analysis)", ParagraphStyle(
        "section", fontSize=13, textColor=DARK_GRAY,
        fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=8
    )))

    shap_data = [["Feature", "SHAP Importance", "Interpretation"]]
    interpretations = {
        "amount_usd"         : "High transaction amounts signal card-not-present fraud",
        "merchant_category"  : "Electronics/ATM categories are high-risk (per policy Sec. 2)",
        "country"            : "Tier-3 country involvement elevates risk (per policy Sec. 3)",
        "device_type"        : "Device pattern anomalies indicate account takeover",
        "hour"               : "Off-hours transactions are weighted 2x in risk score",
    }
    for feat, imp in list(anomalies["feature_importance"].items())[:5]:
        shap_data.append([
            feat,
            f"{imp:.4f}",
            interpretations.get(feat, "")
        ])
    shap_table = Table(shap_data, colWidths=["22%", "18%", "60%"])
    shap_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  DARK_GRAY),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT_BG, colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#D3D1C7")),
        ("PADDING",     (0,0), (-1,-1), 7),
        ("TEXTCOLOR",   (1,1), (1,-1),  MC_RED),
        ("FONTNAME",    (1,1), (1,-1),  "Helvetica-Bold"),
    ]))
    elements.append(shap_table)
    elements.append(Spacer(1, 16))

    # ── Root cause explanation ────────────────────────────────────────────
    elements.append(Paragraph("Root Cause Analysis", ParagraphStyle(
        "section", fontSize=13, textColor=DARK_GRAY,
        fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=8
    )))

    if human_review:
        box_color = colors.HexColor("#FAEEDA")
        box_text_color = colors.HexColor("#633806")
    else:
        box_color = LIGHT_BG
        box_text_color = DARK_GRAY

    explanation_table = Table(
        [[Paragraph(explanation, ParagraphStyle(
            "expl", fontSize=10, textColor=box_text_color, leading=16
        ))]],
        colWidths=["100%"]
    )
    explanation_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), box_color),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#D3D1C7")),
        ("PADDING",    (0,0), (-1,-1), 10),
    ]))
    elements.append(explanation_table)
    elements.append(Spacer(1, 16))

    # ── Responsible AI note ───────────────────────────────────────────────
    elements.append(HRFlowable(width="100%", thickness=0.5, color=MID_GRAY, spaceAfter=8))
    elements.append(Paragraph(
        f"<b>Responsible AI Notice:</b> This report was generated automatically by FraudLens. "
        f"Model confidence: {confidence:.1%}. "
        f"{'⚠️ Low confidence — human review required before any action.' if human_review else 'Confidence above threshold — automated analysis applied.'} "
        f"All automated actions are subject to the 0.5% false positive budget per Section 8 of Fraud Policy.",
        ParagraphStyle("note", fontSize=8, textColor=MID_GRAY, leading=12)
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        f"Generated: {generated_at} | FraudLens v1.0 | Agent pipeline: LangGraph + XGBoost + RAG",
        ParagraphStyle("footer", fontSize=8, textColor=MID_GRAY)
    ))

    doc.build(elements)

    print(f"[Agent 4] Report saved → {fpath}")
    state["report_path"] = fpath
    state["messages"].append({
        "agent"  : "Agent 4 - Report Generator",
        "summary": f"PDF report saved to {fname}"
    })
    return state
