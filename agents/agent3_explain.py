"""
agent3_explain.py — Root Cause Explanation Agent
-------------------------------------------------
Uses Groq (LLaMA 3.1-70B) as the LLM backend.
RAG retrieves relevant policy chunks to ground the explanation.
Confidence guardrail: if confidence < 0.70, skip LLM and flag for human review.
"""

import os
from groq import Groq
from rag.rag_engine import get_rag
from dotenv import load_dotenv

load_dotenv()

CONFIDENCE_THRESHOLD = 0.70


def _build_prompt(raw_data: dict, anomalies: dict, policy_chunks: list[str]) -> str:
    top_features = list(anomalies["feature_importance"].items())[:3]
    top_merchants = list(anomalies["top_risky_merchants"].items())[:3]
    policy_text = "\n\n---\n".join(policy_chunks) if policy_chunks else "No relevant policy found."

    return f"""You are a senior fraud analyst at a payments company.
You have been asked to explain a transaction volume spike detected yesterday.

=== DATA SUMMARY ===
Date              : {raw_data['date']}
Transaction count : {raw_data['txn_count']:,}  (normal average: {raw_data['baseline_avg']:.0f}/day)
Spike ratio       : {raw_data['spike_ratio']}x above normal
Total amount      : ${raw_data['total_amount_usd']:,.2f}
Fraud count       : {raw_data['fraud_count']}

=== ANOMALY DETECTION RESULTS ===
Flagged transactions : {anomalies['flagged_count']} out of {anomalies['total_transactions']}
Alert level          : {anomalies['alert_level']}
Model confidence     : {anomalies['confidence_score']:.1%}

Top fraud-driving features (SHAP analysis):
{chr(10).join(f"  - {k}: {v}" for k, v in top_features)}

Top suspicious merchant categories:
{chr(10).join(f"  - {m}: {d['flagged_count']} flagged txns, avg score {d['avg_score']:.2f}" for m, d in top_merchants)}

=== RELEVANT FRAUD POLICY (from internal knowledge base) ===
{policy_text}

=== YOUR TASK ===
Write a clear, professional explanation (4-6 sentences) covering:
1. What kind of fraud pattern this most likely represents
2. Which features are most concerning and why (reference the SHAP values)
3. What the policy says about this type of spike
4. What immediate action should be taken

Be specific. Do not hallucinate. Only reference what is in the data above.
"""


def _call_groq(prompt: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def run(state: dict) -> dict:
    print("\n[Agent 3] Generating root cause explanation...")

    raw_data   = state["raw_data"]
    anomalies  = state["anomalies"]
    confidence = anomalies.get("confidence_score", 0)

    # ── Responsible AI Guardrail ──────────────────────────────────────────
    if confidence < CONFIDENCE_THRESHOLD:
        explanation = (
            f"⚠️  LOW CONFIDENCE ALERT ({confidence:.1%}) — Automated explanation withheld.\n"
            f"Model confidence is below the {CONFIDENCE_THRESHOLD:.0%} threshold required "
            f"for automated root cause analysis. This investigation has been queued for "
            f"human review by a Level-1 fraud analyst. No automated action will be taken."
        )
        state["explanation"]  = explanation
        state["llm_used"]     = False
        state["human_review"] = True
        print(f"[Agent 3] LOW CONFIDENCE ({confidence:.1%}) — flagged for human review")
        state["messages"].append({
            "agent": "Agent 3 - Explainer",
            "summary": f"Low confidence ({confidence:.1%}). Queued for human review."
        })
        return state

    # ── RAG retrieval ─────────────────────────────────────────────────────
    rag = get_rag()
    top_merchant = list(anomalies["top_risky_merchants"].keys())[0] if anomalies["top_risky_merchants"] else "unknown"
    query = (
        f"fraud spike {top_merchant} merchant high risk country "
        f"transaction volume increase alert {anomalies['alert_level']}"
    )
    policy_chunks = rag.retrieve(query, top_k=4)
    print(f"[Agent 3] RAG retrieved {len(policy_chunks)} policy chunks")

    # ── Call Groq LLM ─────────────────────────────────────────────────────
    prompt = _build_prompt(raw_data, anomalies, policy_chunks)

    try:
        explanation = _call_groq(prompt)
        llm_used = True
        print(f"[Agent 3] Groq LLM response received ({len(explanation)} chars)")
    except Exception as e:
        explanation = f"LLM call failed: {str(e)}. Manual review required."
        llm_used = False
        print(f"[Agent 3] Groq call failed: {e}")

    state["explanation"]   = explanation
    state["policy_chunks"] = policy_chunks
    state["llm_used"]      = llm_used
    state["human_review"]  = False

    state["messages"].append({
        "agent": "Agent 3 - Root Cause Explainer",
        "summary": explanation[:200] + "..."
    })
    return state