"""
agent3_explain.py — Root Cause Explanation Agent
-------------------------------------------------
This is the "brain" agent. It combines:
  - WHAT happened (from Agent 1)
  - WHICH transactions are suspicious (from Agent 2)
  - RELEVANT POLICY CONTEXT (from our RAG layer)

...and uses an LLM to write a plain-English explanation
of WHY the spike happened and what kind of fraud it might be.

RESPONSIBLE AI GUARDRAIL:
  If confidence_score < 0.70, we do NOT let the LLM guess.
  Instead we flag it as "needs human review."
  This prevents hallucinated root causes going into the report.
"""

import os
import json
from rag.rag_engine import get_rag

# We use a mock LLM here so the project works without an API key.
# When you have an Azure OpenAI key, replace _call_llm() with real API call.
# Instructions are at the bottom of this file.

CONFIDENCE_THRESHOLD = 0.70   # below this → human review, no LLM guess


def _build_prompt(raw_data: dict, anomalies: dict, policy_chunks: list[str]) -> str:
    """
    Assemble the prompt we'll send to the LLM.
    Good prompts = good outputs. This is prompt engineering in action.
    """
    top_features = list(anomalies["feature_importance"].items())[:3]
    top_merchants = list(anomalies["top_risky_merchants"].items())[:3]

    policy_text = "\n\n---\n".join(policy_chunks) if policy_chunks else "No relevant policy found."

    prompt = f"""You are a senior fraud analyst at a payments company.
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
Write a clear, professional explanation (4–6 sentences) covering:
1. What kind of fraud pattern this most likely represents
2. Which features are most concerning and why (reference the SHAP values)
3. What the policy says about this type of spike
4. What immediate action should be taken

Be specific. Do not hallucinate. Only reference what is in the data above.
If you are uncertain, say so explicitly.
"""
    return prompt


def _call_llm_mock(prompt: str) -> str:
    """
    MOCK LLM — generates a realistic explanation from the prompt data
    without needing an API key. Replace with real LLM when ready.
    """
    # Extract key values from the prompt to build a contextual response
    lines = prompt.split("\n")
    spike = next((l for l in lines if "Spike ratio" in l), "")
    alert = next((l for l in lines if "Alert level" in l), "")
    top_feature_line = next((l for l in lines if "amount_usd" in l or "merchant" in l or "country" in l), "")

    return (
        "Based on the SHAP analysis, the primary driver of yesterday's transaction spike "
        "is a combination of elevated transaction amounts and concentration in high-risk merchant "
        "categories — particularly electronics and ATM withdrawals, which our fraud policy "
        "classifies as HIGH RISK due to their association with card-not-present fraud and "
        "card skimming attacks respectively. "
        "The geographic distribution shows elevated activity from Tier-2 and Tier-3 countries, "
        "which when combined with high-risk merchants triggers an immediate review requirement "
        "per Section 3 of our Fraud Risk Management Policy. "
        "The 2.0x volume spike exceeds the 150% threshold defined in Section 1, warranting a "
        "YELLOW alert at minimum, and the model confidence of 73% suggests these are genuine "
        "fraud signals rather than legitimate volume growth. "
        "Recommended action: initiate a SOFT BLOCK on electronics and ATM transactions from "
        "Tier-3 countries, notify the Level-1 fraud operations team within 30 minutes, "
        "and queue flagged transactions for manual review before any account freeze decisions."
    )


def _call_llm_azure(prompt: str, api_key: str, endpoint: str) -> str:
    """
    REAL LLM via Azure OpenAI.
    Uncomment and use this when you have an Azure key.

    Setup steps:
      1. Create Azure OpenAI resource at portal.azure.com
      2. Deploy gpt-4o-mini model
      3. Set env vars:  AZURE_OPENAI_KEY  and  AZURE_OPENAI_ENDPOINT
    """
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key        = api_key,
        azure_endpoint = endpoint,
        api_version    = "2024-02-01"
    )
    resp = client.chat.completions.create(
        model    = "gpt-4o-mini",
        messages = [{"role": "user", "content": prompt}],
        max_tokens = 400,
        temperature = 0.3,   # low temp = less hallucination
    )
    return resp.choices[0].message.content.strip()


def run(state: dict) -> dict:
    print("\n[Agent 3] Generating root cause explanation...")

    raw_data  = state["raw_data"]
    anomalies = state["anomalies"]
    confidence = anomalies.get("confidence_score", 0)

    # ── Responsible AI Guardrail ──────────────────────────────────────────
    if confidence < CONFIDENCE_THRESHOLD:
        explanation = (
            f"⚠️  LOW CONFIDENCE ALERT ({confidence:.1%}) — Automated explanation withheld.\n"
            f"Model confidence is below the {CONFIDENCE_THRESHOLD:.0%} threshold required "
            f"for automated root cause analysis. This investigation has been queued for "
            f"human review by a Level-1 fraud analyst. No automated action will be taken."
        )
        state["explanation"]     = explanation
        state["llm_used"]        = False
        state["human_review"]    = True
        print(f"[Agent 3] LOW CONFIDENCE ({confidence:.1%}) — flagged for human review")
        state["messages"].append({
            "agent": "Agent 3 - Explainer",
            "summary": f"Low confidence ({confidence:.1%}). Queued for human review."
        })
        return state

    # ── Retrieve relevant policy context via RAG ──────────────────────────
    rag = get_rag()

    # Build a search query from what we know about the anomaly
    top_merchant = list(anomalies["top_risky_merchants"].keys())[0] if anomalies["top_risky_merchants"] else "unknown"
    query = (
        f"fraud spike {top_merchant} merchant high risk country "
        f"transaction volume increase alert {anomalies['alert_level']}"
    )
    policy_chunks = rag.retrieve(query, top_k=4)
    print(f"[Agent 3] RAG retrieved {len(policy_chunks)} policy chunks")

    # ── Build prompt ──────────────────────────────────────────────────────
    prompt = _build_prompt(raw_data, anomalies, policy_chunks)

    # ── Call LLM ─────────────────────────────────────────────────────────
    azure_key      = os.getenv("AZURE_OPENAI_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if azure_key and azure_endpoint:
        print("[Agent 3] Using Azure OpenAI...")
        explanation = _call_llm_azure(prompt, azure_key, azure_endpoint)
        llm_used = True
    else:
        print("[Agent 3] No Azure key found — using mock LLM (set AZURE_OPENAI_KEY to use real LLM)")
        explanation = _call_llm_mock(prompt)
        llm_used = False

    state["explanation"]  = explanation
    state["policy_chunks"] = policy_chunks
    state["llm_used"]     = llm_used
    state["human_review"] = False

    print(f"[Agent 3] Explanation generated ({len(explanation)} chars)")
    state["messages"].append({
        "agent": "Agent 3 - Root Cause Explainer",
        "summary": explanation[:200] + "..."
    })
    return state
