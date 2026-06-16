"""
pipeline.py
-----------
This is the DIRECTOR of our movie.

LangGraph works like a flowchart with memory:
  - Each node = one agent function
  - The "state" dictionary is passed from agent to agent like a baton
  - Each agent reads from state, adds its findings, passes it on

Think of state like a shared notebook:
  Agent 1 writes: "here's the data"
  Agent 2 reads that, writes: "here are the anomalies"
  Agent 3 reads both, writes: "here's the explanation"
  Agent 4 reads all, writes: "here's the PDF report"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

from agents import agent1_data, agent2_anomaly, agent3_explain, agent4_report


# ── Define the shared state that flows between agents ─────────────────────
class FraudState(TypedDict):
    query       : str          # user's original question
    messages    : list         # running log of what each agent did
    raw_data    : dict         # output of Agent 1
    anomalies   : dict         # output of Agent 2
    explanation : str          # output of Agent 3
    policy_chunks: list        # policy text retrieved by RAG
    report_path : str          # output of Agent 4
    llm_used    : bool
    human_review: bool


def build_graph() -> StateGraph:
    """Construct the LangGraph pipeline."""
    graph = StateGraph(FraudState)

    # ── Add nodes (each node = one agent function) ────────────────────────
    graph.add_node("data_ingestion",   agent1_data.run)
    graph.add_node("anomaly_detection", agent2_anomaly.run)
    graph.add_node("root_cause",        agent3_explain.run)
    graph.add_node("report_generation", agent4_report.run)

    # ── Define the flow: A → B → C → D → END ─────────────────────────────
    graph.set_entry_point("data_ingestion")
    graph.add_edge("data_ingestion",    "anomaly_detection")
    graph.add_edge("anomaly_detection", "root_cause")
    graph.add_edge("root_cause",        "report_generation")
    graph.add_edge("report_generation", END)

    return graph.compile()


def run_investigation(query: str = "Why was transaction volume high yesterday?") -> dict:
    """
    Entry point. Call this with any natural language question.
    Returns the final state with all findings.
    """
    print("=" * 60)
    print("FraudLens Investigation Copilot")
    print("=" * 60)
    print(f"Query: {query}\n")

    graph = build_graph()

    initial_state: FraudState = {
        "query"        : query,
        "messages"     : [],
        "raw_data"     : {},
        "anomalies"    : {},
        "explanation"  : "",
        "policy_chunks": [],
        "report_path"  : "",
        "llm_used"     : False,
        "human_review" : False,
    }

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("INVESTIGATION COMPLETE")
    print("=" * 60)
    for msg in final_state["messages"]:
        print(f"\n[{msg['agent']}]\n  {msg['summary']}")

    print(f"\n Report: {final_state.get('report_path', 'N/A')}")
    return final_state


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Why was transaction volume high yesterday?"
    run_investigation(query)
