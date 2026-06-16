# 🔍 FraudLens — Financial Fraud Investigation Copilot

> A multi-agent AI system that autonomously investigates payment transaction anomalies,
> retrieves context from compliance documents, explains root causes with XGBoost + SHAP,
> and generates executive reports — with built-in hallucination detection.

Built as a portfolio project targeting Mastercard's AI Engineer role (Pune, India).

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  LangGraph Orchestrator              │
│                                                     │
│  Agent 1          Agent 2          Agent 3          Agent 4  │
│  Data             Anomaly          Root Cause       Report   │
│  Ingestion   ──►  Detection   ──►  Explanation ──►  Generator│
│  (SQL)            (XGBoost          (RAG +           (PDF)   │
│                   + SHAP)           Azure OpenAI)            │
│                                         │                    │
│                              ┌──────────┘                    │
│                              │ Responsible AI Guard          │
│                              │ confidence < 70% → human      │
│                              │ review, no automated action   │
└─────────────────────────────────────────────────────┘
    │
    ▼
FastAPI (REST) + Streamlit (Dashboard)
```

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Agent orchestration | LangGraph | Multi-agent state machine |
| LLM backend | Azure OpenAI (gpt-4o-mini) | Root cause explanation |
| ML model | XGBoost + SHAP | Anomaly detection + explainability |
| RAG | TF-IDF / FAISS + LangChain | Policy document retrieval |
| Data | Python + SQL + SQLite | Transaction database |
| API | FastAPI | REST endpoint |
| Dashboard | Streamlit | Visual interface |
| Deployment | Docker + docker-compose | Containerised |
| Responsible AI | Confidence scoring | Hallucination guard |

---

## Quickstart (5 minutes)

### 1. Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/fraudlens.git
cd fraudlens
pip install -r requirements.txt
```

### 2. Generate data + train model
```bash
python data/generate_data.py      # creates transactions.db
python models/train_model.py      # trains XGBoost, saves model
python rag/rag_engine.py          # builds RAG index from policy docs
```

### 3. Run the pipeline
```bash
python pipeline.py
# or with a custom query:
python pipeline.py "Why did fraud spike in electronics yesterday?"
```

### 4. Launch the dashboard
```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

### 5. Launch the API
```bash
uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000/docs  (Swagger UI)
```

### 6. (Optional) Add Azure OpenAI for real LLM
```bash
cp .env.example .env
# Fill in AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT
```

---

## Docker (production-style)

```bash
# Build and start both API + dashboard
docker-compose up --build

# API:       http://localhost:8000
# Dashboard: http://localhost:8501
# API docs:  http://localhost:8000/docs
```

---

## API Usage

```bash
# Trigger an investigation
curl -X POST http://localhost:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"query": "Why was transaction volume high yesterday?"}'

# Download the PDF report
curl http://localhost:8000/report/2026-06-14 --output report.pdf
```

---

## Project Structure

```
fraudlens/
├── data/
│   ├── generate_data.py      # synthetic transaction generator
│   └── transactions.db       # SQLite database (auto-generated)
├── models/
│   ├── train_model.py        # XGBoost + SHAP training
│   ├── fraud_model.pkl       # saved model (auto-generated)
│   └── encoders.pkl          # label encoders (auto-generated)
├── rag/
│   ├── rag_engine.py         # TF-IDF RAG (swap to FAISS for prod)
│   └── tfidf_index.pkl       # saved RAG index (auto-generated)
├── agents/
│   ├── agent1_data.py        # data ingestion agent
│   ├── agent2_anomaly.py     # anomaly detection agent
│   ├── agent3_explain.py     # root cause explanation agent
│   └── agent4_report.py      # executive report agent
├── docs/
│   └── fraud_policy.txt      # policy documents for RAG
├── api/
│   └── main.py               # FastAPI REST API
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── reports/                  # generated PDF reports
├── pipeline.py               # LangGraph orchestrator
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Responsible AI Features

- **Confidence scoring**: Agent 2 computes a model confidence score for every investigation
- **Hallucination guard**: If confidence < 70%, Agent 3 withholds the LLM explanation and flags for human review — no automated action is taken
- **False positive budget**: The system respects a 0.5% FP budget per internal fraud policy (Section 8)
- **SHAP explainability**: Every fraud flag comes with a feature-level explanation — not a black box
- **Policy grounding**: RAG ensures the LLM explanation is anchored in real policy text, not invented context

---

## Resume Bullet Points

> Copy these into your resume after building the project:

- Built a 4-agent LangGraph fraud investigation pipeline over synthetic payment data; integrated XGBoost + SHAP for anomaly attribution and a RAG layer for policy-grounded root cause explanations
- Trained XGBoost classifier on 10,000 synthetic transactions achieving ROC-AUC of 0.9878; used SHAP values to surface top fraud-driving features per investigation
- Implemented a Responsible AI confidence scoring module that withholds LLM-generated explanations below 70% confidence, preventing hallucinated root causes from entering executive reports
- Deployed agent pipeline via FastAPI REST API and Streamlit dashboard, containerised with Docker and docker-compose; designed for Azure Container Apps deployment
- Integrated TF-IDF RAG over internal fraud policy documents to ground LLM explanations in real compliance text, reducing unsupported outputs

---

## Author

**Sanchit Kumar Mahto**
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)
