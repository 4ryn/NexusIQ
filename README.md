# NexusIQ — AI Business Intelligence Platform

> Combines machine learning predictions, retrieval-augmented generation,
> and large language model reasoning to answer business questions with evidence — not guesswork.

---

## What This System Does

| Question | How NexusIQ answers it |
|----------|------------------------|
| *What will Q4 revenue be and why?* | XGBoost time-series model + SHAP feature importance |
| *Which customers will churn in the next 90 days?* | XGBoost classifier (AUC-ROC 0.924) + behavioural drivers |
| *What happens if we reduce churn by 2pp and close 10 more deals?* | LangGraph scenario agent with quantified ARR impact |
| *What should we prioritise in Q4?* | ReAct agent synthesises ML + documents → ranked recommendations |

---

## Architecture

```
Data Sources (UCI Retail · Kaggle Churn · Your PDFs)
              ↓
    ETL Pipeline  →  Processed Parquet files
              ↓                  ↓
    RAG Indexer             ML Models
  (ChromaDB + local         (XGBoost Forecast
   HuggingFace embeddings)   XGBoost Churn
   Business docs, PDFs       SHAP explanations)
              ↓                  ↓
        LangGraph ReAct Agent
     ┌──────────────────────┐
     │ rag_search_tool       │  ← retrieves business context
     │ sales_forecast_tool   │  ← runs ML predictions
     │ churn_analysis_tool   │  ← scores risk + SHAP
     │ scenario_tool         │  ← what-if impact
     └──────────────────────┘
              ↓
    LLM (Groq cloud / Ollama local)
              ↓
    FastAPI REST API  ←→  Streamlit Dashboard
```

---

## Free Datasets — Download First

Place files in `data/raw/` exactly as named. System uses synthetic fallback if absent.

| Dataset | URL | Save as | Used for |
|---------|-----|---------|----------|
| UCI Online Retail | https://archive.ics.uci.edu/dataset/352/online+retail | `online_retail.xlsx` | Sales forecast, RFM |
| Telco Churn | https://kaggle.com/datasets/blastchar/telco-customer-churn | `customer_churn.csv` | Churn classifier |
| Superstore (optional) | https://kaggle.com/datasets/vivek468/superstore-dataset-final | `superstore.csv` | Regional breakdown |

---

## Setup (Python 3.11.9 · Windows · PowerShell)

### Step 1 — Run setup script
```powershell
cd C:\Users\YourName\Projects\nexusiq
.\setup.ps1
```

### Step 2 — Add your API key
Open `configs\.env` and set:
```
GROQ_API_KEY=gsk_your_key_here
```
Free key: https://console.groq.com — 6,000 tokens/minute.

### Step 3 — (Optional) Local LLM
```powershell
# Download from https://ollama.com/download, then:
ollama pull phi3:mini
ollama pull llama3.2     # 2 GB, fast
ollama pull mistral      # 4 GB, best reasoning
```

### Step 4 — Build pipeline and train models

**IMPORTANT: Always use `-m` flag for training — never run train.py directly.**

```powershell
.\venv\Scripts\Activate.ps1

python -m src.pipeline.etl        # builds processed data
python -m src.rag.retriever       # builds ChromaDB vector index
python -m src.ml.train            # trains XGBoost models
```

### Step 5 — Start the system (two terminals)
```powershell
# Terminal 1
uvicorn src.api.main:app --reload --port 8000

# Terminal 2
streamlit run frontend\dashboard.py
```

Open **http://localhost:8501** (dashboard) and **http://localhost:8000/docs** (API).

---

## Why `python -m src.ml.train` Not `python src\ml\train.py`

When you run a file directly, Python registers classes under `__main__`.
Joblib saves that module path in the `.joblib` file.
When the API later loads the model, it looks for `__main__.SalesForecastModel` — which no longer exists.

Using `-m` makes Python register classes under `src.ml.train.SalesForecastModel`, which is
findable at load time. **This is the #1 cause of model-loading errors in this stack.**

If you see `AttributeError: Can't get attribute 'SalesForecastModel'`:
```powershell
Remove-Item data\models\*.joblib
python -m src.ml.train
```

---

## Adding Your Own Documents

Drop any PDF or TXT into `data/raw/documents/`, then rebuild the index:
```powershell
python -m src.rag.retriever
```
The AI Analyst will automatically cite your documents in its answers.

---

## API Endpoints

| Endpoint | Method | Body / Params | Description |
|----------|--------|---------------|-------------|
| `GET /health` | — | — | System status |
| `POST /query` | JSON | `{question, structured}` | Main AI query |
| `POST /forecast` | JSON | `{product, region, months_ahead}` | Sales forecast |
| `POST /churn` | Query | `segment, top_n` | Churn scoring |
| `POST /scenario` | JSON | scenario params | What-if analysis |
| `GET /documents/search` | Query | `q, k` | Semantic doc search |
| `GET /rfm` | — | — | RFM segmentation |
| `GET /docs` | — | — | Swagger UI |

---

## Project Structure

```
nexusiq/
├── src/
│   ├── utils/config.py        ← Central config from .env
│   ├── pipeline/etl.py        ← Data loading + feature engineering
│   ├── ml/
│   │   ├── train.py           ← XGBoost Forecast + Churn + SHAP
│   │   └── loader.py          ← Safe joblib loader (fixes __main__ bug)
│   ├── rag/retriever.py       ← LangChain + ChromaDB knowledge base
│   ├── llm/client.py          ← Groq + Ollama with auto-fallback
│   ├── graph/agent.py         ← LangGraph ReAct agent + 4 tools
│   └── api/main.py            ← FastAPI REST API
├── frontend/dashboard.py      ← Streamlit dashboard (5 pages)
├── data/
│   ├── raw/                   ← Datasets + your documents
│   ├── processed/             ← ETL output (Parquet)
│   ├── embeddings/chroma/     ← ChromaDB index (persistent)
│   └── models/                ← Trained models (joblib)
├── tests/test_all.py          ← Pytest suite (offline-safe)
├── configs/.env               ← API keys + settings
├── requirements.txt           ← Pinned Python 3.11.9 deps
```
