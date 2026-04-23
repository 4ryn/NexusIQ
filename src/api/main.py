"""
NexusIQ REST API — FastAPI
Run: uvicorn src.api.main:app --reload --port 8000
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from src.utils.config import cfg
from src.graph.agent import decision_agent, sales_forecast_tool, churn_analysis_tool, scenario_tool
from src.llm.client import llm_client


app = FastAPI(
    title="NexusIQ Business Intelligence API",
    description="LangGraph · LangChain RAG · Groq/Ollama · XGBoost ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ── Request models ────────────────────────────────────────────────────

class QueryReq(BaseModel):
    question:   str  = Field(..., min_length=5, example="What are our top Q4 priorities?")
    structured: bool = Field(True)

class ForecastReq(BaseModel):
    product:      str = Field("all", example="Widget-A")
    region:       str = Field("all", example="United Kingdom")
    months_ahead: int = Field(3, ge=1, le=12)

class ScenarioReq(BaseModel):
    description:             str   = "Custom scenario"
    churn_rate_change_pp:    float = 0.0
    enterprise_deals_added:  int   = 0
    gross_margin_change_pp:  float = 0.0
    product_launch:          bool  = False
    product_launch_arr_m:    float = 0.0


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"service":"NexusIQ API","version":"1.0.0",
            "llm":cfg.effective_provider,"timestamp":datetime.utcnow().isoformat()}

@app.get("/health")
async def health():
    checks = {}
    checks["groq"]   = "configured" if cfg.has_groq else "no_api_key"
    try:
        import httpx
        r = httpx.get(f"{cfg.ollama_base_url}/api/tags", timeout=2.0)
        checks["ollama"] = "online" if r.status_code==200 else "offline"
    except Exception:
        checks["ollama"] = "offline"
    checks["vector_store"]    = "ready" if Path(cfg.chroma_dir).exists() and any(Path(cfg.chroma_dir).iterdir()) else "not_built"
    checks["forecast_model"]  = "ready" if (cfg.models_dir/"forecast.joblib").exists() else "not_trained"
    checks["churn_model"]     = "ready" if (cfg.models_dir/"churn.joblib").exists()    else "not_trained"
    return {"status":"healthy","checks":checks,"timestamp":datetime.utcnow().isoformat()}

@app.post("/query")
async def query(req: QueryReq):
    t0 = time.time()
    try:
        result = decision_agent.ask_structured(req.question) if req.structured \
                 else decision_agent.ask(req.question)
        result["processing_time_ms"] = round((time.time()-t0)*1000,1)
        result["timestamp"] = datetime.utcnow().isoformat()
        result["llm_provider"] = cfg.effective_provider
        return result
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(500, str(e))

@app.post("/forecast")
async def forecast(req: ForecastReq):
    try:
        raw = sales_forecast_tool.invoke({"product":req.product,"region":req.region,"months_ahead":req.months_ahead})
        return json.loads(raw)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/churn")
async def churn(segment: str = Query("all"), top_n: int = Query(50)):
    try:
        raw = churn_analysis_tool.invoke({"segment":segment,"top_n":top_n})
        return json.loads(raw)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/scenario")
async def scenario(req: ScenarioReq):
    try:
        raw = scenario_tool.invoke({
            "description":req.description,
            "churn_rate_change_pp":req.churn_rate_change_pp,
            "enterprise_deals_added":req.enterprise_deals_added,
            "gross_margin_change_pp":req.gross_margin_change_pp,
            "product_launch":req.product_launch,
            "product_launch_arr_m":req.product_launch_arr_m,
        })
        return json.loads(raw)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/documents/search")
async def search_docs(q: str = Query(..., min_length=3), k: int = Query(4, ge=1, le=10)):
    from src.rag.retriever import get_rag
    rag = get_rag()
    results = rag.retrieve_with_scores(q, k=k)
    return {"query":q,"results":[
        {"source":d.metadata.get("source"),"type":d.metadata.get("type"),
         "score":round(s,4),"excerpt":d.page_content[:300]}
        for d,s in results]}

@app.get("/documents")
async def list_docs():
    from src.rag.retriever import DOCS
    return {"count":len(DOCS),"documents":[{"title":d["title"],"type":d["type"]} for d in DOCS],
            "tip":"Drop PDF/TXT files in data/raw/documents/ then run: python -m src.rag.retriever"}

@app.get("/rfm")
async def rfm():
    p = cfg.processed_dir/"rfm.parquet"
    if not p.exists():
        return {"message":"RFM not computed. Run: python -m src.pipeline.etl"}
    import pandas as pd
    df = pd.read_parquet(p)
    summary = df.groupby("Segment").agg(count=("CustomerID","count"),
        avg_monetary=("Monetary","mean"),avg_recency=("Recency","mean")).round(2).to_dict("index")
    return {"segments":summary,"total_customers":len(df)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=cfg.api_host, port=cfg.api_port, reload=True)
