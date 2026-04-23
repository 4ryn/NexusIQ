"""
NexusIQ — Test Suite
All tests run offline (no API keys needed).
Run: pytest tests\ -v
"""
import sys, json, numpy as np, pandas as pd, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Config ────────────────────────────────────────────────────────────
class TestConfig:
    def test_loads(self):
        from src.utils.config import cfg
        assert hasattr(cfg, "groq_model")
        assert hasattr(cfg, "embedding_model")

    def test_has_groq_is_bool(self):
        from src.utils.config import cfg
        assert isinstance(cfg.has_groq, bool)

    def test_effective_provider(self):
        from src.utils.config import cfg
        assert cfg.effective_provider in ["groq","ollama"]


# ── ETL ───────────────────────────────────────────────────────────────
class TestETL:
    def test_synthetic_retail(self):
        from src.pipeline.etl import _synthetic_retail
        df = _synthetic_retail(12)
        assert len(df) > 0 and (df["Revenue"]>0).all()

    def test_synthetic_churn(self):
        from src.pipeline.etl import _synthetic_churn
        df = _synthetic_churn(200)
        assert set(df["Churn"].unique()).issubset({0,1})

    def test_engineer_sales(self):
        from src.pipeline.etl import _synthetic_retail, engineer_sales
        df = engineer_sales(_synthetic_retail(24))
        assert "lag_1" in df.columns and "month_sin" in df.columns

    def test_engineer_churn(self):
        from src.pipeline.etl import _synthetic_churn, engineer_churn
        df = engineer_churn(_synthetic_churn(200))
        assert "contract_ord" in df.columns

    def test_rfm(self):
        from src.pipeline.etl import _synthetic_retail, compute_rfm
        rfm = compute_rfm(_synthetic_retail(18))
        if not rfm.empty:
            assert "RFM" in rfm.columns and "Segment" in rfm.columns


# ── ML models ─────────────────────────────────────────────────────────
@pytest.fixture
def small_sales():
    from src.pipeline.etl import _synthetic_retail, engineer_sales
    return engineer_sales(_synthetic_retail(24))

@pytest.fixture
def small_churn():
    from src.pipeline.etl import _synthetic_churn, engineer_churn
    return engineer_churn(_synthetic_churn(400))

class TestML:
    def test_forecast_trains(self, small_sales):
        from src.ml.train import SalesForecastModel
        m = SalesForecastModel(); metrics = m.train(small_sales, n_splits=2)
        assert 0 <= metrics["cv_mape"] <= 1.0 and m.fitted

    def test_forecast_predict(self, small_sales):
        from src.ml.train import SalesForecastModel
        import pandas as pd
        m = SalesForecastModel(); m.train(small_sales, n_splits=2)
        df_enc = pd.get_dummies(small_sales, columns=["Description","Country"], dtype=int)
        pt, lo, hi = m.predict_with_intervals(df_enc.tail(3))
        assert len(pt)==3 and (hi>=pt).all() and (pt>=lo).all()

    def test_churn_trains(self, small_churn):
        from src.ml.train import ChurnModel
        m = ChurnModel(); metrics = m.train(small_churn)
        assert 0.5 <= metrics["auc_roc"] <= 1.0 and m.fitted

    def test_churn_scores(self, small_churn):
        from src.ml.train import ChurnModel
        m = ChurnModel(); m.train(small_churn)
        scored = m.score(small_churn)
        assert "churn_prob" in scored.columns
        assert scored["churn_prob"].between(0,1).all()

    def test_churn_shap(self, small_churn):
        from src.ml.train import ChurnModel
        m = ChurnModel(); m.train(small_churn)
        imp = m.shap_importance(small_churn)
        assert isinstance(imp, dict) and len(imp) > 0


# ── RAG ───────────────────────────────────────────────────────────────
class TestRAG:
    def test_build_and_retrieve(self, tmp_path, monkeypatch):
        from src.utils.config import cfg
        monkeypatch.setattr(cfg, "chroma_dir", str(tmp_path/"chroma"))
        from src.rag.retriever import BusinessRAG, DOCS
        rag = BusinessRAG()
        rag.build(extra_docs=DOCS[:2])
        results = rag.retrieve("customer churn", k=2)
        assert len(results) >= 1 and len(results[0].page_content) > 0

    def test_retrieve_with_scores(self, tmp_path, monkeypatch):
        from src.utils.config import cfg
        monkeypatch.setattr(cfg, "chroma_dir", str(tmp_path/"chroma2"))
        from src.rag.retriever import BusinessRAG, DOCS
        rag = BusinessRAG(); rag.build(extra_docs=DOCS[:2])
        results = rag.retrieve_with_scores("revenue forecast", k=2)
        assert all(isinstance(s, float) for _, s in results)

    def test_format_context(self, tmp_path, monkeypatch):
        from src.utils.config import cfg
        monkeypatch.setattr(cfg, "chroma_dir", str(tmp_path/"chroma3"))
        from src.rag.retriever import BusinessRAG, DOCS
        rag = BusinessRAG(); rag.build(extra_docs=DOCS[:2])
        ctx = rag.format_context(rag.retrieve("revenue", k=2))
        assert isinstance(ctx, str) and len(ctx) > 10


# ── LLM Client ────────────────────────────────────────────────────────
class TestLLMClient:
    def test_complete_json_strips_fences(self, monkeypatch):
        from src.llm.client import LLMClient
        client = LLMClient()
        mock = '```json\n{"key":"value"}\n```'
        monkeypatch.setattr(client, "_call_groq", lambda *a,**k: mock)
        monkeypatch.setattr(client, "_call_ollama", lambda *a,**k: mock)
        result = client.complete_json("test")
        assert result.get("key") == "value"

    def test_complete_json_handles_plain(self, monkeypatch):
        from src.llm.client import LLMClient
        client = LLMClient()
        monkeypatch.setattr(client, "_call_groq", lambda *a,**k: '{"answer":"ok","confidence":"High"}')
        monkeypatch.setattr(client, "_call_ollama", lambda *a,**k: '{"answer":"ok","confidence":"High"}')
        result = client.complete_json("test")
        assert result.get("confidence") == "High"


# ── Agent Tools (no LLM needed) ───────────────────────────────────────
class TestTools:
    def test_forecast_tool(self):
        from src.graph.agent import sales_forecast_tool
        r = json.loads(sales_forecast_tool.invoke({"product":"Widget-A","region":"United Kingdom","months_ahead":3}))
        assert "forecast_revenue" in r and "top_predictive_features" in r

    def test_churn_tool(self):
        from src.graph.agent import churn_analysis_tool
        r = json.loads(churn_analysis_tool.invoke({"segment":"SMB","top_n":20}))
        assert "high_risk_customers" in r and "recommended_interventions" in r

    def test_scenario_tool(self):
        from src.graph.agent import scenario_tool
        r = json.loads(scenario_tool.invoke({
            "churn_rate_change_pp":-2.0,"enterprise_deals_added":10,
            "gross_margin_change_pp":2.0,"product_launch":True,
            "product_launch_arr_m":6.0,"description":"Test"}))
        assert "projected_impact" in r and "biggest_lever" in r

    def test_rag_tool(self, tmp_path, monkeypatch):
        from src.utils.config import cfg
        monkeypatch.setattr(cfg, "chroma_dir", str(tmp_path/"tool_chroma"))
        # Pre-build index
        from src.rag.retriever import BusinessRAG, DOCS, _rag_instance
        import src.rag.retriever as rmod
        rag = BusinessRAG(); rag.build(extra_docs=DOCS[:3])
        rmod._rag_instance = rag  # inject so get_rag() finds it
        from src.graph.agent import rag_search_tool
        r = json.loads(rag_search_tool.invoke({"query":"churn reasons","k":2}))
        assert "results" in r and "retrieved_chunks" in r
