"""
RAG System — LangChain 0.2.x + ChromaDB + HuggingFace Embeddings (local, free)

Run:  python -m src.rag.retriever
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Dict, Optional, Tuple
from loguru import logger

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.utils.config import cfg


# ── Sample business documents ─────────────────────────────────────────

DOCS = [
    {
        "title": "Q3 2024 Strategic Business Review",
        "type":  "strategic_report",
        "content": """
Q3 2024 Performance — Executive Summary

Revenue reached £42.3M (+18% YoY). Widget-A led at £16.8M (+22% YoY), driven by enterprise
adoption in the United Kingdom. Widget-B stable at £12.1M despite supply chain headwinds.
Widget-C and Widget-D showing saturation — growth slowing to 5–8%.

Key findings:
1. UK accounts for 58% of total revenue — geographic concentration risk.
2. CAC rose 12% from increased SMB competition; consider reallocating to enterprise.
3. Gross margin improved 3pp to 61% from dual-sourcing in Q2.
4. Month-to-month customers churn at 3× the rate of annual customers (22% vs 7%).

Priorities for Q4:
- Accelerate annual contract upsell for top 500 month-to-month customers.
- Launch Widget-E targeting £2.4B healthcare TAM in November.
- Deploy churn prediction model for 90-day advance warning.
- Reduce data-to-insight lag from 14 days to 48 hours.

Risks: UK macro headwinds, competitor £180M Series D raise, supply chain delays.
        """.strip()
    },
    {
        "title": "Competitive Intelligence Report 2024",
        "type":  "market_intelligence",
        "content": """
Competitive Landscape 2024

AlphaRetail (34% share): Launched freemium in Q3, captured 200K SMB users.
Weakness: poor enterprise support (NPS -12). Not a near-term enterprise threat.

BetaCommerce (22% share): Raised £180M Series D. New AI assistant cuts onboarding by 60%.
Primary threat in enterprise segment. Their pricing is 12% above ours.

Market trends:
- AI features are now table stakes: 78% of enterprise buyers require them in procurement.
- Bundled platforms close 3× faster than point solutions.
- NPS correlates 60% with expansion revenue.
- Annual contracts declining — buyers prefer flexibility.

Opportunities:
- Healthcare: £2.4B TAM, underserved, compliance = high barriers for competitors.
- Government: existing certifications create a defensible moat.

Pricing intelligence:
- AlphaRetail undercut Widget-B pricing by 18% in Q3.
- Price sensitivity highest in Consumer (40% of churn attributed to price).
        """.strip()
    },
    {
        "title": "Customer Retention & Churn Analysis Q3 2024",
        "type":  "customer_insights",
        "content": """
Churn Analysis Q3 2024

Overall monthly churn: 4.2% (up from 3.8%; industry avg 3.1%).
Enterprise: 1.1% (excellent). SMB: 7.8% (critical — target <5%). Consumer: 12.3%.

Exit survey root causes (n=847):
1. Price/value mismatch: 34% — mainly Widget-C SMB.
2. Missing features: 28% — AI reporting cited by 76% of this group.
3. Onboarding friction: 19% — never fully activated in first 30 days.
4. Poor support: 12% — resolution time 4.2h vs competitor 1.8h.
5. Business closure: 7% — external, not addressable.

High-LTV customer profile (top 10%, avg £8,400 ARR):
- Tenure > 18 months, 5+ products, NPS > 8, annual contract.
- Engaged 3+ support sessions in first 90 days.

Proven interventions (A/B tested):
- Proactive CSM outreach when NPS < 7: 34% churn reduction.
- 30-60-90 day onboarding automation: 28% early churn reduction.
- Retention offer (15% discount + feature unlock): 52% acceptance, 71% retained 12mo.
        """.strip()
    },
    {
        "title": "FY2025 Financial Planning & Scenario Assumptions",
        "type":  "financial_planning",
        "content": """
FY2025 Financial Assumptions

Revenue scenarios:
- Base: +22% YoY (£51.6M) — enterprise expansion + Widget-E launch.
- Bull: +35% YoY (£57.1M) — Widget-E hits 50% pipeline by Q2.
- Bear: +9% YoY (£46.1M)  — macro + BetaCommerce feature parity.

Key financial drivers:
- Widget-E projected £8M ARR year 1 (85 signed LOIs in pipeline).
- Enterprise NRR target: 125% (currently 118%).
- CAC payback target: 14 months (currently 21 months).
- Gross margin target: 64% by Q4 via pricing optimisation.

Cost model: R&D 20%, S&M 23%, G&A 7% (targets).

Sensitivities:
- Every 1pp churn reduction → £840K ARR preserved.
- Every 1pp gross margin improvement → £4.2M incremental EBITDA.
- Each enterprise deal (avg £112K ARR) → £112K immediate ARR.
- Widget-E at 80% Q4 target → £6.4M ARR headroom.
        """.strip()
    },
    {
        "title": "Operations & Supply Chain Efficiency Q3 2024",
        "type":  "operations",
        "content": """
Operational Efficiency Q3 2024

Supply chain: Lead times -22% from dual-sourcing (Q1). Inventory turns 8.1×  (benchmark 9×).
Top-3 supplier concentration 68% of COGS — risk remains.

KPIs: Fulfilment accuracy 98.4% (target 99%). SLA breach 2.1% (down from 4.3%).
System uptime 99.92%. Support resolution 4.2h (target 3h).

Automation wins:
- Invoice processing: £1.2M annual savings.
- Demand forecasting ML: 34% excess inventory reduction (£2.8M capital freed).
- Support chatbot: 41% first-contact resolution, £0.9M cost reduction.

Next initiatives:
1. Predictive logistics maintenance: est. £500K savings.
2. Dynamic pricing engine (SMB): +4% margin projected.
3. Real-time inventory rebalancing: reduce stockout from 2.1% to <0.5%.
        """.strip()
    },
]


# ── RAG class ─────────────────────────────────────────────────────────

class BusinessRAG:
    """LangChain 0.2.x + ChromaDB + local HuggingFace embeddings."""

    def __init__(self):
        logger.info(f"Loading embeddings: {cfg.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=cfg.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=64,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.vectorstore: Optional[Chroma] = None

    def build(self, extra_docs: Optional[List[Dict]] = None) -> "BusinessRAG":
        """Build ChromaDB index from sample docs + any PDFs in data/raw/documents/."""
        all_lc_docs: List[Document] = []

        # Sample docs
        for d in (extra_docs or DOCS):
            for i, chunk in enumerate(self.splitter.split_text(d["content"])):
                all_lc_docs.append(Document(
                    page_content=chunk,
                    metadata={"source": d["title"], "type": d["type"], "chunk": i}
                ))

        # User PDFs
        for pdf in cfg.docs_dir.glob("*.pdf"):
            try:
                from langchain_community.document_loaders import PyPDFLoader
                splits = self.splitter.split_documents(PyPDFLoader(str(pdf)).load())
                for doc in splits:
                    doc.metadata["type"] = "user_document"
                all_lc_docs.extend(splits)
                logger.info(f"Loaded PDF: {pdf.name} ({len(splits)} chunks)")
            except Exception as e:
                logger.warning(f"Could not load {pdf.name}: {e}")

        # User TXTs
        for txt in cfg.docs_dir.glob("*.txt"):
            try:
                text = txt.read_text(encoding="utf-8")
                for i, chunk in enumerate(self.splitter.split_text(text)):
                    all_lc_docs.append(Document(
                        page_content=chunk,
                        metadata={"source": txt.name, "type": "user_document", "chunk": i}
                    ))
            except Exception as e:
                logger.warning(f"Could not load {txt.name}: {e}")

        logger.info(f"Indexing {len(all_lc_docs)} chunks into ChromaDB…")
        Path(cfg.chroma_dir).mkdir(parents=True, exist_ok=True)

        self.vectorstore = Chroma.from_documents(
            documents=all_lc_docs,
            embedding=self.embeddings,
            persist_directory=cfg.chroma_dir,
            collection_name="nexusiq_docs",
        )
        self.vectorstore.persist()
        logger.success(f"ChromaDB built: {len(all_lc_docs)} chunks → {cfg.chroma_dir}")
        return self

    def load(self) -> "BusinessRAG":
        self.vectorstore = Chroma(
            persist_directory=cfg.chroma_dir,
            embedding_function=self.embeddings,
            collection_name="nexusiq_docs",
        )
        n = self.vectorstore._collection.count()
        logger.info(f"ChromaDB loaded: {n} chunks")
        return self

    def get_or_build(self) -> "BusinessRAG":
        p = Path(cfg.chroma_dir)
        if p.exists() and any(p.iterdir()):
            try:
                return self.load()
            except Exception as e:
                logger.warning(f"ChromaDB load failed ({e}), rebuilding…")
        return self.build()

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        assert self.vectorstore, "Call get_or_build() first"
        return self.vectorstore.similarity_search(query, k=k)

    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        assert self.vectorstore, "Call get_or_build() first"
        return self.vectorstore.similarity_search_with_relevance_scores(query, k=k)

    def format_context(self, docs: List[Document], max_chars: int = 4000) -> str:
        parts, total = [], 0
        for i, doc in enumerate(docs):
            s = doc.metadata.get("source", "Unknown")
            t = doc.metadata.get("type", "doc")
            part = f"[{i+1}] {s} ({t})\n{doc.page_content}"
            if total + len(part) > max_chars:
                break
            parts.append(part); total += len(part)
        return "\n\n---\n\n".join(parts)

    def as_retriever(self, k: int = 5):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})


# Singleton
_rag_instance: Optional[BusinessRAG] = None

def get_rag() -> BusinessRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = BusinessRAG()
        _rag_instance.get_or_build()
    return _rag_instance


if __name__ == "__main__":
    rag = BusinessRAG()
    rag.build()
    results = rag.retrieve("customer churn main causes", k=3)
    for doc in results:
        print(f"→ {doc.metadata['source']}: {doc.page_content[:120]}…\n")
