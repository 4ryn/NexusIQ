"""Central configuration — reads .env, exposes typed settings."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Try configs/.env first (local dev), then cwd .env (Railway injects here)
_root = Path(__file__).parent.parent.parent
load_dotenv(_root / "configs" / ".env", override=True)
load_dotenv(override=True)


def _p(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _path(key: str, default: str) -> Path:
    p = Path(os.getenv(key, default))
    p.mkdir(parents=True, exist_ok=True)
    return p


class _Config:
    # LLM
    groq_api_key:    str  = _p("GROQ_API_KEY")
    groq_model:      str  = _p("GROQ_MODEL", "llama-3.3-70b-versatile")
    ollama_base_url: str  = _p("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model:    str  = _p("OLLAMA_MODEL", "phi3:mini")
    llm_provider:    str  = _p("LLM_PROVIDER", "auto")

    # Embeddings
    embedding_model: str  = _p("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Paths
    raw_dir:         Path = _path("RAW_DATA_DIR",       "data/raw")
    processed_dir:   Path = _path("PROCESSED_DATA_DIR", "data/processed")
    models_dir:      Path = _path("MODELS_DIR",         "data/models")
    docs_dir:        Path = _path("DOCS_DIR",           "data/raw/documents")
    chroma_dir:      str  = _p("CHROMA_PERSIST_DIR",    "data/embeddings/chroma")

    # API
    api_host:        str  = _p("API_HOST", "0.0.0.0")
    port_raw = _p("PORT") or _p("API_PORT") or "8000"
    pi_port: int = int(port_raw)

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key and not self.groq_api_key.startswith("gsk_your"))

    @property
    def effective_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        return "groq" if self.has_groq else "ollama"


cfg = _Config()
