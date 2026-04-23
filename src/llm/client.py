"""
LLM client — Groq (cloud) with Ollama (local) fallback.
Two interfaces:
  1. get_langchain_llm()  → LangChain BaseChatModel  (used by LangGraph agent)
  2. LLMClient.complete() → raw string               (used for JSON synthesis)
"""
import json
import re
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import cfg


# ── LangChain chat model ──────────────────────────────────────────────

def get_langchain_llm(temperature: float = 0.1):
    """Returns a LangChain-compatible ChatModel. Groq → Ollama fallback."""
    provider = cfg.effective_provider

    if provider == "groq" and cfg.has_groq:
        try:
            from langchain_groq import ChatGroq
            logger.info(f"LLM: Groq / {cfg.groq_model}")
            return ChatGroq(
                groq_api_key=cfg.groq_api_key,
                model_name=cfg.groq_model,
                temperature=temperature,
                max_tokens=4096,
            )
        except Exception as e:
            logger.warning(f"Groq LangChain init failed ({e}), trying Ollama…")

    # Ollama fallback
    try:
        from langchain_community.chat_models import ChatOllama
        logger.info(f"LLM: Ollama / {cfg.ollama_model}")
        return ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            temperature=temperature,
            num_ctx=4096,
        )
    except Exception as e:
        raise RuntimeError(
            f"No LLM available. Error: {e}\n"
            "Options:\n"
            "  1. Add GROQ_API_KEY to configs\\.env  (free at https://console.groq.com)\n"
            "  2. Install Ollama from https://ollama.com, then: ollama pull llama3.2"
        )


# ── Raw completion client ─────────────────────────────────────────────

class LLMClient:
    """Direct LLM client for structured JSON synthesis (no LangChain overhead)."""

    def __init__(self):
        self._groq = None
        if cfg.has_groq:
            try:
                import groq as groq_sdk
                self._groq = groq_sdk.Groq(api_key=cfg.groq_api_key)
                logger.info("Groq SDK client ready")
            except Exception as e:
                logger.warning(f"Groq SDK init: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
    def _call_groq(self, messages: list, max_tokens: int, temperature: float) -> str:
        resp = self._groq.chat.completions.create(
            model=cfg.groq_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    def _call_ollama(self, messages: list, max_tokens: int, temperature: float) -> str:
        import httpx
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user   = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        r = httpx.post(
            f"{cfg.ollama_base_url}/api/generate",
            json={"model": cfg.ollama_model, "prompt": user, "system": system,
                  "stream": False, "options": {"temperature": temperature, "num_predict": max_tokens}},
            timeout=120.0,
        )
        r.raise_for_status()
        return r.json()["response"]

    def complete(self, user_prompt: str, system_prompt: str = "",
                 max_tokens: int = 2048, temperature: float = 0.1) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        if cfg.effective_provider == "groq" and self._groq:
            try:
                return self._call_groq(messages, max_tokens, temperature)
            except Exception as e:
                logger.warning(f"Groq failed ({e}), falling back to Ollama…")

        return self._call_ollama(messages, max_tokens, temperature)

    def complete_json(self, user_prompt: str, system_prompt: str = "",
                      max_tokens: int = 2048) -> dict:
        sys_p = (system_prompt or "") + "\nIMPORTANT: respond ONLY with valid JSON. No markdown, no backticks."
        raw = self.complete(user_prompt, sys_p, max_tokens, temperature=0.05)
        # Strip markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
            logger.error(f"JSON parse failed. Raw: {raw[:300]}")
            return {"error": "JSON parse failed", "raw": raw[:500]}


llm_client = LLMClient()
