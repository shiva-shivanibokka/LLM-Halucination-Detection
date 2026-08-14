"""
llm/generator.py

Generates LLM responses via a provider-agnostic OpenAI-SDK wrapper.

All providers expose an OpenAI-compatible REST API, so the same
client.chat.completions.create() call works everywhere — only the
base_url and api_key change. No provider-specific SDK is needed
except for the standard `openai` package.

Supported providers:
  - openai    — GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
  - anthropic — Claude Opus/Sonnet/Haiku (via Anthropic's OpenAI-compat endpoint)
  - groq      — Llama 3, Mixtral, Gemma2  (free tier)
  - mistral   — Mistral Large/Small, Open-Mistral  (free tier)
  - gemini    — Gemini 2.0/1.5 Flash/Pro  (free tier)
  - ollama    — Any locally-pulled model, no API key needed

Two generation modes:
  - grounded:   RAG-style, retrieves source chunks from ChromaDB
  - ungrounded: no context, more likely to hallucinate
"""

import os

from openai import OpenAI

TOP_K_CONTEXT = 5
MAX_CONTEXT_WORDS = 1500

PROVIDERS: dict[str, dict] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "env_key": "ANTHROPIC_API_KEY",
        # Required when calling Anthropic via the OpenAI-compatible endpoint
        "extra_headers": {"anthropic-version": "2023-06-01"},
        "models": [
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
        ],
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ],
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "env_key": "MISTRAL_API_KEY",
        "models": [
            "mistral-large-latest",
            "mistral-small-latest",
            "open-mistral-7b",
        ],
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GOOGLE_API_KEY",
        "models": [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
    },
    "ollama": {
        # Run `ollama serve` locally before using these.
        # Pull a model first, e.g. `ollama pull phi3`
        "base_url": "http://localhost:11434/v1",
        "env_key": None,  # no API key needed for local Ollama
        "models": [
            "llama3.2",
            "llama3.2:1b",
            "mistral",
            "deepseek-r1:1.5b",
            "deepseek-r1:7b",
            "phi3",
            "phi3:mini",
            "gemma3",
            "gemma3:4b",
        ],
    },
}

# Default model per provider (used when no model is specified)
DEFAULT_MODEL: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-haiku-4-5",
    "groq": "llama-3.3-70b-versatile",
    "mistral": "mistral-small-latest",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.2",
}


def _get_client(provider: str, api_key: str | None = None) -> OpenAI:
    """Return an OpenAI SDK client for the given provider.
    Key resolution (BYOK): the caller-supplied api_key wins; otherwise fall back
    to the server env key (so free providers work out-of-the-box)."""
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            f"Choose from: {', '.join(PROVIDERS.keys())}"
        )
    cfg = PROVIDERS[provider]

    if cfg["env_key"] is None:
        # Ollama — local, no key needed
        resolved_key = "ollama"
    else:
        resolved_key = api_key or os.getenv(cfg["env_key"])
        if not resolved_key:
            raise ValueError(
                f"No API key for '{provider}'. Paste your own key (BYOK) or set "
                f"{cfg['env_key']} in the server environment."
            )

    return OpenAI(
        base_url=cfg["base_url"],
        api_key=resolved_key,
        default_headers=cfg.get("extra_headers", {}),
    )


def _call_llm(
    system: str,
    prompt: str,
    provider: str,
    model: str | None,
    api_key: str | None = None,
) -> str:
    """Core call: routes to the right provider and returns the response text."""
    provider = provider.lower()
    resolved_model = model or DEFAULT_MODEL.get(provider, "")
    if not resolved_model:
        raise ValueError(
            f"No model specified and no default found for provider '{provider}'."
        )

    client = _get_client(provider, api_key)
    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        # content can be None on a content-filter / safety finish — don't crash.
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        raise RuntimeError(
            f"LLM call failed ({provider} | {resolved_model}): {e}"
        ) from e


def generate_grounded(
    question: str,
    vector_store,
    provider: str = "openai",
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Generate a response grounded in source documents from the vector store."""
    candidates = vector_store.query(question, k=TOP_K_CONTEXT)
    if not candidates:
        context = "(No source documents loaded — answering from general knowledge.)"
    else:
        chunks = [c["chunk"] for c in candidates]
        joined = "\n\n".join(chunks)
        words = joined.split()
        if len(words) > MAX_CONTEXT_WORDS:
            joined = " ".join(words[:MAX_CONTEXT_WORDS]) + "…"
        context = joined

    system = (
        "You are a factual assistant. You are given source document excerpts and a question. "
        "Answer the question using the provided source documents as your primary reference. "
        "Cite specific details from the documents. If the documents contain relevant information, "
        "use it — do not refuse to answer just because the excerpts are incomplete. "
        "Only state something is missing if the documents genuinely contain nothing relevant."
    )
    prompt = f"Source documents:\n\n{context}\n\nQuestion: {question}"
    return _call_llm(system, prompt, provider, model, api_key)
