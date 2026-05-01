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
from typing import Optional
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
            "mixtral-8x7b-32768",
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

# Reverse lookup: model name -> provider name
MODEL_TO_PROVIDER: dict[str, str] = {
    model: provider
    for provider, details in PROVIDERS.items()
    for model in details["models"]
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


def get_models_for_provider(provider: str) -> list[str]:
    """Return the list of available models for a given provider name."""
    return PROVIDERS.get(provider.lower(), {}).get("models", [])


def get_all_providers() -> list[str]:
    """Return a list of all supported provider names."""
    return list(PROVIDERS.keys())


def _get_client(provider: str, runtime_api_key: Optional[str] = None) -> OpenAI:
    """
    Return an OpenAI SDK client configured for the given provider.
    runtime_api_key (from the request body) takes priority over env vars.
    """
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            f"Choose from: {', '.join(PROVIDERS.keys())}"
        )
    cfg = PROVIDERS[provider]

    if cfg["env_key"] is None:
        # Ollama — local, no key needed
        api_key = "ollama"
    else:
        api_key = runtime_api_key or os.getenv(cfg["env_key"])
        if not api_key:
            raise ValueError(
                f"API key for '{provider}' not provided. "
                f"Set {cfg['env_key']} in your .env file or paste it in the UI."
            )

    return OpenAI(
        base_url=cfg["base_url"],
        api_key=api_key,
        default_headers=cfg.get("extra_headers", {}),
    )


def _call_llm(
    system: str,
    prompt: str,
    provider: str,
    model: Optional[str],
    api_key: Optional[str],
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
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(
            f"LLM call failed ({provider} | {resolved_model}): {e}"
        ) from e


def generate_grounded(
    question: str,
    vector_store,
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
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


def generate_ungrounded(
    question: str,
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Generate a response with no source grounding (baseline for hallucination comparison)."""
    system = "You are a helpful assistant. Answer the user's question concisely."
    return _call_llm(system, question, provider, model, api_key)


def extract_claims(
    response: str,
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> list[str]:
    """
    Use the LLM to extract discrete, verifiable factual claims from a response.
    Returns a list of individual claim strings, each self-contained and checkable.
    """
    system = (
        "You are a fact-checking assistant. "
        "Given a text, extract every discrete factual claim it makes. "
        "A claim is a single, self-contained statement that can be verified as true or false. "
        "Break compound sentences into individual claims. "
        "Return ONLY a numbered list, one claim per line, no commentary. "
        "Example format:\n1. The Eiffel Tower is located in Paris.\n2. It was built in 1889."
    )
    prompt = f"Extract all factual claims from this text:\n\n{response}"
    raw = _call_llm(system, prompt, provider, model, api_key)

    claims = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering like "1." or "1)"
        import re

        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if cleaned:
            claims.append(cleaned)
    return claims
