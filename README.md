# LLM Hallucination Detection & Guardrails System

NLI-based hallucination detection pipeline. Flags unsupported LLM claims using DeBERTa-v3, assigns a hallucination confidence score per sentence, and grounds responses against source documents via ChromaDB.

---

## What It Does

Every company deploying LLMs in production faces hallucination — models confidently stating wrong or unsupported facts. This pipeline takes an LLM-generated response and its source documents, uses a Natural Language Inference (NLI) model to detect factual inconsistencies, assigns a hallucination confidence score, and flags specific sentences that are not grounded in the source.

**Three-stage pipeline:**

```
Source Documents (PDF / URL / Text)
        ↓
  ChromaDB Vector Store (sentence-transformers embeddings)
        ↓
  For each sentence in the LLM response:
    → Retrieve top-K most relevant source chunks
    → DeBERTa-v3 NLI: entailment / contradiction / neutral scores
    → Classify: GROUNDED | UNGROUNDED | CONTRADICTED
        ↓
  Overall hallucination score + per-sentence breakdown
```

---

## Features

- **DeBERTa-v3-large NLI** — `cross-encoder/nli-deberta-v3-large` for sentence-level factual grounding
- **ChromaDB vector store** — semantic retrieval of most relevant source chunks per sentence
- **Multi-source ingestion** — plain text, PDF upload, or URL scraping (Playwright fallback for JS-rendered pages)
- **Dual LLM provider** — OpenAI (GPT-4o) or Anthropic (Claude) selectable in the UI
- **Side-by-side demo** — generates and analyzes a grounded vs ungrounded response simultaneously
- **FastAPI backend** — REST API with full OpenAPI docs at `/docs`
- **Gradio frontend** — three-tab UI deployable on Hugging Face Spaces

---

## Project Structure

```
LLM-Halucination_Detection/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI app — all endpoints
├── detector/
│   ├── __init__.py
│   └── hallucination_detector.py   # NLI pipeline core
├── ingestor/
│   ├── __init__.py
│   ├── ingestor.py          # PDF / URL / text extraction and chunking
│   └── vector_store.py      # ChromaDB wrapper
├── llm/
│   ├── __init__.py
│   └── generator.py         # OpenAI + Anthropic response generation
├── app.py                   # Gradio frontend
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For URL scraping with Playwright (JS-rendered pages):
```bash
playwright install chromium
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Start the FastAPI backend

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

### 4. Start the Gradio frontend

```bash
python app.py
```

UI available at: `http://localhost:7860`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + chunk count |
| `POST` | `/ingest/text` | Ingest plain text |
| `POST` | `/ingest/pdf` | Ingest uploaded PDF |
| `POST` | `/ingest/url` | Scrape and ingest a URL |
| `POST` | `/generate` | Generate LLM response (grounded or ungrounded) |
| `POST` | `/analyze` | Analyze a response for hallucinations |
| `POST` | `/full-pipeline` | Ingest + generate both + analyze both in one call |
| `DELETE` | `/reset` | Clear all source documents |

---

## Hallucination Labels

| Label | Meaning |
|-------|---------|
| `GROUNDED` | Sentence is supported by source documents (entailment score ≥ 0.5) |
| `UNGROUNDED` | No source chunk entails or contradicts the sentence |
| `CONTRADICTED` | A source chunk explicitly contradicts the sentence (contradiction score ≥ 0.5) |

**Overall response labels:**

| Score Range | Label |
|-------------|-------|
| 0.0 – 0.3 | `GROUNDED` |
| 0.3 – 0.6 | `PARTIALLY_GROUNDED` |
| 0.6 – 1.0 | `HALLUCINATED` |

---

## Stack

| Component | Technology |
|-----------|-----------|
| NLI model | `cross-encoder/nli-deberta-v3-large` (HuggingFace) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (in-memory, ephemeral per session) |
| LLM providers | OpenAI GPT-4o, Anthropic Claude 3.5 Haiku |
| PDF parsing | pypdf |
| Web scraping | requests + BeautifulSoup, Playwright fallback |
| Backend | FastAPI + Pydantic + Uvicorn |
| Frontend | Gradio |
| Deploy | Hugging Face Spaces |

---

## Deploy on Hugging Face Spaces

1. Push this repo to a Hugging Face Space (SDK: Gradio)
2. Set `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` in Space secrets
3. Set `API_BASE_URL` to your FastAPI deployment URL (or run both in the same process)
4. The Gradio app is the entry point — set `app.py` as the Space entry file

---

## New Skills Demonstrated

- NLI models (Natural Language Inference)
- DeBERTa-v3 for factual grounding
- Sentence-level hallucination scoring
- ChromaDB semantic retrieval
- LLM output guardrails
- HuggingFace transformers inference pipeline
- FastAPI async REST API with Pydantic validation
- Multi-provider LLM integration (OpenAI + Anthropic)
- Gradio multi-tab UI with file upload and live analysis
