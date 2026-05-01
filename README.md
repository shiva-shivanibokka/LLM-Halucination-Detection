# LLM Eval Platform

A hallucination evaluation and benchmarking platform for large language models. Designed for teams that need to systematically test and compare LLM models for factual accuracy — not just check one response at a time.

## What it does

Most LLM evaluation tools check a single response in isolation. This platform is built around the workflow that actually matters in production: define a benchmark, run multiple models against it, compare their hallucination rates, and track regressions over time.

The core scoring engine uses a local NLI model (DeBERTa-v3-large) to check every sentence in an LLM's response against its reference document. Sentences are classified as GROUNDED, UNGROUNDED, or CONTRADICTED based on entailment scores. Results are stored in SQLite and queryable across runs.

## Workflow

```
1. Upload a PDF
   └── Name the benchmark, set how many questions, click Create
   └── The LLM reads the PDF and generates factual test questions automatically

2. Run the benchmark against a model
   └── Pick a provider and model, click Start
   └── System gets the model's answer for each question, scores every sentence
       against the reference document using the NLI model, saves to DB

3. Run again with a different model or a newer version

4. Compare the two runs
   └── Overall score delta, per-question breakdown, domain scores, source type split
```

## Features

- **PDF-first benchmark creation** — upload a PDF, name the benchmark, and the LLM auto-generates factual test questions from it. No manual question writing required
- **Manual test case management** — add questions one at a time, bulk import via CSV, or delete individual cases from the My Benchmarks tab
- **Multi-provider support** — OpenAI, Anthropic, Groq, Mistral, Gemini, Ollama (local) — all via the OpenAI-compatible SDK
- **NLI scoring engine** — DeBERTa-v3-large scores every sentence in the LLM response against the reference document
- **Domain tagging** — tag test cases by domain (medical, legal, finance, technical, etc.) and see domain-level score breakdowns
- **Source type tracking** — mark each reference document as `internal` (private, reliable) or `public` (Wikipedia, papers — contamination risk). The comparison report splits results by source type so you know which scores to trust
- **Run history** — all runs are stored in SQLite, persists across restarts
- **Model comparison** — pick two runs and get a side-by-side diff: overall delta, which questions improved or regressed, flagged by source type
- **Configurable thresholds** — tune entailment and contradiction thresholds without touching code
- **FastAPI backend** — full REST API with OpenAPI docs at `/docs`
- **Gradio frontend** — four-tab UI

## Project Structure

```
LLM-Halucination-Detection/
├── core/
│   ├── generator.py         LLM call wrapper (OpenAI-compatible, all providers)
│   ├── detector.py          NLI scoring engine (DeBERTa-v3-large)
│   ├── ingestor.py          PDF / URL / text extraction and chunking
│   └── vector_store.py      ChromaDB wrapper (per-run, ephemeral)
├── db/
│   ├── database.py          SQLite connection and schema init
│   └── models.py            CRUD for benchmarks, test cases, runs, results
├── eval/
│   └── runner.py            Runs a full benchmark end-to-end, writes to DB
├── api/
│   └── main.py              FastAPI — 15 endpoints
├── app.py                   Gradio frontend — 5 tabs
├── eval_platform.db         Created automatically on first run
├── requirements.txt
└── .env.example
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For URL scraping with JavaScript-rendered pages:
```bash
playwright install chromium
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and add the keys for the providers you want to use. You only need the ones you'll actually use — leaving others blank just means that provider will error if selected.

```
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...          # free tier
GOOGLE_API_KEY=AI...          # free tier
MISTRAL_API_KEY=...           # free tier
OPENAI_API_KEY=sk-...
```

For Ollama (local, no key needed): install from https://ollama.com, then pull a model:
```bash
ollama pull llama3.2
```

### 3. Start the FastAPI backend

```bash
uvicorn api.main:app --reload --port 8000
```

Keep this terminal open. API docs at `http://localhost:8000/docs`.

### 4. Start the Gradio frontend

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

## UI Tabs

| Tab | What it does |
|---|---|
| New Benchmark | Upload a PDF, name the benchmark, click Create — questions are auto-generated from the PDF |
| My Benchmarks | View existing benchmarks, inspect test cases, add questions manually, delete cases |
| Run Eval | Select a benchmark and model, click Start — progress updates live, results saved to DB |
| Results | Browse all completed runs, per-question breakdown, domain-level hallucination scores |
| Compare Models | Diff two runs side by side — overall delta, per-question verdict, source type split |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/benchmarks` | List all benchmarks |
| `POST` | `/benchmarks` | Create a benchmark |
| `DELETE` | `/benchmarks/{id}` | Delete benchmark and all data |
| `GET` | `/benchmarks/{id}/cases` | List test cases |
| `POST` | `/benchmarks/{id}/cases` | Add a single test case |
| `POST` | `/benchmarks/{id}/cases/bulk` | Bulk import from CSV |
| `POST` | `/benchmarks/{id}/generate-cases` | Auto-generate test cases from a document |
| `DELETE` | `/cases/{id}` | Delete a test case |
| `GET` | `/runs` | List all runs |
| `POST` | `/runs` | Start an eval run (async, background) |
| `GET` | `/runs/{id}` | Run status and summary |
| `GET` | `/runs/{id}/results` | Per-question results |
| `GET` | `/runs/{id}/domains` | Domain-level score breakdown |
| `GET` | `/compare?run_a={id}&run_b={id}` | Diff two runs |
| `GET` | `/providers` | List supported LLM providers and models |

## Hallucination Labels

| Label | Meaning |
|---|---|
| `GROUNDED` | Sentence is supported by the reference document |
| `UNGROUNDED` | No source chunk supports or contradicts the sentence |
| `CONTRADICTED` | A source chunk explicitly contradicts the sentence |

Overall run labels:

| Score range | Label |
|---|---|
| 0.0 – 0.3 | `GROUNDED` |
| 0.3 – 0.6 | `PARTIALLY_GROUNDED` |
| 0.6 – 1.0 | `HALLUCINATED` |

## Source Type and Contamination

Each test case is tagged as `internal` or `public`:

- **Internal** — private documents the LLM has never seen (company policies, internal reports, product specs). Results from these cases are reliable.
- **Public** — documents from publicly available sources (Wikipedia, research papers, news). The LLM may already know the answers from pretraining, which can inflate grounding scores. The comparison report splits results by source type and flags public-document scores as potentially unreliable.

## Stack

| Component | Technology |
|---|---|
| NLI model | `cross-encoder/nli-deberta-v3-large` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (in-memory, per eval run) |
| Database | SQLite (persistent across restarts) |
| LLM providers | OpenAI, Anthropic, Groq, Mistral, Gemini, Ollama |
| LLM client | OpenAI Python SDK (OpenAI-compatible endpoint for all providers) |
| PDF parsing | pypdf |
| Web scraping | requests + BeautifulSoup, Playwright fallback |
| Backend | FastAPI + Pydantic + Uvicorn |
| Frontend | Gradio |
