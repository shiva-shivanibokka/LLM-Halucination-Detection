# LLM Hallucination Eval Platform

> [!IMPORTANT]
> **The hosted demo is temporary.** This project's backend runs on Google Cloud
> Run under a Google Cloud free trial that ends **around 19 September 2026**.
> When the trial closes the service is stopped, and every `run.app` link below
> stops responding. The hosted frontend will keep loading after that date, but it
> will not be able to reach its API, so it will show no data.
>
> Nothing in this repository depends on that. The code, tests and results are
> complete, and the instructions below run the whole thing locally.


> Score an LLM's answers against reference documents with an NLI detector — and measure that detector against **human** hallucination labels.

[![CI](https://github.com/shiva-shivanibokka/llm-hallucination-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/shiva-shivanibokka/llm-hallucination-detection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![Next.js 16](https://img.shields.io/badge/Next.js-16-black.svg)

**Live:** [llm-hallucination-detection.vercel.app](https://llm-hallucination-detection.vercel.app) · **API:** [Cloud Run](https://llm-eval-backend-1033307298880.us-east1.run.app/health) (docs at [`/docs`](https://llm-eval-backend-1033307298880.us-east1.run.app/docs))

---

## Recruiter TL;DR

- **What it is:** a full-stack platform that grades an LLM answer sentence-by-sentence for hallucination against a reference document, using a local NLI (natural-language-inference) model — then scores that detector against the human-labeled **RAGTruth** corpus.
- **Hardest problem solved:** most hallucination tools emit a number with nothing to calibrate it against. This one closes the loop — it treats the detector as the thing under test and reports how often it *agrees with human judgments* (precision / recall / F1), which is the honest measure of whether the detector works.
- **Shape:** Next.js on Vercel → FastAPI on GCP Cloud Run (serving DeBERTa-v3 NLI + ChromaDB retrieval) → Neon Postgres. Strict bring-your-own-key for any live model call; the RAGTruth path needs no key at all.

---

## Overview — the problem

A hallucination checker that scores a single answer and stops is easy to build and impossible to trust: there's no ground truth to say whether its verdict was *right*. This project is built around the workflow that actually matters:

1. Define a **benchmark** — questions paired with the reference documents their answers must stay faithful to.
2. **Run** a model (or a dataset's stored answers) against it; the detector grades every sentence.
3. **Compare** models, and break scores out by document source to flag training-data contamination.
4. Against a labeled dataset (RAGTruth), **quantify the detector itself** — precision, recall, and F1 versus human annotations.

Step 4 is the part portfolios usually skip, and it's the headline here: the detector is the subject of the evaluation, not just its author.

## Features

A single-page app with six tabs, each a stage in that workflow:

- **About** — a built-in guide: the concepts (benchmark, gold label, internal-vs-public contamination, the NLI detector, P/R/F1, BYOK) and how to use every tab.
- **RAGTruth** — seed the human-labeled [RAGTruth](https://huggingface.co/datasets/wandb/RAGTruth-processed) corpus as a benchmark. No API key required.
- **Run Eval** — score a benchmark sentence-by-sentence; tune the detector's entail/contradict/ceiling thresholds; watch progress live and see the average score + grounded rate on completion.
- **Results** — per-run domain breakdown, per-question verdicts, and — for labeled runs — the detector's **F1 against human labels**.
- **Compare** — two runs side by side, with an internal-vs-public source split that surfaces contamination risk.
- **New Benchmark** — upload a PDF (text is extracted in the browser) and have a model generate grounded questions from it.

**Strict BYOK:** the server holds no provider keys. Any live model call (generating questions, or generating fresh answers to score) uses a key you paste in; it's kept in memory for the session, cleared on refresh, and never persisted. Scoring RAGTruth's stored answers needs no key.

## Architecture

```mermaid
flowchart LR
    U([Browser])

    subgraph Vercel [Vercel · free tier]
      FE[Next.js App Router<br/>6-tab SPA]
      PROXY[Route handler<br/>attaches bearer token]
    end

    subgraph GCP [GCP Cloud Run · Docker, scale-to-zero]
      API[FastAPI]
      DET[DeBERTa-v3 NLI<br/>sentence detector]
      VEC[sentence-transformers<br/>+ ChromaDB, isolated per case]
      API --> DET
      API --> VEC
    end

    DB[(Neon Postgres<br/>benchmarks · cases · runs · results · metrics)]
    RT[[RAGTruth<br/>human labels]]
    LLM[[LLM providers<br/>OpenAI · Anthropic · Groq · Mistral · Gemini]]

    U -->|GET, direct| API
    U -->|POST / DELETE| PROXY
    PROXY -->|Bearer token| API
    U --- FE
    API -->|psycopg pool| DB
    API -->|load once| RT
    API -.->|BYOK, per request| LLM
```

**Why this shape:**

- **Two services, not one.** The frontend is static and free on Vercel; the backend needs ~4 GB RAM to hold DeBERTa-large and can't run on a static host. Splitting them lets each scale (and cost) independently — Cloud Run scales to **zero** when idle.
- **The token never reaches the browser.** GET requests hit the backend directly, but mutations go through a same-origin Next.js route handler that injects the bearer token server-side. The browser only ever talks to `/api/proxy/*`.
- **Retrieval is isolated per test case.** Each case builds its own ChromaDB collection (unique name, torn down after) so one case's reference text can't leak into another's scoring, and concurrent runs don't collide.
- **Postgres, not a document store.** The data is inherently relational (benchmarks → cases → runs → results → metrics) and the app leans on JOINs and aggregates for the domain/source breakdowns.

Full design rationale and the label-mapping decisions: [`docs/adr/0001-frontend-backend-split.md`](docs/adr/0001-frontend-backend-split.md).

## How the detector is scored

RAGTruth ships each answer with a human hallucination verdict. A RAGTruth run scores those **stored** answers with the NLI detector and compares its verdicts to the human labels, reporting **precision / recall / F1 / accuracy** (positive class = *hallucinated*; `PARTIALLY_GROUNDED` and `UNGROUNDED` both map to hallucinated). Seed from the **RAGTruth** tab, run it on **Run Eval**, and read the metric card on **Results**.

## Tech stack

| Layer | Technology | Note |
|---|---|---|
| NLI detector | `cross-encoder/nli-deberta-v3-large` | per-sentence entailment/contradiction vs retrieved chunks |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | for chunk retrieval |
| Vector store | ChromaDB (in-memory) | one isolated collection per test case |
| Dataset | RAGTruth (`wandb/RAGTruth-processed`) | real human hallucination labels |
| Backend | FastAPI · Pydantic · Uvicorn · Python 3.12 | async background tasks for eval runs |
| Database | Neon Postgres (`psycopg` + pool) | SQL migrations, JSONB for sentence results |
| Frontend | Next.js 16 (App Router) · React 19 · TypeScript | custom CSS design system (aurora glass) |
| LLM providers | OpenAI · Anthropic · Groq · Mistral · Gemini · Ollama | OpenAI-compatible clients, **BYOK** |
| Infra | GCP Cloud Run (Docker, 4 GiB / 2 vCPU, scale-to-zero) · Vercel · Neon | all free-tier |
| CI | GitHub Actions | ruff + pytest (Postgres service) + `next build` + `tsc` |

## Skills demonstrated

- **Production ML deployment / MLOps** — a transformer NLI model served behind a REST API, containerized for a serverless CPU runtime, separate from any notebook.
- **LLM application development & RAG** — sentence-level retrieval-augmented grounding with a vector store and a provider-agnostic BYOK client layer.
- **Cloud deployment (GCP)** — Cloud Run via source deploy, scale-to-zero, no CPU throttling for background work, scoped CORS.
- **RESTful API design** — a versioned FastAPI surface (benchmarks, cases, runs, results, metrics, compare, dataset seeding).
- **Database design & schema migrations** — a normalized Postgres schema with an idempotent file-based migration runner and JSONB payloads.
- **Asynchronous / concurrent systems** — eval runs execute as background tasks with per-case failure isolation and a validated connection pool.
- **System design & architecture** — a documented ADR weighing the frontend/backend split, data model, and load behavior.
- **Application security (authz)** — bearer-token auth with constant-time comparison; the token never reaches the browser.
- **Observability** — structured logging and a `/health` check that verifies DB connectivity.
- **CI/CD** — GitHub Actions running lint, tests (against a Postgres service), a production build, and a type check on every push.
- **Data engineering** — a loader that pulls the RAGTruth corpus into a model-ready, labeled benchmark.

## Getting started

**Prerequisites:** Python 3.12, Node 20+, and a Postgres database (local, or a free [Neon](https://neon.tech) instance).

**Backend** (install CPU torch first — do not install tensorflow):

```bash
cd backend
python -m venv .venv && source .venv/Scripts/activate   # .venv/bin/activate on macOS/Linux
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export DATABASE_URL="postgresql://user:pass@host/db?sslmode=require"
export APP_API_TOKEN="dev-token"
export FRONTEND_ORIGIN="http://localhost:3000"
uvicorn api.main:app --reload --port 8000
```

Migrations apply automatically on startup. The first run downloads the DeBERTa and embedding models (~1.5 GB).

**Frontend:**

```bash
cd frontend
npm install
# .env.local:
#   NEXT_PUBLIC_API_BASE=http://localhost:8000
#   APP_API_TOKEN=dev-token          # server-only; same value as the backend
npm run dev
```

Open `http://localhost:3000`, go to the **RAGTruth** tab, seed a benchmark, and run it — no API key needed for that path.

## Usage

The public read endpoints need no auth:

```bash
curl https://llm-eval-backend-1033307298880.us-east1.run.app/health
# {"status":"ok","db":"ok","model":"lazy"}

curl https://llm-eval-backend-1033307298880.us-east1.run.app/providers
# {"openai":{"models":["gpt-4o", ...],"requires_key":true}, ...}
```

Mutating endpoints (`POST /datasets/ragtruth/seed`, `POST /runs`, …) require the bearer token and are normally driven through the UI, which proxies the token server-side. The end-to-end no-key demo flow is: **RAGTruth → seed → Run Eval → Start → Results (F1)**.

## Project structure

```
backend/
  api/main.py          FastAPI app: endpoints, bearer auth, CORS, lifespan
  core/
    detector.py        DeBERTa NLI detector + sentence classification
    generator.py       provider-agnostic BYOK LLM client (OpenAI-compatible)
    vector_store.py    ChromaDB wrapper, isolated per test case
    ingestor.py        text cleaning + chunking
  eval/
    runner.py          background eval orchestration (per-case isolation)
    scoring.py         label→binary mapping + P/R/F1 metrics
  data/ragtruth.py     RAGTruth loader → labeled benchmark
  db/                  Postgres models + connection pool
  migrations/*.sql     schema migrations
  tests/               pytest (scoring, detector logic, DB, isolation)
frontend/
  src/app/             Next.js App Router: page + 6 tab components
  src/app/api/proxy/   token-injecting reverse proxy route
  src/lib/             typed API client, PDF extraction, in-memory key store
docs/                  ADR + implementation plan
.github/workflows/     CI
```

## Testing

```bash
pytest backend -m "not slow"        # DB integration tests skip without DATABASE_URL
cd frontend && npm run build && npx tsc --noEmit
```

Backend tests cover the scoring math, the detector's pure classification logic, DB round-trips, and vector-store isolation. CI runs the full backend suite against a real Postgres service plus the frontend build and type check on every push and PR. Coverage isn't formally measured, so no coverage figure is claimed.

## Deployment

All three tiers run on free plans.

1. **Neon** — provision Postgres (e.g. via the Vercel Neon integration) and copy the `DATABASE_URL`.
2. **GCP Cloud Run** (backend) — source deploy from `backend/`:
   ```bash
   gcloud run deploy llm-eval-backend --source backend --region us-east1 \
     --allow-unauthenticated --memory 4Gi --cpu 2 --no-cpu-throttling \
     --min-instances 0 --timeout 3600 \
     --set-env-vars "DATABASE_URL=...,APP_API_TOKEN=...,FRONTEND_ORIGIN=https://<your>.vercel.app"
   ```
   `--no-cpu-throttling` lets background eval runs finish; `--min-instances 0` scales to zero when idle. Copy the printed service URL.
3. **Vercel** (frontend) — set the project root to `frontend/`, and set `NEXT_PUBLIC_API_BASE` (the Cloud Run URL) and `APP_API_TOKEN` (server-only, matching the backend).

## Results

This is a working, deployed system, not a static demo: seed a labeled benchmark, run it, and the Results tab reports the detector's precision / recall / F1 against the human labels. The numbers you get are computed live from your run — no figures are baked into this README, because a benchmark result is only meaningful with its sample size and configuration attached, and those are yours to choose.

## Roadmap

- Larger, reported RAGTruth evaluations (the F1 story is strongest with a few hundred labeled cases, run and recorded).
- Stream the RAGTruth download instead of materializing the full split.
- A UI for managing individual test cases (the API supports it; the app currently manages whole benchmarks).
- Auth is a single shared bearer token — fine for a single-owner demo, not multi-tenant.
- Optional persisted (disk-backed) vector store for very large reference documents.

## License

[MIT](LICENSE) © 2026 Shivani Bokka
