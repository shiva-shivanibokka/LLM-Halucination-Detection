"""
api/main.py

FastAPI backend for the LLM Eval Platform.

Auth: mutating endpoints (POST/DELETE) require `Authorization: Bearer <APP_API_TOKEN>`.
LLM provider keys are read from the server environment only — never the request.

Run with:
  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import csv
import hmac
import io
import os
import re
from contextlib import asynccontextmanager

from core.detector import (
    DEFAULT_CONTRADICT_THRESHOLD,
    DEFAULT_ENTAIL_THRESHOLD,
    DEFAULT_GROUNDED_CEILING,
    DEFAULT_PARTIAL_CEILING,
)
from core.generator import PROVIDERS, _call_llm
from core.logging_config import get_logger
from data.ragtruth import seed_ragtruth_benchmark
from db import models as db
from db.database import get_connection, init_db
from eval.runner import run_benchmark
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

log = get_logger(__name__)

MAX_REFERENCE_CHARS = 200_000
MAX_GENERATE_CASES = 50


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail loudly at startup if required config is missing.
    missing = [k for k in ("DATABASE_URL", "APP_API_TOKEN") if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
    init_db()
    log.info("startup_complete")
    yield


app = FastAPI(
    title="LLM Eval Platform",
    description="Benchmark and compare LLM models for hallucination and factual accuracy.",
    version="2.0.0",
    lifespan=lifespan,
)

# Scoped CORS — only the configured frontend origin(s), not "*".
_origins = [o.strip() for o in os.getenv("FRONTEND_ORIGIN", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_token(authorization: str = Header(default="")) -> None:
    """Guard for mutating endpoints. Expects `Authorization: Bearer <APP_API_TOKEN>`."""
    expected = os.getenv("APP_API_TOKEN", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")


# --------------------------------------------------------------------------- #
# Request models (no api_key — keys are server-side)
# --------------------------------------------------------------------------- #
class BenchmarkCreate(BaseModel):
    name: str
    description: str = ""


class TestCaseCreate(BaseModel):
    question: str
    reference_text: str
    domain: str = "general"
    source_type: str = "internal"


class BulkImportRequest(BaseModel):
    csv_text: str


class GenerateCasesRequest(BaseModel):
    reference_text: str
    num_cases: int = 10
    domain: str = "general"
    source_type: str = "internal"
    provider: str = "openai"
    model: str | None = None
    api_key: str | None = None  # BYOK; falls back to the server env key


class StartRunRequest(BaseModel):
    benchmark_id: int
    provider: str
    model: str
    api_key: str | None = None  # BYOK; falls back to the server env key
    entail_threshold: float = DEFAULT_ENTAIL_THRESHOLD
    contradict_threshold: float = DEFAULT_CONTRADICT_THRESHOLD
    grounded_ceiling: float = DEFAULT_GROUNDED_CEILING
    partial_ceiling: float = DEFAULT_PARTIAL_CEILING


class SeedRagtruthRequest(BaseModel):
    split: str = "train"
    limit: int = 50
    name: str | None = None


# --------------------------------------------------------------------------- #
# Health & providers
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    db_ok = True
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")
    except Exception:  # noqa: BLE001 - health probe: any failure means "down"
        db_ok = False
    status = "ok" if db_ok else "down"
    code = 200 if db_ok else 503
    if not db_ok:
        raise HTTPException(status_code=code, detail={"status": status, "db": "down"})
    return {"status": status, "db": "ok", "model": "lazy"}


@app.get("/providers")
def list_providers():
    return {
        provider: {
            "models": details["models"],
            "requires_key": details["env_key"] is not None,
        }
        for provider, details in PROVIDERS.items()
    }


# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #
@app.get("/benchmarks")
def list_benchmarks():
    with get_connection() as conn:
        return db.list_benchmarks(conn)


@app.post("/benchmarks", status_code=201, dependencies=[Depends(require_token)])
def create_benchmark(req: BenchmarkCreate):
    with get_connection() as conn:
        try:
            return db.create_benchmark(conn, req.name, req.description)
        except Exception as e:  # noqa: BLE001 - mapped to 409/500 below
            if "unique" in str(e).lower():
                raise HTTPException(status_code=409, detail=f"A benchmark named '{req.name}' already exists.")
            raise HTTPException(status_code=500, detail=str(e))


@app.delete("/benchmarks/{benchmark_id}", status_code=204, dependencies=[Depends(require_token)])
def delete_benchmark(benchmark_id: int):
    with get_connection() as conn:
        if not db.get_benchmark(conn, benchmark_id):
            raise HTTPException(status_code=404, detail="Benchmark not found.")
        db.delete_benchmark(conn, benchmark_id)


# --------------------------------------------------------------------------- #
# Test cases
# --------------------------------------------------------------------------- #
@app.get("/benchmarks/{benchmark_id}/cases")
def list_cases(benchmark_id: int):
    with get_connection() as conn:
        if not db.get_benchmark(conn, benchmark_id):
            raise HTTPException(status_code=404, detail="Benchmark not found.")
        return db.get_test_cases(conn, benchmark_id)


@app.post("/benchmarks/{benchmark_id}/cases", status_code=201, dependencies=[Depends(require_token)])
def add_case(benchmark_id: int, req: TestCaseCreate):
    with get_connection() as conn:
        if not db.get_benchmark(conn, benchmark_id):
            raise HTTPException(status_code=404, detail="Benchmark not found.")
        return db.add_test_case(
            conn, benchmark_id, req.question, req.reference_text, req.domain, req.source_type,
        )


@app.post("/benchmarks/{benchmark_id}/cases/bulk", status_code=201, dependencies=[Depends(require_token)])
def bulk_import_cases(benchmark_id: int, req: BulkImportRequest):
    with get_connection() as conn:
        if not db.get_benchmark(conn, benchmark_id):
            raise HTTPException(status_code=404, detail="Benchmark not found.")

        reader = csv.DictReader(io.StringIO(req.csv_text.strip()))
        required = {"question", "reference_text"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise HTTPException(
                status_code=422,
                detail=f"CSV must have columns: question, reference_text. Optional: domain, source_type. Got: {reader.fieldnames}",
            )
        added = 0
        for row in reader:
            question = (row.get("question") or "").strip()
            ref = (row.get("reference_text") or "").strip()
            domain = (row.get("domain") or "general").strip() or "general"
            source_type = (row.get("source_type") or "internal").strip() or "internal"
            if source_type not in ("internal", "public"):
                source_type = "internal"
            if question and ref:
                db.add_test_case(conn, benchmark_id, question, ref, domain, source_type)
                added += 1
        return {"added": added}


@app.post("/benchmarks/{benchmark_id}/generate-cases", status_code=201, dependencies=[Depends(require_token)])
def generate_cases(benchmark_id: int, req: GenerateCasesRequest):
    if len(req.reference_text) > MAX_REFERENCE_CHARS:
        raise HTTPException(status_code=422, detail=f"reference_text exceeds {MAX_REFERENCE_CHARS} chars.")
    num = max(1, min(req.num_cases, MAX_GENERATE_CASES))

    with get_connection() as conn:
        if not db.get_benchmark(conn, benchmark_id):
            raise HTTPException(status_code=404, detail="Benchmark not found.")

    system = (
        "You are an expert evaluator building a hallucination detection benchmark. "
        "Given a reference document, generate realistic factual questions that can be answered directly from it. "
        f"Generate exactly {num} questions. "
        "Return ONLY a numbered list, one question per line. No explanations, no preamble."
    )
    prompt = (
        f"Reference document:\n\n{req.reference_text}\n\n"
        f"Generate {num} factual questions about this document."
    )
    try:
        raw = _call_llm(system, prompt, req.provider, req.model, req.api_key)
    except Exception as e:  # noqa: BLE001 - external call, mapped to 502
        log.warning("generate_cases_llm_failed provider=%s model=%s: %s", req.provider, req.model, e)
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    questions = []
    for line in raw.strip().splitlines():
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip()).strip()
        if cleaned:
            questions.append(cleaned)

    rows = [
        {"question": q, "reference_text": req.reference_text,
         "domain": req.domain, "source_type": req.source_type}
        for q in questions
    ]
    with get_connection() as conn:
        db.add_test_cases(conn, benchmark_id, rows)  # atomic: all or nothing
    return {"generated": len(questions), "questions": questions}


@app.delete("/cases/{case_id}", status_code=204, dependencies=[Depends(require_token)])
def delete_case(case_id: int):
    with get_connection() as conn:
        # run_results FK is ON DELETE CASCADE, so deleting a case that's been
        # run would silently wipe its rows from completed runs and desync their
        # stored avg_score. Refuse instead of corrupting history.
        if db.case_has_results(conn, case_id):
            raise HTTPException(
                status_code=409,
                detail="This test case is part of completed run(s); deleting it "
                       "would corrupt their results. Delete the runs or the whole "
                       "benchmark instead.",
            )
        db.delete_test_case(conn, case_id)


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
@app.get("/runs")
def list_runs(benchmark_id: int | None = None):
    with get_connection() as conn:
        return db.list_runs(conn, benchmark_id)


@app.post("/runs", status_code=202, dependencies=[Depends(require_token)])
def start_run(req: StartRunRequest, background_tasks: BackgroundTasks):
    with get_connection() as conn:
        if not db.get_benchmark(conn, req.benchmark_id):
            raise HTTPException(status_code=404, detail="Benchmark not found.")
        cases = db.get_test_cases(conn, req.benchmark_id)
        if not cases:
            raise HTTPException(status_code=422, detail="Benchmark has no test cases. Add test cases before running.")
        run = db.create_run(conn, req.benchmark_id, req.provider, req.model)

    background_tasks.add_task(
        run_benchmark,
        run_id=run["id"],
        api_key=req.api_key,
        entail_threshold=req.entail_threshold,
        contradict_threshold=req.contradict_threshold,
        grounded_ceiling=req.grounded_ceiling,
        partial_ceiling=req.partial_ceiling,
    )
    return {"run_id": run["id"], "status": "running", "message": f"Eval started for {len(cases)} test cases."}


@app.get("/runs/{run_id}")
def get_run_status(run_id: int):
    with get_connection() as conn:
        run = db.get_run(conn, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found.")
        completed = conn.execute(
            "SELECT COUNT(*) AS c FROM run_results WHERE run_id = %s", (run_id,)
        ).fetchone()["c"]
        total = conn.execute(
            "SELECT COUNT(*) AS c FROM test_cases WHERE benchmark_id = %s", (run["benchmark_id"],)
        ).fetchone()["c"]
    return {**run, "completed_cases": completed, "total_cases": total}


@app.get("/runs/{run_id}/results")
def get_run_results(run_id: int):
    with get_connection() as conn:
        if not db.get_run(conn, run_id):
            raise HTTPException(status_code=404, detail="Run not found.")
        return db.get_run_results(conn, run_id)


@app.get("/runs/{run_id}/domains")
def get_domain_breakdown(run_id: int):
    with get_connection() as conn:
        if not db.get_run(conn, run_id):
            raise HTTPException(status_code=404, detail="Run not found.")
        return db.get_domain_scores(conn, run_id)


@app.get("/runs/{run_id}/metrics")
def get_run_metrics(run_id: int):
    with get_connection() as conn:
        if not db.get_run(conn, run_id):
            raise HTTPException(status_code=404, detail="Run not found.")
        metrics = db.get_run_metrics(conn, run_id)
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics for this run (benchmark is not labeled).")
    return metrics


# --------------------------------------------------------------------------- #
# Dataset seeding
# --------------------------------------------------------------------------- #
MAX_SEED_LIMIT = 200  # seeding is synchronous; keep it under the HF Spaces request timeout


@app.post("/datasets/ragtruth/seed", status_code=201, dependencies=[Depends(require_token)])
def seed_ragtruth(req: SeedRagtruthRequest):
    limit = max(1, min(req.limit, MAX_SEED_LIMIT))
    name = req.name or f"RAGTruth {req.split} (n={limit})"
    try:
        # seed_ragtruth_benchmark manages its own connection AFTER the dataset
        # download, so no pooled connection is held idle across the network fetch.
        return seed_ragtruth_benchmark(name, split=req.split, limit=limit)
    except Exception as e:
        log.exception("ragtruth_seed_failed")
        raise HTTPException(status_code=500, detail=f"RAGTruth seed failed: {e}")


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
@app.get("/compare")
def compare_runs(run_a: int, run_b: int):
    with get_connection() as conn:
        ra = db.get_run(conn, run_a)
        rb = db.get_run(conn, run_b)
        if not ra:
            raise HTTPException(status_code=404, detail=f"Run {run_a} not found.")
        if not rb:
            raise HTTPException(status_code=404, detail=f"Run {run_b} not found.")

        results_a = {r["test_case_id"]: r for r in db.get_run_results(conn, run_a)}
        results_b = {r["test_case_id"]: r for r in db.get_run_results(conn, run_b)}
        source_scores_a = db.get_source_type_scores(conn, run_a)
        source_scores_b = db.get_source_type_scores(conn, run_b)

    shared_ids = set(results_a.keys()) & set(results_b.keys())
    per_case = []
    for tc_id in sorted(shared_ids):
        a = results_a[tc_id]
        b = results_b[tc_id]
        delta = b["hallucination_score"] - a["hallucination_score"]
        per_case.append({
            "test_case_id": tc_id,
            "question": a["question"],
            "domain": a["domain"],
            "source_type": a["source_type"],
            "score_a": a["hallucination_score"],
            "label_a": a["overall_label"],
            "score_b": b["hallucination_score"],
            "label_b": b["overall_label"],
            "delta": round(delta, 4),
            "verdict": "improved" if delta < -0.05 else "regressed" if delta > 0.05 else "stable",
        })

    avg_a = ra["avg_score"] or 0.0
    avg_b = rb["avg_score"] or 0.0
    return {
        "run_a": ra,
        "run_b": rb,
        "avg_score_a": avg_a,
        "avg_score_b": avg_b,
        "overall_delta": round(avg_b - avg_a, 4),
        "improved_count": sum(1 for c in per_case if c["verdict"] == "improved"),
        "regressed_count": sum(1 for c in per_case if c["verdict"] == "regressed"),
        "stable_count": sum(1 for c in per_case if c["verdict"] == "stable"),
        "source_type_scores_a": source_scores_a,
        "source_type_scores_b": source_scores_b,
        "per_case": per_case,
    }
