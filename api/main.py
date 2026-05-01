"""
api/main.py

FastAPI backend for the LLM Eval Platform.

Endpoints:

  Benchmarks
    GET    /benchmarks                       — list all benchmarks
    POST   /benchmarks                       — create a benchmark
    DELETE /benchmarks/{id}                  — delete benchmark and all its data

  Test Cases
    GET    /benchmarks/{id}/cases            — list test cases
    POST   /benchmarks/{id}/cases            — add a single test case
    POST   /benchmarks/{id}/cases/bulk       — bulk import from CSV text
    DELETE /cases/{id}                       — delete a test case
    POST   /benchmarks/{id}/generate-cases   — auto-generate test cases from a document

  Runs
    GET    /runs                             — list all runs
    POST   /runs                             — start an eval run (runs in background)
    GET    /runs/{id}                        — get run status and summary
    GET    /runs/{id}/results                — get per-question results
    GET    /runs/{id}/domains                — get domain-level score breakdown

  Comparison
    GET    /compare?run_a={id}&run_b={id}    — diff two runs side by side

  Providers
    GET    /providers                        — list supported LLM providers and models

Run with:
  uvicorn api.main:app --reload --port 8000
"""

import csv
import io
import re
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.detector import (
    DEFAULT_ENTAIL_THRESHOLD,
    DEFAULT_CONTRADICT_THRESHOLD,
    DEFAULT_GROUNDED_CEILING,
    DEFAULT_PARTIAL_CEILING,
)
from core.generator import PROVIDERS, _call_llm
from db.database import init_db, get_connection
from db import models as db
from eval.runner import run_benchmark


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="LLM Eval Platform",
    description="Benchmark and compare LLM models for hallucination and factual accuracy.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    model: Optional[str] = None
    api_key: Optional[str] = None


class StartRunRequest(BaseModel):
    benchmark_id: int
    provider: str
    model: str
    api_key: Optional[str] = None
    entail_threshold: float = DEFAULT_ENTAIL_THRESHOLD
    contradict_threshold: float = DEFAULT_CONTRADICT_THRESHOLD
    grounded_ceiling: float = DEFAULT_GROUNDED_CEILING
    partial_ceiling: float = DEFAULT_PARTIAL_CEILING


@app.get("/providers")
def list_providers():
    return {
        provider: {
            "models": details["models"],
            "requires_key": details["env_key"] is not None,
            "env_key": details["env_key"],
        }
        for provider, details in PROVIDERS.items()
    }


@app.get("/benchmarks")
def list_benchmarks():
    with get_connection() as conn:
        return db.list_benchmarks(conn)


@app.post("/benchmarks", status_code=201)
def create_benchmark(req: BenchmarkCreate):
    with get_connection() as conn:
        try:
            return db.create_benchmark(conn, req.name, req.description)
        except Exception as e:
            if "UNIQUE" in str(e):
                raise HTTPException(
                    status_code=409,
                    detail=f"A benchmark named '{req.name}' already exists.",
                )
            raise HTTPException(status_code=500, detail=str(e))


@app.delete("/benchmarks/{benchmark_id}", status_code=204)
def delete_benchmark(benchmark_id: int):
    with get_connection() as conn:
        if not db.get_benchmark(conn, benchmark_id):
            raise HTTPException(status_code=404, detail="Benchmark not found.")
        db.delete_benchmark(conn, benchmark_id)


@app.get("/benchmarks/{benchmark_id}/cases")
def list_cases(benchmark_id: int):
    with get_connection() as conn:
        if not db.get_benchmark(conn, benchmark_id):
            raise HTTPException(status_code=404, detail="Benchmark not found.")
        return db.get_test_cases(conn, benchmark_id)


@app.post("/benchmarks/{benchmark_id}/cases", status_code=201)
def add_case(benchmark_id: int, req: TestCaseCreate):
    with get_connection() as conn:
        if not db.get_benchmark(conn, benchmark_id):
            raise HTTPException(status_code=404, detail="Benchmark not found.")
        return db.add_test_case(
            conn,
            benchmark_id,
            req.question,
            req.reference_text,
            req.domain,
            req.source_type,
        )


@app.post("/benchmarks/{benchmark_id}/cases/bulk", status_code=201)
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
            question = row.get("question", "").strip()
            ref = row.get("reference_text", "").strip()
            domain = row.get("domain", "general").strip() or "general"
            source_type = row.get("source_type", "internal").strip() or "internal"
            if source_type not in ("internal", "public"):
                source_type = "internal"
            if question and ref:
                db.add_test_case(conn, benchmark_id, question, ref, domain, source_type)
                added += 1

        return {"added": added}


@app.post("/benchmarks/{benchmark_id}/generate-cases", status_code=201)
def generate_cases(benchmark_id: int, req: GenerateCasesRequest):
    with get_connection() as conn:
        if not db.get_benchmark(conn, benchmark_id):
            raise HTTPException(status_code=404, detail="Benchmark not found.")

    system = (
        "You are an expert evaluator building a hallucination detection benchmark. "
        "Given a reference document, generate realistic factual questions that can be answered directly from it. "
        f"Generate exactly {req.num_cases} questions. "
        "Return ONLY a numbered list, one question per line. No explanations, no preamble."
    )
    prompt = (
        f"Reference document:\n\n{req.reference_text}\n\n"
        f"Generate {req.num_cases} factual questions about this document."
    )

    try:
        raw = _call_llm(system, prompt, req.provider, req.model, req.api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    questions = []
    for line in raw.strip().splitlines():
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip()).strip()
        if cleaned:
            questions.append(cleaned)

    with get_connection() as conn:
        added = 0
        for q in questions:
            db.add_test_case(
                conn, benchmark_id, q, req.reference_text, req.domain, req.source_type
            )
            added += 1

    return {"generated": added, "questions": questions}


@app.delete("/cases/{case_id}", status_code=204)
def delete_case(case_id: int):
    with get_connection() as conn:
        db.delete_test_case(conn, case_id)


@app.get("/runs")
def list_runs(benchmark_id: Optional[int] = None):
    with get_connection() as conn:
        return db.list_runs(conn, benchmark_id)


@app.post("/runs", status_code=202)
def start_run(req: StartRunRequest, background_tasks: BackgroundTasks):
    with get_connection() as conn:
        if not db.get_benchmark(conn, req.benchmark_id):
            raise HTTPException(status_code=404, detail="Benchmark not found.")
        cases = db.get_test_cases(conn, req.benchmark_id)
        if not cases:
            raise HTTPException(
                status_code=422,
                detail="Benchmark has no test cases. Add test cases before running.",
            )
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

    return {
        "run_id": run["id"],
        "status": "running",
        "message": f"Eval started for {len(cases)} test cases.",
    }


@app.get("/runs/{run_id}")
def get_run_status(run_id: int):
    with get_connection() as conn:
        run = db.get_run(conn, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found.")
        completed = conn.execute(
            "SELECT COUNT(*) FROM run_results WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        total = conn.execute(
            "SELECT COUNT(*) FROM test_cases WHERE benchmark_id = ?",
            (run["benchmark_id"],),
        ).fetchone()[0]
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
        per_case.append(
            {
                "test_case_id": tc_id,
                "question": a["question"],
                "domain": a["domain"],
                "source_type": a["source_type"],
                "score_a": a["hallucination_score"],
                "label_a": a["overall_label"],
                "score_b": b["hallucination_score"],
                "label_b": b["overall_label"],
                "delta": round(delta, 4),
                "verdict": "improved"
                if delta < -0.05
                else "regressed"
                if delta > 0.05
                else "stable",
            }
        )

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
