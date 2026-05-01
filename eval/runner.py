"""
eval/runner.py

Runs a full benchmark against a model and writes results to the database.

For each test case in the benchmark:
  1. Build a per-case in-memory vector store from the reference document.
  2. Generate a grounded response (LLM sees the reference document).
  3. Score the response with the NLI detector against the reference.
  4. Write the result to run_results.

At the end, compute aggregate stats and mark the run as completed.

The runner is synchronous and designed to be called from a background thread
by the FastAPI endpoint so the HTTP response returns immediately while work proceeds.
Progress is polled via GET /runs/{run_id}.
"""

import json
from dataclasses import asdict
from typing import Optional, Callable

from core.generator import generate_grounded
from core.detector import (
    HallucinationDetector,
    DEFAULT_ENTAIL_THRESHOLD,
    DEFAULT_CONTRADICT_THRESHOLD,
    DEFAULT_GROUNDED_CEILING,
    DEFAULT_PARTIAL_CEILING,
)
from core.ingestor import extract_text_chunks
from core.vector_store import VectorStore
from db.database import get_connection
from db.models import (
    get_test_cases,
    add_run_result,
    complete_run,
    fail_run,
    get_run,
)


def run_benchmark(
    run_id: int,
    api_key: Optional[str] = None,
    entail_threshold: float = DEFAULT_ENTAIL_THRESHOLD,
    contradict_threshold: float = DEFAULT_CONTRADICT_THRESHOLD,
    grounded_ceiling: float = DEFAULT_GROUNDED_CEILING,
    partial_ceiling: float = DEFAULT_PARTIAL_CEILING,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Execute a benchmark run end-to-end.
    Called in a background thread — does not return anything meaningful.
    Results are written directly to the database as they complete.
    """
    conn = get_connection()
    run = get_run(conn, run_id)
    if not run:
        return

    provider = run["provider"]
    model = run["model"]
    benchmark_id = run["benchmark_id"]
    test_cases = get_test_cases(conn, benchmark_id)

    if not test_cases:
        fail_run(conn, run_id, "Benchmark has no test cases.")
        conn.close()
        return

    detector = HallucinationDetector()
    scores = []
    grounded_verdicts = 0

    try:
        for i, tc in enumerate(test_cases):
            if on_progress:
                on_progress(i, len(test_cases))

            store = VectorStore()
            chunks = extract_text_chunks(tc["reference_text"])
            if chunks:
                store.add_chunks(chunks, source_label=f"test_case_{tc['id']}")

            try:
                response = generate_grounded(
                    question=tc["question"],
                    vector_store=store,
                    provider=provider,
                    model=model,
                    api_key=api_key,
                )
            except Exception as e:
                response = f"[ERROR: {e}]"

            analysis = detector.analyze(
                response,
                store,
                entail_threshold=entail_threshold,
                contradict_threshold=contradict_threshold,
                grounded_ceiling=grounded_ceiling,
                partial_ceiling=partial_ceiling,
            )

            sentence_results = [asdict(sr) for sr in analysis.sentence_results]

            add_run_result(
                conn,
                run_id=run_id,
                test_case_id=tc["id"],
                response=response,
                overall_label=analysis.overall_label,
                hallucination_score=analysis.overall_hallucination_score,
                grounded_count=analysis.grounded_count,
                ungrounded_count=analysis.ungrounded_count,
                contradicted_count=analysis.contradicted_count,
                total_sentences=analysis.total_sentences,
                sentence_results=sentence_results,
            )

            scores.append(analysis.overall_hallucination_score)
            if analysis.overall_label == "GROUNDED":
                grounded_verdicts += 1

        avg_score = sum(scores) / len(scores) if scores else 0.0
        grounded_pct = grounded_verdicts / len(test_cases) if test_cases else 0.0
        complete_run(conn, run_id, avg_score, grounded_pct)

        if on_progress:
            on_progress(len(test_cases), len(test_cases))

    except Exception as e:
        fail_run(conn, run_id, str(e))
    finally:
        conn.close()
