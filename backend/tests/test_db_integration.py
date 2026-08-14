"""
Postgres integration tests: connection guard + full CRUD + fail_run reason +
metrics round-trip. Requires DATABASE_URL (Neon or local Postgres); skipped
otherwise via the `db` fixture. Runs in CI against a Postgres service.
"""

import pytest
from db import models as m


def test_dsn_required(monkeypatch):
    """get_pool() must fail loudly when DATABASE_URL is unset."""
    import db.database as d

    monkeypatch.delenv("DATABASE_URL", raising=False)
    d._pool = None
    with pytest.raises(RuntimeError):
        d.get_pool()
    d._pool = None  # let later tests reopen with the real DSN


def test_benchmark_and_case_crud(db):
    bm = m.create_benchmark(db, "unit-bench", "desc")
    assert bm["id"] and bm["name"] == "unit-bench"

    tc = m.add_test_case(
        db, bm["id"], "Q?", "reference text", domain="general",
        source_type="public", gold_label="hallucinated", answer="an answer",
    )
    assert tc["gold_label"] == "hallucinated"
    assert tc["answer"] == "an answer"

    cases = m.get_test_cases(db, bm["id"])
    assert len(cases) == 1 and cases[0]["id"] == tc["id"]

    listing = m.list_benchmarks(db)
    assert any(b["id"] == bm["id"] and b["case_count"] == 1 for b in listing)


def test_run_result_and_fail_reason(db):
    bm = m.create_benchmark(db, "run-bench")
    tc = m.add_test_case(db, bm["id"], "Q?", "ref")
    run = m.create_run(db, bm["id"], "openai", "gpt-4o")

    m.add_run_result(
        db, run_id=run["id"], test_case_id=tc["id"], response="resp",
        overall_label="GROUNDED", hallucination_score=0.1,
        grounded_count=1, ungrounded_count=0, contradicted_count=0,
        total_sentences=1, sentence_results=[{"s": "resp", "label": "GROUNDED"}],
        predicted_label="grounded",
    )
    results = m.get_run_results(db, run["id"])
    assert len(results) == 1
    assert results[0]["predicted_label"] == "grounded"
    assert results[0]["sentence_results"] == [{"s": "resp", "label": "GROUNDED"}]  # JSONB round-trip

    # fail_run must persist the reason (regression test for the discarded-reason bug).
    m.fail_run(db, run["id"], "boom happened")
    got = m.get_run(db, run["id"])
    assert got["status"] == "failed"
    assert got["error"] == "boom happened"


def test_metrics_roundtrip(db):
    bm = m.create_benchmark(db, "metric-bench")
    run = m.create_run(db, bm["id"], "detector", "deberta")
    metrics = {"precision": 0.8, "recall": 0.75, "f1": 0.774, "accuracy": 0.9, "n": 20}
    m.save_run_metrics(db, run["id"], metrics)
    got = m.get_run_metrics(db, run["id"])
    assert got["n"] == 20 and round(got["f1"], 3) == 0.774
    # upsert: saving again with new numbers replaces, not duplicates
    m.save_run_metrics(db, run["id"], {**metrics, "n": 21})
    assert m.get_run_metrics(db, run["id"])["n"] == 21
