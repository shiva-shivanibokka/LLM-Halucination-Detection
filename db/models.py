"""
db/models.py

CRUD operations for all database tables.
Each function takes a sqlite3.Connection and returns plain dicts or lists of dicts.
No ORM — just SQL. Keeps it readable and dependency-free.
"""

import json
import sqlite3
from dataclasses import asdict
from typing import Optional


def create_benchmark(
    conn: sqlite3.Connection, name: str, description: str = ""
) -> dict:
    cur = conn.execute(
        "INSERT INTO benchmarks (name, description) VALUES (?, ?) RETURNING *",
        (name, description),
    )
    row = cur.fetchone()
    conn.commit()
    return dict(row)


def get_benchmark(conn: sqlite3.Connection, benchmark_id: int) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM benchmarks WHERE id = ?", (benchmark_id,)
    ).fetchone()
    return dict(row) if row else None


def list_benchmarks(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("""
        SELECT b.*, COUNT(tc.id) AS case_count
        FROM benchmarks b
        LEFT JOIN test_cases tc ON tc.benchmark_id = b.id
        GROUP BY b.id
        ORDER BY b.created_at DESC
    """).fetchall()
    return [dict(r) for r in rows]


def delete_benchmark(conn: sqlite3.Connection, benchmark_id: int) -> None:
    conn.execute("DELETE FROM benchmarks WHERE id = ?", (benchmark_id,))
    conn.commit()


def add_test_case(
    conn: sqlite3.Connection,
    benchmark_id: int,
    question: str,
    reference_text: str,
    domain: str = "general",
    source_type: str = "internal",
) -> dict:
    cur = conn.execute(
        "INSERT INTO test_cases (benchmark_id, question, reference_text, domain, source_type) VALUES (?, ?, ?, ?, ?) RETURNING *",
        (benchmark_id, question, reference_text, domain, source_type),
    )
    row = cur.fetchone()
    conn.commit()
    return dict(row)


def get_test_cases(conn: sqlite3.Connection, benchmark_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM test_cases WHERE benchmark_id = ? ORDER BY id",
        (benchmark_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_test_case(conn: sqlite3.Connection, test_case_id: int) -> None:
    conn.execute("DELETE FROM test_cases WHERE id = ?", (test_case_id,))
    conn.commit()


def create_run(
    conn: sqlite3.Connection,
    benchmark_id: int,
    provider: str,
    model: str,
) -> dict:
    cur = conn.execute(
        "INSERT INTO eval_runs (benchmark_id, provider, model, status) VALUES (?, ?, ?, 'running') RETURNING *",
        (benchmark_id, provider, model),
    )
    row = cur.fetchone()
    conn.commit()
    return dict(row)


def complete_run(
    conn: sqlite3.Connection,
    run_id: int,
    avg_score: float,
    grounded_pct: float,
) -> None:
    conn.execute(
        """UPDATE eval_runs
           SET status = 'completed', avg_score = ?, grounded_pct = ?,
               completed_at = datetime('now')
           WHERE id = ?""",
        (avg_score, grounded_pct, run_id),
    )
    conn.commit()


def fail_run(conn: sqlite3.Connection, run_id: int, reason: str) -> None:
    conn.execute(
        "UPDATE eval_runs SET status = 'failed', completed_at = datetime('now') WHERE id = ?",
        (run_id,),
    )
    conn.commit()


def get_run(conn: sqlite3.Connection, run_id: int) -> Optional[dict]:
    row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def list_runs(
    conn: sqlite3.Connection, benchmark_id: Optional[int] = None
) -> list[dict]:
    if benchmark_id is not None:
        rows = conn.execute(
            "SELECT r.*, b.name AS benchmark_name FROM eval_runs r JOIN benchmarks b ON b.id = r.benchmark_id WHERE r.benchmark_id = ? ORDER BY r.run_at DESC",
            (benchmark_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT r.*, b.name AS benchmark_name FROM eval_runs r JOIN benchmarks b ON b.id = r.benchmark_id ORDER BY r.run_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def add_run_result(
    conn: sqlite3.Connection,
    run_id: int,
    test_case_id: int,
    response: str,
    overall_label: str,
    hallucination_score: float,
    grounded_count: int,
    ungrounded_count: int,
    contradicted_count: int,
    total_sentences: int,
    sentence_results: list,
) -> dict:
    cur = conn.execute(
        """INSERT INTO run_results
           (run_id, test_case_id, response, overall_label, hallucination_score,
            grounded_count, ungrounded_count, contradicted_count, total_sentences, sentence_results)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING *""",
        (
            run_id,
            test_case_id,
            response,
            overall_label,
            hallucination_score,
            grounded_count,
            ungrounded_count,
            contradicted_count,
            total_sentences,
            json.dumps(sentence_results),
        ),
    )
    row = cur.fetchone()
    conn.commit()
    return dict(row)


def get_run_results(conn: sqlite3.Connection, run_id: int) -> list[dict]:
    rows = conn.execute(
        """SELECT rr.*, tc.question, tc.domain, tc.source_type, tc.reference_text
           FROM run_results rr
           JOIN test_cases tc ON tc.id = rr.test_case_id
           WHERE rr.run_id = ?
           ORDER BY rr.id""",
        (run_id,),
    ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["sentence_results"] = json.loads(d["sentence_results"])
        results.append(d)
    return results


def get_source_type_scores(conn: sqlite3.Connection, run_id: int) -> dict:
    """
    Return avg hallucination scores split by source_type (internal vs public).
    Used in comparison reports to flag which results are trustworthy.
    """
    rows = conn.execute(
        """SELECT tc.source_type,
                  COUNT(*) AS total,
                  AVG(rr.hallucination_score) AS avg_score,
                  SUM(CASE WHEN rr.overall_label = 'GROUNDED' THEN 1 ELSE 0 END) AS grounded,
                  SUM(CASE WHEN rr.overall_label = 'HALLUCINATED' THEN 1 ELSE 0 END) AS hallucinated
           FROM run_results rr
           JOIN test_cases tc ON tc.id = rr.test_case_id
           WHERE rr.run_id = ?
           GROUP BY tc.source_type""",
        (run_id,),
    ).fetchall()
    result = {"internal": None, "public": None}
    for r in rows:
        result[r["source_type"]] = dict(r)
    return result


def get_domain_scores(conn: sqlite3.Connection, run_id: int) -> list[dict]:
    rows = conn.execute(
        """SELECT tc.domain,
                  COUNT(*) AS total,
                  AVG(rr.hallucination_score) AS avg_score,
                  SUM(CASE WHEN rr.overall_label = 'GROUNDED' THEN 1 ELSE 0 END) AS grounded,
                  SUM(CASE WHEN rr.overall_label = 'PARTIALLY_GROUNDED' THEN 1 ELSE 0 END) AS partial,
                  SUM(CASE WHEN rr.overall_label = 'HALLUCINATED' THEN 1 ELSE 0 END) AS hallucinated
           FROM run_results rr
           JOIN test_cases tc ON tc.id = rr.test_case_id
           WHERE rr.run_id = ?
           GROUP BY tc.domain
           ORDER BY avg_score DESC""",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]
