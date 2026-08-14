"""
db/models.py

CRUD operations for all tables. Each function takes a psycopg Connection
(dict_row) and returns plain dicts or lists of dicts. No ORM — just SQL.
"""


from psycopg.types.json import Jsonb


# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #
def create_benchmark(conn, name: str, description: str = "") -> dict:
    row = conn.execute(
        "INSERT INTO benchmarks (name, description) VALUES (%s, %s) RETURNING *",
        (name, description),
    ).fetchone()
    conn.commit()
    return dict(row)


def get_benchmark(conn, benchmark_id: int) -> dict | None:
    row = conn.execute(
        "SELECT * FROM benchmarks WHERE id = %s", (benchmark_id,)
    ).fetchone()
    return dict(row) if row else None


def list_benchmarks(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT b.*, COUNT(tc.id) AS case_count
        FROM benchmarks b
        LEFT JOIN test_cases tc ON tc.benchmark_id = b.id
        GROUP BY b.id
        ORDER BY b.created_at DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def delete_benchmark(conn, benchmark_id: int) -> None:
    conn.execute("DELETE FROM benchmarks WHERE id = %s", (benchmark_id,))
    conn.commit()


# --------------------------------------------------------------------------- #
# Test cases
# --------------------------------------------------------------------------- #
def add_test_case(
    conn,
    benchmark_id: int,
    question: str,
    reference_text: str,
    domain: str = "general",
    source_type: str = "internal",
    gold_label: str | None = None,
    answer: str | None = None,
) -> dict:
    row = conn.execute(
        """INSERT INTO test_cases
             (benchmark_id, question, reference_text, domain, source_type, gold_label, answer)
           VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *""",
        (benchmark_id, question, reference_text, domain, source_type, gold_label, answer),
    ).fetchone()
    conn.commit()
    return dict(row)


def add_test_cases(conn, benchmark_id: int, rows: list[dict]) -> int:
    """Insert many test cases atomically (one transaction, one commit). A failure
    mid-way rolls the whole batch back instead of leaving a half-populated
    benchmark. Each row: question, reference_text, and optional domain,
    source_type, gold_label, answer."""
    if not rows:
        return 0
    params = [
        (
            benchmark_id, r["question"], r["reference_text"],
            r.get("domain", "general"), r.get("source_type", "internal"),
            r.get("gold_label"), r.get("answer"),
        )
        for r in rows
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO test_cases
                 (benchmark_id, question, reference_text, domain, source_type, gold_label, answer)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            params,
        )
    conn.commit()
    return len(rows)


def case_has_results(conn, test_case_id: int) -> bool:
    """True if any run_results row references this case (deleting it would
    corrupt the history of completed runs via ON DELETE CASCADE)."""
    row = conn.execute(
        "SELECT 1 FROM run_results WHERE test_case_id = %s LIMIT 1", (test_case_id,)
    ).fetchone()
    return row is not None


def get_test_cases(conn, benchmark_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM test_cases WHERE benchmark_id = %s ORDER BY id",
        (benchmark_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_test_case(conn, test_case_id: int) -> None:
    conn.execute("DELETE FROM test_cases WHERE id = %s", (test_case_id,))
    conn.commit()


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
def create_run(conn, benchmark_id: int, provider: str, model: str,
               dataset: str | None = None) -> dict:
    row = conn.execute(
        """INSERT INTO eval_runs (benchmark_id, provider, model, status, dataset)
           VALUES (%s, %s, %s, 'running', %s) RETURNING *""",
        (benchmark_id, provider, model, dataset),
    ).fetchone()
    conn.commit()
    return dict(row)


def complete_run(conn, run_id: int, avg_score: float, grounded_pct: float) -> None:
    conn.execute(
        """UPDATE eval_runs
           SET status = 'completed', avg_score = %s, grounded_pct = %s,
               completed_at = now()
           WHERE id = %s""",
        (avg_score, grounded_pct, run_id),
    )
    conn.commit()


def fail_run(conn, run_id: int, reason: str) -> None:
    conn.execute(
        "UPDATE eval_runs SET status = 'failed', error = %s, completed_at = now() WHERE id = %s",
        (reason[:2000] if reason else None, run_id),
    )
    conn.commit()


def get_run(conn, run_id: int) -> dict | None:
    row = conn.execute(
        """SELECT r.*, b.name AS benchmark_name
           FROM eval_runs r JOIN benchmarks b ON b.id = r.benchmark_id
           WHERE r.id = %s""",
        (run_id,),
    ).fetchone()
    return dict(row) if row else None


def list_runs(conn, benchmark_id: int | None = None) -> list[dict]:
    if benchmark_id is not None:
        rows = conn.execute(
            """SELECT r.*, b.name AS benchmark_name
               FROM eval_runs r JOIN benchmarks b ON b.id = r.benchmark_id
               WHERE r.benchmark_id = %s ORDER BY r.run_at DESC""",
            (benchmark_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT r.*, b.name AS benchmark_name
               FROM eval_runs r JOIN benchmarks b ON b.id = r.benchmark_id
               ORDER BY r.run_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Run results
# --------------------------------------------------------------------------- #
def add_run_result(
    conn,
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
    predicted_label: str | None = None,
) -> dict:
    row = conn.execute(
        """INSERT INTO run_results
             (run_id, test_case_id, response, overall_label, hallucination_score,
              grounded_count, ungrounded_count, contradicted_count, total_sentences,
              sentence_results, predicted_label)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *""",
        (
            run_id, test_case_id, response, overall_label, hallucination_score,
            grounded_count, ungrounded_count, contradicted_count, total_sentences,
            Jsonb(sentence_results), predicted_label,
        ),
    ).fetchone()
    conn.commit()
    return dict(row)


def get_run_results(conn, run_id: int) -> list[dict]:
    rows = conn.execute(
        """SELECT rr.*, tc.question, tc.domain, tc.source_type, tc.reference_text,
                  tc.gold_label
           FROM run_results rr
           JOIN test_cases tc ON tc.id = rr.test_case_id
           WHERE rr.run_id = %s
           ORDER BY rr.id""",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]   # sentence_results already parsed from JSONB


def get_source_type_scores(conn, run_id: int) -> dict:
    rows = conn.execute(
        """SELECT tc.source_type,
                  COUNT(*)                    AS total,
                  AVG(rr.hallucination_score) AS avg_score,
                  SUM(CASE WHEN rr.overall_label = 'GROUNDED'  THEN 1 ELSE 0 END) AS grounded,
                  SUM(CASE WHEN rr.overall_label <> 'GROUNDED' THEN 1 ELSE 0 END) AS hallucinated
           FROM run_results rr
           JOIN test_cases tc ON tc.id = rr.test_case_id
           WHERE rr.run_id = %s
           GROUP BY tc.source_type""",
        (run_id,),
    ).fetchall()
    result: dict = {"internal": None, "public": None}
    for r in rows:
        d = dict(r)
        d["avg_score"] = float(d["avg_score"]) if d["avg_score"] is not None else None
        result[r["source_type"]] = d
    return result


def get_domain_scores(conn, run_id: int) -> list[dict]:
    rows = conn.execute(
        """SELECT tc.domain,
                  COUNT(*)                    AS total,
                  AVG(rr.hallucination_score) AS avg_score,
                  SUM(CASE WHEN rr.overall_label = 'GROUNDED'                 THEN 1 ELSE 0 END) AS grounded,
                  SUM(CASE WHEN rr.overall_label = 'PARTIALLY_GROUNDED'       THEN 1 ELSE 0 END) AS partial,
                  -- UNGROUNDED (empty-response short-circuit) counts as hallucinated,
                  -- matching the F1 mapping, so grounded+partial+hallucinated = total.
                  SUM(CASE WHEN rr.overall_label IN ('HALLUCINATED','UNGROUNDED') THEN 1 ELSE 0 END) AS hallucinated
           FROM run_results rr
           JOIN test_cases tc ON tc.id = rr.test_case_id
           WHERE rr.run_id = %s
           GROUP BY tc.domain
           ORDER BY avg_score DESC""",
        (run_id,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["avg_score"] = float(d["avg_score"]) if d["avg_score"] is not None else None
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# Metrics (detector vs gold labels)
# --------------------------------------------------------------------------- #
def save_run_metrics(conn, run_id: int, metrics: dict) -> None:
    conn.execute(
        """INSERT INTO run_metrics (run_id, precision, recall, f1, accuracy, n)
           VALUES (%s, %s, %s, %s, %s, %s)
           ON CONFLICT (run_id) DO UPDATE SET
             precision = EXCLUDED.precision, recall = EXCLUDED.recall,
             f1 = EXCLUDED.f1, accuracy = EXCLUDED.accuracy, n = EXCLUDED.n""",
        (run_id, metrics["precision"], metrics["recall"], metrics["f1"],
         metrics["accuracy"], metrics["n"]),
    )
    conn.commit()


def get_run_metrics(conn, run_id: int) -> dict | None:
    row = conn.execute(
        "SELECT * FROM run_metrics WHERE run_id = %s", (run_id,)
    ).fetchone()
    return dict(row) if row else None
