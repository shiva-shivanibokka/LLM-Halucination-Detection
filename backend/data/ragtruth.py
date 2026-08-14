"""
data/ragtruth.py

Load the RAGTruth hallucination corpus (real, human-annotated) into the
platform as a labeled benchmark.

Dataset: `wandb/RAGTruth-processed` (train 15,090 / test 2,700). Each row has a
`query`, the retrieved `context`, an LLM `output`, and human hallucination
annotations. We store the RAGTruth `output` as the case's `answer` and derive a
binary `gold_label` (hallucinated | grounded) from the annotations.

Because RAGTruth's labels annotate ITS OWN `output`, a run over a RAGTruth
benchmark scores that stored answer directly (the runner skips generation when
`answer` is present) — so the resulting precision/recall/F1 measure how well the
NLI detector agrees with human judgments, which is the headline metric.
"""

from dataclasses import dataclass

DATASET_ID = "wandb/RAGTruth-processed"
DATASET_REVISION = None  # pin a commit hash here for full reproducibility if needed


@dataclass
class RagCase:
    question: str
    reference_text: str
    answer: str
    gold_label: str  # "hallucinated" | "grounded"
    domain: str = "general"
    source_type: str = "public"  # RAGTruth is public -> contamination-aware


def _is_hallucinated(row: dict) -> bool:
    proc = row.get("hallucination_labels_processed")
    if isinstance(proc, dict):
        return bool(proc.get("evident_conflict") or proc.get("baseless_info"))
    labels = row.get("hallucination_labels")
    return labels not in (None, "", "[]", [], {})


def load_ragtruth(split: str = "train", limit: int | None = 50) -> list[RagCase]:
    """Load up to `limit` RAGTruth rows as labeled cases. Skips rows missing
    query/context/output. Requires network + the `datasets` package."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=split, revision=DATASET_REVISION)
    cases: list[RagCase] = []
    for row in ds:
        q = (row.get("query") or "").strip()
        ctx = (row.get("context") or "").strip()
        out = (row.get("output") or "").strip()
        if not (q and ctx and out):
            continue
        cases.append(
            RagCase(
                question=q,
                reference_text=ctx,
                answer=out,
                gold_label="hallucinated" if _is_hallucinated(row) else "grounded",
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def seed_ragtruth_benchmark(name: str, split: str = "train",
                            limit: int | None = 50) -> dict:
    """Create a benchmark named `name` and populate it with RAGTruth cases.

    Opens its own DB connection only AFTER the (multi-minute) dataset download,
    so it never holds a pooled connection idle across the network fetch — the
    same pattern the eval runner uses for the model load.
    """
    from db import models as db
    from db.database import get_connection

    # Network download happens with NO DB connection held.
    cases = load_ragtruth(split=split, limit=limit)
    if not cases:
        raise ValueError("RAGTruth returned no usable rows — check dataset id/split.")
    # Fail loudly on a likely schema mismatch: if a healthy sample has zero
    # hallucinated labels, the annotation fields were probably misread and every
    # case silently became 'grounded', which would poison the F1 metric.
    if len(cases) >= 10 and not any(c.gold_label == "hallucinated" for c in cases):
        raise ValueError(
            "No 'hallucinated' gold labels in the RAGTruth sample — the "
            "hallucination-annotation field names likely changed. Verify the "
            "'wandb/RAGTruth-processed' schema in data/ragtruth.py before seeding."
        )
    rows = [
        {
            "question": c.question, "reference_text": c.reference_text,
            "domain": c.domain, "source_type": c.source_type,
            "gold_label": c.gold_label, "answer": c.answer,
        }
        for c in cases
    ]
    with get_connection() as conn:
        bm = db.create_benchmark(conn, name, f"RAGTruth {split} (n={len(cases)}) — human-labeled")
        db.add_test_cases(conn, bm["id"], rows)
    return {"benchmark_id": bm["id"], "added": len(cases)}
