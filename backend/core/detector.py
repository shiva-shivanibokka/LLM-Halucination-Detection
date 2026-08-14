"""
detector/hallucination_detector.py

NLI-based hallucination detection using DeBERTa-v3-large.

For each sentence in the LLM response:
  1. Retrieve the top-K most relevant source chunks from ChromaDB.
  2. Run each (source_chunk, sentence) pair through the NLI model.
  3. Aggregate entailment / contradiction / neutral scores.
  4. Classify the sentence as GROUNDED, CONTRADICTED, or UNGROUNDED.

Overall hallucination score and label are derived from sentence-level results.
"""

import re
from dataclasses import dataclass

import torch
from transformers import pipeline

NLI_MODEL = "cross-encoder/nli-deberta-v3-large"

# Default thresholds — can be overridden per-call
DEFAULT_ENTAIL_THRESHOLD = 0.5
DEFAULT_CONTRADICT_THRESHOLD = 0.5
DEFAULT_GROUNDED_CEILING = 0.3
DEFAULT_PARTIAL_CEILING = 0.6


@dataclass
class SentenceResult:
    sentence: str
    label: str
    hallucination_score: float
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    best_source_chunk: str | None
    best_source_similarity: float


@dataclass
class AnalysisResult:
    overall_label: str
    overall_hallucination_score: float
    total_sentences: int
    grounded_count: int
    ungrounded_count: int
    contradicted_count: int
    sentence_results: list[SentenceResult]


class HallucinationDetector:
    def __init__(self) -> None:
        device = 0 if torch.cuda.is_available() else -1
        self._nli = pipeline(
            "text-classification",
            model=NLI_MODEL,
            device=device,
            top_k=None,
        )

    def analyze(
        self,
        response: str,
        vector_store,
        entail_threshold: float = DEFAULT_ENTAIL_THRESHOLD,
        contradict_threshold: float = DEFAULT_CONTRADICT_THRESHOLD,
        grounded_ceiling: float = DEFAULT_GROUNDED_CEILING,
        partial_ceiling: float = DEFAULT_PARTIAL_CEILING,
    ) -> AnalysisResult:
        sentences = _split_sentences(response)
        sentence_results: list[SentenceResult] = []

        for sentence in sentences:
            if not sentence.strip():
                continue
            result = self._analyze_sentence(
                sentence, vector_store, entail_threshold, contradict_threshold
            )
            sentence_results.append(result)

        return _aggregate(sentence_results, grounded_ceiling, partial_ceiling)

    def _analyze_sentence(
        self,
        sentence: str,
        vector_store,
        entail_threshold: float,
        contradict_threshold: float,
    ) -> SentenceResult:
        candidates = vector_store.query(sentence)

        if not candidates:
            return SentenceResult(
                sentence=sentence,
                label="UNGROUNDED",
                hallucination_score=1.0,
                entailment_score=0.0,
                contradiction_score=0.0,
                neutral_score=1.0,
                best_source_chunk=None,
                best_source_similarity=0.0,
            )

        scored = [
            self._score_pair(candidate["chunk"][:512], sentence[:256])
            for candidate in candidates
        ]
        # Grounding is judged by the best-supporting chunk, but contradiction is
        # taken as the STRONGEST across all chunks — a sentence contradicted by
        # chunk B must not be missed just because chunk A entails it more.
        best_i, best_entail, best_contradict, best_neutral = _reduce_candidates(scored)
        best_chunk = candidates[best_i]["chunk"]
        best_similarity = candidates[best_i]["similarity"]

        label, hallucination_score = _classify(
            best_entail, best_contradict, entail_threshold, contradict_threshold
        )

        return SentenceResult(
            sentence=sentence,
            label=label,
            hallucination_score=hallucination_score,
            entailment_score=round(best_entail, 4),
            contradiction_score=round(best_contradict, 4),
            neutral_score=round(best_neutral, 4),
            best_source_chunk=best_chunk,
            best_source_similarity=round(best_similarity, 4),
        )

    def _score_pair(self, premise: str, hypothesis: str) -> dict[str, float]:
        raw = self._nli(
            f"{premise} [SEP] {hypothesis}", truncation=True, max_length=512
        )
        scores: dict[str, float] = {
            "entailment": 0.0,
            "contradiction": 0.0,
            "neutral": 0.0,
        }
        for item in raw[0]:
            label = item["label"].lower()
            if label in scores:
                scores[label] = item["score"]
        return scores


def _reduce_candidates(scored: list[dict]) -> tuple[int, float, float, float]:
    """Reduce per-candidate NLI scores to (best_entail_index, best_entailment,
    best_contradiction, neutral_of_best_entail_chunk).

    Entailment/neutral come from the single best-supporting chunk; contradiction
    is the MAX across all chunks so a contradiction in a non-top chunk still counts.
    """
    best_i = 0
    for i in range(1, len(scored)):
        if scored[i]["entailment"] > scored[best_i]["entailment"]:
            best_i = i
    best_entail = scored[best_i]["entailment"]
    best_neutral = scored[best_i]["neutral"]
    best_contradict = max(s["contradiction"] for s in scored)
    return best_i, best_entail, best_contradict, best_neutral


def _classify(
    best_entail: float,
    best_contradict: float,
    entail_threshold: float,
    contradict_threshold: float,
) -> tuple[str, float]:
    """Label a sentence and its hallucination score from its best entail/contradict."""
    if best_entail >= entail_threshold:
        return "GROUNDED", round(1.0 - best_entail, 4)
    if best_contradict >= contradict_threshold:
        return "CONTRADICTED", round(best_contradict, 4)
    return "UNGROUNDED", round(1.0 - best_entail, 4)


def _split_sentences(text: str) -> list[str]:
    pattern = r"(?<=[.!?])\s+"
    parts = re.split(pattern, text.strip())
    return [s.strip() for s in parts if s.strip()]


def _aggregate(
    results: list[SentenceResult],
    grounded_ceiling: float = DEFAULT_GROUNDED_CEILING,
    partial_ceiling: float = DEFAULT_PARTIAL_CEILING,
) -> AnalysisResult:
    if not results:
        return AnalysisResult(
            overall_label="UNGROUNDED",
            overall_hallucination_score=1.0,
            total_sentences=0,
            grounded_count=0,
            ungrounded_count=0,
            contradicted_count=0,
            sentence_results=[],
        )

    grounded = sum(1 for r in results if r.label == "GROUNDED")
    ungrounded = sum(1 for r in results if r.label == "UNGROUNDED")
    contradicted = sum(1 for r in results if r.label == "CONTRADICTED")
    total = len(results)

    overall_score = sum(r.hallucination_score for r in results) / total

    if overall_score <= grounded_ceiling:
        overall_label = "GROUNDED"
    elif overall_score <= partial_ceiling:
        overall_label = "PARTIALLY_GROUNDED"
    else:
        overall_label = "HALLUCINATED"

    return AnalysisResult(
        overall_label=overall_label,
        overall_hallucination_score=round(overall_score, 4),
        total_sentences=total,
        grounded_count=grounded,
        ungrounded_count=ungrounded,
        contradicted_count=contradicted,
        sentence_results=results,
    )
