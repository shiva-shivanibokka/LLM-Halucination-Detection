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
from dataclasses import dataclass, field
from typing import Optional

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
    best_source_chunk: Optional[str]
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

        best_chunk = candidates[0]["chunk"]
        best_similarity = candidates[0]["similarity"]
        best_entail = 0.0
        best_contradict = 0.0
        best_neutral = 0.0

        for candidate in candidates:
            chunk = candidate["chunk"]
            premise = chunk[:512]
            hypothesis = sentence[:256]
            scores = self._score_pair(premise, hypothesis)
            if scores["entailment"] > best_entail:
                best_entail = scores["entailment"]
                best_contradict = scores["contradiction"]
                best_neutral = scores["neutral"]
                best_chunk = chunk
                best_similarity = candidate["similarity"]

        if best_entail >= entail_threshold:
            label = "GROUNDED"
            hallucination_score = 1.0 - best_entail
        elif best_contradict >= contradict_threshold:
            label = "CONTRADICTED"
            hallucination_score = best_contradict
        else:
            label = "UNGROUNDED"
            hallucination_score = 1.0 - best_entail

        return SentenceResult(
            sentence=sentence,
            label=label,
            hallucination_score=round(hallucination_score, 4),
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
