"""Unit tests for the detector's pure scoring logic (no model load required)."""

from core.detector import _classify, _reduce_candidates


def test_contradiction_in_non_top_entail_chunk_is_caught():
    # Chunk A weakly entails but strongly contradicts; chunk B entails more but
    # is neutral. Old code read contradiction only from the max-entail chunk (B)
    # and missed A's contradiction. Now contradiction is the max across chunks.
    scored = [
        {"entailment": 0.05, "contradiction": 0.95, "neutral": 0.00},  # A
        {"entailment": 0.30, "contradiction": 0.10, "neutral": 0.60},  # B
    ]
    best_i, best_entail, best_contradict, _best_neutral = _reduce_candidates(scored)
    assert best_i == 1               # entailment still comes from chunk B
    assert best_entail == 0.30
    assert best_contradict == 0.95   # contradiction taken from chunk A
    label, score = _classify(best_entail, best_contradict, 0.5, 0.5)
    assert label == "CONTRADICTED"
    assert score == 0.95


def test_grounded_when_entailment_clears_threshold():
    label, score = _classify(0.80, 0.10, 0.5, 0.5)
    assert label == "GROUNDED"
    assert score == round(1.0 - 0.80, 4)


def test_ungrounded_when_neither_threshold_met():
    label, score = _classify(0.20, 0.20, 0.5, 0.5)
    assert label == "UNGROUNDED"
    assert score == round(1.0 - 0.20, 4)


def test_reduce_single_candidate():
    best_i, e, c, n = _reduce_candidates([{"entailment": 0.4, "contradiction": 0.3, "neutral": 0.3}])
    assert (best_i, e, c, n) == (0, 0.4, 0.3, 0.3)
