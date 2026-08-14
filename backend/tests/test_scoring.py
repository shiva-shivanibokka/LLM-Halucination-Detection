"""Pure-logic tests for detector-vs-gold metrics. No DB, runs anywhere."""

from eval.scoring import GROUNDED, HALLUCINATED, binary_metrics, label_to_binary


def test_f1_on_known_pairs():
    pairs = [
        ("hallucinated", "hallucinated"),  # TP
        ("hallucinated", "grounded"),      # FP
        ("grounded", "grounded"),          # TN
        ("grounded", "hallucinated"),      # FN
    ]
    m = binary_metrics(pairs)
    assert m["n"] == 4
    assert round(m["precision"], 3) == 0.5
    assert round(m["recall"], 3) == 0.5
    assert round(m["f1"], 3) == 0.5
    assert round(m["accuracy"], 3) == 0.5


def test_perfect_and_empty():
    perfect = [("hallucinated", "hallucinated"), ("grounded", "grounded")]
    m = binary_metrics(perfect)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    empty = binary_metrics([])
    assert empty == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "n": 0}


def test_label_mapping():
    assert label_to_binary("GROUNDED") == GROUNDED
    assert label_to_binary("HALLUCINATED") == HALLUCINATED
    assert label_to_binary("UNGROUNDED") == HALLUCINATED
    assert label_to_binary("PARTIALLY_GROUNDED") == HALLUCINATED
    assert label_to_binary("PARTIALLY_GROUNDED", partial_is_hallucinated=False) == GROUNDED


if __name__ == "__main__":
    test_f1_on_known_pairs()
    test_perfect_and_empty()
    test_label_mapping()
    print("PASS")
