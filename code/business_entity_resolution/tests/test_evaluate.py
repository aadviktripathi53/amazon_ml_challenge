"""Unit tests for src.metrics, including the worked example and singleton edge cases."""
import pytest

from src.evaluate import (
    avg_candidates_per_entity,
    blocking_recall,
    evaluate,
    fbeta_for_entity,
    macro_fbeta,
)


def test_worked_example_pred3_true2_correct2():
    """Predicted {A,B,C}, true {A,C}, 2 correct -> P=2/3, R=1, F0.5 ~= 0.714."""
    score = fbeta_for_entity({"A", "B", "C"}, {"A", "C"})
    assert score == pytest.approx(0.714, abs=1e-3)
    assert score == pytest.approx(5 / 7)


def test_perfect_prediction_is_one():
    """Exactly matching a non-singleton scores 1.0."""
    assert fbeta_for_entity({"a", "b"}, {"a", "b"}) == 1.0


def test_singleton_empty_prediction_is_one():
    """Empty prediction on a true singleton scores 1.0."""
    assert fbeta_for_entity(set(), set()) == 1.0


def test_singleton_any_prediction_is_zero():
    """Any prediction on a true singleton scores 0.0."""
    assert fbeta_for_entity({"x"}, set()) == 0.0
    assert fbeta_for_entity({"x", "y"}, set()) == 0.0


def test_non_singleton_empty_prediction_is_zero():
    """Empty prediction on a non-singleton scores 0.0."""
    assert fbeta_for_entity(set(), {"a"}) == 0.0


def test_no_correct_predictions_is_zero():
    """Disjoint non-empty prediction scores 0.0."""
    assert fbeta_for_entity({"x"}, {"a"}) == 0.0


def test_precision_weighted_more_than_recall():
    """F0.5 penalises a wrong extra prediction more than a missed match."""
    extra = fbeta_for_entity({"a", "x"}, {"a"})  # P=0.5, R=1
    missed = fbeta_for_entity({"a"}, {"a", "b"})  # P=1, R=0.5
    assert missed > extra


def test_accepts_lists_with_duplicates():
    """Duplicate IDs in a list input do not change the score."""
    assert fbeta_for_entity(["a", "a", "b"], ["a", "b"]) == 1.0


def test_macro_average_over_entities():
    """Macro score averages per-entity scores, incl. singletons and missing predictions."""
    truth = {"e1": {"a", "b"}, "e2": set(), "e3": {"c"}, "e4": set()}
    preds = {"e1": {"a", "b", "z"}, "e2": set(), "e4": {"q"}}  # e3 missing -> empty
    expected = (5 / 7 + 1.0 + 0.0 + 0.0) / 4
    assert macro_fbeta(preds, truth) == pytest.approx(expected)


def test_macro_ignores_predictions_outside_truth():
    """Predictions for IDs not in truth do not affect the score."""
    assert macro_fbeta({"e1": {"a"}, "other": {"z"}}, {"e1": {"a"}}) == 1.0


def test_macro_empty_truth_raises():
    """An empty truth mapping is an error rather than a silent 0/1."""
    with pytest.raises(ValueError):
        macro_fbeta({}, {})


def test_blocking_recall():
    """Recall counts true pairs found among candidates, pooled across entities."""
    truth = {"e1": {"a", "b"}, "e2": {"c"}, "e3": set()}
    cands = {"e1": {"a", "x"}, "e2": {"c"}, "e3": {"y"}}
    assert blocking_recall(cands, truth) == pytest.approx(2 / 3)


def test_blocking_recall_missing_candidates_and_no_pairs():
    """Missing candidate entries count as misses; no true pairs gives 1.0."""
    assert blocking_recall({}, {"e1": {"a"}}) == 0.0
    assert blocking_recall({}, {"e1": set()}) == 1.0


def test_avg_candidates_per_entity():
    """Average counts distinct candidates and treats missing entities as 0."""
    cands = {"e1": ["a", "a", "b"], "e2": ["c"]}
    assert avg_candidates_per_entity(cands, ["e1", "e2", "e3"]) == pytest.approx(1.0)
    assert avg_candidates_per_entity(cands, []) == 0.0


def test_evaluate_bundle():
    """evaluate() returns all three metrics keyed by name."""
    truth = {"e1": {"a"}, "e2": set()}
    result = evaluate({"e1": {"a"}}, {"e1": {"a", "b"}, "e2": set()}, truth)
    assert result == {"macro_f05": 1.0, "blocking_recall": 1.0, "avg_candidates": 1.0}
