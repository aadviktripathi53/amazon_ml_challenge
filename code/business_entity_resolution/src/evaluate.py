"""Evaluation: macro F0.5 over Source 1 entities, blocking recall and average candidates per entity."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Set

BETA = 0.5


def fbeta_for_entity(predicted: Iterable[str], true: Iterable[str], beta: float = BETA) -> float:
    """F-beta score for a single Source 1 entity.

    Rules:
        * true singleton, empty prediction -> 1.0
        * true singleton, any prediction -> 0.0
        * non-singleton, empty prediction -> 0.0
        * otherwise standard F-beta of precision/recall (0.0 if nothing correct)

    Args:
        predicted: Predicted matched IDs.
        true: Ground-truth matched IDs (empty for a singleton).
        beta: F-beta weight; 0.5 favours precision over recall.

    Returns:
        Score in [0, 1].
    """
    predicted, true = set(predicted), set(true)
    if not true:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    correct = len(predicted & true)
    if correct == 0:
        return 0.0
    precision = correct / len(predicted)
    recall = correct / len(true)
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def macro_fbeta(
    predictions: Mapping[str, Iterable[str]],
    truth: Mapping[str, Iterable[str]],
    beta: float = BETA,
) -> float:
    """Macro-average of per-entity F-beta over every Source 1 entity in ``truth``.

    Entities missing from ``predictions`` count as empty predictions; entities
    in ``predictions`` but not in ``truth`` are ignored.

    Args:
        predictions: Mapping from Source 1 ID to predicted matched IDs.
        truth: Mapping from Source 1 ID to true matched IDs.
        beta: F-beta weight.

    Returns:
        Mean per-entity score.

    Raises:
        ValueError: If ``truth`` is empty.
    """
    if not truth:
        raise ValueError("truth must contain at least one Source 1 entity")
    scores = [fbeta_for_entity(predictions.get(eid, ()), true, beta) for eid, true in truth.items()]
    return sum(scores) / len(scores)


def blocking_recall(candidates: Mapping[str, Iterable[str]], truth: Mapping[str, Iterable[str]]) -> float:
    """Fraction of true (Source 1, matched) pairs that appear in the candidate dict.

    Args:
        candidates: Mapping from Source 1 ID to candidate IDs.
        truth: Mapping from Source 1 ID to true matched IDs.

    Returns:
        Recall in [0, 1]; 1.0 if there are no true pairs at all.
    """
    total = found = 0
    for eid, true in truth.items():
        true = set(true)
        total += len(true)
        found += len(true & set(candidates.get(eid, ())))
    return found / total if total else 1.0


def avg_candidates_per_entity(candidates: Mapping[str, Iterable[str]], entity_ids: Iterable[str]) -> float:
    """Average number of distinct candidates per Source 1 entity.

    Args:
        candidates: Mapping from Source 1 ID to candidate IDs.
        entity_ids: Source 1 IDs to average over (missing ones count as 0).

    Returns:
        Mean candidate count; 0.0 if ``entity_ids`` is empty.
    """
    counts = [len(set(candidates.get(eid, ()))) for eid in entity_ids]
    return sum(counts) / len(counts) if counts else 0.0


def evaluate(
    predictions: Mapping[str, Iterable[str]],
    candidates: Mapping[str, Iterable[str]],
    truth: Mapping[str, Set[str]],
) -> Dict[str, float]:
    """Compute macro F0.5, blocking recall and average candidates per entity.

    Args:
        predictions: Mapping from Source 1 ID to predicted matched IDs.
        candidates: Mapping from Source 1 ID to candidate IDs.
        truth: Mapping from Source 1 ID to true matched IDs.

    Returns:
        Dict with keys ``macro_f05``, ``blocking_recall``, ``avg_candidates``.
    """
    return {
        "macro_f05": macro_fbeta(predictions, truth),
        "blocking_recall": blocking_recall(candidates, truth),
        "avg_candidates": avg_candidates_per_entity(candidates, truth.keys()),
    }
