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


# ---------------------------------------------------------------------------------------------------------------
# Command line: `python -m src.evaluate --split val`. The metric functions above are unchanged; this only loads the
# pipeline files and reports macro F0.5 with the extra breakdowns.
# ---------------------------------------------------------------------------------------------------------------
def _pairs_to_dict(table_path) -> Dict[str, Set[str]]:
    """Read an ``(s1_id, cand_id, ...)`` parquet into ``{s1_id: {cand ids}}``, streaming by row group.

    Args:
        table_path: Parquet path.

    Returns:
        Mapping from S1 id to the set of its ids.
    """
    import pyarrow.parquet as pq

    out: Dict[str, Set[str]] = {}
    pf = pq.ParquetFile(table_path)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["s1_id", "cand_id"])
        for s, c in zip(t.column("s1_id").to_pylist(), t.column("cand_id").to_pylist()):
            out.setdefault(s, set()).add(c)
    return out


def evaluate_split(split: str = "val") -> Dict[str, object]:
    """Evaluate a labelled split: macro F0.5, precision, recall, singleton / non-singleton / per-country F0.5.

    Args:
        split: ``"val"`` (or ``"train"`` for a sanity check with the OOF-thresholded matches).

    Returns:
        Metrics dict (also printed and saved to ``data/interim/metrics/evaluate_<split>.json``).
    """
    import pyarrow.parquet as pq

    from .block import candidates_path
    from .config import TRAIN_DIR
    from .decide import matches_path
    from .io_utils import GROUND_TRUTH_SUFFIX, load_ground_truth_subset
    from .normalize import records_path
    from .perf import SAMPLE_CAVEAT, save_metrics
    from .split import load_split_ids

    s1_ids = load_split_ids(split)
    truth = load_ground_truth_subset(TRAIN_DIR / f"train_{GROUND_TRUTH_SUFFIX}", s1_ids)
    candidates = _pairs_to_dict(candidates_path(split))
    predictions = _pairs_to_dict(matches_path(split))
    base = evaluate(predictions, candidates, truth)

    n_pred = sum(len(v) for v in predictions.values())
    n_true = sum(len(v) for v in truth.values())
    n_correct = sum(len(set(predictions.get(s, ())) & t) for s, t in truth.items())
    s1_country = pq.read_table(records_path(split), columns=["entity_id", "country"], filters=[("source", "=", "S1")]).to_pandas()
    country_of = dict(zip(s1_country["entity_id"], s1_country["country"]))
    by_country: Dict[str, Dict[str, Set[str]]] = {}
    for s, t in truth.items():
        by_country.setdefault(country_of.get(s, "?"), {})[s] = t
    singles = {s: t for s, t in truth.items() if not t}
    others = {s: t for s, t in truth.items() if t}
    result = {
        "macro_f05": base["macro_f05"],
        "precision_micro": n_correct / n_pred if n_pred else 1.0,
        "recall_micro": n_correct / n_true if n_true else 1.0,
        "f05_singletons": macro_fbeta(predictions, singles) if singles else None,
        "f05_non_singletons": macro_fbeta(predictions, others) if others else None,
        "f05_by_country": {c: macro_fbeta(predictions, t) for c, t in sorted(by_country.items())},
        "blocking_recall": base["blocking_recall"],
        "avg_candidates": base["avg_candidates"],
        "n_s1": len(truth),
        "n_singletons": len(singles),
        "n_predicted_pairs": n_pred,
    }
    print(f"evaluate[{split}]: macro F0.5 {result['macro_f05']:.4f} over {len(truth)} S1 "
          f"(micro precision {result['precision_micro']:.4f}, micro recall {result['recall_micro']:.4f})")
    print(f"evaluate[{split}]: F0.5 on singletons only {result['f05_singletons']} ({len(singles)} S1), "
          f"non-singletons only {result['f05_non_singletons']:.4f} ({len(others)} S1)")
    print(f"evaluate[{split}]: F0.5 per country: {{{', '.join(f'{c}: {v:.4f}' for c, v in result['f05_by_country'].items())}}}")
    print(f"evaluate[{split}]: blocking recall {result['blocking_recall']:.4f}, avg candidates per S1 {result['avg_candidates']:.1f}")
    print(SAMPLE_CAVEAT)
    save_metrics(f"evaluate_{split}", result)
    return result


def main(argv=None) -> None:
    """Entry point for ``python -m src.evaluate --split val``.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    import argparse

    from .perf import stage_timer

    parser = argparse.ArgumentParser(description="Evaluate a labelled split (macro F0.5 and breakdowns).")
    parser.add_argument("--split", default="val", choices=("val", "train"))
    split = parser.parse_args(argv).split
    with stage_timer("evaluate", split):
        evaluate_split(split)


if __name__ == "__main__":
    main()
