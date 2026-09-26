"""Blocking-quality report for train/val: recall, per-country recall, full-set capture and missed-pair analysis.

Streams the candidate parquet one row group at a time and keeps only counters plus the set of found true
pairs, so it stays usable at full scale (memory ~ number of true pairs, not number of candidates).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from .config import SEED, TRAIN_DIR
from .io_utils import GROUND_TRUTH_SUFFIX, load_ground_truth_subset
from .normalize import records_path
from .perf import save_metrics
from .split import load_split_ids

TARGET_RECALL = 0.97
N_EXAMPLES = 20


def stream_recall(cand_path, truth: Dict[str, Set[str]]) -> Tuple[Set[Tuple[str, str]], Counter, int, int]:
    """Scan candidates once and collect which true pairs were found.

    Args:
        cand_path: Candidate parquet.
        truth: ``{s1_id: {true ids}}`` restricted to the evaluated S1s.

    Returns:
        ``(found_pairs, found_per_s1, n_candidates, n_candidate_s1)``.
    """
    found: Set[Tuple[str, str]] = set()
    per_s1: Counter = Counter()
    n_cand = 0
    pf = pq.ParquetFile(cand_path)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["s1_id", "cand_id"])
        n_cand += t.num_rows
        for s, c in zip(t.column("s1_id").to_pylist(), t.column("cand_id").to_pylist()):
            if c in truth.get(s, ()):
                found.add((s, c))
                per_s1[s] += 1
    return found, per_s1, n_cand, len(per_s1)


def classify_misses(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Likely cause of each missed true pair from the raw records of both sides.

    Causes, checked in this order: empty candidate address; non-Latin script on either side's name (transliteration);
    names very different (token_set_ratio < 50: DBA / trade name / heavy typo); partly different (50-80);
    otherwise the names are similar and the pair was lost to top-K crowding or n-gram pruning.

    Args:
        a: S1 records of the missed pairs (``name_raw``, ``addr_raw``, ``name_core``).
        b: Candidate-side records, same alignment.

    Returns:
        ``(cause labels, token_set_ratio of the core names)``, both aligned with the pairs.
    """
    sim = process.cpdist(a["name_core"].tolist(), b["name_core"].tolist(), scorer=fuzz.token_set_ratio, workers=1)
    non_latin = np.array([not (x.isascii() and y.isascii()) for x, y in zip(a["name_raw"], b["name_raw"])])
    kind = np.where(b["addr_raw"].str.strip() == "", "candidate address empty",
            np.where(non_latin, "non-Latin script name (transliteration)",
            np.where(sim < 50, "names very different (DBA / trade name / heavy typo)",
            np.where(sim < 80, "names partly different (token_set_ratio 50-80)",
                     "names similar (>=80) but not retrieved (top-K crowding / pruning)"))))
    return kind, sim


def describe_misses(missed: List[Tuple[str, str]], rec_path) -> None:
    """Print a breakdown of missed true pairs by likely cause, then ``N_EXAMPLES`` examples grouped by cause.

    Args:
        missed: True pairs not present in the candidates.
        rec_path: Records parquet used to look up names/addresses.
    """
    rng = np.random.RandomState(SEED)
    ids = sorted({i for pair in missed for i in pair})
    rec = pq.read_table(rec_path, columns=["entity_id", "source", "name_raw", "addr_raw", "name_core"],
                        filters=[("entity_id", "in", ids)]).to_pandas().set_index("entity_id")
    a = rec.loc[[m[0] for m in missed]].reset_index()
    b = rec.loc[[m[1] for m in missed]].reset_index()
    kind, sim = classify_misses(a, b)
    counts = Counter(kind)
    print(f"missed true pairs ({len(missed)}) by likely cause:")
    for label, n in counts.most_common():
        print(f"   {label}: {n} ({n / len(missed):.1%})")
    print("   by candidate source:", dict(Counter(b["source"])))
    per_cause = {label: max(1, round(N_EXAMPLES * n / len(missed))) for label, n in counts.items()}
    print(f"example missed pairs (about {N_EXAMPLES}, grouped by cause; S1 -> candidate | token_set_ratio):")
    for label, _ in counts.most_common():
        pool = np.flatnonzero(kind == label)
        print(f"  [{label}]")
        for i in rng.choice(pool, size=min(per_cause[label], len(pool)), replace=False):
            print(f"   {a['name_raw'][i]!r} @ {a['addr_raw'][i]!r}\n      -> {b['entity_id'][i]} {b['name_raw'][i]!r} @ {b['addr_raw'][i]!r} | {sim[i]:.0f}")


def report_blocking(split: str, cand_path: Optional[str] = None) -> Dict[str, object]:
    """Print and save blocking recall (overall, per country), full-set capture and avg candidates per S1.

    Args:
        split: ``"train"`` or ``"val"``.
        cand_path: Candidate parquet; defaults to ``candidates_<split>.parquet``.

    Returns:
        The metrics dict (also saved to ``data/interim/metrics/block_<split>.json``).
    """
    from .block import candidates_path  # local import: block imports this module lazily

    cand_path = cand_path or candidates_path(split)
    s1_ids = load_split_ids(split)
    truth = load_ground_truth_subset(TRAIN_DIR / f"train_{GROUND_TRUTH_SUFFIX}", s1_ids)
    rec_path = records_path(split)
    s1_rec = pq.read_table(rec_path, columns=["entity_id", "country"], filters=[("source", "=", "S1")]).to_pandas()
    country_of = dict(zip(s1_rec["entity_id"], s1_rec["country"]))

    found, per_s1, n_cand, _ = stream_recall(cand_path, truth)
    total_by_c, found_by_c = defaultdict(int), defaultdict(int)
    for s, true in truth.items():
        total_by_c[country_of.get(s, "?")] += len(true)
    for s, _ in found:
        found_by_c[country_of.get(s, "?")] += 1
    n_true = sum(len(v) for v in truth.values())
    non_single = [s for s, v in truth.items() if v]
    full_capture = sum(1 for s in non_single if per_s1[s] == len(truth[s])) / max(len(non_single), 1)
    recall = len(found) / max(n_true, 1)
    metrics = {
        "blocking_recall": recall,
        "recall_by_country": {c: found_by_c[c] / total_by_c[c] for c in sorted(total_by_c) if total_by_c[c]},
        "full_set_capture": full_capture,
        "avg_candidates": n_cand / max(len(truth), 1),
        "n_pairs": n_cand,
        "n_s1": len(truth),
        "n_true_pairs": n_true,
    }
    print(f"blocking[{split}]: recall {recall:.4f} ({len(found)}/{n_true} true pairs), "
          f"S1s with the FULL true set captured {full_capture:.4f}, avg candidates per S1 {metrics['avg_candidates']:.1f}")
    print("blocking recall per country:", {c: round(v, 4) for c, v in metrics["recall_by_country"].items()})
    if recall < TARGET_RECALL:
        print(f"blocking recall {recall:.4f} < target {TARGET_RECALL}: analysing missed pairs")
        missed = [(s, c) for s, v in truth.items() for c in sorted(v) if (s, c) not in found]
        describe_misses(missed, rec_path)
    save_metrics(f"block_{split}", metrics)
    return metrics
