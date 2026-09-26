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
from .io_utils import GROUND_TRUTH_SUFFIX, load_ground_truth
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


def describe_misses(missed: List[Tuple[str, str]], rec_path) -> None:
    """Print a breakdown of missed true pairs and ``N_EXAMPLES`` random examples.

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
    sim = process.cpdist(a["name_core"].tolist(), b["name_core"].tolist(), scorer=fuzz.token_set_ratio, workers=1)
    kind = np.where(b["addr_raw"].str.strip() == "", "candidate address empty",
                    np.where(sim < 50, "names very different (score<50: DBA / transliteration / heavy typo)",
                             np.where(sim < 80, "names partly different (50-80)", "names similar (>=80) but not retrieved")))
    print("missed true pairs by kind / by source:")
    for label, n in Counter(kind).most_common():
        print(f"   {label}: {n} ({n / len(missed):.1%})")
    print("   by candidate source:", dict(Counter(b["source"])))
    print(f"{min(N_EXAMPLES, len(missed))} example missed pairs (S1 -> candidate | token_set_ratio):")
    for i in rng.choice(len(missed), size=min(N_EXAMPLES, len(missed)), replace=False):
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
    full_truth = load_ground_truth(TRAIN_DIR / f"train_{GROUND_TRUTH_SUFFIX}")
    truth = {s: full_truth[s] for s in s1_ids}
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
