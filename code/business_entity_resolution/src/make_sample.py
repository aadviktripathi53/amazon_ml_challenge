"""Build a small, deterministic, self-consistent sample of the FULL dataset (streaming, memory-safe).

Train:
  * pick ``--n-s1`` Source 1 entities, stratified (proportional) by (country, singleton flag);
  * keep ALL ground-truth S2/S3 matches of those S1s;
  * add non-matching S2/S3 distractors: ``--distractor-ratio`` x (matched count of that source),
    chosen as the records with the smallest hash of ``seed:entity_id`` (a bounded "bottom-k"
    kept during ONE chunked pass, so the choice is independent of chunk size and file order);
  * ground truth filtered to the sampled S1s.
Test: a deterministic hash-based ``--test-frac`` of each test source file.
Output uses the same file names/format under ``<out>/train`` and ``<out>/test`` (so it can be
used as ``ER_DATA_DIR``). Ends with built-in PASS/FAIL verification.

Run from ``code/business_entity_resolution/``::

    python -m src.make_sample [--out ../../dataset_sample] [--n-s1 2000] [--distractor-ratio 3] [--test-frac 0.05]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .config import DATA_DIR, SAMPLE_DIR, SEED
from .io_utils import COUNTRY_COL, GROUND_TRUTH_SUFFIX, ID_COL, MATCHED_COL, SOURCE_SUFFIXES, TRUTH_ID_COL, parse_id_list
from .streaming import hash_ids, isin_sorted, iter_chunks, read_header, scan_source

PREFIXES = ("S1-", "S2-", "S3-")


def source_path(base: Path, split: str, n: int) -> Path:
    """Path of ``<base>/<split>/<split>_source<n>.tsv``.

    Args:
        base: Dataset root.
        split: ``"train"`` or ``"test"``.
        n: Source number (1-3).

    Returns:
        File path.
    """
    return base / split / f"{split}_{SOURCE_SUFFIXES[n - 1]}"


def truth_path(base: Path) -> Path:
    """Path of ``<base>/train/train_ground_truth.tsv``.

    Args:
        base: Dataset root.

    Returns:
        File path.
    """
    return base / "train" / f"train_{GROUND_TRUTH_SUFFIX}"


def allocate(sizes: Dict[tuple, int], n: int) -> Dict[tuple, int]:
    """Proportionally allocate ``n`` picks over strata (largest-remainder, capped at stratum size).

    Args:
        sizes: Stratum key -> number of available entities.
        n: Total number of entities wanted.

    Returns:
        Stratum key -> number to pick (sums to ``min(n, total)``).
    """
    total = sum(sizes.values())
    if n >= total:
        return dict(sizes)
    quota = {k: n * v / total for k, v in sizes.items()}
    take = {k: int(math.floor(q)) for k, q in quota.items()}
    for k in sorted(sizes, key=lambda k: (-(quota[k] - take[k]), str(k))):
        if sum(take.values()) >= n:
            break
        if take[k] < sizes[k]:
            take[k] += 1
    return take


def select_s1(base: Path, n_s1: int, seed: int) -> Tuple[Set[str], Counter, int]:
    """Choose the sampled S1 entities (stratified by country x singleton) in chunked passes.

    Args:
        base: Source dataset root.
        n_s1: Number of S1 entities wanted.
        seed: Hash seed for within-stratum selection.

    Returns:
        ``(selected_ids, original_country_counts, s1_without_ground_truth)``.
    """
    singleton: Dict[str, bool] = {}
    for c in iter_chunks(truth_path(base), usecols=[TRUTH_ID_COL, MATCHED_COL]):
        singleton.update(zip(c[TRUTH_ID_COL].str.strip(), (c[MATCHED_COL].str.strip() == "")))
    strata: Dict[tuple, List[Tuple[int, str]]] = defaultdict(list)
    countries: Counter = Counter()
    skipped = 0
    for c in iter_chunks(source_path(base, "train", 1), usecols=[ID_COL, COUNTRY_COL]):
        countries.update(c[COUNTRY_COL].value_counts().to_dict())
        h = hash_ids(c[ID_COL], seed, "s1")
        for eid, country, hv in zip(c[ID_COL].str.strip(), c[COUNTRY_COL], h):
            if eid not in singleton:
                skipped += 1
                continue
            strata[(country, singleton[eid])].append((int(hv), eid))
    take = allocate({k: len(v) for k, v in strata.items()}, n_s1)
    chosen: Set[str] = set()
    for k, items in strata.items():
        chosen.update(eid for _, eid in sorted(items)[: take[k]])
    return chosen, countries, skipped


def filter_rows(path: Path, ids: Set[str], id_col: str) -> pd.DataFrame:
    """Return all rows of a TSV whose ``id_col`` is in ``ids`` (chunked scan, all columns).

    Args:
        path: TSV path.
        ids: Wanted IDs.
        id_col: Column to match on.

    Returns:
        DataFrame of the matching rows in file order.
    """
    parts = [c[c[id_col].str.strip().isin(ids)] for c in iter_chunks(path)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=read_header(path))


def pass_source(path: Path, matched: Set[str], k: int, seed: int, salt: str) -> Tuple[pd.DataFrame, int]:
    """One chunked pass over an S2/S3 file: keep matched rows plus a bottom-k hash distractor set.

    Args:
        path: Source TSV path.
        matched: IDs that must be kept (ground-truth matches of the sampled S1s).
        k: Number of distractors (non-matched rows with the smallest ``seed:id`` hash) to keep.
        seed: Hash seed.
        salt: Hash salt (per source).

    Returns:
        ``(rows, total_rows_in_file)``; rows are shuffled by hash so matches are not grouped.
    """
    keep: List[pd.DataFrame] = []
    best: Optional[pd.DataFrame] = None
    total = 0
    for c in iter_chunks(path):
        total += len(c)
        is_m = c[ID_COL].str.strip().isin(matched)
        h = hash_ids(c[ID_COL], seed, salt)
        keep.append(c[is_m].assign(_h=h[is_m.to_numpy()]))
        if k > 0:
            cand = c[~is_m].assign(_h=h[(~is_m).to_numpy()])
            best = cand if best is None else pd.concat([best, cand])
            best = best.sort_values(["_h", ID_COL]).head(k)
    if not keep:
        return pd.DataFrame(columns=read_header(path)), 0
    if best is not None:
        keep.append(best)
    df = pd.concat(keep, ignore_index=True)
    df = df.sort_values(["_h", ID_COL]).drop(columns="_h").reset_index(drop=True)
    return df, total


def sample_test_file(src: Path, dst: Path, test_frac: float, seed: int) -> Tuple[int, int, Counter, Counter]:
    """Write the deterministic hash-based ``test_frac`` subset of one test file, streaming.

    Args:
        src: Original test TSV.
        dst: Output TSV.
        test_frac: Fraction of rows to keep (a row is kept iff its hash < frac * 2^64).
        seed: Hash seed.

    Returns:
        ``(rows_before, rows_after, country_counts_before, country_counts_after)``.
    """
    thr = None if test_frac >= 1 else np.uint64(int(test_frac * 2**64))
    before = after = 0
    cb: Counter = Counter()
    ca: Counter = Counter()
    dst.parent.mkdir(parents=True, exist_ok=True)
    cols = read_header(src)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        first = True
        for c in iter_chunks(src):
            before += len(c)
            cb.update(c[COUNTRY_COL].value_counts().to_dict())
            sub = c if thr is None else c[hash_ids(c[ID_COL], seed, "test") < thr]
            after += len(sub)
            ca.update(sub[COUNTRY_COL].value_counts().to_dict())
            sub.to_csv(f, sep="\t", index=False, header=first, lineterminator="\n", quoting=csv.QUOTE_NONE)
            first = False
        if first:
            pd.DataFrame(columns=cols).to_csv(f, sep="\t", index=False, lineterminator="\n", quoting=csv.QUOTE_NONE)
    return before, after, cb, ca


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame as TSV (same conventions as the competition files).

    Args:
        df: Rows to write.
        path: Output path (parent created).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n", quoting=csv.QUOTE_NONE)


def _share_dev(orig: Counter, samp: Counter, statistical: bool) -> Tuple[float, float]:
    """Max absolute country-share deviation between original and sample, and its tolerance.

    Args:
        orig: Original country counts.
        samp: Sampled country counts.
        statistical: True for random (hash) samples: tolerance is 4 binomial standard errors;
            False for stratified samples: a fixed 5 points plus 1/n.

    Returns:
        ``(max_deviation, tolerance)``.
    """
    n_o, n_s = sum(orig.values()), sum(samp.values())
    if n_o == 0 or n_s == 0:
        return 0.0, 1.0
    worst = (-math.inf, 0.0, 0.0)  # (margin, deviation, tolerance) of the least-passing country
    for c in set(orig) | set(samp):
        p, q = orig[c] / n_o, samp[c] / n_s
        tol = 4 * math.sqrt(p * (1 - p) / n_s) + 1 / n_s if statistical else 0.05 + 1 / n_s
        worst = max(worst, (abs(p - q) - tol, abs(p - q), tol))
    return worst[1], worst[2]


def verify(out: Path, s1_orig_countries: Counter, test_orig: Dict[int, Counter]) -> List[Tuple[str, bool, str]]:
    """Verify a written sample; every check is streaming/hash-based.

    Checks: (1) every ground-truth ID exists in the sampled sources, (2) ID prefixes match their
    files, (3) no duplicate entity_ids in any file, (4) country distribution roughly preserved.

    Args:
        out: Sample root.
        s1_orig_countries: Country counts of the ORIGINAL train S1 file.
        test_orig: Source number -> country counts of the ORIGINAL test file.

    Returns:
        List of ``(check name, passed, detail)``.
    """
    scans = {(sp, n): scan_source(source_path(out, sp, n), PREFIXES[n - 1], keep_ids=True) for sp in ("train", "test") for n in (1, 2, 3)}
    res: List[Tuple[str, bool, str]] = []

    ref = {n: np.sort(scans[("train", n)].id_hashes) for n in (1, 2, 3)}
    missing = 0
    gt_rows = 0
    for c in iter_chunks(truth_path(out)):
        gt_rows += len(c)
        missing += int((~isin_sorted(hash_ids(c[TRUTH_ID_COL]), ref[1])).sum())
        for n in (2, 3):
            ids = c[MATCHED_COL].str.split(",").explode().str.strip()
            ids = ids[ids.str.startswith(PREFIXES[n - 1])]
            missing += int((~isin_sorted(hash_ids(ids), ref[n])).sum())
    res.append(("every ground-truth id exists in the sampled sources", missing == 0, f"{missing} missing over {gt_rows} gt rows"))

    bad = {f"{sp}_source{n}": s.bad_prefix for (sp, n), s in scans.items() if s.bad_prefix}
    res.append(("id prefixes match their files", not bad, f"violations: {bad or 0}"))

    dups = {f"{sp}_source{n}": s.duplicate_ids for (sp, n), s in scans.items() if s.duplicate_ids}
    res.append(("no duplicate entity_ids", not dups, f"duplicates: {dups or 0}"))

    dev, tol = _share_dev(s1_orig_countries, scans[("train", 1)].country_counts, statistical=False)
    ok, detail = dev <= tol, [f"train S1 max share diff {dev:.3f} (tol {tol:.3f})"]
    for n in (1, 2, 3):
        dev, tol = _share_dev(test_orig[n], scans[("test", n)].country_counts, statistical=True)
        ok &= dev <= tol
        detail.append(f"test S{n} {dev:.3f} (tol {tol:.3f})")
    res.append(("country distribution roughly preserved", bool(ok), "; ".join(detail)))
    return res


def make_sample(data_dir: Path, out: Path, n_s1: int = 2000, distractor_ratio: float = 3, test_frac: float = 0.05,
                seed: int = SEED) -> dict:
    """Create the sample dataset and verify it.

    Args:
        data_dir: Full dataset root (never modified).
        out: Output root; must not be ``data_dir`` or inside it.
        n_s1: Number of train S1 entities.
        distractor_ratio: Distractors per matched record, per source.
        test_frac: Fraction of each test file to keep.
        seed: Hash seed.

    Returns:
        Dict with ``counts`` (file -> (before, after)) and ``checks`` (verification results).

    Raises:
        ValueError: If ``out`` would overwrite or sit inside ``data_dir``.
    """
    data_dir, out = data_dir.resolve(), out.resolve()
    if out == data_dir or data_dir in out.parents:
        raise ValueError(f"refusing to write the sample into the source dataset: {out}")
    counts: Dict[str, Tuple[int, int]] = {}

    selected, s1_countries, skipped = select_s1(data_dir, n_s1, seed)
    s1_rows = filter_rows(source_path(data_dir, "train", 1), selected, ID_COL)
    gt_rows = filter_rows(truth_path(data_dir), selected, TRUTH_ID_COL)
    write_tsv(s1_rows, source_path(out, "train", 1))
    write_tsv(gt_rows, truth_path(out))
    counts["train_source1.tsv"] = (sum(s1_countries.values()), len(s1_rows))
    n_gt = sum(len(c) for c in iter_chunks(truth_path(data_dir), usecols=[TRUTH_ID_COL]))
    counts["train_ground_truth.tsv"] = (n_gt, len(gt_rows))

    matched: Set[str] = set()
    for cell in gt_rows[MATCHED_COL]:
        matched |= parse_id_list(cell)
    for n in (2, 3):
        m_n = {i for i in matched if i.startswith(PREFIXES[n - 1])}
        rows, total = pass_source(source_path(data_dir, "train", n), m_n, int(round(distractor_ratio * len(m_n))), seed, f"s{n}")
        write_tsv(rows, source_path(out, "train", n))
        counts[f"train_source{n}.tsv"] = (total, len(rows))

    test_orig: Dict[int, Counter] = {}
    for n in (1, 2, 3):
        b, a, cb, _ = sample_test_file(source_path(data_dir, "test", n), source_path(out, "test", n), test_frac, seed)
        counts[f"test_source{n}.tsv"] = (b, a)
        test_orig[n] = cb
    return {"counts": counts, "skipped_s1_without_gt": skipped, "checks": verify(out, s1_countries, test_orig)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: sample DATA_DIR into ``--out``, print row counts and PASS/FAIL checks.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 if every check passes, else 1.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=SAMPLE_DIR)
    ap.add_argument("--n-s1", type=int, default=2000)
    ap.add_argument("--distractor-ratio", type=float, default=3)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    result = make_sample(DATA_DIR, args.out, args.n_s1, args.distractor_ratio, args.test_frac, args.seed)
    print(f"Sampled {DATA_DIR} -> {args.out.resolve()}\n")
    print(f"{'file':28} {'before':>12} {'after':>10}")
    for name, (b, a) in result["counts"].items():
        print(f"{name:28} {b:>12,} {a:>10,}")
    if result["skipped_s1_without_gt"]:
        print(f"\nnote: {result['skipped_s1_without_gt']} train S1 records had no ground-truth row and were not sampled")
    print("\nVerification:")
    for name, ok, detail in result["checks"]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} - {detail}")
    passed = all(ok for _, ok, _ in result["checks"])
    print(f"\nOVERALL: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
