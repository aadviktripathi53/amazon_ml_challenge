"""Profile every file in DATA_DIR/train and DATA_DIR/test (streaming, chunked) -> reports/data_profile.md.

Per file: size on disk, row count, columns; for source files also country counts and the
share of empty business_name / business_address. For ``train_ground_truth.tsv``: singleton
rate, matches-per-S1 histogram, S2 vs S3 share of matches, S2/S3 IDs used by more than one
ground-truth row, and whether every ground-truth ID exists in the source files.

Memory: files are read in chunks of ``streaming.CHUNKSIZE`` rows; IDs are kept as 8-byte hashes.

Run from ``code/business_entity_resolution/``::

    python -m src.profile_data [--out ../../reports/data_profile.md]
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import streaming
from .config import DATA_DIR, REPORTS_DIR
from .io_utils import ID_COL, MATCHED_COL, S2_PREFIX, S3_PREFIX, TRUTH_ID_COL
from .streaming import ScanResult, hash_ids, human_size, isin_sorted, iter_chunks, read_header, scan_source

HIST_CAP = 10  # matches-per-S1 histogram is bucketed as 0..HIST_CAP-1 and "HIST_CAP+"
EXAMPLES = 5


def list_files(data_dir: Path) -> List[Path]:
    """List every regular, non-hidden file in ``data_dir/train`` and ``data_dir/test``.

    Args:
        data_dir: Dataset root.

    Returns:
        Sorted list of paths (train first, then test).
    """
    files: List[Path] = []
    for sub in ("train", "test"):
        d = data_dir / sub
        if d.is_dir():
            files += sorted(p for p in d.iterdir() if p.is_file() and not p.name.startswith("."))
    return files


def classify(path: Path) -> str:
    """Classify a file by its header as ``source``, ``truth`` or ``other``.

    Args:
        path: TSV path.

    Returns:
        ``"source"`` if it has ``entity_id``, ``"truth"`` if it has ``source1_entity_id``, else ``"other"``.
    """
    cols = read_header(path)
    if ID_COL in cols:
        return "source"
    if TRUTH_ID_COL in cols:
        return "truth"
    return "other"


def count_rows(path: Path) -> int:
    """Count data rows of an arbitrary TSV by streaming its first column.

    Args:
        path: TSV path.

    Returns:
        Number of data rows.
    """
    return sum(len(c) for c in iter_chunks(path, usecols=[read_header(path)[0]]))


def profile_ground_truth(path: Path, source_ids: Dict[str, np.ndarray]) -> dict:
    """Stream the ground-truth file and compute the matching statistics.

    Args:
        path: ``train_ground_truth.tsv`` path.
        source_ids: ``{"S1"|"S2"|"S3": uint64 ID-hash array}`` from the train source files; a
            missing key means that existence check is skipped.

    Returns:
        Dict of statistics (see keys used in ``render_markdown``).
    """
    ref = {k: np.sort(v) for k, v in source_ids.items()}
    hist: Counter = Counter()
    rows = n_s2 = n_s3 = n_other = both = 0
    s1_hashes: List[np.ndarray] = []
    match_hashes: List[np.ndarray] = []
    missing = {"S1": 0, "S2": 0, "S3": 0}
    missing_examples: Dict[str, List[str]] = {"S1": [], "S2": [], "S3": []}

    def check(key: str, ids: pd.Series) -> None:
        """Count (and sample) IDs of one source that are absent from the source file."""
        if key not in ref or ids.empty:
            return
        absent = ~isin_sorted(hash_ids(ids), ref[key])
        missing[key] += int(absent.sum())
        if len(missing_examples[key]) < EXAMPLES:
            missing_examples[key] += ids[absent].head(EXAMPLES - len(missing_examples[key])).tolist()

    for chunk in iter_chunks(path, usecols=[TRUTH_ID_COL, MATCHED_COL]):
        rows += len(chunk)
        matched = chunk[MATCHED_COL].str.strip()
        parts = matched.str.split(",")
        n = np.where(matched == "", 0, parts.str.len())
        hist.update(pd.Series(n).value_counts().to_dict())
        c2, c3 = matched.str.count(S2_PREFIX), matched.str.count(S3_PREFIX)
        both += int(((c2 > 0) & (c3 > 0)).sum())
        ex = parts.explode().str.strip()
        ex = ex[ex != ""]
        is2, is3 = ex.str.startswith(S2_PREFIX), ex.str.startswith(S3_PREFIX)
        n_s2, n_s3, n_other = n_s2 + int(is2.sum()), n_s3 + int(is3.sum()), n_other + int((~is2 & ~is3).sum())
        s1_hashes.append(hash_ids(chunk[TRUTH_ID_COL]))
        match_hashes.append(hash_ids(ex[is2 | is3]))
        check("S1", chunk[TRUTH_ID_COL])
        check("S2", ex[is2])
        check("S3", ex[is3])

    all_s1 = np.concatenate(s1_hashes) if s1_hashes else np.empty(0, dtype=np.uint64)
    all_m = np.concatenate(match_hashes) if match_hashes else np.empty(0, dtype=np.uint64)
    _, mult = np.unique(all_m, return_counts=True)
    s1_without_gt = None
    if "S1" in ref:
        s1_without_gt = int((~isin_sorted(ref["S1"], np.sort(all_s1))).sum())
    binned: Counter = Counter()
    for k, v in hist.items():
        binned[k if k < HIST_CAP else HIST_CAP] += v
    return {
        "rows": rows, "singletons": int(hist.get(0, 0)), "hist": binned, "n_s2": n_s2, "n_s3": n_s3, "n_other": n_other,
        "s1_with_both": both, "dup_gt_s1_rows": int(len(all_s1) - len(np.unique(all_s1))),
        "shared_match_ids": int((mult > 1).sum()), "max_multiplicity": int(mult.max()) if len(mult) else 0,
        "missing": missing, "missing_examples": missing_examples, "checked": sorted(ref), "s1_without_gt": s1_without_gt,
    }


def profile(data_dir: Path) -> dict:
    """Profile all files under ``data_dir/{train,test}`` in a streaming fashion.

    Args:
        data_dir: Dataset root.

    Returns:
        Dict with ``files`` (per-file info), ``scans`` (ScanResult per source file) and ``truth`` (stats or None).
    """
    files, scans, other_rows, kinds = list_files(data_dir), {}, {}, {}
    train_ids: Dict[str, np.ndarray] = {}
    truth_path: Optional[Path] = None
    for path in files:
        kinds[path] = classify(path)
        if kinds[path] == "source":
            m = re.search(r"source([123])", path.name)
            is_train = path.parent.name == "train"
            prefix = f"S{m.group(1)}-" if m else None
            scans[path] = scan_source(path, expect_prefix=prefix, keep_ids=True)
            if is_train and m:
                train_ids[f"S{m.group(1)}"] = scans[path].id_hashes
        elif kinds[path] == "truth":
            truth_path = truth_path or path
        else:
            other_rows[path] = count_rows(path)
    truth = profile_ground_truth(truth_path, train_ids) if truth_path else None
    return {"data_dir": data_dir, "files": files, "kinds": kinds, "scans": scans, "other_rows": other_rows,
            "truth": truth, "truth_path": truth_path}


def _pct(part: int, whole: int) -> str:
    """Format ``part/whole`` as a percentage string (``"n/a"`` when ``whole`` is 0)."""
    return f"{100 * part / whole:.2f}%" if whole else "n/a"


def render_markdown(result: dict) -> str:
    """Render the profile dict as a Markdown report.

    Args:
        result: Output of ``profile``.

    Returns:
        Markdown text.
    """
    L: List[str] = ["# Data profile", "", f"Data dir: `{result['data_dir']}`  ",
                    f"Streaming chunk size: {streaming.CHUNKSIZE:,} rows. IDs compared via 64-bit hashes.", "",
                    "## Files", "", "| file | size | rows | columns |", "|---|---|---|---|"]
    root = result["data_dir"]
    for p in result["files"]:
        s = result["scans"].get(p)
        rows = s.rows if s else (result["truth"]["rows"] if p == result["truth_path"] else result["other_rows"].get(p, "?"))
        cols = s.columns if s else read_header(p)
        L.append(f"| {p.relative_to(root)} | {human_size(p.stat().st_size)} | {rows:,} | {', '.join(cols)} |")
    L += ["", "## Source files", ""]
    for p, s in result["scans"].items():
        L += [f"### {p.relative_to(root)}", "",
              f"- rows: {s.rows:,}; duplicate entity_ids: {s.duplicate_ids:,}; ids with unexpected prefix: {s.bad_prefix:,}",
              f"- empty business_name: {s.empty_name:,} ({_pct(s.empty_name, s.rows)}); "
              f"empty business_address: {s.empty_address:,} ({_pct(s.empty_address, s.rows)})", "",
              "| country | rows | share |", "|---|---|---|"]
        L += [f"| {c} | {n:,} | {_pct(n, s.rows)} |" for c, n in s.country_counts.most_common()]
        L.append("")
    t = result["truth"]
    if t:
        rows = t["rows"]
        L += [f"## Ground truth (`{result['truth_path'].relative_to(root)}`)", "",
              f"- S1 rows: {rows:,}; duplicate source1_entity_id rows: {t['dup_gt_s1_rows']:,}",
              f"- singleton rate (no matches): {t['singletons']:,} ({_pct(t['singletons'], rows)})",
              f"- S1 with matches from both S2 and S3: {t['s1_with_both']:,} ({_pct(t['s1_with_both'], rows)})",
              f"- total matches: {t['n_s2'] + t['n_s3']:,} - from S2: {t['n_s2']:,} "
              f"({_pct(t['n_s2'], t['n_s2'] + t['n_s3'])}), from S3: {t['n_s3']:,} ({_pct(t['n_s3'], t['n_s2'] + t['n_s3'])}); "
              f"IDs with another prefix: {t['n_other']:,}",
              f"- S2/S3 IDs appearing in more than one ground-truth row: **{t['shared_match_ids']:,}** "
              f"(max rows per ID: {t['max_multiplicity']})", "",
              "Matches per S1 entity:", "", "| matches | S1 entities | share |", "|---|---|---|"]
        for k in sorted(t["hist"]):
            label = f"{HIST_CAP}+" if k == HIST_CAP else str(k)
            L.append(f"| {label} | {t['hist'][k]:,} | {_pct(t['hist'][k], rows)} |")
        L += ["", "Existence of ground-truth IDs in the train source files:", ""]
        for key in ("S1", "S2", "S3"):
            if key in t["checked"]:
                ex = f" e.g. {t['missing_examples'][key]}" if t["missing"][key] else ""
                L.append(f"- {key}: {t['missing'][key]:,} missing{ex}")
            else:
                L.append(f"- {key}: not checked (train source file not found)")
        if t["s1_without_gt"] is not None:
            L.append(f"- train S1 records without a ground-truth row: {t['s1_without_gt']:,}")
        verdict = all(t["missing"][k] == 0 for k in t["checked"]) and len(t["checked"]) == 3
        L += ["", f"**Every ground-truth ID exists in the source files: {'YES' if verdict else 'NO / not fully checked'}**", ""]
    else:
        L += ["## Ground truth", "", "No file with a `source1_entity_id` column found.", ""]
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point: profile DATA_DIR, write the report and print it.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "data_profile.md")
    args = ap.parse_args(argv)
    md = render_markdown(profile(DATA_DIR))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
