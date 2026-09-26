"""Print the end-of-run summary table (called by ``run_all.sh``): quality metrics, then time and peak memory per stage.

Reads ``data/interim/metrics/*.json`` and ``data/interim/stage_metrics.jsonl`` written by the stages.
"""
from __future__ import annotations

import json
from typing import Dict, List

from .config import DATA_DIR, INTERIM_DIR
from .perf import SAMPLE_CAVEAT, STAGE_LOG, load_metrics

STAGE_ORDER = ["split", "normalize", "block", "features", "train", "predict", "decide", "evaluate", "write_submission"]


def _fmt(value, spec: str = ".4f") -> str:
    """Format a number, or ``-`` when it is missing.

    Args:
        value: Number or None.
        spec: Format spec.

    Returns:
        Formatted string.
    """
    return "-" if value is None else format(value, spec)


def stage_rows() -> List[Dict]:
    """Load the per-stage time / memory records of the current run (last record wins per stage+split).

    Returns:
        Records sorted by pipeline order.
    """
    path = INTERIM_DIR / STAGE_LOG
    latest: Dict[tuple, Dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            latest[(rec["stage"], rec["split"])] = rec
    return sorted(latest.values(), key=lambda r: (STAGE_ORDER.index(r["stage"]) if r["stage"] in STAGE_ORDER else 99, r["split"]))


def main() -> None:
    """Print quality and resource summary tables."""
    train, ev, dec = load_metrics("train"), load_metrics("evaluate_val"), load_metrics("decide_train")
    print("=" * 96)
    print(f"RUN SUMMARY   (ER_DATA_DIR = {DATA_DIR})")
    print(SAMPLE_CAVEAT)
    print("-" * 96)
    print(f"{'blocking':<10}{'recall':>9}{'full-set':>10}{'avg cands':>11}{'pairs':>12}    recall per country")
    for split in ("train", "val", "test"):
        b = load_metrics(f"block_{split}")
        if not b:
            continue
        per_c = ", ".join(f"{c}={v:.3f}" for c, v in b.get("recall_by_country", {}).items())
        print(f"{split:<10}{_fmt(b.get('blocking_recall')):>9}{_fmt(b.get('full_set_capture')):>10}"
              f"{_fmt(b.get('avg_candidates'), '.1f'):>11}{b.get('n_pairs', 0):>12}    {per_c}")
    print("-" * 96)
    print(f"model      OOF AUC {_fmt(train.get('auc'))}   AP {_fmt(train.get('average_precision'))}   "
          f"threshold {_fmt(dec.get('threshold'), '.2f')} (OOF macro F0.5 {_fmt(dec.get('oof_macro_f05'))})")
    print(f"validation macro F0.5 {_fmt(ev.get('macro_f05'))}   precision {_fmt(ev.get('precision_micro'))}   "
          f"recall {_fmt(ev.get('recall_micro'))}   singletons {_fmt(ev.get('f05_singletons'))}   "
          f"non-singletons {_fmt(ev.get('f05_non_singletons'))}")
    if ev.get("f05_by_country"):
        print("           F0.5 per country: " + ", ".join(f"{c}={v:.4f}" for c, v in ev["f05_by_country"].items()))
    print("-" * 96)
    print(f"{'stage':<18}{'split':<8}{'seconds':>10}{'peak MB':>10}{'pairs/records':>16}")
    for r in stage_rows():
        n = r.get("pairs") or r.get("records") or r.get("s1") or ""
        print(f"{r['stage']:<18}{r['split']:<8}{r['seconds']:>10.1f}{r['peak_mb']:>10.0f}{n:>16}")
    print("=" * 96)


if __name__ == "__main__":
    main()
