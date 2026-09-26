"""Per-stage wall-time / peak-memory logging and small JSON metric files shared between stages.

Every stage wraps its work in ``stage_timer("<stage>", split)``; on exit it prints one line and
appends a record to ``data/interim/stage_metrics.jsonl``. Stages also save their headline numbers
(blocking recall, AUC, threshold, F0.5, ...) with ``save_metrics`` so ``src.summary`` (called by
``run_all.sh``) can print one table at the end.

Peak memory is the process high-water mark (``ru_maxrss``) of the stage's own process; ``worker_peak_mb`` is the
largest forked worker (``ER_N_JOBS``), whose memory is additional (mostly shared copy-on-write).
"""
from __future__ import annotations

import json
import resource
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from .config import INTERIM_DIR, ensure_dirs

METRICS_DIR_NAME = "metrics"
STAGE_LOG = "stage_metrics.jsonl"

# The sample's S2/S3 pools are true matches + random distractors, so every metric computed on it is
# optimistic. Printed next to each reported metric.
SAMPLE_CAVEAT = (
    "CAVEAT: on dataset_sample the train S2/S3 pools are true matches + RANDOM distractors, so blocking recall "
    "and F0.5 are optimistic; the test sample lacks most true matches (format check only)."
)


def peak_rss_mb() -> float:
    """Return the peak resident set size of this process in MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


@contextmanager
def stage_timer(stage: str, split: str = "-", extra: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
    """Time a stage, record its wall time and peak memory, and print a one-line report.

    Args:
        stage: Stage name (e.g. ``"block"``).
        split: Split the stage ran on (``"train"``, ``"val"``, ``"test"`` or ``"-"``).
        extra: Optional initial key/values to store in the log record.

    Yields:
        A dict the stage may fill with extra counts (rows, pairs, ...) to be logged.
    """
    info: Dict[str, Any] = dict(extra or {})
    start = time.perf_counter()
    print(f"[{stage}:{split}] start", flush=True)
    try:
        yield info
    finally:
        seconds = time.perf_counter() - start
        children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        children_mb = children / (1024 * 1024) if sys.platform == "darwin" else children / 1024
        record = {"stage": stage, "split": split, "seconds": round(seconds, 2), "peak_mb": round(peak_rss_mb(), 1),
                  "worker_peak_mb": round(children_mb, 1), **info}
        print(f"[{stage}:{split}] done in {seconds:.1f}s, peak RSS {record['peak_mb']:.0f} MB", flush=True)
        ensure_dirs()
        with open(INTERIM_DIR / STAGE_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


def save_metrics(name: str, values: Dict[str, Any]) -> None:
    """Write a stage's headline metrics to ``data/interim/metrics/<name>.json`` (overwriting).

    Args:
        name: File stem, e.g. ``"block_val"`` or ``"train"``.
        values: JSON-serialisable metric dict.
    """
    ensure_dirs()
    out_dir = INTERIM_DIR / METRICS_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.json", "w", encoding="utf-8") as fh:
        json.dump(values, fh, indent=2, default=float)


def load_metrics(name: str) -> Dict[str, Any]:
    """Load ``data/interim/metrics/<name>.json``; returns ``{}`` when the file does not exist.

    Args:
        name: File stem used with ``save_metrics``.

    Returns:
        The stored dict, or an empty dict.
    """
    path = INTERIM_DIR / METRICS_DIR_NAME / f"{name}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
