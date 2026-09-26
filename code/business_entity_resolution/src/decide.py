"""Stage 5 (decision v0): pick the probability threshold on out-of-fold train predictions, apply it to val/test.

Run as ``python -m src.decide --split <train|val|test>`` from ``code/business_entity_resolution/``.

* ``train``: sweep the threshold 0.05..0.95 (step 0.05) on the OUT-OF-FOLD train predictions and keep the one that
  maximises macro F0.5 (ties -> the higher threshold, i.e. the more precise one). Saved to ``threshold.json``.
* ``val`` / ``test``: keep the candidate pairs with ``prob >= threshold`` -> ``matches_<split>.parquet``
  (``s1_id, cand_id, prob``). An S1 with no candidates, or none above the threshold, gets an empty prediction.

The sweep uses a vectorised closed form of the per-S1 F0.5, which equals ``src.evaluate.fbeta_for_entity``
(unit-tested): with tp true positives, n_pred predictions and n_true true matches,
``F0.5 = 1.25 * tp / (n_pred + 0.25 * n_true)``; a singleton scores 1 iff n_pred == 0.
"""
from __future__ import annotations

import json
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .cli import parse_split
from .config import INTERIM_DIR, TRAIN_DIR, ensure_dirs
from .io_utils import GROUND_TRUTH_SUFFIX, load_ground_truth
from .perf import SAMPLE_CAVEAT, load_metrics, save_metrics, stage_timer
from .split import load_split_ids
from .train import preds_path

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
THRESHOLD_PATH = INTERIM_DIR / "threshold.json"
BETA2 = 0.25  # beta ** 2 for beta = 0.5
MATCH_SCHEMA = pa.schema([("s1_id", pa.string()), ("cand_id", pa.string()), ("prob", pa.float32())])


def matches_path(split: str):
    """Path of the kept-match parquet of a split.

    Args:
        split: ``"val"`` or ``"test"``.

    Returns:
        ``data/interim/matches_<split>.parquet``.
    """
    return INTERIM_DIR / f"matches_{split}.parquet"


def macro_f05_vectorised(codes: np.ndarray, keep: np.ndarray, label: np.ndarray, n_true: np.ndarray) -> float:
    """Macro F0.5 over S1s from pair arrays, without Python dictionaries.

    Args:
        codes: Integer S1 code (0..n_s1-1) of every scored pair.
        keep: Boolean, True where the pair is predicted a match.
        label: 0/1 ground-truth label of every scored pair.
        n_true: Number of TRUE matches of every S1 (including those blocking missed); its length is n_s1.

    Returns:
        Mean per-S1 F0.5 (S1s without any kept pair count as empty predictions).
    """
    n_s1 = len(n_true)
    n_pred = np.bincount(codes[keep], minlength=n_s1).astype(np.float64)
    tp = np.bincount(codes[keep & (label == 1)], minlength=n_s1).astype(np.float64)
    denom = n_pred + BETA2 * n_true
    f = np.divide((1 + BETA2) * tp, denom, out=np.zeros(n_s1), where=denom > 0)
    f = np.where(n_true == 0, (n_pred == 0).astype(np.float64), f)
    return float(f.mean())


def sweep_thresholds(probs: np.ndarray, codes: np.ndarray, label: np.ndarray, n_true: np.ndarray) -> Dict[float, float]:
    """Macro F0.5 for every threshold in ``THRESHOLDS``.

    Args:
        probs: Pair probabilities.
        codes: S1 code per pair.
        label: 0/1 label per pair.
        n_true: True match count per S1.

    Returns:
        ``{threshold: macro F0.5}``.
    """
    return {float(t): macro_f05_vectorised(codes, probs >= t, label, n_true) for t in THRESHOLDS}


def best_threshold(sweep: Dict[float, float]) -> Tuple[float, float]:
    """Threshold with the highest score (ties go to the larger, more precise threshold).

    Args:
        sweep: Output of ``sweep_thresholds``.

    Returns:
        ``(threshold, score)``.
    """
    t = max(sweep, key=lambda k: (round(sweep[k], 12), k))
    return t, sweep[t]


def tune_on_oof() -> Dict[str, object]:
    """Sweep the threshold on ``preds_train.parquet`` (out-of-fold) and save ``threshold.json``.

    Returns:
        ``{"threshold", "oof_macro_f05", "sweep", "n_s1"}``.
    """
    s1_ids = load_split_ids("train")
    truth = load_ground_truth(TRAIN_DIR / f"train_{GROUND_TRUTH_SUFFIX}")
    preds = pq.read_table(preds_path("train")).to_pandas()
    if load_metrics("train").get("sampled", False):  # the train stage sampled whole S1 groups: evaluate on those only
        s1_ids = sorted(set(preds["s1_id"]))
    index = pd.Index(s1_ids)
    codes = index.get_indexer(preds["s1_id"])
    ok = codes >= 0
    n_true = np.array([len(truth[s]) for s in s1_ids], dtype=np.float64)
    sweep = sweep_thresholds(preds["prob"].to_numpy()[ok], codes[ok], preds["label"].to_numpy()[ok], n_true)
    t, score = best_threshold(sweep)
    result = {"threshold": t, "oof_macro_f05": score, "sweep": {str(k): v for k, v in sweep.items()}, "n_s1": len(s1_ids)}
    ensure_dirs()
    with open(THRESHOLD_PATH, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return result


def load_threshold() -> float:
    """Read the saved threshold.

    Returns:
        The threshold from ``threshold.json``.

    Raises:
        FileNotFoundError: If ``decide --split train`` has not been run.
    """
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"{THRESHOLD_PATH} not found - run `python -m src.decide --split train` first")
    with open(THRESHOLD_PATH, encoding="utf-8") as fh:
        return float(json.load(fh)["threshold"])


def apply_threshold(split: str, threshold: float) -> Tuple[int, int]:
    """Keep the pairs with ``prob >= threshold`` and write ``matches_<split>.parquet`` (streamed by batch).

    Args:
        split: ``"val"`` or ``"test"``.
        threshold: Probability cut-off.

    Returns:
        ``(n_kept, n_scored)``.
    """
    kept = scored = 0
    ensure_dirs()
    with pq.ParquetWriter(matches_path(split), MATCH_SCHEMA, compression="zstd") as writer:
        for batch in pq.ParquetFile(preds_path(split)).iter_batches(batch_size=1_000_000):
            t = pa.Table.from_batches([batch])
            mask = t.column("prob").to_numpy() >= threshold
            scored += t.num_rows
            kept += int(mask.sum())
            writer.write_table(t.filter(pa.array(mask)).cast(MATCH_SCHEMA))
    return kept, scored


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point: tune the threshold (train) or apply it (val/test).

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    split = parse_split(__doc__.splitlines()[0], argv)
    with stage_timer("decide", split) as info:
        if split == "train":
            res = tune_on_oof()
            print("decide: threshold sweep on out-of-fold train predictions (macro F0.5):")
            for t, f in res["sweep"].items():
                print(f"   {float(t):.2f}  {f:.4f}{'  <- best' if float(t) == res['threshold'] else ''}")
            print(f"decide: threshold {res['threshold']:.2f}, OOF macro F0.5 {res['oof_macro_f05']:.4f} over {res['n_s1']} S1")
            print(SAMPLE_CAVEAT)
            save_metrics("decide_train", {"threshold": res["threshold"], "oof_macro_f05": res["oof_macro_f05"]})
        else:
            thr = load_threshold()
            kept, scored = apply_threshold(split, thr)
            info["pairs"] = scored
            print(f"decide: threshold {thr:.2f}: kept {kept}/{scored} scored pairs -> {matches_path(split)}")


if __name__ == "__main__":
    main()
