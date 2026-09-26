"""Stage 5 (decision): calibrate, choose each S1's answer by expected F0.5, resolve duplicate claims (exclusivity).

Run as ``python -m src.decide --split <train|val|test>`` from ``code/business_entity_resolution/``.

* ``train``: sweep the threshold 0.05..0.95 (step 0.05) on the OUT-OF-FOLD train predictions and keep the one that
  maximises macro F0.5 (ties -> the higher threshold, i.e. the more precise one). Saved to ``threshold.json``.
  Also fits isotonic calibration on the same out-of-fold predictions -> ``calibration.json``.
* ``val`` / ``test``: calibrate, then per S1 (``ER_DECISION``):
  ``expf`` (default) = the top-k with the highest expected F0.5 (``src.decision.choose_expected_f05``), or
  ``threshold`` = the flat cut-off from ``threshold.json`` on the raw probability (the v0 rule);
  then, with ``ER_EXCLUSIVE=1`` (default), a candidate chosen by several S1s is kept only for the most probable one.
  For ``val`` the train S1s' out-of-fold decisions take part in the exclusivity contest (more rival S1s = closer to
  test, where every S1 is scored). -> ``matches_<split>.parquet`` (``s1_id, cand_id, prob, prob_cal``).

The sweep uses a vectorised closed form of the per-S1 F0.5, which equals ``src.evaluate.fbeta_for_entity``
(unit-tested): with tp true positives, n_pred predictions and n_true true matches,
``F0.5 = 1.25 * tp / (n_pred + 0.25 * n_true)``; a singleton scores 1 iff n_pred == 0.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .cli import parse_split
from .config import INTERIM_DIR, TRAIN_DIR, ensure_dirs
from .io_utils import GROUND_TRUTH_SUFFIX, load_ground_truth_subset
from .perf import SAMPLE_CAVEAT, load_metrics, save_metrics, stage_timer
from .split import load_split_ids
from .decision import apply_calibration, choose_expected_f05, exclusive_mask, fit_isotonic
from .train import preds_path

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
THRESHOLD_PATH = INTERIM_DIR / "threshold.json"
BETA2 = 0.25  # beta ** 2 for beta = 0.5
MATCH_SCHEMA = pa.schema([("s1_id", pa.string()), ("cand_id", pa.string()), ("prob", pa.float32()), ("prob_cal", pa.float32())])
CALIB_PATH = INTERIM_DIR / "calibration.json"
# "hybrid" (default): the flat threshold decides empty vs non-empty, expected F0.5 picks how many to predict;
# "expf": expected F0.5 decides everything; "threshold": the flat v0 rule.
DECISION = os.environ.get("ER_DECISION", "hybrid")
EXCLUSIVE = os.environ.get("ER_EXCLUSIVE", "1") == "1"
if DECISION not in ("hybrid", "expf", "threshold"):
    raise ValueError(f"ER_DECISION must be 'hybrid', 'expf' or 'threshold', got {DECISION!r}")
CHUNK_PAIRS = 2_000_000  # prediction rows decided at once (S1s are never split across chunks)
MC_DRAWS = int(os.environ.get("ER_MC_DRAWS", "2000"))
MC_MIN_P = float(os.environ.get("ER_MC_MIN_P", "0.001"))


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
    preds = pq.read_table(preds_path("train"), columns=["s1_id", "prob", "label"]).to_pandas()
    if load_metrics("train").get("sampled", False):  # the train stage sampled whole S1 groups: evaluate on those only
        s1_ids = sorted(set(preds["s1_id"]))
    truth = load_ground_truth_subset(TRAIN_DIR / f"train_{GROUND_TRUTH_SUFFIX}", s1_ids)
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


def fit_calibration() -> Dict[str, object]:
    """Fit isotonic calibration on the out-of-fold train predictions and save ``calibration.json``.

    Returns:
        The breakpoints plus the Brier score before/after (in-sample on the OOF predictions).
    """
    preds = pq.read_table(preds_path("train"), columns=["prob", "label"]).to_pandas()
    prob, label = preds["prob"].to_numpy(), preds["label"].to_numpy()
    calib = fit_isotonic(prob, label)
    cal = apply_calibration(prob, calib)
    calib["brier_raw"] = float(np.mean((prob - label) ** 2))
    calib["brier_calibrated"] = float(np.mean((cal - label) ** 2))
    ensure_dirs()
    with open(CALIB_PATH, "w", encoding="utf-8") as fh:
        json.dump(calib, fh)
    return calib


def load_calibration() -> Dict[str, list]:
    """Read ``calibration.json``.

    Returns:
        Breakpoints for ``apply_calibration``.

    Raises:
        FileNotFoundError: If ``decide --split train`` has not been run.
    """
    if not CALIB_PATH.exists():
        raise FileNotFoundError(f"{CALIB_PATH} not found - run `python -m src.decide --split train` first")
    with open(CALIB_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def decide_frame(preds: pd.DataFrame, calib: Dict[str, list], threshold: float, mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply calibration and the per-S1 decision rule to one prediction frame (no exclusivity yet).

    Args:
        preds: ``s1_id, cand_id, prob``.
        calib: Calibration breakpoints.
        threshold: Flat threshold (used by ``mode="threshold"`` and for the switch statistics).
        mode: ``"hybrid"``, ``"expf"`` or ``"threshold"``.

    Returns:
        ``(keep mask of the chosen rule, keep mask of the flat threshold rule, calibrated probabilities)``.
    """
    p_raw = preds["prob"].to_numpy(dtype=np.float32)
    p_cal = apply_calibration(p_raw, calib)
    flat = p_raw >= threshold
    if mode == "threshold":
        return flat, flat, p_cal
    codes = pd.factorize(preds["s1_id"])[0]
    keep, _ = choose_expected_f05(codes, p_cal, p_raw, n_draws=MC_DRAWS, min_p=MC_MIN_P)
    if mode == "hybrid":  # empty vs non-empty from the flat rule; the number predicted from expected F0.5
        n_s1 = int(codes.max()) + 1 if len(codes) else 0
        flat_any = (np.bincount(codes[flat], minlength=n_s1) > 0)[codes]
        expf_any = (np.bincount(codes[keep], minlength=n_s1) > 0)[codes]
        keep = (keep & flat_any) | (flat & ~expf_any)
    return keep, flat, p_cal


def iter_s1_chunks(path, chunk_pairs: int = CHUNK_PAIRS):
    """Stream a prediction parquet as frames that never split an S1 (its rows are contiguous in the file).

    Args:
        path: ``preds_<split>.parquet`` (same row order as the candidates, so each S1's rows are contiguous).
        chunk_pairs: Target rows per frame.

    Returns:
        Iterator of ``s1_id, cand_id, prob`` frames.
    """
    carry = None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk_pairs, columns=["s1_id", "cand_id", "prob"]):
        frame = batch.to_pandas()
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
        last = frame["s1_id"].iat[-1]
        tail = (frame["s1_id"] == last).to_numpy()
        tail_start = len(frame) - int(np.argmin(tail[::-1])) if not tail.all() else 0
        carry = frame.iloc[tail_start:].reset_index(drop=True)
        if tail_start:
            yield frame.iloc[:tail_start].reset_index(drop=True)
    if carry is not None and len(carry):
        yield carry


def run_decision(split: str) -> Dict[str, object]:
    """Decide the matches of ``split`` (``val``/``test``) and write ``matches_<split>.parquet``.

    Args:
        split: ``"val"`` or ``"test"``.

    Returns:
        Statistics: kept pairs, S1 answer switches versus the flat threshold rule, pairs removed by exclusivity,
        decision time.
    """
    calib, thr = load_calibration(), load_threshold()
    sources = [split] + (["train"] if split == "val" and EXCLUSIVE else [])  # train S1s' OOF decisions compete too
    chosen, stats = [], {"mode": DECISION, "exclusive": EXCLUSIVE, "threshold": thr, "s1_empty_to_predict": 0,
                         "s1_predict_to_empty": 0, "pairs_kept_before_exclusivity": 0, "pairs_kept_flat_rule": 0,
                         "n_pairs_decided": 0, "n_s1_decided": 0}
    t0 = time.perf_counter()
    for name in sources:
        for preds in iter_s1_chunks(preds_path(name)):
            keep, flat, p_cal = decide_frame(preds, calib, thr, DECISION)
            stats["n_pairs_decided"] += len(preds)
            stats["n_s1_decided"] += preds["s1_id"].nunique()
            if name == split:
                s1_any = pd.Series(keep).groupby(preds["s1_id"].to_numpy()).any()
                s1_flat = pd.Series(flat).groupby(preds["s1_id"].to_numpy()).any()
                stats["s1_empty_to_predict"] += int((~s1_flat & s1_any).sum())
                stats["s1_predict_to_empty"] += int((s1_flat & ~s1_any).sum())
                stats["pairs_kept_before_exclusivity"] += int(keep.sum())
                stats["pairs_kept_flat_rule"] += int(flat.sum())
            chosen.append(preds[keep].assign(prob_cal=p_cal[keep], target=(name == split)))
    stats["decision_seconds"] = round(time.perf_counter() - t0, 1)
    sel = pd.concat(chosen, ignore_index=True)
    if EXCLUSIVE:
        ex = exclusive_mask(sel["s1_id"].to_numpy(), sel["cand_id"].to_numpy(), sel["prob_cal"].to_numpy(), sel["prob"].to_numpy())
        stats["exclusivity_removed_target"] = int((~ex & sel["target"].to_numpy()).sum())
        stats["exclusivity_removed_all"] = int((~ex).sum())
        sel = sel[ex]
    out = sel[sel["target"]].drop(columns="target")
    stats["pairs_kept"] = len(out)
    ensure_dirs()
    pq.write_table(pa.Table.from_pandas(out[MATCH_SCHEMA.names], schema=MATCH_SCHEMA, preserve_index=False), matches_path(split), compression="zstd")
    return stats


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
            calib = fit_calibration()
            print(f"decide: isotonic calibration on OOF predictions: {len(calib['x'])} breakpoints, Brier "
                  f"{calib['brier_raw']:.5f} -> {calib['brier_calibrated']:.5f} (in-sample) -> {CALIB_PATH}")
            save_metrics("decide_train", {"threshold": res["threshold"], "oof_macro_f05": res["oof_macro_f05"],
                                          "brier_raw": calib["brier_raw"], "brier_calibrated": calib["brier_calibrated"]})
        else:
            stats = run_decision(split)
            info["pairs"] = stats["n_pairs_decided"]
            print(f"decide[{split}]: rule={stats['mode']} exclusive={stats['exclusive']}: kept {stats['pairs_kept']} pairs "
                  f"(flat-threshold rule would keep {stats['pairs_kept_flat_rule']}); decision over {stats['n_s1_decided']} S1 / "
                  f"{stats['n_pairs_decided']} pairs took {stats['decision_seconds']}s")
            print(f"decide[{split}]: S1 answers vs the flat rule: empty -> predict {stats['s1_empty_to_predict']}, "
                  f"predict -> empty {stats['s1_predict_to_empty']}")
            if stats["exclusive"]:
                print(f"decide[{split}]: exclusivity removed {stats['exclusivity_removed_target']} {split} pairs "
                      f"({stats['exclusivity_removed_all']} incl. competing S1s)")
            save_metrics(f"decide_{split}", stats)

if __name__ == "__main__":
    main()
