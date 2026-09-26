"""Stage 4: train the pair classifier -> data/interim/model.txt and out-of-fold data/interim/preds_train.parquet
(``s1_id, cand_id, prob`` plus the ``label`` used for training, so the threshold sweep needs no other file).

Run as ``python -m src.train --split train`` from ``code/business_entity_resolution/`` (other splits are scored by
``src.predict``). LightGBM binary classifier, 5-fold ``GroupKFold`` grouped by ``s1_id`` on the TRAIN split only, so
no S1's pairs are ever split across folds. The out-of-fold probabilities are what ``src.decide`` tunes the threshold
on; the final model is then refit on all train rows.

Scaling: features are float32; if the train feature file has more than ``MAX_TRAIN_PAIRS`` rows, whole S1 groups are
sampled (seed 42) until the cap is reached (out-of-fold predictions then exist only for the sampled S1s).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .cli import parse_split
from .config import INTERIM_DIR, SEED, ensure_dirs, max_mem_gb
from .features import FEATURE_COLUMNS, features_path
from .perf import SAMPLE_CAVEAT, save_metrics, stage_timer

MODEL_PATH = INTERIM_DIR / "model.txt"
N_FOLDS = 5
# Default row cap scales with the memory budget (float32 matrix + Arrow table + LightGBM copies ~ 0.8 GB per million rows).
MAX_TRAIN_PAIRS = int(os.environ.get("ER_MAX_TRAIN_PAIRS") or 1_250_000 * max_mem_gb())
PARAMS: Dict[str, object] = {
    "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05, "num_leaves": 63,
    "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "seed": SEED, "bagging_seed": SEED, "feature_fraction_seed": SEED, "deterministic": True, "force_row_wise": True,
    "verbosity": -1,
}
MAX_ROUNDS = 1000
EARLY_STOP = 50


def preds_path(split: str):
    """Path of the prediction parquet of a split.

    Args:
        split: ``"train"``, ``"val"`` or ``"test"``.

    Returns:
        ``data/interim/preds_<split>.parquet``.
    """
    return INTERIM_DIR / f"preds_{split}.parquet"


def choose_s1_sample(path, max_pairs: int) -> Optional[pa.Array]:
    """Pick whole S1 groups (seed 42) until ``max_pairs`` rows are reached, reading only the ``s1_id`` column.

    Args:
        path: Feature parquet.
        max_pairs: Row cap; when the file is not larger, no sampling is needed.

    Returns:
        Arrow array of the chosen S1 ids, or None when every row fits.
    """
    if pq.ParquetFile(path).metadata.num_rows <= max_pairs:
        return None
    counts = pc.value_counts(pq.read_table(path, columns=["s1_id"]).column("s1_id"))
    ids, sizes = counts.field("values"), counts.field("counts").to_numpy()
    order = np.random.RandomState(SEED).permutation(len(sizes))
    chosen = order[np.cumsum(sizes[order]) <= max_pairs]
    print(f"train: feature file exceeds the cap of {max_pairs} pairs: sampling {len(chosen)}/{len(sizes)} S1 groups")
    return ids.take(pa.array(chosen))


def load_train_arrays(path=None, max_pairs: int = MAX_TRAIN_PAIRS) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pa.Table]:
    """Read the train features as a float32 matrix without going through a pandas frame of Python strings.

    Args:
        path: Feature parquet (defaults to ``features_train.parquet``).
        max_pairs: Row cap; whole S1 groups are sampled when the file is larger (see ``choose_s1_sample``).

    Returns:
        ``(X float32 [n, n_features], y int8, group code per row, Arrow table with s1_id and cand_id)``.
    """
    path = path or features_path("train")
    keep = choose_s1_sample(path, max_pairs)
    pf = pq.ParquetFile(path)
    parts = []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg)
        parts.append(t.filter(pc.is_in(t.column("s1_id"), value_set=keep)) if keep is not None else t)
    table = pa.concat_tables(parts)
    x = np.empty((table.num_rows, len(FEATURE_COLUMNS)), dtype=np.float32)
    for j, name in enumerate(FEATURE_COLUMNS):
        x[:, j] = table.column(name).to_numpy()
    y = table.column("label").to_numpy().astype(np.int8)
    groups = table.column("s1_id").combine_chunks().dictionary_encode().indices.to_numpy().astype(np.int64)
    return x, y, groups, table.select(["s1_id", "cand_id"])


def cross_validate(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int = N_FOLDS) -> Tuple[np.ndarray, List[int]]:
    """5-fold GroupKFold out-of-fold predictions (groups = S1) with early stopping per fold.

    Args:
        x: Feature matrix.
        y: 0/1 labels.
        groups: Integer S1 code per row (no S1 is ever split across folds).
        n_folds: Number of folds.

    Returns:
        ``(oof_probabilities aligned with x, best_iteration per fold)``.
    """
    oof = np.zeros(len(y), dtype=np.float32)
    best: List[int] = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(x, y, groups)):
        dtr = lgb.Dataset(x[tr], y[tr], feature_name=FEATURE_COLUMNS)
        dva = lgb.Dataset(x[va], y[va], reference=dtr)
        booster = lgb.train(PARAMS, dtr, MAX_ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        oof[va] = booster.predict(x[va], num_iteration=booster.best_iteration)
        best.append(int(booster.best_iteration))
        print(f"train: fold {fold}: best iteration {booster.best_iteration}, "
              f"AUC {roc_auc_score(y[va], oof[va]):.4f}", flush=True)
    return oof, best


def fit_final(x: np.ndarray, y: np.ndarray, rounds: int) -> lgb.Booster:
    """Fit the final model on all train rows.

    Args:
        x: Feature matrix.
        y: 0/1 labels.
        rounds: Boosting rounds (mean best fold iteration, slightly inflated for the larger training set).

    Returns:
        Trained booster.
    """
    return lgb.train(PARAMS, lgb.Dataset(x, y, feature_name=FEATURE_COLUMNS), rounds)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point: cross-validate, save OOF predictions, refit and save the final model (train split only).

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    split = parse_split(__doc__.splitlines()[0], argv)
    if split != "train":
        print(f"train: nothing to do for '{split}' (models are trained on the train split; use src.predict to score it)")
        return
    with stage_timer("train", split) as info:
        sampled = pq.ParquetFile(features_path("train")).metadata.num_rows > MAX_TRAIN_PAIRS
        x, y, groups, ids = load_train_arrays()
        info["pairs"] = len(y)
        oof, best = cross_validate(x, y, groups)
        auc, ap = float(roc_auc_score(y, oof)), float(average_precision_score(y, oof))
        rounds = max(int(np.mean(best) * 1.1), 10)
        booster = fit_final(x, y, rounds)
        ensure_dirs()
        booster.save_model(str(MODEL_PATH))
        pq.write_table(
            pa.table({"s1_id": ids.column("s1_id"), "cand_id": ids.column("cand_id"), "prob": oof, "label": y}),
            preds_path("train"), compression="zstd",
        )
        imp = pd.Series(booster.feature_importance("gain"), index=FEATURE_COLUMNS).sort_values(ascending=False)
        top = {k: float(v) for k, v in imp.head(15).items()}
        print(f"train: OOF AUC {auc:.4f}, average precision {ap:.4f} ({len(y)} pairs, {int(y.sum())} positives), final rounds {rounds}")
        print("train: top 15 features by gain:")
        for name, gain in top.items():
            print(f"   {name:18s} {gain:14.1f}")
        print(SAMPLE_CAVEAT)
        save_metrics("train", {"auc": auc, "average_precision": ap, "rounds": rounds, "best_iterations": best,
                               "n_pairs": len(y), "sampled": sampled, "top_features": top})


if __name__ == "__main__":
    main()
