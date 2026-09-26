"""Stage 4b: score candidate pairs with the trained model -> data/interim/preds_{split}.parquet.

Run as ``python -m src.predict --split <val|test>`` from ``code/business_entity_resolution/``. Feature rows are
streamed in batches of ``BATCH_ROWS`` so memory does not depend on the number of pairs. The out-of-fold train
predictions are written by ``src.train`` (``train`` is a no-op here on purpose: in-sample scores would be optimistic).
Output rows are in the same order as ``features_<split>.parquet`` / ``candidates_<split>.parquet``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .cli import parse_split
from .config import ensure_dirs
from .features import FEATURE_COLUMNS, features_path
from .perf import stage_timer
from .train import MODEL_PATH, preds_path

BATCH_ROWS = 1_000_000
PRED_SCHEMA = pa.schema([("s1_id", pa.string()), ("cand_id", pa.string()), ("prob", pa.float32())])


def predict_split(split: str) -> int:
    """Score every feature row of a split and write the prediction parquet.

    Args:
        split: ``"val"`` or ``"test"``.

    Returns:
        Number of pairs scored.
    """
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    if booster.feature_name() != FEATURE_COLUMNS:
        raise ValueError("model feature names differ from src.features.FEATURE_COLUMNS - retrain with `python -m src.train`")
    ensure_dirs()
    n = 0
    with pq.ParquetWriter(preds_path(split), PRED_SCHEMA, compression="zstd") as writer:
        for batch in pq.ParquetFile(features_path(split)).iter_batches(batch_size=BATCH_ROWS, columns=["s1_id", "cand_id", *FEATURE_COLUMNS]):
            df = batch.to_pandas()
            prob = booster.predict(df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            writer.write_table(pa.table({"s1_id": df["s1_id"], "cand_id": df["cand_id"], "prob": prob.astype(np.float32)}, schema=PRED_SCHEMA))
            n += len(df)
    return n


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point: score a split (``train`` is a no-op, its OOF predictions come from ``src.train``).

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    split = parse_split(__doc__.splitlines()[0], argv)
    if split == "train":
        print("predict: 'train' predictions are the out-of-fold ones written by src.train - nothing to do")
        return
    with stage_timer("predict", split) as info:
        info["pairs"] = predict_split(split)
        print(f"predict: {info['pairs']} pairs -> {preds_path(split)}")


if __name__ == "__main__":
    main()
