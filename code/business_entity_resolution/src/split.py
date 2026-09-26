"""Deterministic train/validation split of Source 1 entity IDs."""
from __future__ import annotations

import json
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .config import INTERIM_DIR, SEED, TRAIN_DIR, ensure_dirs
from .io_utils import COUNTRY_COL, GROUND_TRUTH_SUFFIX, ID_COL, SOURCE_SUFFIXES, load_ground_truth, read_tsv
from .perf import stage_timer

VAL_FRAC = 0.2
SPLIT_PATH = INTERIM_DIR / "split.json"


def make_split(
    source1: pd.DataFrame,
    truth: Mapping[str, Set[str]],
    val_frac: float = VAL_FRAC,
    seed: int = SEED,
    id_col: str = ID_COL,
    country_col: str = COUNTRY_COL,
) -> Tuple[List[str], List[str]]:
    """Split Source 1 entity IDs into train/validation, stratified by country and singleton-ness.

    Strata are (country, is_singleton) pairs. Each stratum is shuffled with a
    seeded RNG (after sorting, so the result does not depend on row order) and
    ``round(val_frac * size)`` of its IDs go to validation. Strata with a single
    member therefore stay in train.

    Args:
        source1: Source 1 DataFrame containing ``id_col`` and ``country_col``.
        truth: Mapping from Source 1 ID to its true matched ID set; an empty
            set marks a singleton.
        val_frac: Fraction of each stratum assigned to validation.
        seed: Seed for the shuffling RNG.
        id_col: Name of the Source 1 ID column.
        country_col: Name of the country column.

    Returns:
        Tuple ``(train_ids, val_ids)`` of sorted ID lists.

    Raises:
        KeyError: If a Source 1 ID is missing from ``truth``.
    """
    strata: Dict[Tuple[str, bool], List[str]] = {}
    for eid, country in zip(source1[id_col], source1[country_col]):
        strata.setdefault((country, len(truth[eid]) == 0), []).append(eid)

    rng = np.random.RandomState(seed)
    train_ids: List[str] = []
    val_ids: List[str] = []
    for key in sorted(strata):
        ids = sorted(strata[key])
        order = rng.permutation(len(ids))
        n_val = int(np.floor(len(ids) * val_frac + 0.5))
        val_ids.extend(ids[i] for i in order[:n_val])
        train_ids.extend(ids[i] for i in order[n_val:])
    return sorted(train_ids), sorted(val_ids)


def load_split_ids(split: str) -> Optional[List[str]]:
    """Return the S1 ids of a pipeline split, or None for ``test`` (which uses every S1 of the test file).

    Args:
        split: ``"train"``, ``"val"`` or ``"test"``.

    Returns:
        Sorted S1 id list from ``data/interim/split.json`` for train/val; None for test.

    Raises:
        FileNotFoundError: If ``split.json`` has not been created (run ``python -m src.split``).
    """
    if split == "test":
        return None
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"{SPLIT_PATH} not found - run `python -m src.split` first")
    with open(SPLIT_PATH, encoding="utf-8") as fh:
        return json.load(fh)[split]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Create ``data/interim/split.json`` from the train S1 file and ground truth (80/20, seed 42).

    Validation S1s are later matched against the FULL train S2/S3 pool (see ``src.block``); the split
    only decides which S1 ids are used for training and which for validation.

    Args:
        argv: Unused (the stage takes no arguments); kept for a uniform ``main`` signature.
    """
    with stage_timer("split"):
        s1 = read_tsv(TRAIN_DIR / f"train_{SOURCE_SUFFIXES[0]}")  # only S1 + truth: S2/S3 are not needed here
        truth = load_ground_truth(TRAIN_DIR / f"train_{GROUND_TRUTH_SUFFIX}")
        train_ids, val_ids = make_split(s1, truth)
        ensure_dirs()
        with open(SPLIT_PATH, "w", encoding="utf-8") as fh:
            json.dump({"train": train_ids, "val": val_ids}, fh)
        n_single = sum(1 for e in val_ids if not truth[e])
        print(f"split: {len(train_ids)} train S1, {len(val_ids)} val S1 ({n_single} val singletons) -> {SPLIT_PATH}")


if __name__ == "__main__":
    main()
