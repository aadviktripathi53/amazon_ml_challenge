"""Deterministic train/validation split of Source 1 entity IDs."""
from __future__ import annotations

from typing import Dict, List, Mapping, Set, Tuple

import numpy as np
import pandas as pd

from .config import SEED
from .io_utils import COUNTRY_COL, ID_COL

VAL_FRAC = 0.2


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
