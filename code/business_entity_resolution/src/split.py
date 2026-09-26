"""Deterministic train/validation split of Source 1 entity IDs."""
from __future__ import annotations

import json
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .config import INTERIM_DIR, SEED, TRAIN_DIR, ensure_dirs, train_s1_frac
from .io_utils import COUNTRY_COL, GROUND_TRUTH_SUFFIX, ID_COL, SOURCE_SUFFIXES, ground_truth_has_matches, read_tsv
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


_NO_MATCH: frozenset = frozenset()
_HAS_MATCH: frozenset = frozenset({"<match>"})


def sample_s1_fraction(
    source1: pd.DataFrame,
    has_match: Mapping[str, bool],
    frac: float,
    seed: int = SEED,
    id_col: str = ID_COL,
    country_col: str = COUNTRY_COL,
) -> pd.DataFrame:
    """Stratified random fraction of the S1 rows (strata = country x singleton flag).

    Each stratum keeps ``round(frac * size)`` rows, but at least one row when the stratum is non-empty, chosen with a
    seeded RNG after sorting by id (so the result does not depend on row order).

    Args:
        source1: S1 DataFrame with ``id_col`` and ``country_col``.
        has_match: ``{s1_id: True if the S1 has at least one true match}``.
        frac: Fraction in (0, 1]; 1.0 returns ``source1`` unchanged.
        seed: RNG seed.
        id_col: S1 id column.
        country_col: Country column.

    Returns:
        The sampled rows (a copy), sorted by id.
    """
    if frac >= 1.0:
        return source1
    strata: Dict[Tuple[str, bool], List[int]] = {}
    ordered = source1.sort_values(id_col).reset_index(drop=True)
    for pos, (eid, country) in enumerate(zip(ordered[id_col], ordered[country_col])):
        strata.setdefault((country, not has_match[eid]), []).append(pos)
    rng = np.random.RandomState(seed)
    keep: List[int] = []
    for key in sorted(strata):
        rows = strata[key]
        n = max(1, int(np.floor(len(rows) * frac + 0.5)))
        keep.extend(np.array(rows)[rng.permutation(len(rows))[:n]].tolist())
    return ordered.iloc[sorted(keep)].reset_index(drop=True)


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

    With ``ER_TRAIN_S1_FRAC < 1`` a stratified (country x singleton) random fraction of the train S1 ids is drawn first
    (seed 42) and only that fraction is split 80/20. Blocking still searches the FULL train S2/S3 pool for those S1s,
    so negatives are realistic hard negatives.

    Args:
        argv: Unused (the stage takes no arguments); kept for a uniform ``main`` signature.
    """
    with stage_timer("split"):
        s1 = read_tsv(TRAIN_DIR / f"train_{SOURCE_SUFFIXES[0]}")  # only S1 + truth flags: S2/S3 are not needed here
        has_match = ground_truth_has_matches(TRAIN_DIR / f"train_{GROUND_TRUTH_SUFFIX}")
        frac = train_s1_frac()
        s1 = sample_s1_fraction(s1, has_match, frac)
        truth_like = {eid: (_HAS_MATCH if has_match[eid] else _NO_MATCH) for eid in s1[ID_COL]}  # make_split only needs emptiness
        train_ids, val_ids = make_split(s1, truth_like)
        ensure_dirs()
        with open(SPLIT_PATH, "w", encoding="utf-8") as fh:
            json.dump({"train": train_ids, "val": val_ids, "frac": frac}, fh)
        n_single = sum(1 for e in val_ids if not has_match[e])
        print(f"split: S1 fraction {frac}: {len(train_ids)} train S1, {len(val_ids)} val S1 ({n_single} val singletons) -> {SPLIT_PATH}")


if __name__ == "__main__":
    main()
