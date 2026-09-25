"""Loading of the three source TSVs and the ground-truth file.

File layout (from the problem statement)::

    dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
    dataset/test/test_source{1,2,3}.tsv

Source files have columns entity_id, business_name, business_address, country;
the ground truth has source1_entity_id and matched_entity_ids. A record's
source is given by its entity_id prefix (S1-/S2-/S3-).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd

from .config import split_dir

# Files are named "<split>_<suffix>", e.g. train_source1.tsv / test_source1.tsv.
SOURCE_SUFFIXES: Tuple[str, str, str] = ("source1.tsv", "source2.tsv", "source3.tsv")
GROUND_TRUTH_SUFFIX = "ground_truth.tsv"

# Columns of the source files.
ID_COL = "entity_id"
NAME_COL = "business_name"
ADDRESS_COL = "business_address"
COUNTRY_COL = "country"

# Columns of the ground-truth file.
TRUTH_ID_COL = "source1_entity_id"
MATCHED_COL = "matched_entity_ids"

# entity_id prefixes identifying the source of a record.
S1_PREFIX, S2_PREFIX, S3_PREFIX = "S1-", "S2-", "S3-"


def read_tsv(path) -> pd.DataFrame:
    """Read a TSV keeping every column as a string and empty cells as "".

    Args:
        path: Path to the tab-separated file.

    Returns:
        DataFrame with ``str`` dtype everywhere and no NaN values.
    """
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def parse_id_list(value: str) -> Set[str]:
    """Parse a comma-separated ID string into a set of IDs.

    Args:
        value: Raw cell such as ``"a1,b2"``; may be empty or contain stray spaces.

    Returns:
        Set of non-empty, whitespace-stripped IDs (empty set for an empty cell).
    """
    return {part.strip() for part in value.split(",") if part.strip()}


def load_sources(data_dir, split: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Source 1, Source 2 and Source 3 from a dataset split directory.

    Args:
        data_dir: Directory such as ``dataset/train`` or ``dataset/test``.
        split: File-name prefix (``"train"`` or ``"test"``); defaults to the
            directory name.

    Returns:
        Tuple ``(source1, source2, source3)`` of string DataFrames.
    """
    data_dir = Path(data_dir)
    split = split or data_dir.name
    s1, s2, s3 = (read_tsv(data_dir / f"{split}_{suffix}") for suffix in SOURCE_SUFFIXES)
    return s1, s2, s3


def load_ground_truth(path, id_col: str = TRUTH_ID_COL, matched_col: str = MATCHED_COL) -> Dict[str, Set[str]]:
    """Load ground truth as a mapping from Source 1 ID to its matched ID set.

    Args:
        path: Path to the ground-truth TSV.
        id_col: Column with the Source 1 entity ID.
        matched_col: Column with the comma-separated matched IDs (may be empty).

    Returns:
        Dict ``{source1_id: {matched ids}}``; a true singleton maps to ``set()``.

    Raises:
        ValueError: If a Source 1 ID appears on more than one row.
    """
    df = read_tsv(path)
    duplicated = df.loc[df[id_col].duplicated(), id_col].unique()
    if len(duplicated):
        raise ValueError(f"Duplicate Source 1 IDs in ground truth, e.g. {list(duplicated[:5])}")
    return {eid: parse_id_list(m) for eid, m in zip(df[id_col], df[matched_col])}


def load_split(data_dir=None, with_truth: Optional[bool] = None, split: Optional[str] = None):
    """Load a dataset split directory: the three sources and, if present, ground truth.

    Args:
        data_dir: Directory such as ``dataset/train`` or ``dataset/test``;
            when None, ``config.split_dir(split)`` is used (``split`` required).
        with_truth: Force loading (True) or skipping (False) the ground truth.
            When None, it is loaded only if the ground-truth file exists.
        split: File-name prefix; defaults to the directory name.

    Returns:
        Tuple ``(source1, source2, source3, truth)`` where ``truth`` is None
        when no ground truth was loaded.
    """
    if data_dir is None:
        if split is None:
            raise ValueError("load_split needs data_dir or split")
        data_dir = split_dir(split)
    data_dir = Path(data_dir)
    if split == "val":  # the val split is carved out of the train files
        split = "train"
    split = split or data_dir.name
    s1, s2, s3 = load_sources(data_dir, split)
    truth_path = data_dir / f"{split}_{GROUND_TRUTH_SUFFIX}"
    if with_truth is None:
        with_truth = truth_path.exists()
    truth = load_ground_truth(truth_path) if with_truth else None
    return s1, s2, s3, truth
