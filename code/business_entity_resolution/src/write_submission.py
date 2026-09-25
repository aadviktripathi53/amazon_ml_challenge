"""Writers for matching_results.tsv and candidate_pairs.tsv."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Set

import pandas as pd

from .config import OUTPUT_DIR
from .io_utils import MATCHED_COL, S2_PREFIX, S3_PREFIX

MATCHING_FILE = "matching_results.tsv"
CANDIDATES_FILE = "candidate_pairs.tsv"
OUT_ID_COL = "source1_entity_id"
MATCH_COL = MATCHED_COL
CANDIDATE_COL = "candidate_entity_ids"


def _join_ids(ids: Iterable[str]) -> str:
    """Deduplicate IDs, sort them, and join with commas ("" when there are none).

    Args:
        ids: Any iterable of ID strings, possibly with duplicates.

    Returns:
        Comma-joined string of unique IDs.
    """
    return ",".join(sorted(set(ids)))


def check_target_ids(mapping: Mapping[str, Iterable[str]], valid_ids: Optional[Set[str]] = None, what: str = "IDs") -> None:
    """Check that every listed ID is a Source 2 / Source 3 ID (and exists, if ``valid_ids`` is given).

    S1- IDs (self-matches) and anything without an S2-/S3- prefix are rejected,
    as are IDs missing from ``valid_ids`` (typically the S2 + S3 IDs of the test set).

    Args:
        mapping: Mapping from Source 1 ID to a list of target IDs.
        valid_ids: Optional set of allowed IDs.
        what: Label used in the error message (e.g. "matches").

    Raises:
        ValueError: If any ID is invalid.
    """
    bad = set()
    for ids in mapping.values():
        for i in ids:
            if not i.startswith((S2_PREFIX, S3_PREFIX)) or (valid_ids is not None and i not in valid_ids):
                bad.add(i)
    if bad:
        raise ValueError(f"{len(bad)} invalid {what} (must be existing S2-/S3- IDs), e.g. {sorted(bad)[:5]}")


def check_matches_are_candidates(
    matches: Mapping[str, Iterable[str]], candidates: Mapping[str, Iterable[str]]
) -> None:
    """Assert that every predicted match is also a candidate for the same entity.

    Args:
        matches: Mapping from Source 1 ID to predicted matched IDs.
        candidates: Mapping from Source 1 ID to candidate IDs.

    Raises:
        AssertionError: If any match is absent from that entity's candidates.
    """
    bad = {}
    for eid, matched in matches.items():
        missing = set(matched) - set(candidates.get(eid, ()))
        if missing:
            bad[eid] = sorted(missing)
    assert not bad, f"{len(bad)} entities have matches that are not candidates, e.g. {dict(list(bad.items())[:3])}"


def write_results(
    source1_ids: Sequence[str],
    matches: Mapping[str, Iterable[str]],
    candidates: Mapping[str, Iterable[str]],
    out_dir=OUTPUT_DIR,
    valid_ids: Optional[Set[str]] = None,
    id_col: str = OUT_ID_COL,
) -> List[Path]:
    """Write matching_results.tsv and candidate_pairs.tsv, one row per Source 1 ID.

    Rows follow the order of ``source1_ids``. ID lists are deduplicated,
    sorted and comma-joined; entities with no matches/candidates get "".

    Args:
        source1_ids: Every Source 1 ID that must appear in the output.
        matches: Mapping from Source 1 ID to predicted matched IDs.
        candidates: Mapping from Source 1 ID to candidate IDs.
        out_dir: Output directory (created if missing); defaults to ``config.OUTPUT_DIR``.
        valid_ids: S2 + S3 IDs present in the evaluated split; when given, any
            other ID in matches/candidates is rejected.
        id_col: Name of the Source 1 ID column in the output files.

    Returns:
        Paths of the two files written, ``[matching_results, candidate_pairs]``.

    Raises:
        AssertionError: If a match is not also a candidate.
        ValueError: If ``source1_ids`` has duplicates, the dicts contain keys
            outside ``source1_ids``, or a listed ID is not a valid S2-/S3- ID.
    """
    ids = list(source1_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("source1_ids contains duplicates")
    unknown = (set(matches) | set(candidates)) - set(ids)
    if unknown:
        raise ValueError(f"{len(unknown)} IDs in matches/candidates are not Source 1 IDs, e.g. {sorted(unknown)[:5]}")
    check_target_ids(matches, valid_ids, "matches")
    check_target_ids(candidates, valid_ids, "candidates")
    check_matches_are_candidates(matches, candidates)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, col_name, mapping in (
        (MATCHING_FILE, MATCH_COL, matches),
        (CANDIDATES_FILE, CANDIDATE_COL, candidates),
    ):
        df = pd.DataFrame({id_col: ids, col_name: [_join_ids(mapping.get(eid, ())) for eid in ids]})
        path = out_dir / filename
        df.to_csv(path, sep="\t", index=False, lineterminator="\n")
        paths.append(path)
    return paths
