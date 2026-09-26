"""Writers for matching_results.tsv and candidate_pairs.tsv (``python -m src.write_submission --split test``)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

import pandas as pd
import pyarrow.parquet as pq

from .cli import parse_split
from .config import OUTPUT_DIR, TEST_DIR
from .io_utils import ID_COL, MATCHED_COL, S2_PREFIX, S3_PREFIX, SOURCE_SUFFIXES
from .perf import stage_timer
from .streaming import iter_chunks

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


def write_results_streaming(
    source1_ids: Sequence[str],
    candidates_parquet,
    matches_parquet,
    out_dir=OUTPUT_DIR,
    valid_ids: Optional[pd.Index] = None,
    id_col: str = OUT_ID_COL,
) -> List[Path]:
    """Write both TSVs from the pipeline parquet files without building an all-pairs dictionary.

    ``candidates_parquet`` is read one row group at a time (``src.block`` never splits an S1 across row groups), so
    memory is bounded by the row group size plus the (small) match table. Rows come out in candidate order; S1s that
    have no candidate are appended with empty lists, so every id of ``source1_ids`` gets exactly one row.

    Args:
        source1_ids: Every Source 1 id of the test set.
        candidates_parquet: Parquet with ``s1_id, cand_id`` = the exact set the model scored.
        matches_parquet: Parquet with ``s1_id, cand_id`` = the pairs kept by the decision layer.
        out_dir: Output directory (created if missing).
        valid_ids: Optional index of all S2/S3 ids of the test set; any other id is rejected.
        id_col: Name of the Source 1 id column in the outputs.

    Returns:
        Paths ``[matching_results, candidate_pairs]``.

    Raises:
        ValueError: On duplicate S1 ids, an S1 that is not in ``source1_ids``, an S1 split across row groups, or an
            id that is not an existing S2-/S3- id.
        AssertionError: If a match is not also a candidate of the same S1.
    """
    ids = list(source1_ids)
    required = set(ids)
    if len(required) != len(ids):
        raise ValueError("source1_ids contains duplicates")
    m = pq.read_table(matches_parquet, columns=["s1_id", "cand_id"]).to_pandas()
    matches: Dict[str, List[str]] = {s: list(g) for s, g in m.groupby("s1_id", sort=False)["cand_id"]}
    check_target_ids(matches, None, "matches")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    match_path, cand_path = out_dir / MATCHING_FILE, out_dir / CANDIDATES_FILE
    seen: Set[str] = set()
    with open(match_path, "w", encoding="utf-8", newline="\n") as fm, open(cand_path, "w", encoding="utf-8", newline="\n") as fc:
        fm.write(f"{id_col}\t{MATCH_COL}\n")
        fc.write(f"{id_col}\t{CANDIDATE_COL}\n")
        pf = pq.ParquetFile(candidates_parquet)
        for rg in range(pf.num_row_groups):
            df = pf.read_row_group(rg, columns=["s1_id", "cand_id"]).to_pandas()
            if not df["cand_id"].str.startswith((S2_PREFIX, S3_PREFIX)).all():
                raise ValueError("candidate ids must be S2-/S3- ids")
            if valid_ids is not None and (valid_ids.get_indexer(df["cand_id"]) < 0).any():
                raise ValueError("candidate ids that do not exist in the test S2/S3 files")
            for s1, group in df.groupby("s1_id", sort=False)["cand_id"]:
                if s1 in seen:
                    raise ValueError(f"S1 {s1} appears in more than one candidate row group")
                if s1 not in required:
                    raise ValueError(f"candidate S1 id {s1} is not in the test source1 file")
                seen.add(s1)
                cands = _join_ids(group)
                missing = set(matches.get(s1, ())) - set(group)
                assert not missing, f"matches of {s1} that are not candidates: {sorted(missing)[:3]}"
                fc.write(f"{s1}\t{cands}\n")
                fm.write(f"{s1}\t{_join_ids(matches.get(s1, ()))}\n")
        stray = set(matches) - seen
        assert not stray, f"{len(stray)} S1s have matches but no candidates, e.g. {sorted(stray)[:3]}"
        for s1 in ids:  # S1s blocking found nothing for: empty prediction
            if s1 not in seen:
                fc.write(f"{s1}\t\n")
                fm.write(f"{s1}\t\n")
    return [match_path, cand_path]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point: write ``output/matching_results.tsv`` and ``output/candidate_pairs.tsv`` for the test split.

    The candidate file is exactly ``candidates_test.parquet`` (what the model scored); the row counts of the
    candidate, feature and prediction files are asserted equal first.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    from .block import candidates_path
    from .decide import matches_path
    from .features import features_path
    from .train import preds_path

    split = parse_split(__doc__.splitlines()[0], argv)
    if split != "test":
        print(f"write_submission: only the test split is written to output/ (got '{split}') - nothing to do")
        return
    with stage_timer("write_submission", split) as info:
        n_cand = pq.ParquetFile(candidates_path(split)).metadata.num_rows
        for name, path in (("features", features_path(split)), ("preds", preds_path(split))):
            n = pq.ParquetFile(path).metadata.num_rows
            assert n == n_cand, f"{name} has {n} rows but candidates has {n_cand}: the model must score exactly the candidate set"
        source1 = [i for chunk in iter_chunks(TEST_DIR / f"test_{SOURCE_SUFFIXES[0]}", usecols=[ID_COL]) for i in chunk[ID_COL].str.strip()]
        valid = pd.Index([i for suffix in SOURCE_SUFFIXES[1:] for chunk in iter_chunks(TEST_DIR / f"test_{suffix}", usecols=[ID_COL]) for i in chunk[ID_COL].str.strip()])
        paths = write_results_streaming(source1, candidates_path(split), matches_path(split), valid_ids=valid)
        info["s1"] = len(source1)
        print(f"write_submission: {len(source1)} S1 rows ({n_cand} candidate pairs) -> {', '.join(str(p) for p in paths)}")


if __name__ == "__main__":
    main()
