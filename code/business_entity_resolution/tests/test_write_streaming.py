"""Output rules of the streaming submission writer (and agreement with the organisers' validator)."""
import importlib.util
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.write_submission import write_results_streaming

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_pairs(path, groups):
    """Write ``groups`` (list of row groups, each a list of (s1, cand)) as a parquet file."""
    schema = pa.schema([("s1_id", pa.string()), ("cand_id", pa.string())])
    with pq.ParquetWriter(path, schema) as w:
        for g in groups:
            w.write_table(pa.table({"s1_id": [a for a, _ in g], "cand_id": [b for _, b in g]}, schema=schema))


def _read(path):
    """Read a written TSV as {s1: raw list string}."""
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    assert lines[-1] == ""  # file ends with a newline
    return lines[0], dict(line.split("\t") for line in lines[1:-1])


@pytest.fixture
def files(tmp_path):
    """Candidates over two row groups (S1-3 has none), matches for S1-1 and S1-2 (S1-4 has a candidate but no match)."""
    cand = tmp_path / "cand.parquet"
    match = tmp_path / "match.parquet"
    _write_pairs(cand, [
        [("S1-1", "S3-9"), ("S1-1", "S2-5"), ("S1-1", "S2-5"), ("S1-1", "S2-7")],
        [("S1-2", "S2-1"), ("S1-4", "S3-2")],
    ])
    _write_pairs(match, [[("S1-1", "S3-9"), ("S1-1", "S2-5"), ("S1-2", "S2-1")]])
    return cand, match, tmp_path / "out"


def test_one_row_per_s1_sorted_deduped_no_spaces_or_quotes(files):
    """Every S1 gets one row; lists are sorted, deduplicated, comma-joined without spaces or quotes; empties are ''."""
    cand, match, out = files
    ids = ["S1-1", "S1-2", "S1-3", "S1-4"]
    m_path, c_path = write_results_streaming(ids, cand, match, out_dir=out)
    header, matches = _read(m_path)
    header_c, cands = _read(c_path)
    assert header == "source1_entity_id\tmatched_entity_ids" and header_c == "source1_entity_id\tcandidate_entity_ids"
    assert set(matches) == set(ids) == set(cands) and len(matches) == 4
    assert matches["S1-1"] == "S2-5,S3-9" and cands["S1-1"] == "S2-5,S2-7,S3-9"
    assert matches["S1-3"] == "" and cands["S1-3"] == ""  # no candidates at all -> empty rows, still present
    assert matches["S1-4"] == "" and cands["S1-4"] == "S3-2"  # candidate but no match
    text = Path(m_path).read_text() + Path(c_path).read_text()
    assert '"' not in text and ", " not in text and " " not in text.replace("\t", "")


def test_only_s2_s3_ids_and_matches_subset_of_candidates(files):
    """Matches are a subset of candidates and all ids carry S2-/S3- prefixes."""
    cand, match, out = files
    m_path, c_path = write_results_streaming(["S1-1", "S1-2", "S1-3", "S1-4"], cand, match, out_dir=out)
    _, matches = _read(m_path)
    _, cands = _read(c_path)
    for s1, ml in matches.items():
        assert set(filter(None, ml.split(","))) <= set(filter(None, cands[s1].split(",")))
        assert all(i.startswith(("S2-", "S3-")) for i in filter(None, ml.split(",")))


def test_match_that_is_not_a_candidate_is_rejected(tmp_path):
    """A kept match absent from that S1's candidates is a pipeline bug and must fail loudly."""
    cand, match = tmp_path / "c.parquet", tmp_path / "m.parquet"
    _write_pairs(cand, [[("S1-1", "S2-1")]])
    _write_pairs(match, [[("S1-1", "S2-2")]])
    with pytest.raises(AssertionError):
        write_results_streaming(["S1-1"], cand, match, out_dir=tmp_path / "o")


def test_invalid_inputs_are_rejected(tmp_path):
    """Self-matches, unknown ids, unknown S1s, duplicate S1 ids and split row groups all raise."""
    cand, match = tmp_path / "c.parquet", tmp_path / "m.parquet"
    _write_pairs(match, [[]])
    _write_pairs(cand, [[("S1-1", "S1-2")]])  # an S1 id as a candidate
    with pytest.raises(ValueError):
        write_results_streaming(["S1-1"], cand, match, out_dir=tmp_path / "o")
    _write_pairs(cand, [[("S1-1", "S2-1")]])
    with pytest.raises(ValueError):  # candidate id that does not exist in the test files
        write_results_streaming(["S1-1"], cand, match, out_dir=tmp_path / "o", valid_ids=pd.Index(["S2-99"]))
    with pytest.raises(ValueError):  # candidate row for an S1 that is not in the test source1 file
        write_results_streaming(["S1-2"], cand, match, out_dir=tmp_path / "o")
    with pytest.raises(ValueError):  # duplicate S1 ids in the required list
        write_results_streaming(["S1-1", "S1-1"], cand, match, out_dir=tmp_path / "o")
    _write_pairs(cand, [[("S1-1", "S2-1")], [("S1-1", "S2-2")]])  # same S1 in two row groups
    with pytest.raises(ValueError):
        write_results_streaming(["S1-1"], cand, match, out_dir=tmp_path / "o")


def test_output_passes_the_official_validator(files, tmp_path):
    """The written files pass utils/validate_submission.py (including the ID-existence check)."""
    spec = importlib.util.spec_from_file_location("validate_submission", REPO_ROOT / "utils" / "validate_submission.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    cand, match, out = files
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    (test_dir / "test_source1.tsv").write_text(header + "".join(f"S1-{i}\tn\ta\tUS\n" for i in (1, 2, 3, 4)))
    (test_dir / "test_source2.tsv").write_text(header + "".join(f"S2-{i}\tn\ta\tUS\n" for i in (1, 5, 7)))
    (test_dir / "test_source3.tsv").write_text(header + "".join(f"S3-{i}\tn\ta\tUS\n" for i in (2, 9)))
    m_path, c_path = write_results_streaming(["S1-1", "S1-2", "S1-3", "S1-4"], cand, match, out_dir=out)
    errors, _ = validator.validate(str(m_path), str(c_path), str(test_dir), check_ids=True)
    assert errors == []
