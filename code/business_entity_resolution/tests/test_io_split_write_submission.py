"""Smoke tests for io_utils, split and write_submission."""
import pandas as pd
import pytest

from src.io_utils import load_ground_truth, load_split, parse_id_list, read_tsv
from src.split import make_split
from src.write_submission import check_matches_are_candidates, write_results


def test_parse_id_list():
    """Comma-separated cells parse to sets; empty cells to the empty set."""
    assert parse_id_list("a, b,,c") == {"a", "b", "c"}
    assert parse_id_list("") == set()


def test_ground_truth_roundtrip(tmp_path):
    """Ground truth loads with string dtype, keeping empty cells as singletons."""
    p = tmp_path / "gt.tsv"
    p.write_text("source1_entity_id\tmatched_entity_ids\nS1-001\tS2-1,S3-2\nS1-002\t\n")
    assert load_ground_truth(p) == {"S1-001": {"S2-1", "S3-2"}, "S1-002": set()}
    assert read_tsv(p)["source1_entity_id"].tolist() == ["S1-001", "S1-002"]


def test_ground_truth_duplicate_ids_raise(tmp_path):
    """A repeated Source 1 ID in ground truth is rejected."""
    p = tmp_path / "gt.tsv"
    p.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-1\tS2-2\n")
    with pytest.raises(ValueError):
        load_ground_truth(p)


def _toy():
    """Build a toy Source 1 frame and truth over two countries."""
    ids = [f"S1-{i:05d}" for i in range(200)]
    s1 = pd.DataFrame({"entity_id": ids, "country": ["US" if i % 2 else "IN" for i in range(200)]})
    truth = {e: (set() if i % 3 == 0 else {f"S2-{i}"}) for i, e in enumerate(ids)}
    return s1, truth


def test_split_is_deterministic_disjoint_and_stratified():
    """Same seed gives the same split, order-independent, ~80/20 in every stratum."""
    s1, truth = _toy()
    s1["country"] = s1["country"].where(s1.index % 7 != 0, "France")  # open-set country labels
    train, val = make_split(s1, truth)
    assert (train, val) == make_split(s1, truth)
    assert (train, val) == make_split(s1.sample(frac=1, random_state=0), truth)
    assert not set(train) & set(val) and len(train) + len(val) == len(s1)
    assert abs(len(val) / len(s1) - 0.2) < 0.02
    val_df = s1[s1["entity_id"].isin(val)]
    assert set(val_df["country"]) == {"US", "IN", "France"}
    assert any(len(truth[e]) == 0 for e in val) and any(len(truth[e]) > 0 for e in val)


def test_load_split_uses_split_prefixed_files(tmp_path):
    """load_split reads <split>_source{1,2,3}.tsv and the optional ground truth."""
    d = tmp_path / "train"
    d.mkdir()
    hdr = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    for n in (1, 2, 3):
        (d / f"train_source{n}.tsv").write_text(hdr + f"S{n}-1\tAcme, Inc\t1 Main St, NY\tUS\n")
    s1, s2, s3, truth = load_split(d)
    assert truth is None and s1["business_address"][0] == "1 Main St, NY" and s3["entity_id"][0] == "S3-1"
    (d / "train_ground_truth.tsv").write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\n")
    assert load_split(d)[3] == {"S1-1": {"S2-1"}}


def test_writer_output_and_dedup(tmp_path):
    """One row per ID, deduplicated comma-joined lists, empty string when none."""
    ids = ["S1-1", "S1-2", "S1-3"]
    valid = {"S2-1", "S2-2", "S3-1", "S3-9"}
    write_results(
        ids,
        {"S1-1": ["S3-1", "S2-1", "S2-1"]},
        {"S1-1": ["S2-1", "S3-1", "S2-2"], "S1-2": ["S3-9"]},
        tmp_path,
        valid_ids=valid,
    )
    m = read_tsv(tmp_path / "matching_results.tsv")
    c = read_tsv(tmp_path / "candidate_pairs.tsv")
    assert list(m.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert list(c.columns) == ["source1_entity_id", "candidate_entity_ids"]
    assert m["source1_entity_id"].tolist() == ids and c["source1_entity_id"].tolist() == ids
    assert m["matched_entity_ids"].tolist() == ["S2-1,S3-1", "", ""]
    assert c["candidate_entity_ids"].tolist() == ["S2-1,S2-2,S3-1", "S3-9", ""]


def test_writer_asserts_matches_subset_of_candidates(tmp_path):
    """A match that is not a candidate fails the assertion and nothing is written."""
    with pytest.raises(AssertionError):
        check_matches_are_candidates({"S1-1": ["S2-1"]}, {"S1-1": ["S2-2"]})
    with pytest.raises(AssertionError):
        write_results(["S1-1"], {"S1-1": ["S2-1"]}, {}, tmp_path)
    assert not (tmp_path / "matching_results.tsv").exists()


def test_writer_rejects_invalid_ids(tmp_path):
    """Keys outside Source 1, S1- self-matches, non-S2/S3 IDs and unknown IDs are rejected."""
    with pytest.raises(ValueError):
        write_results(["S1-1"], {}, {"S1-9": ["S2-1"]}, tmp_path)  # unknown Source 1 key
    with pytest.raises(ValueError):
        write_results(["S1-1"], {"S1-1": ["S1-1"]}, {"S1-1": ["S1-1"]}, tmp_path)  # self-match
    with pytest.raises(ValueError):
        write_results(["S1-1"], {}, {"S1-1": ["X-1"]}, tmp_path)  # bad prefix
    with pytest.raises(ValueError):
        write_results(["S1-1"], {}, {"S1-1": ["S2-5"]}, tmp_path, valid_ids={"S2-1"})  # not in test set
    with pytest.raises(ValueError):
        write_results(["S1-1", "S1-1"], {}, {}, tmp_path)  # duplicate S1 rows
