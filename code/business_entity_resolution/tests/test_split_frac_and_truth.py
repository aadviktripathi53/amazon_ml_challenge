"""Fractional S1 sampling, chunked ground-truth readers and the environment knobs."""
import pandas as pd
import pytest

from src.config import max_mem_gb, train_s1_frac
from src.io_utils import ground_truth_has_matches, load_ground_truth_subset
from src.split import make_split, sample_s1_fraction


def _s1(n_us=400, n_in=200):
    """S1 frame plus has-match flags (every 10th S1 is a singleton)."""
    rows = [(f"S1-{i}", "n", "a", "US") for i in range(n_us)] + [(f"S1-{n_us + i}", "n", "a", "India") for i in range(n_in)]
    df = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])
    return df, {e: int(e.split("-")[1]) % 10 != 0 for e in df["entity_id"]}


def test_sample_fraction_is_stratified_seeded_and_order_independent():
    """Each (country, singleton) stratum keeps ~frac of its rows; same seed -> same rows; row order is irrelevant."""
    df, has = _s1()
    a = sample_s1_fraction(df, has, 0.25)
    b = sample_s1_fraction(df.sample(frac=1.0, random_state=1), has, 0.25)
    assert list(a["entity_id"]) == list(b["entity_id"])
    assert 0.24 < len(a) / len(df) < 0.26
    for country in ("US", "India"):
        for single in (True, False):
            full = [e for e, c in zip(df["entity_id"], df["country"]) if c == country and (not has[e]) == single]
            kept = [e for e, c in zip(a["entity_id"], a["country"]) if c == country and (not has[e]) == single]
            assert abs(len(kept) - 0.25 * len(full)) <= 1
    assert list(sample_s1_fraction(df, has, 0.25, seed=7)["entity_id"]) != list(a["entity_id"])


def test_sample_fraction_edge_cases():
    """frac=1 returns everything; a tiny fraction still keeps one row of every non-empty stratum."""
    df, has = _s1()
    assert len(sample_s1_fraction(df, has, 1.0)) == len(df)
    tiny = sample_s1_fraction(df, has, 0.0001)
    assert len(tiny) == 4  # US/India x singleton/non-singleton


def test_sampled_fraction_then_80_20_split():
    """The sampled subset is split 80/20 by make_split with matching strata."""
    df, has = _s1(4000, 2000)
    sub = sample_s1_fraction(df, has, 0.1)
    truth_like = {e: (frozenset({"x"}) if has[e] else frozenset()) for e in sub["entity_id"]}
    train, val = make_split(sub, truth_like)
    assert set(train) | set(val) == set(sub["entity_id"]) and not set(train) & set(val)
    assert abs(len(val) / len(sub) - 0.2) < 0.02


def test_truth_readers_are_chunked_and_consistent(tmp_path):
    """Chunked subset loader and singleton flags agree with the file, across chunk boundaries."""
    lines = ["source1_entity_id\tmatched_entity_ids"] + [f"S1-{i}\t" + ("" if i % 3 == 0 else f"S2-{i},S3-{i}") for i in range(10)]
    path = tmp_path / "gt.tsv"
    path.write_text("\n".join(lines) + "\n")
    sub = load_ground_truth_subset(path, ["S1-1", "S1-3", "S1-9", "S1-404"], chunksize=3)
    assert sub == {"S1-1": {"S2-1", "S3-1"}, "S1-3": set(), "S1-9": set()}
    flags = ground_truth_has_matches(path, chunksize=4)
    assert flags["S1-3"] is False or flags["S1-3"] == False  # noqa: E712 (numpy bool)
    assert sum(1 for v in flags.values() if not v) == 4 and len(flags) == 10


def test_env_knobs():
    """ER_MAX_MEM_GB defaults to 8; ER_TRAIN_S1_FRAC defaults to 1.0; invalid values are rejected."""
    assert max_mem_gb({}) == 8.0 and max_mem_gb({"ER_MAX_MEM_GB": "4"}) == 4.0
    assert train_s1_frac({}) == 1.0 and train_s1_frac({"ER_TRAIN_S1_FRAC": "0.05"}) == 0.05
    with pytest.raises(ValueError):
        train_s1_frac({"ER_TRAIN_S1_FRAC": "0"})
    with pytest.raises(ValueError):
        max_mem_gb({"ER_MAX_MEM_GB": "-1"})
