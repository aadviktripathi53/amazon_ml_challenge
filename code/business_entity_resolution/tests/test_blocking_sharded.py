"""Sharded, memory-bounded blocking: shard-size invariance, brute-force agreement, budget planning, feature batch."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import src.block as B
from src.tfidf import prune_common


def _records(n_q=30, n_pool=150, seed=0):
    """Synthetic records: names/addresses drawn from a small vocabulary so many pool docs compete for the top-K."""
    rng = np.random.RandomState(seed)
    words = ["acme", "foods", "global", "tech", "blue", "river", "bakery", "sun", "star", "metro", "care", "green"]
    streets = ["oak", "pine", "maple", "cedar", "elm", "lake", "hill", "park"]

    def name():
        return " ".join(rng.choice(words, size=rng.randint(1, 4), replace=True))

    def addr():
        return f"{rng.randint(1, 99)} {rng.choice(streets)} street {rng.choice(['springfield', 'portland', 'denver'])}"

    rows = []
    for i in range(n_q):
        rows.append((f"S1-{i}", "S1", "US", name(), "", "springfield", addr()))
    for src in ("S2", "S3"):
        for i in range(n_pool):
            rows.append((f"{src}-{i}", src, "US", name(), "", rng.choice(["springfield", "portland", "denver"]), addr()))
    rows.append(("S1-other", "S1", "Atlantis", "acme foods", "", "elsewhere", "1 far street elsewhere"))
    rows.append(("S2-other", "S2", "Atlantis", "acme foods", "", "elsewhere", "1 far street elsewhere"))
    df = pd.DataFrame(rows, columns=["entity_id", "source", "country", "name_core", "postcode", "city_token", "addr_norm"])
    return df


@pytest.fixture
def records_path(tmp_path):
    """A small records parquet written in several row groups."""
    path = tmp_path / "records.parquet"
    pq.write_table(pa.Table.from_pandas(_records(), preserve_index=False), path, row_group_size=40)
    return path


def run_partition(path, country, shard_docs, n_jobs=1, query_rows=None):
    """Blocking of one partition through the real sharded code path; returns the candidate frame."""
    idf = B.fit_idf(path, country)
    q = pd.concat([B.to_frame(b) for b in B.iter_partition_batches(path, country, ("S1",), ["entity_id", "country", "source", *B.TEXT_COLUMNS])], ignore_index=True)
    if query_rows is not None:
        q = q.iloc[query_rows].reset_index(drop=True)
    block = B.build_query_block(q, np.zeros(len(q), np.int8), idf)
    states, pool_ids = B.block_query_block(path, country, block, idf, B.Budget(shard_docs, 10 ** 6), n_jobs, "test")
    return B.emit_candidates(block, states, pool_ids, 0, len(block.ids)).to_pandas(), idf, block, states


@pytest.mark.parametrize("shard_docs", [7, 25, 60, 10 ** 6])
def test_result_is_independent_of_shard_size(records_path, monkeypatch, shard_docs):
    """Any shard size (down to a few documents) gives exactly the same candidates, ranks and cosines."""
    monkeypatch.setattr(B, "BATCH_ROWS", 7)  # shards close at 7-row batch boundaries
    reference, *_ = run_partition(records_path, "US", 10 ** 6)
    got, *_ = run_partition(records_path, "US", shard_docs)
    pd.testing.assert_frame_equal(reference.reset_index(drop=True), got.reset_index(drop=True))
    assert len(got) > 0


def test_parallel_workers_give_the_same_result(records_path, monkeypatch):
    """Forked workers over query chunks reproduce the serial result."""
    monkeypatch.setattr(B, "CHUNK_ROWS", 5)
    serial, *_ = run_partition(records_path, "US", 40, n_jobs=1)
    parallel, *_ = run_partition(records_path, "US", 40, n_jobs=3)
    pd.testing.assert_frame_equal(serial.reset_index(drop=True), parallel.reset_index(drop=True))


def test_merged_topk_matches_brute_force(records_path, monkeypatch):
    """The merged per-shard top-K scores equal a dense brute-force top-K over the whole S2 pool (name channel)."""
    monkeypatch.setattr(B, "BATCH_ROWS", 7)
    _, idf, block, states = run_partition(records_path, "US", 20)
    pool = pd.concat([B.to_frame(b) for b in B.iter_partition_batches(records_path, "US", ("S2",), ["entity_id", "source", *B.TEXT_COLUMNS])], ignore_index=True)
    from src.tfidf import hashed_counts, tfidf_weight

    x = tfidf_weight(hashed_counts(B.channel_text(pool, "name")), idf.idf["name"])
    dense = (block.search["name"] @ prune_common(x, idf.df["name"], idf.max_df).T).toarray()
    k = B.CHANNELS["name"]
    expected = -np.sort(-dense, axis=1)[:, :k]
    got = states[("name", "S2")].score
    np.testing.assert_allclose(np.where(got < 0, 0, got), expected, atol=1e-6)  # empty slots (score -1) are zeros in the dense result


def test_candidates_schema_partitioning_and_ids(records_path):
    """Output has the documented schema, only same-partition pool ids, exact cosines and consistent flags."""
    table, *_ = run_partition(records_path, "US", 30)
    assert list(table.columns) == B.CANDIDATE_SCHEMA.names
    assert table["cand_id"].str.startswith(("S2-", "S3-")).all() and not table["cand_id"].eq("S2-other").any()
    assert table["s1_id"].str.startswith("S1-").all() and not table["s1_id"].eq("S1-other").any()
    assert not table.duplicated(["s1_id", "cand_id"]).any()
    assert (table["ch_name"] == table["rank_name"].notna()).all()
    assert table["cos_name"].between(0, 1.0001).all() and (table["block_score"] >= table[["cos_name", "cos_ctx", "cos_addr"]].max(axis=1) - 1e-6).all()
    other, *_ = run_partition(records_path, "Atlantis", 30)
    assert set(other["cand_id"]) == {"S2-other"}  # an unseen country label is just another partition


def test_plan_budget_scales_with_the_memory_budget_and_is_clamped():
    """More memory -> bigger shards; tiny budgets are clamped to a sane minimum."""
    small, big = B.plan_budget(2, 3), B.plan_budget(16, 3)
    assert small.shard_docs < B.plan_budget(8, 3).shard_docs < big.shard_docs
    assert small.query_block < big.query_block
    assert B.plan_budget(0.01, 3).shard_docs == 20_000 and B.plan_budget(1000, 3).shard_docs == 3_000_000
    assert B.plan_budget(8, 1).shard_docs > B.plan_budget(8, 3).shard_docs


def test_feature_batch_values():
    """compute_batch gives sane numbers for a true pair, a missing address and a number conflict."""
    from src.features import FEATURE_COLUMNS, compute_batch
    from src.normalize import normalize_frame

    raw = pd.DataFrame(
        {
            "entity_id": ["S1-1", "S2-1", "S3-1"],
            "business_name": ["Acme Foods LLC", "ACME FOODS", "Acme Foods Inc"],
            "business_address": ["12 Main Street, Springfield, IL", "12 MAIN ST, SPRINGFIELD, IL", ""],
            "country": ["US", "US", "US"],
        }
    )
    rec = normalize_frame(raw)
    cands = pd.DataFrame(
        {"s1_id": ["S1-1", "S1-1"], "cand_id": ["S2-1", "S3-1"], "ch_name": [True, True], "ch_ctx": [True, False], "ch_addr": [True, False],
         "rank_name": [1.0, 2.0], "rank_ctx": [1.0, np.nan], "rank_addr": [1.0, np.nan], "cos_name": [1.0, 1.0], "cos_ctx": [0.9, 0.5],
         "cos_addr": [0.9, 0.0], "block_score": [1.0, 1.0]}
    )
    feats = compute_batch(cands, rec, pd.Index(rec["entity_id"])).set_index("cand_id")
    assert list(feats.columns) == ["s1_id"] + FEATURE_COLUMNS
    assert feats.loc["S2-1", "core_exact"] == 1 and feats.loc["S2-1", "num_shared"] >= 1 and feats.loc["S2-1", "num_conflict"] == 0
    assert feats.loc["S2-1", "ad_tset"] > 0.7 and feats.loc["S2-1", "cand_is_s3"] == 0
    assert feats.loc["S3-1", "addr_empty_cand"] == 1 and feats.loc["S3-1", "ad_ratio"] == 0 and feats.loc["S3-1", "cand_is_s3"] == 1
    assert feats.loc["S3-1", "num_conflict"] == 0  # a missing address is not a number conflict
    assert (feats.drop(columns="s1_id").dtypes == np.float32).all()
