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


def run_partition(path, country, shard_docs, n_jobs=1, query_rows=None, rerank=True):
    """Blocking of one partition through the real sharded code path; returns the candidate frame."""
    idf = B.fit_idf(path, country)
    q = pd.concat([B.to_frame(b) for b in B.iter_partition_batches(path, country, ("S1",), ["entity_id", "country", "source", *B.TEXT_COLUMNS])], ignore_index=True)
    if query_rows is not None:
        q = q.iloc[query_rows].reset_index(drop=True)
    block = B.build_query_block(q, np.zeros(len(q), np.int8), idf)
    states, pool_ids = B.block_query_block(path, country, block, idf, B.Budget(shard_docs, 10 ** 6), n_jobs, "test", rerank=rerank)
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
    """The merged per-shard top-K' (retrieval, before rerank) equals a dense brute-force top-K' over the S2 pool."""
    monkeypatch.setattr(B, "BATCH_ROWS", 7)
    _, idf, block, states = run_partition(records_path, "US", 20, rerank=False)
    pool = pd.concat([B.to_frame(b) for b in B.iter_partition_batches(records_path, "US", ("S2",), ["entity_id", "source", *B.TEXT_COLUMNS])], ignore_index=True)
    from src.tfidf import hashed_counts, tfidf_weight

    x = tfidf_weight(hashed_counts(B.channel_text(pool, "name")), idf.idf["name"])
    dense = (block.search["name"] @ prune_common(x, idf.df["name"], idf.max_df).T).toarray()
    k = B.retrieve_k("name")
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


def _common_trigram_corpus(n_pool=1500):
    """Pool where nearly every S2 name contains 'inc' (its trigrams exceed the cap) plus queries with rare-word names."""
    rng = np.random.RandomState(1)
    rare = ["zorblax", "quentin", "marigold", "vespera", "thornbury", "kestrel", "lumina", "obsidian"]
    rows = [("S1-inc", "S1", "US", "inc inc inc", "", "x", "1 road x")]
    for i in range(20):
        rows.append((f"S1-r{i}", "S1", "US", f"{rng.choice(rare)} {rng.choice(rare)}", "", "x", f"{i} road x"))
    for i in range(n_pool):
        w = rare[i % len(rare)] if i % 3 == 0 else "common"
        rows.append((f"S2-{i}", "S2", "US", f"inc {w} {i % 7}", "", "x", f"{i} road x"))
    rows.append(("S3-0", "S3", "US", "zorblax marigold", "", "x", "5 road x"))
    return pd.DataFrame(rows, columns=["entity_id", "source", "country", "name_core", "postcode", "city_token", "addr_norm"])


def test_zero_survivor_row_gets_candidates_through_the_rare_ngram_fallback(tmp_path, monkeypatch):
    """A name made only of over-cap trigrams ('inc inc inc') had 0 name candidates; the fallback gives it some."""
    path = tmp_path / "records.parquet"
    pq.write_table(pa.Table.from_pandas(_common_trigram_corpus(), preserve_index=False), path, row_group_size=300)
    monkeypatch.setattr(B, "RARE_FALLBACK_N", 0)
    before, idf, block_off, _ = run_partition(path, "US", 400)
    assert idf.max_df == 500 and block_off.n_fallback["name"] == 0
    assert not before[(before.s1_id == "S1-inc") & before.ch_name].shape[0], "premise: no name candidates without the fallback"
    monkeypatch.setattr(B, "RARE_FALLBACK_N", 5)
    after, _, block_on, _ = run_partition(path, "US", 400)
    got = after[(after.s1_id == "S1-inc") & after.ch_name]
    assert block_on.n_fallback["name"] == 1 and len(got) >= 1
    assert (got["cos_name"] > 0).all() and got["cand_id"].str.startswith(("S2-", "S3-")).all()


def test_rows_with_surviving_ngrams_are_unchanged_by_the_fallback(tmp_path, monkeypatch):
    """The fallback is not a general relaxation: every other query gets exactly the same candidates as without it."""
    path = tmp_path / "records.parquet"
    pq.write_table(pa.Table.from_pandas(_common_trigram_corpus(), preserve_index=False), path, row_group_size=300)
    monkeypatch.setattr(B, "RARE_FALLBACK_N", 0)
    off, *_ = run_partition(path, "US", 400)
    monkeypatch.setattr(B, "RARE_FALLBACK_N", 5)
    on, *_ = run_partition(path, "US", 400)
    key = lambda d: d[d.s1_id != "S1-inc"].sort_values(["s1_id", "cand_id"]).reset_index(drop=True)  # noqa: E731
    # name/ctx channel content of the other rows must be identical (extra pool buckets only exist for the fallback row)
    cols = ["s1_id", "cand_id", "ch_name", "ch_ctx", "ch_addr", "rank_name", "rank_ctx", "rank_addr", "cos_name", "cos_ctx", "cos_addr"]
    pd.testing.assert_frame_equal(key(off)[cols], key(on)[cols])


def test_rare_fallback_helper():
    """Only zero-survivor rows change: they get their n_rare rarest buckets; the exempt mask lists exactly those."""
    import scipy.sparse as sp

    from src.tfidf import N_FEATURES, prune_common, rare_fallback

    df = np.zeros(N_FEATURES, dtype=np.int64)
    df[[10, 11, 12, 13]] = [900, 700, 800, 600]  # all above the cap of 500 -> pruned
    df[[20, 21]] = [5, 6]  # rare -> survive
    rows = [[10, 11, 12, 13], [10, 20], [], [12, 13]]  # row 0 and 3: zero survivors; row 1: survivor; row 2: empty
    data, ind, ptr = [], [], [0]
    for r in rows:
        ind += r
        data += [1.0] * len(r)
        ptr.append(len(ind))
    x = sp.csr_matrix((np.array(data, np.float32), np.array(ind), np.array(ptr)), shape=(4, N_FEATURES))
    pruned = prune_common(x, df, 500)
    assert list(np.diff(pruned.indptr)) == [0, 1, 0, 0]
    out, exempt, n = rare_fallback(x, pruned, df, 2)
    assert n == 2  # rows 0 and 3 only
    assert set(out[0].indices) == {13, 11} and set(out[3].indices) == {13, 12}  # the 2 rarest by df (600, 700 / 600, 800)
    assert set(out[1].indices) == {20} and out[2].nnz == 0  # untouched rows
    assert set(np.flatnonzero(exempt)) == {11, 12, 13}
    same, _, n0 = rare_fallback(x, pruned, df, 0)
    assert n0 == 0 and (same != pruned).nnz == 0
    pool = prune_common(x, df, 500, keep=exempt)  # the pool side keeps the exempt buckets
    assert set(pool[0].indices) == {11, 12, 13} | ({10} & set())  # 10 (df 900) is not exempt here


def test_rerank_keeps_k_sorted_by_exact_cosine(records_path):
    """After rerank every (channel, source) has K slots whose rerank key is non-increasing."""
    _, _, _, states = run_partition(records_path, "US", 30)
    for (ch, _src), st in states.items():
        assert st.idx.shape[1] == B.CHANNELS[ch]
        key = np.nan_to_num(st.cos[:, :, B.ENABLED.index(ch)], nan=0.0)
        if ch in B.ADDR_TIEBREAK_CHANNELS:
            key = key + B.ADDR_TIEBREAK_WEIGHT * np.nan_to_num(st.cos[:, :, B.ENABLED.index("addr")], nan=0.0)
        for row_key, row_idx in zip(key, st.idx):
            filled = row_key[row_idx >= 0]
            assert (np.diff(filled) <= 1e-6).all() and (row_idx[len(filled):] == -1).all()  # sorted, empty slots last


def _duplicate_name_corpus(n_dup):
    """One query 'ridgeline' at a Waxahachie address; ``n_dup`` identical names elsewhere; the true record last."""
    rows = [("S1-q", "S1", "US", "ridgeline", "", "waxahachie", "152 north grove boulevard waxahachie tx")]
    streets = ["oak lane", "pine road", "elm court", "maple drive", "cedar way", "birch avenue", "lake street", "hill road"]
    for i in range(n_dup):
        rows.append((f"S2-{i}", "S2", "US", "ridgeline", "", "dallas", f"{100 + i} {streets[i % 8]} dallas tx"))
    rows.append(("S2-true", "S2", "US", "ridgeline", "", "waxahachie", "152 north grove boulevard waxahachie texas"))
    rows.append(("S3-0", "S3", "US", "other name", "", "x", "1 road x"))
    return pd.DataFrame(rows, columns=["entity_id", "source", "country", "name_core", "postcode", "city_token", "addr_norm"])


def test_address_tiebreak_ranks_matching_address_first(tmp_path, monkeypatch):
    """With fewer identical names than K', the one at the query's address is ranked 1st on the name channel."""
    path = tmp_path / "records.parquet"
    pq.write_table(pa.Table.from_pandas(_duplicate_name_corpus(40), preserve_index=False), path, row_group_size=13)
    table, *_ = run_partition(path, "US", 20)
    top = table[table.ch_name].sort_values("rank_name")
    assert top.iloc[0]["cand_id"] == "S2-true" and len(top) <= B.CHANNELS["name"]
    monkeypatch.setattr(B, "ADDR_TIEBREAK_WEIGHT", 0.0)
    plain, *_ = run_partition(path, "US", 20)
    assert plain[plain.ch_name].sort_values("rank_name").iloc[0]["cand_id"] != "S2-true"  # all names tie -> lowest index wins


def test_more_duplicates_than_k_prime_are_recovered_by_ctx(tmp_path):
    """Known limit: with more identical names than K' the name channel cannot see the true one; ctx (name+city) does."""
    path = tmp_path / "records.parquet"
    pq.write_table(pa.Table.from_pandas(_duplicate_name_corpus(3 * B.retrieve_k("name")), preserve_index=False), path, row_group_size=13)
    table, *_ = run_partition(path, "US", 50)
    true_row = table[table.cand_id == "S2-true"]
    assert len(true_row) == 1 and bool(true_row["ch_ctx"].iloc[0])
