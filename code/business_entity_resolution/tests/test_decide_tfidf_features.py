"""Vectorised F0.5 == evaluate.py; TF-IDF helpers; blocking and feature functions on tiny in-memory data."""
import numpy as np
import pandas as pd
import pytest

from src.decide import best_threshold, macro_f05_vectorised, sweep_thresholds
from src.evaluate import fbeta_for_entity, macro_fbeta
from src.tfidf import (
    document_frequency, hashed_counts, idf_vector, pair_cosine, prune_common, tfidf_weight, topk_per_row,
)


def test_worked_example_from_the_problem_statement():
    """Predicted 3, true 2, 2 correct -> 0.714 (also through the vectorised closed form)."""
    assert fbeta_for_entity({"a", "b", "c"}, {"a", "b"}) == pytest.approx(0.7142857, abs=1e-6)
    f = macro_f05_vectorised(np.array([0, 0, 0]), np.array([True, True, True]), np.array([1, 1, 0]), np.array([2.0]))
    assert f == pytest.approx(0.7142857, abs=1e-6)


def test_vectorised_macro_f05_equals_evaluate_module():
    """Random predictions: the closed form matches src.evaluate.macro_fbeta exactly, singletons included."""
    rng = np.random.RandomState(42)
    n_s1 = 60
    truth, cand_rows = {}, []
    for s in range(n_s1):
        true = {f"c{s}_{j}" for j in range(rng.choice([0, 0, 1, 2, 4]))}
        truth[f"s{s}"] = true
        pool = sorted(true) + [f"x{s}_{j}" for j in range(rng.randint(0, 5))]
        rng.shuffle(pool)
        cand_rows += [(s, c) for c in pool]
    codes = np.array([r[0] for r in cand_rows])
    label = np.array([int(c in truth[f"s{s}"]) for s, c in cand_rows])
    probs = rng.rand(len(cand_rows)) * 0.5 + label * 0.5
    n_true = np.array([len(truth[f"s{s}"]) for s in range(n_s1)], dtype=float)
    for t in (0.3, 0.5, 0.8):
        keep = probs >= t
        preds = {f"s{s}": {c for (s2, c), k in zip(cand_rows, keep) if k and s2 == s} for s in range(n_s1)}
        assert macro_f05_vectorised(codes, keep, label, n_true) == pytest.approx(macro_fbeta(preds, truth), abs=1e-12)


def test_singleton_edge_cases_and_threshold_choice():
    """A singleton with no kept pair scores 1, with any kept pair 0; ties choose the higher threshold."""
    codes, label, n_true = np.array([0, 1]), np.array([0, 0]), np.array([0.0, 0.0])
    assert macro_f05_vectorised(codes, np.array([False, False]), label, n_true) == 1.0
    assert macro_f05_vectorised(codes, np.array([True, False]), label, n_true) == 0.5
    sweep = sweep_thresholds(np.array([0.1, 0.1]), codes, label, n_true)
    assert best_threshold(sweep)[0] == 0.95  # every threshold scores 1.0 -> the most precise one wins


def test_tfidf_cosine_and_topk():
    """Identical texts have cosine 1, unrelated ~0; per-row top-k returns ranks 1..k by score."""
    texts = ["acme foods", "acme foods", "zzz qqq", "acme food"]
    counts = hashed_counts(texts)
    idf = idf_vector(document_frequency(counts), len(texts))
    x = tfidf_weight(counts, idf)
    cos = pair_cosine(x, np.array([0, 0, 0]), x, np.array([1, 2, 3]))
    assert cos[0] == pytest.approx(1.0, abs=1e-5) and cos[1] < 0.05 and 0.5 < cos[2] < 1.0
    scores = (x[:1] @ x.T).tocsr()
    row, col, val, rank = topk_per_row(scores, 2)
    assert list(rank) == [1, 2] and set(col) <= {0, 1, 3} and val[0] >= val[1]
    empty = tfidf_weight(hashed_counts([""]), idf)
    assert empty.nnz == 0  # an empty text stays an all-zero row


def test_prune_common_keeps_exact_weights_of_rare_ngrams():
    """Pruning removes only n-grams above the df cap and leaves the other weights untouched."""
    counts = hashed_counts(["aaa bbb", "aaa ccc", "aaa ddd"])
    df = document_frequency(counts)
    x = tfidf_weight(counts, idf_vector(df, 3))
    pruned = prune_common(x, df, max_df=1)
    assert 0 < pruned.nnz < x.nnz
    assert pruned.data.max() <= x.data.max()


def _tiny_records():
    """Three sources, one country, one true triple plus distractors."""
    rows = [
        ("S1-1", "S1", "US", "acme foods", "", "springfield", "12 main street springfield il"),
        ("S1-2", "S1", "US", "zenith robotics", "", "portland", "9 oak avenue portland or"),
        ("S2-1", "S2", "US", "acme foods", "", "springfield", "12 main street springfield il"),
        ("S2-2", "S2", "US", "blue river bakery", "", "denver", "77 pine road denver co"),
        ("S3-1", "S3", "US", "acme food", "", "springfield", "12 main st springfield illinois"),
        ("S3-2", "S3", "US", "unrelated name", "", "boston", "1 far street boston ma"),
    ]
    return pd.DataFrame(rows, columns=["entity_id", "source", "country", "name_core", "postcode", "city_token", "addr_norm"])


def test_blocking_finds_true_pairs_and_writes_the_schema():
    """build_partition + search_chunk retrieve the true S2/S3 records with the documented columns."""
    from src.block import CANDIDATE_SCHEMA, build_partition, search_chunk

    part = build_partition(_tiny_records(), None, "US")
    table = search_chunk(part, 0, len(part.s1_ids)).to_pandas()
    assert list(table.columns) == CANDIDATE_SCHEMA.names
    found = set(zip(table["s1_id"], table["cand_id"]))
    assert ("S1-1", "S2-1") in found and ("S1-1", "S3-1") in found
    assert not (table["s1_id"] == "S1-x").any()
    row = table[(table["s1_id"] == "S1-1") & (table["cand_id"] == "S2-1")].iloc[0]
    assert row["ch_name"] and row["cos_name"] > 0.99 and row["block_score"] > 0.99


def test_blocking_query_restriction_keeps_the_full_pool():
    """Restricting the queries (validation S1s) does not shrink the S2/S3 pool."""
    from src.block import build_partition

    part = build_partition(_tiny_records(), {"S1-2"}, "US")
    assert list(part.s1_ids) == ["S1-2"] and len(part.pool_ids) == 4


def test_feature_batch_values():
    """compute_batch gives sane numbers for a true pair, a conflicting-number pair and empty addresses."""
    from src.block import build_partition, search_chunk
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
    part = build_partition(rec, None, "US")
    cands = search_chunk(part, 0, 1).to_pandas()
    feats = compute_batch(cands, rec, pd.Index(rec["entity_id"])).set_index("cand_id")
    assert list(feats.columns) == ["s1_id"] + FEATURE_COLUMNS
    assert feats.loc["S2-1", "core_exact"] == 1 and feats.loc["S2-1", "num_shared"] >= 1 and feats.loc["S2-1", "num_conflict"] == 0
    assert feats.loc["S2-1", "ad_tset"] > 0.7 and feats.loc["S2-1", "cand_is_s3"] == 0
    assert feats.loc["S3-1", "addr_empty_cand"] == 1 and feats.loc["S3-1", "ad_ratio"] == 0 and feats.loc["S3-1", "cand_is_s3"] == 1
    assert feats.loc["S3-1", "num_conflict"] == 0  # a missing address is not a number conflict
    assert (feats.drop(columns="s1_id").dtypes == np.float32).all()
