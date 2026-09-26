"""Calibration, expected-F0.5 choice and exclusivity (src.decision)."""
import itertools

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from src.decision import apply_calibration, choose_expected_f05, exclusive_mask, fit_isotonic


def exact_expected_f05(p, k):
    """Brute-force expected F0.5 of predicting the top-k of sorted probabilities p (independent Bernoullis)."""
    total = 0.0
    for outcome in itertools.product([0, 1], repeat=len(p)):
        prob = np.prod([pi if o else 1 - pi for pi, o in zip(p, outcome)])
        n_true, tp = sum(outcome), sum(outcome[:k])
        f = (1.0 if k == 0 else 0.0) if n_true == 0 else (0.0 if k == 0 else 1.25 * tp / (k + 0.25 * n_true))
        total += prob * f
    return total


def test_isotonic_breakpoints_reproduce_sklearn():
    """np.interp on the stored breakpoints equals IsotonicRegression.predict, and the map is monotone."""
    rng = np.random.RandomState(0)
    p = rng.rand(2000)
    y = (rng.rand(2000) < p ** 2).astype(int)
    calib = fit_isotonic(p, y)
    ref = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
    grid = np.linspace(-0.1, 1.1, 500)
    np.testing.assert_allclose(apply_calibration(grid, calib), ref.predict(grid), atol=1e-6)
    assert (np.diff(apply_calibration(np.sort(p), calib)) >= -1e-7).all()


@pytest.mark.parametrize("probs", [[0.9, 0.8, 0.1], [0.55, 0.5, 0.45, 0.4], [0.2, 0.15], [0.97], [0.6, 0.6, 0.6, 0.05, 0.02]])
def test_chosen_k_matches_brute_force_expectation(probs):
    """The Monte Carlo choice is (near-)optimal against the exact expectation for small n."""
    codes = np.zeros(len(probs), dtype=np.int64)
    p = np.array(probs, dtype=np.float32)
    keep, best_k = choose_expected_f05(codes, p, p, n_draws=20000, min_p=0.0)
    exact = [exact_expected_f05(sorted(probs, reverse=True), k) for k in range(len(probs) + 1)]
    assert exact[best_k[0]] >= max(exact) - 0.01  # MC noise may pick a near-tie
    assert keep.sum() == best_k[0]
    assert set(np.flatnonzero(keep)) == set(np.argsort(-p, kind="stable")[: best_k[0]])  # keeps the TOP-k


def test_k0_is_exact_product_and_low_probability_s1s_predict_empty():
    """An S1 whose candidates are all unlikely predicts nothing (prod(1-p) wins); a confident one predicts."""
    codes = np.array([0, 0, 0, 1, 1])
    p = np.array([0.05, 0.03, 0.01, 0.95, 0.9], dtype=np.float32)
    keep, best_k = choose_expected_f05(codes, p, p)
    assert best_k[0] == 0 and best_k[1] == 2
    assert list(keep) == [False, False, False, True, True]


def test_choice_is_reproducible_and_order_independent():
    """Same seed -> same answer; shuffling the input rows does not change which pairs are kept."""
    rng = np.random.RandomState(3)
    codes = np.repeat(np.arange(300), 5)
    p = rng.beta(0.5, 0.5, size=len(codes)).astype(np.float32)
    k1, b1 = choose_expected_f05(codes, p, p)
    k2, b2 = choose_expected_f05(codes, p, p)
    assert (k1 == k2).all() and (b1 == b2).all()
    perm = rng.permutation(len(codes))
    k3, b3 = choose_expected_f05(codes[perm], p[perm], p[perm])
    assert (b3 == b1).mean() > 0.97  # blocks are drawn in a different row order; only near-ties may flip
    assert (k3[np.argsort(perm)] == k1).mean() > 0.97


def test_exclusivity_keeps_the_most_probable_s1():
    """A candidate claimed twice stays only with the higher calibrated (then raw) probability."""
    s1 = np.array(["S1-a", "S1-b", "S1-a", "S1-c", "S1-d"])
    cand = np.array(["S2-1", "S2-1", "S3-9", "S3-5", "S3-5"])
    p_cal = np.array([0.8, 0.9, 0.7, 0.6, 0.6], dtype=np.float32)
    p_raw = np.array([0.8, 0.9, 0.7, 0.61, 0.65], dtype=np.float32)
    keep = exclusive_mask(s1, cand, p_cal, p_raw)
    assert list(keep) == [False, True, True, False, True]


def test_iter_s1_chunks_never_splits_an_s1(tmp_path):
    """Streaming the prediction file yields every row once, in order, with each S1 in exactly one chunk."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.decide import iter_s1_chunks

    rng = np.random.RandomState(0)
    s1 = np.repeat([f"S1-{i:03d}" for i in range(50)], rng.randint(1, 9, size=50))
    path = tmp_path / "p.parquet"
    pq.write_table(pa.table({"s1_id": s1, "cand_id": [f"S2-{i}" for i in range(len(s1))], "prob": rng.rand(len(s1)).astype(np.float32)}), path, row_group_size=7)
    chunks = list(iter_s1_chunks(path, chunk_pairs=10))
    assert [c for ch in chunks for c in ch["cand_id"]] == [f"S2-{i}" for i in range(len(s1))]
    seen = [set(ch["s1_id"]) for ch in chunks]
    assert all(not (a & b) for i, a in enumerate(seen) for b in seen[i + 1 :])


def test_hybrid_never_flips_empty_to_predict(monkeypatch):
    """Hybrid: an S1 the flat rule leaves empty stays empty; an S1 it predicts for gets the expected-F0.5 top-k."""
    import pandas as pd

    from src.decide import decide_frame

    calib = {"x": [0.0, 1.0], "y": [0.0, 1.0]}  # identity calibration
    preds = pd.DataFrame({"s1_id": ["a", "a", "b", "b", "b"], "cand_id": list("vwxyz"), "prob": [0.55, 0.5, 0.95, 0.9, 0.5]})
    keep_h, flat, _ = decide_frame(preds, calib, 0.65, "hybrid")
    keep_e, _, _ = decide_frame(preds, calib, 0.65, "expf")
    assert not keep_h[:2].any() and keep_e[:2].any()  # 'a': flat says empty; expf alone would predict
    assert keep_h[2] and keep_h[3] and list(flat) == [False, False, True, True, False]


def test_monte_carlo_blocks_respect_the_element_budget(monkeypatch):
    """Each simulated block (draws x rows x widest row) stays within MC_ELEMENT_BUDGET, and every S1 is decided."""
    import src.decision as D

    sizes = []
    orig = D._mc_block

    def spy(p, n_valid, n_draws, rng):
        sizes.append(n_draws * p.size)
        return orig(p, n_valid, n_draws, rng)

    monkeypatch.setattr(D, "_mc_block", spy)
    monkeypatch.setattr(D, "MC_ELEMENT_BUDGET", 200 * 60)
    rng = np.random.RandomState(1)
    n_per = rng.choice([1, 1, 1, 2, 3, 11, 30], size=400)
    codes = np.repeat(np.arange(400), n_per)
    p = rng.rand(len(codes)).astype(np.float32)
    keep, best_k = D.choose_expected_f05(codes, p, p, n_draws=200)
    assert max(sizes) <= 200 * 60 or max(sizes) == 200 * 30  # a single 30-wide row may exceed a tiny budget alone
    assert len(best_k) == 400 and (np.bincount(codes[keep], minlength=400) == best_k).all()
