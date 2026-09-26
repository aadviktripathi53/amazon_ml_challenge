"""Decision layer building blocks: isotonic calibration, expected-F0.5 choice per S1, exclusivity.

All functions are pure (arrays in, arrays out) so ``src.decide`` can combine them and tests can check them.

* ``fit_isotonic`` / ``apply_calibration``: isotonic regression of the label on the OUT-OF-FOLD probability, stored as
  its breakpoints (``np.interp`` reproduces ``IsotonicRegression.predict`` exactly), so it can be saved as JSON.
* ``choose_expected_f05``: for every S1, sort its candidates by calibrated probability and pick the prefix length k
  (0..n) with the highest expected F0.5, treating each candidate as an independent Bernoulli(p) "is a true match".
  k = 0 is exact (``prod(1 - p)``); k >= 1 is Monte Carlo (``n_draws`` draws, seeded). Candidates with p < ``min_p``
  are never worth predicting and are left out of the simulation (their chance of being true is ignored).
* ``exclusive_mask``: a candidate selected by several S1s is kept only for the S1 with the highest calibrated
  probability (ties: higher raw probability, then smaller S1 id).

Known blind spot: expected F0.5 only sees the CANDIDATES, so true matches that blocking missed are invisible; it is
therefore somewhat optimistic about "predict something" versus "predict empty" when recall of blocking is < 1.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

BETA2 = 0.25
MC_ELEMENT_BUDGET = 40_000_000  # draws x S1s x max candidates simulated at once (~40 MB of booleans)


def fit_isotonic(prob: np.ndarray, label: np.ndarray) -> Dict[str, list]:
    """Fit isotonic regression label ~ prob (increasing, clipped to [0, 1]) and return its breakpoints.

    Args:
        prob: Out-of-fold raw probabilities.
        label: 0/1 labels.

    Returns:
        ``{"x": [...], "y": [...]}`` breakpoints for ``apply_calibration``.
    """
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(prob.astype(np.float64), label.astype(np.float64))
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def apply_calibration(prob: np.ndarray, calib: Dict[str, list]) -> np.ndarray:
    """Map raw probabilities to calibrated ones (piecewise-linear between the isotonic breakpoints, clipped).

    Args:
        prob: Raw probabilities.
        calib: Output of ``fit_isotonic``.

    Returns:
        float32 calibrated probabilities.
    """
    return np.interp(prob.astype(np.float64), np.asarray(calib["x"]), np.asarray(calib["y"])).astype(np.float32)


def sort_within_s1(codes: np.ndarray, p_cal: np.ndarray, p_raw: np.ndarray) -> np.ndarray:
    """Order of the pairs: by S1 code, then calibrated probability desc, then raw probability desc.

    Args:
        codes: Integer S1 code per pair.
        p_cal: Calibrated probability per pair.
        p_raw: Raw probability per pair (tie-break of isotonic plateaus).

    Returns:
        Permutation array.
    """
    return np.lexsort((-p_raw, -p_cal, codes))


def _mc_block(p: np.ndarray, n_valid: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Monte Carlo expected F0.5 of predicting the top-k (k = 1..width) for a block of S1s.

    Args:
        p: ``(m, width)`` probabilities sorted desc per row, zero-padded.
        n_valid: Number of real candidates per row.
        n_draws: Simulations.
        rng: Random generator (consumed in a fixed order, so results are reproducible).

    Returns:
        ``(m, width)`` expected F0.5 for k = 1..width (``-inf`` where k > n_valid).
    """
    m, width = p.shape
    x = rng.random((n_draws, m, width), dtype=np.float32) < p[None, :, :]  # bool, n_draws x m x width
    tp = np.cumsum(x, axis=2, dtype=np.int16)
    del x
    total = tp[:, :, -1].astype(np.float32)
    k = np.arange(1, width + 1, dtype=np.float32)
    f = np.float32(1 + BETA2) * tp.astype(np.float32)
    del tp
    f /= k[None, None, :] + np.float32(BETA2) * total[:, :, None]
    ef = f.mean(axis=0)
    ef[k[None, :] > n_valid[:, None]] = -np.inf
    return ef


def choose_expected_f05(
    codes: np.ndarray, p_cal: np.ndarray, p_raw: np.ndarray, n_draws: int = 2000, seed: int = 42, min_p: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick, for every S1, the number k of top candidates that maximises expected F0.5.

    Args:
        codes: Integer S1 code per pair (0..n_s1-1).
        p_cal: Calibrated probability per pair.
        p_raw: Raw probability per pair (tie-break when sorting).
        n_draws: Monte Carlo draws for k >= 1.
        seed: RNG seed.
        min_p: Candidates below this probability are never predicted and not simulated.

    Returns:
        ``(keep mask aligned with the input pairs, chosen k per S1 code)``; ties between k values go to the smaller k.
    """
    n_s1 = int(codes.max()) + 1 if len(codes) else 0
    order = sort_within_s1(codes, p_cal, p_raw)
    c, p = codes[order], p_cal[order].astype(np.float64)
    # k = 0: exact probability that none of the candidates is a true match
    log_none = np.bincount(c, weights=np.log1p(-np.clip(p, 0.0, 1.0 - 1e-12)), minlength=n_s1)
    e0 = np.exp(log_none)
    live = p >= min_p
    c_live, p_live = c[live], p[live].astype(np.float32)
    n_live = np.bincount(c_live, minlength=n_s1)
    start = np.concatenate([[0], np.cumsum(n_live)[:-1]])
    pos = np.arange(len(c_live)) - start[c_live]  # rank of each live candidate within its S1
    best_k = np.zeros(n_s1, dtype=np.int32)
    rng = np.random.default_rng(seed)
    s1_live = np.flatnonzero(n_live > 0)
    # group S1s by candidate count so each block is a dense rectangle without much padding
    s1_live = s1_live[np.argsort(n_live[s1_live], kind="stable")]
    widths = n_live[s1_live]  # ascending
    max_cells = max(1, MC_ELEMENT_BUDGET // n_draws)  # rows x width allowed per block
    i = 0
    while i < len(s1_live):
        # largest block whose padded size (rows x width of its WIDEST row, the last one) fits the budget
        remaining = len(s1_live) - i
        m = 1
        while m < remaining and min(2 * m, remaining) * widths[i + min(2 * m, remaining) - 1] <= max_cells:
            m = min(2 * m, remaining)
        lo_m, hi_m = m, min(2 * m, remaining)
        while lo_m < hi_m:  # binary search between m and 2m
            mid = (lo_m + hi_m + 1) // 2
            if mid * widths[i + mid - 1] <= max_cells:
                lo_m = mid
            else:
                hi_m = mid - 1
        m = lo_m
        block = s1_live[i : i + m]
        width = int(widths[i + m - 1])
        m_rows = len(block)
        mat = np.zeros((m_rows, width), dtype=np.float32)
        row_of = np.full(n_s1, -1, dtype=np.int64)
        row_of[block] = np.arange(m_rows)
        sel = row_of[c_live] >= 0
        mat[row_of[c_live[sel]], pos[sel]] = p_live[sel]
        ef = _mc_block(mat, n_live[block], n_draws, rng)
        full = np.concatenate([e0[block][:, None], ef], axis=1)  # column 0 = k 0
        best_k[block] = np.argmax(full, axis=1)  # argmax returns the first maximum -> smaller k on ties
        i += m_rows
    rank = np.arange(len(c)) - np.concatenate([[0], np.cumsum(np.bincount(c, minlength=n_s1))[:-1]])[c]
    keep_sorted = rank < best_k[c]
    keep = np.zeros(len(codes), dtype=bool)
    keep[order] = keep_sorted
    return keep, best_k


def exclusive_mask(s1_id: np.ndarray, cand_id: np.ndarray, p_cal: np.ndarray, p_raw: np.ndarray) -> np.ndarray:
    """Among SELECTED pairs, keep each candidate only for its most probable S1.

    Args:
        s1_id: S1 id per selected pair.
        cand_id: Candidate id per selected pair.
        p_cal: Calibrated probability.
        p_raw: Raw probability (tie-break).

    Returns:
        Boolean mask (True = keep) aligned with the inputs.
    """
    frame = pd.DataFrame({"s1": s1_id, "c": cand_id, "pc": p_cal, "pr": p_raw})
    frame = frame.sort_values(["c", "pc", "pr", "s1"], ascending=[True, False, False, True], kind="mergesort")
    winners = ~frame.duplicated("c", keep="first")
    keep = np.zeros(len(frame), dtype=bool)
    keep[frame.index.to_numpy()[winners.to_numpy()]] = True
    return keep
