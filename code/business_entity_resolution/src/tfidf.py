"""Streaming char-3-gram TF-IDF helpers for blocking (sparse, chunked, fixed memory).

Vectorising uses ``HashingVectorizer`` (stateless, ``char_wb`` 3-grams, 2**18 buckets) so texts can be
turned into count matrices chunk by chunk with no growing vocabulary. IDF is then computed from the
document frequencies of the (S1+S2+S3) partition and applied with sublinear tf and L2 normalisation
(the classic scikit-learn TF-IDF formula, ``idf = ln((1+N)/(1+df)) + 1``).

``char_wb`` (n-grams inside word boundaries) is used on purpose: swapped word order then yields the same
n-grams. Matrices are float32 CSR; nothing here builds a dense matrix.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

N_FEATURES = 2 ** 18
TEXT_CHUNK = 200_000

_VECTORIZER = HashingVectorizer(
    analyzer="char_wb", ngram_range=(3, 3), n_features=N_FEATURES, alternate_sign=False, norm=None,
    lowercase=False, dtype=np.float32,
)


def hashed_counts(texts: Union[Sequence[str], pd.Series], chunk: int = TEXT_CHUNK) -> sp.csr_matrix:
    """Char 3-gram count matrix of ``texts`` (rows aligned with ``texts``), built in chunks.

    Args:
        texts: Already-normalised strings (list, or a pandas Series whose slices are converted to Python
            strings one chunk at a time so a multi-million-row column is never materialised as objects).
        chunk: Texts vectorised per step.

    Returns:
        float32 CSR matrix of shape ``(len(texts), N_FEATURES)``.
    """
    if len(texts) == 0:
        return sp.csr_matrix((0, N_FEATURES), dtype=np.float32)
    as_list = (lambda i: texts.iloc[i : i + chunk].tolist()) if isinstance(texts, pd.Series) else (lambda i: texts[i : i + chunk])
    parts = [_VECTORIZER.transform(as_list(i)) for i in range(0, len(texts), chunk)]
    return sp.vstack(parts, format="csr", dtype=np.float32) if len(parts) > 1 else parts[0].tocsr()


def document_frequency(*count_matrices: sp.csr_matrix) -> np.ndarray:
    """Number of documents containing each n-gram bucket, summed over several count matrices.

    Args:
        *count_matrices: CSR count matrices with ``N_FEATURES`` columns.

    Returns:
        int64 vector of length ``N_FEATURES``.
    """
    df = np.zeros(N_FEATURES, dtype=np.int64)
    for m in count_matrices:
        df += np.bincount(m.indices, minlength=N_FEATURES)
    return df


def idf_vector(df: np.ndarray, n_docs: int) -> np.ndarray:
    """Smoothed IDF: ``ln((1 + N) / (1 + df)) + 1``.

    Args:
        df: Document frequencies from ``document_frequency``.
        n_docs: Number of documents ``df`` was computed over.

    Returns:
        float32 vector of length ``N_FEATURES``.
    """
    return (np.log((1.0 + n_docs) / (1.0 + df)) + 1.0).astype(np.float32)


def tfidf_weight(counts: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    """Apply sublinear tf (``1 + ln tf``), IDF and row-wise L2 normalisation (all-zero rows stay zero).

    Args:
        counts: CSR count matrix.
        idf: IDF vector from ``idf_vector``.

    Returns:
        New float32 CSR matrix with unit-norm (or zero) rows.
    """
    x = counts.tocsr().copy()
    x.data = ((1.0 + np.log(x.data)) * idf[x.indices]).astype(np.float32)
    sq = x.data.astype(np.float64) ** 2
    row_of = np.repeat(np.arange(x.shape[0]), np.diff(x.indptr))  # row index of every stored value
    norms = np.sqrt(np.bincount(row_of, weights=sq, minlength=x.shape[0]))
    norms[norms == 0] = 1.0
    x.data /= norms[row_of].astype(np.float32)
    return x


def prune_common(x: sp.csr_matrix, df: np.ndarray, max_df: int, keep: Optional[np.ndarray] = None) -> sp.csr_matrix:
    """Zero out n-grams whose document frequency exceeds ``max_df`` (search-time pruning only).

    The weights of the remaining n-grams are left unchanged, so a dot product of two pruned rows is a
    partial sum of the exact cosine that ignores very common n-grams. Bounds the cost of the sparse
    product (a posting list is at most ``max_df`` long) without changing the exact cosines used later.

    Args:
        x: Weighted CSR matrix.
        df: Document frequencies.
        max_df: Maximum allowed document frequency.
        keep: Optional boolean mask over the buckets that are exempt from the cap (used on the POOL side for the
            buckets that zero-survivor query rows fall back to, see ``rare_fallback``).

    Returns:
        New CSR matrix without the pruned entries.
    """
    y = x.copy()
    drop = df[y.indices] > max_df
    if keep is not None:
        drop &= ~keep[y.indices]
    y.data[drop] = 0.0
    y.eliminate_zeros()
    return y


def rare_fallback(x: sp.csr_matrix, pruned: sp.csr_matrix, df: np.ndarray, n_rare: int) -> Tuple[sp.csr_matrix, np.ndarray, int]:
    """Give every query row that lost ALL its n-grams to the frequency cap its ``n_rare`` rarest n-grams back.

    Only rows that have n-grams in ``x`` but none left in ``pruned`` are touched (rows that still have a surviving
    n-gram are returned unchanged). For such a row the ``n_rare`` buckets with the lowest document frequency (ties: lowest
    bucket id) are restored with their original weights. Those buckets exceed the cap by construction, so they are absent
    from the (pruned) pool matrices: the returned mask lists them so the pool side can exempt exactly these buckets.

    Args:
        x: Full weighted query matrix.
        pruned: ``prune_common(x, df, max_df)``.
        df: Document frequencies.
        n_rare: Buckets restored per zero-survivor row (0 disables the fallback).

    Returns:
        ``(pruned matrix with the fallback entries, bool mask over buckets to exempt on the pool side, number of rows)``.
    """
    exempt = np.zeros(x.shape[1], dtype=bool)
    if n_rare <= 0:
        return pruned, exempt, 0
    rows = np.flatnonzero((np.diff(pruned.indptr) == 0) & (np.diff(x.indptr) > 0))
    if len(rows) == 0:
        return pruned, exempt, 0
    r_out, c_out, v_out = [], [], []
    for r in rows:
        lo, hi = x.indptr[r], x.indptr[r + 1]
        cols, vals = x.indices[lo:hi], x.data[lo:hi]
        pick = np.lexsort((cols, df[cols]))[:n_rare]
        r_out.append(np.full(len(pick), r, dtype=np.int64))
        c_out.append(cols[pick])
        v_out.append(vals[pick])
    r_all, c_all, v_all = np.concatenate(r_out), np.concatenate(c_out), np.concatenate(v_out)
    exempt[c_all] = True
    extra = sp.csr_matrix((v_all, (r_all, c_all)), shape=x.shape, dtype=np.float32)
    return (pruned + extra).tocsr(), exempt, len(rows)


def pair_cosine(a: sp.csr_matrix, ia: np.ndarray, b: sp.csr_matrix, ib: np.ndarray) -> np.ndarray:
    """Row-wise dot product of ``a[ia[k]]`` and ``b[ib[k]]`` for every pair ``k`` (cosine for unit rows).

    Args:
        a: CSR matrix of query rows.
        ia: Row indices into ``a``.
        b: CSR matrix of pool rows (same columns as ``a``).
        ib: Row indices into ``b``.

    Returns:
        float32 vector of length ``len(ia)``.
    """
    if len(ia) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(a[ia].multiply(b[ib]).sum(axis=1), dtype=np.float32).ravel()


def topk_per_row(scores: sp.csr_matrix, k: int):
    """Top-``k`` entries of every row of a sparse score matrix (ties broken by lowest column index, deterministically).

    Args:
        scores: CSR matrix ``(n_query, n_pool)`` of similarity scores (zeros are absent).
        k: Number of entries kept per row.

    Returns:
        ``(row, col, score, rank)`` arrays, rank starting at 1 within each row, rows ascending.
    """
    rows, cols, vals, ranks = [], [], [], []
    indptr, indices, data = scores.indptr, scores.indices, scores.data
    for r in range(scores.shape[0]):
        lo, hi = indptr[r], indptr[r + 1]
        if hi == lo:
            continue
        d, c = data[lo:hi], indices[lo:hi]
        if hi - lo > k:
            # deterministic top-k: everything above the k-th score, then the LOWEST column indices among the ties at
            # the boundary (argpartition alone would pick an arbitrary tied subset, making results depend on layout)
            kth = np.partition(d, len(d) - k)[len(d) - k]
            above = np.flatnonzero(d > kth)
            tied = np.flatnonzero(d == kth)
            tied = tied[np.argsort(c[tied], kind="stable")][: k - len(above)]
            keep = np.concatenate([above, tied])
            d, c = d[keep], c[keep]
        order = np.lexsort((c, -d))
        n = len(order)
        rows.append(np.full(n, r, dtype=np.int32))
        cols.append(c[order])
        vals.append(d[order])
        ranks.append(np.arange(1, n + 1, dtype=np.int16))
    if not rows:
        return (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32), np.zeros(0, np.int16))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), np.concatenate(ranks)
