"""Stage 2: candidate generation -> data/interim/candidates_{split}.parquet (memory-bounded, sharded).

Run as ``python -m src.block --split <train|val|test>`` from ``code/business_entity_resolution/``.
``train`` also produces ``candidates_val.parquet`` in the same pass over the pool (they share it); ``val`` then only
checks that file exists.

Method (per country partition when true pairs share the country label >= 99.5% of the time, else one global partition):

* three channels on char-3-gram TF-IDF cosine: ``name`` (name_core, top ``K_NAME`` per source), ``ctx``
  (name_core + postcode + city_token, top ``K_CTX``), ``addr`` (addr_norm, top ``K_ADDR``); the union is kept with
  per-channel flags, ranks and EXACT cosines.
* the IDF of every channel is fitted ONCE on the whole partition (S1+S2+S3) in a streaming pass: texts are hashed
  (``HashingVectorizer``, 2**18 buckets) batch by batch and only the document-frequency vector is kept.

Memory bound (``ER_MAX_MEM_GB``, default 8): nothing that grows with the pool is held at once.

* the S2 and S3 pools are streamed from the records parquet and cut into SHARDS of ``shard_docs`` documents (sized from
  the budget, see ``plan_budget``); one shard's float32 sparse matrices are alive at a time;
* for each shard every S1 query chunk is multiplied against it (sparse x sparse, pruned of very common n-grams, never
  dense), the shard's top-K per query is extracted, and it is MERGED into the running global top-K of that
  (channel, source); exact cosines (all channels) are computed only for entries that entered the running top-K;
* the running top-K state is ``n_queries x K`` arrays, so memory depends on the number of queries in a block (also
  bounded by the budget), not on the pool size.

The result is identical for any shard size (unit-tested). Chunks of one shard are independent, so ``ER_N_JOBS=<n>`` runs
them in forked workers that share the shard matrices copy-on-write. One Parquet row group is written per emitted query
chunk; a row group never splits an S1's candidates. Validation S1s are searched against the FULL train pool.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import scipy.sparse as sp

from .cli import parse_split
from .config import INTERIM_DIR, TRAIN_DIR, ensure_dirs, max_mem_gb
from .io_utils import GROUND_TRUTH_SUFFIX
from .normalize import records_path
from .perf import SAMPLE_CAVEAT, peak_rss_mb, save_metrics, stage_timer
from .split import SPLIT_PATH, load_split_ids
from .tfidf import (
    N_FEATURES, document_frequency, hashed_counts, idf_vector, pair_cosine, prune_common, tfidf_weight,
    topk_per_row,
)

K_NAME = 15  # top-K per source on the name channel
K_CTX = 10  # top-K per source on the name+postcode+city channel
K_ADDR = 10  # top-K per source on the address channel (recovers pairs whose NAMES differ: scripts, DBA names)
CHANNELS: Dict[str, int] = {"name": K_NAME, "ctx": K_CTX, "addr": K_ADDR}
ENABLED = [c for c in os.environ.get("ER_CHANNELS", "name,ctx,addr").split(",") if c in CHANNELS]
CHUNK_ROWS = int(os.environ.get("ER_BLOCK_CHUNK", "1000"))  # S1 queries per sparse product
EMIT_ROWS = 2000  # S1 queries per written Parquet row group
BATCH_ROWS = 100_000  # rows per streamed record batch
MAX_DF_FRAC = 0.002  # n-gram pruned from the SEARCH matrices when in > this share of the partition's documents ...
MAX_DF_FLOOR = 500  # ... but never below this many documents
COUNTRY_EQUAL_MIN = 0.995
META_PATH = INTERIM_DIR / "block_meta.json"
TEXT_COLUMNS = ["name_core", "postcode", "city_token", "addr_norm"]
SOURCES = ("S2", "S3")

# Budget model: bytes per document per channel, measured with tracemalloc on the sample (exact + pruned/transposed
# matrices + transients, x2 safety) and the share of the budget given to each consumer.
SHARD_BYTES_PER_DOC_PER_CHANNEL = 1500
QUERY_BYTES_PER_QUERY = 2600  # exact + pruned query matrices + 2 sources x sum(K) x (idx, score, 3 cosines)
SHARD_BUDGET_SHARE = 0.35
QUERY_BUDGET_SHARE = 0.25

CANDIDATE_SCHEMA = pa.schema(
    [
        ("s1_id", pa.string()), ("cand_id", pa.string()), ("country", pa.string()), ("ch_name", pa.bool_()),
        ("ch_ctx", pa.bool_()), ("ch_addr", pa.bool_()), ("rank_name", pa.float32()), ("rank_ctx", pa.float32()),
        ("rank_addr", pa.float32()), ("cos_name", pa.float32()), ("cos_ctx", pa.float32()), ("cos_addr", pa.float32()),
        ("block_score", pa.float32()),
    ]
)


def candidates_path(split: str):
    """Path of the candidate parquet of a split.

    Args:
        split: ``"train"``, ``"val"`` or ``"test"``.

    Returns:
        ``data/interim/candidates_<split>.parquet``.
    """
    return INTERIM_DIR / f"candidates_{split}.parquet"


@dataclass
class Budget:
    """Sizes derived from the memory budget.

    Attributes:
        shard_docs: Pool documents per shard (all enabled channels).
        query_block: S1 queries handled together (their matrices + running top-K state stay in memory).
    """

    shard_docs: int
    query_block: int


def plan_budget(mem_gb: Optional[float] = None, n_channels: Optional[int] = None) -> Budget:
    """Turn the memory budget into a shard size and a query-block size.

    ``SHARD_BUDGET_SHARE`` of the budget goes to one pool shard, ``QUERY_BUDGET_SHARE`` to the query block; the rest
    is headroom for record batches, sparse-product temporaries, Arrow buffers and the OS.

    Args:
        mem_gb: Budget in GB (defaults to ``config.max_mem_gb()``).
        n_channels: Number of enabled channels (defaults to ``len(ENABLED)``).

    Returns:
        The ``Budget`` (shard clamped to 20k..3M documents, query block to 20k..2M queries).
    """
    mem_gb = max_mem_gb() if mem_gb is None else mem_gb
    n_channels = len(ENABLED) if n_channels is None else n_channels
    total = mem_gb * 1024 ** 3
    shard_docs = int(SHARD_BUDGET_SHARE * total / (SHARD_BYTES_PER_DOC_PER_CHANNEL * n_channels))
    query_block = int(QUERY_BUDGET_SHARE * total / QUERY_BYTES_PER_QUERY)
    return Budget(shard_docs=int(np.clip(shard_docs, 20_000, 3_000_000)), query_block=int(np.clip(query_block, 20_000, 2_000_000)))


def country_equality_share() -> Dict[str, object]:
    """Share of true train pairs whose S1 and S2/S3 country labels are identical strings.

    Streams the ground truth in chunks and joins (Arrow hash join) against the ``(entity_id, country)`` columns of
    ``records_train.parquet``, so memory stays small even for millions of pairs. Pairs with an id missing from the
    records are ignored. The result is cached in ``data/interim/block_meta.json`` so val/test runs reuse it.

    Returns:
        ``{"country_equal_share", "n_true_pairs", "partition_by_country"}``.
    """
    rec = pq.read_table(records_path("train"), columns=["entity_id", "country"])
    left = rec.rename_columns(["s1_id", "c1"])
    right = rec.rename_columns(["cand_id", "c2"])
    equal = total = 0
    gt_path = TRAIN_DIR / f"train_{GROUND_TRUTH_SUFFIX}"
    for chunk in pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, quoting=3, chunksize=200_000):
        pairs = chunk.assign(cand_id=chunk["matched_entity_ids"].str.split(",")).explode("cand_id")
        pairs = pairs[pairs["cand_id"].fillna("").str.strip() != ""]
        table = pa.table({"s1_id": pairs["source1_entity_id"].str.strip().to_numpy(), "cand_id": pairs["cand_id"].str.strip().to_numpy()})
        joined = table.join(left, keys="s1_id").join(right, keys="cand_id")
        total += joined.num_rows
        equal += int(pc.sum(pc.equal(joined.column("c1"), joined.column("c2")).cast(pa.int64())).as_py() or 0)
    share = equal / total if total else 1.0
    meta = {"country_equal_share": share, "n_true_pairs": int(total), "partition_by_country": share >= COUNTRY_EQUAL_MIN}
    ensure_dirs()
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return meta


def partition_decision(split: str) -> Dict[str, object]:
    """Decide (once, from train) whether blocking is partitioned by the country string.

    Args:
        split: Current split; ``train`` recomputes and stores the share, val/test read the stored decision
            (recomputing it if the file is missing but train records exist).

    Returns:
        The meta dict from ``country_equality_share``.
    """
    if split != "train" and META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return country_equality_share()


def list_countries(path) -> List[str]:
    """Distinct country strings of a records parquet, read one batch at a time.

    Args:
        path: Records parquet.

    Returns:
        Sorted list of country labels (an open set; nothing is hard-coded).
    """
    seen = set()
    for batch in pq.ParquetFile(path).iter_batches(columns=["country"], batch_size=500_000):
        seen.update(batch.column(0).unique().to_pylist())
    return sorted(seen)


def iter_partition_batches(
    path, country: Optional[str], sources: Sequence[str], columns: Sequence[str], id_filter: Optional[Sequence[str]] = None
) -> Iterator[pa.RecordBatch]:
    """Stream the records of one country partition and some sources as Arrow record batches (predicate push-down).

    Args:
        path: Records parquet.
        country: Country label, or None for all countries.
        sources: Sources to keep (``"S1"``/``"S2"``/``"S3"``).
        columns: Columns to read.
        id_filter: Optional entity ids to keep.

    Returns:
        Iterator of record batches of about ``BATCH_ROWS`` rows (fewer after filtering).
    """
    expr = ds.field("source").isin(list(sources))
    if country is not None:
        expr = expr & (ds.field("country") == country)
    if id_filter is not None:
        expr = expr & ds.field("entity_id").isin(list(id_filter))
    scanner = ds.dataset(str(path), format="parquet").scanner(columns=list(columns), filter=expr, batch_size=BATCH_ROWS)
    yield from scanner.to_batches()


def channel_text(frame: pd.DataFrame, channel: str) -> pd.Series:
    """The text vectorised for one channel: ``name`` (name_core), ``ctx`` (name + postcode + city) or ``addr``.

    Args:
        frame: Records with ``name_core``, ``postcode``, ``city_token``, ``addr_norm``.
        channel: ``"name"``, ``"ctx"`` or ``"addr"``.

    Returns:
        String Series aligned with ``frame``.
    """
    if channel == "name":
        return frame["name_core"]
    if channel == "ctx":
        return (frame["name_core"] + " " + frame["postcode"] + " " + frame["city_token"]).str.strip()
    return frame["addr_norm"]


def to_frame(batch: pa.RecordBatch) -> pd.DataFrame:
    """Arrow record batch -> pandas frame with Arrow-backed strings (compact, no Python string objects).

    Args:
        batch: Record batch.

    Returns:
        DataFrame.
    """
    return batch.to_pandas(types_mapper=pd.ArrowDtype)


@dataclass
class Idf:
    """Whole-partition IDF statistics of every enabled channel.

    Attributes:
        df: Per channel document frequency of each hash bucket.
        idf: Per channel IDF vector (float32).
        n_docs: Number of documents (S1+S2+S3) in the partition.
        max_df: Buckets with ``df > max_df`` are removed from the SEARCH matrices (not from the exact cosines).
        n_pool: Number of pool documents per source.
    """

    df: Dict[str, np.ndarray]
    idf: Dict[str, np.ndarray]
    n_docs: int
    max_df: int
    n_pool: Dict[str, int]


def fit_idf(path, country: Optional[str]) -> Idf:
    """Fit the IDF of every channel on the whole partition in one streaming pass (only df vectors are kept).

    Args:
        path: Records parquet.
        country: Partition label (None = everything).

    Returns:
        The fitted ``Idf``.
    """
    df = {ch: np.zeros(N_FEATURES, dtype=np.int64) for ch in ENABLED}
    n_pool = {s: 0 for s in SOURCES}
    n_docs = 0
    columns = ["source", *TEXT_COLUMNS]
    for batch in iter_partition_batches(path, country, ("S1", "S2", "S3"), columns):
        frame = to_frame(batch)
        n_docs += len(frame)
        for s in SOURCES:
            n_pool[s] += int((frame["source"] == s).sum())
        for ch in ENABLED:
            df[ch] += document_frequency(hashed_counts(channel_text(frame, ch)))
    return Idf(df=df, idf={ch: idf_vector(df[ch], n_docs) for ch in ENABLED}, n_docs=n_docs,
               max_df=max(MAX_DF_FLOOR, int(MAX_DF_FRAC * n_docs)), n_pool=n_pool)


@dataclass
class QueryBlock:
    """A block of S1 queries with their TF-IDF matrices (exact and search-pruned) per channel.

    Attributes:
        ids: S1 ids.
        country: Country label of each query.
        split_code: Index of the output split (0 = first requested split) of each query.
        exact: Per channel full-vector matrix (for exact cosines).
        search: Per channel matrix pruned of very common n-grams (for the sparse product).
    """

    ids: np.ndarray
    country: np.ndarray
    split_code: np.ndarray
    exact: Dict[str, sp.csr_matrix] = field(default_factory=dict)
    search: Dict[str, sp.csr_matrix] = field(default_factory=dict)


def build_query_block(frame: pd.DataFrame, split_code: np.ndarray, idf: Idf) -> QueryBlock:
    """Weight and prune the query matrices of a frame of S1 records.

    Args:
        frame: S1 records (``entity_id``, ``country`` and the text columns).
        split_code: Output split index per row.
        idf: Fitted IDF statistics.

    Returns:
        The ``QueryBlock``.
    """
    block = QueryBlock(ids=frame["entity_id"].to_numpy(dtype=object), country=frame["country"].to_numpy(dtype=object), split_code=split_code)
    for ch in ENABLED:
        x = tfidf_weight(hashed_counts(channel_text(frame, ch)), idf.idf[ch])
        block.exact[ch] = x
        block.search[ch] = prune_common(x, idf.df[ch], idf.max_df)
    return block


@dataclass
class Shard:
    """One slice of the S2 or S3 pool of a partition (float32 sparse matrices per channel).

    Attributes:
        ids: Arrow string array of the pool entity ids of this shard.
        offset: Index (within the partition's pool of this source) of the shard's first document.
        exact: Per channel full-vector matrix ``(n, N_FEATURES)``.
        search_t: Per channel pruned matrix, TRANSPOSED to ``(N_FEATURES, n)`` CSR for ``query @ shard``.
    """

    ids: pa.Array
    offset: int
    exact: Dict[str, sp.csr_matrix]
    search_t: Dict[str, sp.csr_matrix]


def iter_shards(path, country: Optional[str], source: str, shard_docs: int, idf: Idf) -> Iterator[Shard]:
    """Stream the pool of one source into shards of about ``shard_docs`` documents.

    Args:
        path: Records parquet.
        country: Partition label (None = everything).
        source: ``"S2"`` or ``"S3"``.
        shard_docs: Target documents per shard (a shard is closed at the first batch boundary past this size).
        idf: Fitted IDF statistics.

    Returns:
        Iterator of ``Shard`` (only the current one should be kept alive by the caller).
    """
    parts: Dict[str, List[sp.csr_matrix]] = {ch: [] for ch in ENABLED}
    ids: List[pa.Array] = []
    n = offset = 0

    def close() -> Shard:
        """Assemble the shard from the accumulated batches."""
        exact = {ch: sp.vstack(parts[ch], format="csr", dtype=np.float32) if len(parts[ch]) > 1 else parts[ch][0] for ch in ENABLED}
        search_t = {ch: prune_common(exact[ch], idf.df[ch], idf.max_df).T.tocsr() for ch in ENABLED}
        return Shard(ids=pa.concat_arrays(ids), offset=offset, exact=exact, search_t=search_t)

    for batch in iter_partition_batches(path, country, (source,), ["entity_id", "source", *TEXT_COLUMNS]):
        frame = to_frame(batch)
        for ch in ENABLED:
            parts[ch].append(tfidf_weight(hashed_counts(channel_text(frame, ch)), idf.idf[ch]))
        ids.append(batch.column("entity_id").cast(pa.string()))
        n += len(frame)
        if n >= shard_docs:
            yield close()
            offset += n
            parts, ids, n = {ch: [] for ch in ENABLED}, [], 0
    if n:
        yield close()


@dataclass
class TopK:
    """Running global top-K of one (channel, source) for every query of a block, sorted by descending score.

    Attributes:
        idx: ``(n_queries, K)`` int32 index into the partition's pool of the source (-1 = empty slot).
        score: ``(n_queries, K)`` float32 search (pruned) score; -1 marks an empty slot.
        cos: ``(n_queries, K, n_channels)`` float32 EXACT cosines of the pair on every enabled channel.
    """

    idx: np.ndarray
    score: np.ndarray
    cos: np.ndarray


def empty_topk(n_queries: int, k: int) -> TopK:
    """An empty running top-K.

    Args:
        n_queries: Queries in the block.
        k: Slots per query.

    Returns:
        A ``TopK`` with every slot empty.
    """
    return TopK(np.full((n_queries, k), -1, np.int32), np.full((n_queries, k), -1.0, np.float32),
                np.full((n_queries, k, len(ENABLED)), np.nan, np.float32))


_CTX: Optional[Tuple[Dict[str, sp.csr_matrix], Dict[str, sp.csr_matrix]]] = None  # (query search matrices, shard search_t) shared by fork


def _search_range(bounds: Tuple[int, int]) -> Tuple[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Top-K of query rows ``[a, b)`` against the current shard, for every channel.

    Args:
        bounds: ``(a, b)`` query row range.

    Returns:
        ``(a, {channel: (shard-local index [b-a, K] int32 (-1 empty), score [b-a, K] float32 (-1 empty))})``.
    """
    a, b = bounds
    query_search, shard_t = _CTX
    out = {}
    for ch in ENABLED:
        k = CHANNELS[ch]
        row, col, val, rank = topk_per_row((query_search[ch][a:b] @ shard_t[ch]).tocsr(), k)
        idx = np.full((b - a, k), -1, np.int32)
        score = np.full((b - a, k), -1.0, np.float32)
        idx[row, rank - 1] = col
        score[row, rank - 1] = val
        out[ch] = (idx, score)
    return a, out


def search_shard(block: QueryBlock, shard: Shard, n_jobs: int, chunk_rows: Optional[int] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Top-K of every query of the block against one shard, per channel (chunked sparse products).

    Args:
        block: Query block.
        shard: Pool shard.
        n_jobs: Forked worker processes (1 = in-process); results do not depend on it.
        chunk_rows: Queries per sparse product (defaults to ``CHUNK_ROWS``).

    Returns:
        ``{channel: (local idx [n_queries, K], score [n_queries, K])}``.
    """
    global _CTX
    chunk_rows = chunk_rows or CHUNK_ROWS
    n = len(block.ids)
    out = {ch: (np.full((n, CHANNELS[ch]), -1, np.int32), np.full((n, CHANNELS[ch]), -1.0, np.float32)) for ch in ENABLED}
    ranges = [(a, min(a + chunk_rows, n)) for a in range(0, n, chunk_rows)]
    _CTX = (block.search, shard.search_t)
    try:
        if n_jobs <= 1 or len(ranges) < 2:
            results = map(_search_range, ranges)
            for a, res in results:
                for ch, (idx, score) in res.items():
                    out[ch][0][a : a + len(idx)], out[ch][1][a : a + len(idx)] = idx, score
        else:
            with multiprocessing.get_context("fork").Pool(min(n_jobs, len(ranges))) as pool:
                for a, res in pool.imap(_search_range, ranges):
                    for ch, (idx, score) in res.items():
                        out[ch][0][a : a + len(idx)], out[ch][1][a : a + len(idx)] = idx, score
    finally:
        _CTX = None
    return out


def merge_shard_result(state: TopK, new_idx: np.ndarray, new_score: np.ndarray, shard: Shard, block: QueryBlock, pair_chunk: int = 400_000) -> TopK:
    """Merge one shard's top-K into the running global top-K and compute exact cosines for the new entries.

    Old entries come first in the stable sort, so ties keep the earlier (lower-index) document: the merged result equals
    the top-K of the concatenated pool, whatever the shard size.

    Args:
        state: Running top-K of this (channel, source).
        new_idx: Shard-local indices ``(n, K)`` (-1 empty).
        new_score: Scores ``(n, K)`` (-1 empty).
        shard: The shard (for its offset and exact matrices).
        block: Query block (for the exact query matrices).
        pair_chunk: Pairs per exact-cosine computation (bounds temporary memory).

    Returns:
        The merged ``TopK`` (fresh arrays).
    """
    k = state.idx.shape[1]
    glob = np.where(new_idx >= 0, new_idx + shard.offset, -1).astype(np.int32)
    all_idx = np.concatenate([state.idx, glob], axis=1)
    all_score = np.concatenate([state.score, new_score], axis=1)
    order = np.argsort(-all_score, axis=1, kind="stable")[:, :k]
    idx = np.take_along_axis(all_idx, order, axis=1)
    score = np.take_along_axis(all_score, order, axis=1)
    all_cos = np.concatenate([state.cos, np.full((len(idx), k, state.cos.shape[2]), np.nan, np.float32)], axis=1)
    cos = np.take_along_axis(all_cos, order[:, :, None], axis=1)
    rows, cols = np.nonzero((order >= k) & (idx >= 0))  # entries that entered the top-K from this shard
    local = idx[rows, cols] - shard.offset
    for lo in range(0, len(rows), pair_chunk):
        r, c, l = rows[lo : lo + pair_chunk], cols[lo : lo + pair_chunk], local[lo : lo + pair_chunk]
        for ci, ch in enumerate(ENABLED):
            cos[r, c, ci] = pair_cosine(block.exact[ch], r, shard.exact[ch], l)
    return TopK(idx, score, cos)


def emit_candidates(block: QueryBlock, states: Dict[Tuple[str, str], TopK], pool_ids: Dict[str, pa.Array], lo: int, hi: int) -> pa.Table:
    """Union the channels/sources of query rows ``[lo, hi)`` into one candidate table (``CANDIDATE_SCHEMA``).

    Args:
        block: Query block.
        states: Final ``TopK`` per (channel, source).
        pool_ids: Arrow id arrays of the whole pool per source (indexed by the states' idx).
        lo: First query row.
        hi: One past the last query row.

    Returns:
        Arrow table sorted by S1 id, descending ``block_score`` and candidate id.
    """
    frames = []
    for (ch, src), st in states.items():
        idx = st.idx[lo:hi]
        q, slot = np.nonzero(idx >= 0)
        f = {"q": q.astype(np.int32), "src": np.full(len(q), 0 if src == "S2" else 1, np.int8), "idx": idx[q, slot].astype(np.int64),
             f"rank_{ch}": (slot + 1).astype(np.float32)}
        for ci, c2 in enumerate(ENABLED):
            f[f"cos_{c2}"] = st.cos[lo:hi][q, slot, ci]
        frames.append(pd.DataFrame(f))
    allp = pd.concat(frames, ignore_index=True)
    rank_cols = [f"rank_{ch}" for ch in ENABLED]
    cos_cols = [f"cos_{ch}" for ch in ENABLED]
    agg = {**{c: "min" for c in rank_cols}, **{c: "first" for c in cos_cols}}
    pairs = allp.groupby(["q", "src", "idx"], sort=False).agg(agg).reset_index()
    cand = np.empty(len(pairs), dtype=object)
    for code, src in enumerate(SOURCES):
        m = (pairs["src"] == code).to_numpy()
        if m.any():
            cand[m] = pool_ids[src].take(pa.array(pairs.loc[m, "idx"].to_numpy())).to_numpy(zero_copy_only=False)
    q = pairs["q"].to_numpy()
    out = {"s1_id": block.ids[lo + q], "cand_id": cand, "country": block.country[lo + q]}
    for ch in CHANNELS:
        rank = pairs[f"rank_{ch}"].to_numpy(dtype=np.float32) if ch in ENABLED else np.full(len(pairs), np.nan, np.float32)
        out[f"ch_{ch}"] = ~np.isnan(rank)
        out[f"rank_{ch}"] = rank
    for ch in CHANNELS:
        out[f"cos_{ch}"] = pairs[f"cos_{ch}"].to_numpy(dtype=np.float32) if ch in ENABLED else np.full(len(pairs), np.nan, np.float32)
    frame = pd.DataFrame(out)
    frame["block_score"] = frame[[f"cos_{ch}" for ch in ENABLED]].max(axis=1).astype(np.float32)
    frame = frame.sort_values(["s1_id", "block_score", "cand_id"], ascending=[True, False, True], kind="mergesort")
    return pa.Table.from_pandas(frame[CANDIDATE_SCHEMA.names], schema=CANDIDATE_SCHEMA, preserve_index=False)


def block_query_block(path, country: Optional[str], block: QueryBlock, idf: Idf, budget: Budget, n_jobs: int, label: str) -> Tuple[Dict, Dict]:
    """Run all pool shards of both sources against one query block and return the final top-K states.

    Args:
        path: Records parquet.
        country: Partition label.
        block: Query block.
        idf: Fitted IDF statistics.
        budget: Shard sizing.
        n_jobs: Forked workers for the sparse products.
        label: Text used in progress lines.

    Returns:
        ``(states {(channel, source): TopK}, pool_ids {source: Arrow id array of the whole pool})``.
    """
    n = len(block.ids)
    states = {(ch, s): empty_topk(n, CHANNELS[ch]) for ch in ENABLED for s in SOURCES}
    pool_ids: Dict[str, List[pa.Array]] = {s: [] for s in SOURCES}
    for src in SOURCES:
        for i, shard in enumerate(iter_shards(path, country, src, budget.shard_docs, idf)):
            t0 = time.perf_counter()
            res = search_shard(block, shard, n_jobs)
            for ch in ENABLED:
                states[(ch, src)] = merge_shard_result(states[(ch, src)], res[ch][0], res[ch][1], shard, block)
            pool_ids[src].append(shard.ids)
            print(f"block: {label} {src} shard {i} ({len(shard.ids)} docs, offset {shard.offset}) searched by {n} queries in "
                  f"{time.perf_counter() - t0:.1f}s, peak RSS {peak_rss_mb():.0f} MB", flush=True)
            del shard, res
    return states, {s: pa.concat_arrays(v) if v else pa.array([], pa.string()) for s, v in pool_ids.items()}


def run_block(splits: Sequence[str], n_jobs: int = 1, budget: Optional[Budget] = None, path=None) -> Dict[str, object]:
    """Generate candidates for one or more splits that share the pool and write ``candidates_<split>.parquet`` files.

    Args:
        splits: Splits to produce in one pass (``["train", "val"]``, ``["val"]`` or ``["test"]``).
        n_jobs: Forked worker processes for the sparse products.
        budget: Sizing (defaults to ``plan_budget()``).
        path: Records parquet (defaults to the one of the first split); mainly for tests.

    Returns:
        Dict with ``n_pairs``/``n_queries`` per split and the partition decision.
    """
    budget = budget or plan_budget()
    meta = partition_decision(splits[0])
    print(f"block: true pairs with equal country label = {meta['country_equal_share']:.4%} "
          f"({meta['n_true_pairs']} pairs) -> partition by country: {meta['partition_by_country']}")
    print(f"block: memory budget {max_mem_gb():.1f} GB -> shards of {budget.shard_docs} docs, query blocks of {budget.query_block}, channels {ENABLED}")
    path = path or records_path(splits[0])
    id_to_code: Optional[Dict[str, int]] = None
    if splits[0] != "test":
        id_to_code = {i: code for code, sp_name in enumerate(splits) for i in load_split_ids(sp_name)}
    partitions = list_countries(path) if meta["partition_by_country"] else [None]
    ensure_dirs()
    writers = {s: pq.ParquetWriter(candidates_path(s), CANDIDATE_SCHEMA, compression="zstd") for s in splits}
    n_pairs = {s: 0 for s in splits}
    n_queries = {s: 0 for s in splits}
    try:
        for country in partitions:
            label = repr(country)
            t0 = time.perf_counter()
            idf = fit_idf(path, country)
            print(f"block: partition {label}: idf fitted on {idf.n_docs} docs (pool {idf.n_pool}), search cap max_df={idf.max_df}, "
                  f"{time.perf_counter() - t0:.0f}s, peak RSS {peak_rss_mb():.0f} MB", flush=True)
            id_filter = list(id_to_code) if id_to_code is not None else None
            q_batches = [to_frame(b) for b in iter_partition_batches(path, country, ("S1",), ["entity_id", "country", "source", *TEXT_COLUMNS], id_filter)]
            if not q_batches:
                continue
            queries = pd.concat(q_batches, ignore_index=True)
            del q_batches
            codes = queries["entity_id"].astype(object).map(id_to_code).to_numpy(dtype=np.int8) if id_to_code is not None else np.zeros(len(queries), np.int8)
            for lo in range(0, len(queries), budget.query_block):
                sub = queries.iloc[lo : lo + budget.query_block].reset_index(drop=True)
                block = build_query_block(sub, codes[lo : lo + budget.query_block], idf)
                states, pool_ids = block_query_block(path, country, block, idf, budget, n_jobs, label)
                for e in range(0, len(block.ids), EMIT_ROWS):
                    table = emit_candidates(block, states, pool_ids, e, min(e + EMIT_ROWS, len(block.ids)))
                    s1_code = pd.Series(block.split_code[e : e + EMIT_ROWS], index=block.ids[e : e + EMIT_ROWS])
                    row_code = s1_code.reindex(table.column("s1_id").to_pandas()).to_numpy()
                    for code, s in enumerate(splits):
                        part = table.filter(pa.array(row_code == code))
                        if part.num_rows:
                            writers[s].write_table(part)
                            n_pairs[s] += part.num_rows
                for code, s in enumerate(splits):
                    n_queries[s] += int((block.split_code == code).sum())
                del block, states, pool_ids
            print(f"block: partition {label} done in {time.perf_counter() - t0:.0f}s, peak RSS {peak_rss_mb():.0f} MB", flush=True)
    finally:
        for w in writers.values():
            w.close()
    return {"n_pairs": n_pairs, "n_queries": n_queries, **meta}


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point: run blocking for a split and, for train/val, print the blocking-recall report.

    ``train`` also produces the val candidates (same pool pass); ``val`` then only verifies they exist.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    split = parse_split(__doc__.splitlines()[0], argv)
    n_jobs = int(os.environ.get("ER_N_JOBS", "1"))
    if split == "val" and candidates_path("val").exists() and SPLIT_PATH.exists() and candidates_path("val").stat().st_mtime >= SPLIT_PATH.stat().st_mtime:
        print("block: candidates_val.parquet was produced together with the train candidates - nothing to do")
        return
    splits = ["train", "val"] if split == "train" else [split]
    with stage_timer("block", split) as info:
        result = run_block(splits, n_jobs)
        info["pairs"] = sum(result["n_pairs"].values())
        for s in splits:
            print(f"block: {result['n_pairs'][s]} candidate pairs for {result['n_queries'][s]} S1 -> {candidates_path(s)}")
    for s in splits:
        if s != "test":
            from .block_report import report_blocking  # imported here: block_report needs candidates_path from this module

            if "sample" in str(os.environ.get("ER_DATA_DIR", "")):
                print(SAMPLE_CAVEAT)
            report_blocking(s)
        else:
            save_metrics(f"block_{s}", {"n_pairs": result["n_pairs"][s], "n_queries": result["n_queries"][s],
                                        "avg_candidates": result["n_pairs"][s] / max(result["n_queries"][s], 1)})


if __name__ == "__main__":
    main()
