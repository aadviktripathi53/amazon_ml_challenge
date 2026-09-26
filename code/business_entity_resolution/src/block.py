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
* retrieval keeps ``K' = 4 x K`` per (channel, source) by the pruned score; after the last shard the K' are reranked by
  EXACT cosine (name/ctx: + ``ADDR_TIEBREAK_WEIGHT`` x exact address cosine) and the top K are kept;
* the running top-K' state is ``n_queries x K'`` arrays, so memory depends on the number of queries in a block (also
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
from .streaming import isin_sorted
from .tfidf import (
    N_FEATURES, document_frequency, hashed_counts, idf_vector, pair_cosine, prune_common, rare_fallback, tfidf_weight,
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
RARE_FALLBACK_N = int(os.environ.get("ER_RARE_FALLBACK_N", "5"))  # rarest n-grams a zero-survivor query row gets back (0 = off)
RARE_FALLBACK_CHANNELS = ("name", "ctx", "addr")  # channels the fallback applies to
RETRIEVE_MULT = int(os.environ.get("ER_RETRIEVE_MULT", "4"))  # K' = RETRIEVE_MULT x K retrieved by pruned score, reranked to K
RETRIEVE_MULT_BY_CHANNEL = {"name": int(os.environ.get("ER_RETRIEVE_MULT_NAME") or 6)}  # per-channel override (name: 6 x K)
DUP_N = int(os.environ.get("ER_DUP_N", "20"))  # duplicate-name channel: query core names shared by > DUP_N pool docs (0 = off)
K_DUP = int(os.environ.get("ER_K_DUP", "5"))  # per source, ranked by exact address cosine within the same-name cluster
DUP_ON = DUP_N > 0 and "addr" in ENABLED
ACTIVE = ENABLED + (["dup"] if DUP_ON else [])  # channels that produce top-K states
RANK_CHANNELS = [*CHANNELS, "dup"]  # channels with a flag + rank column in the candidate file
ADDR_TIEBREAK_WEIGHT = float(os.environ.get("ER_ADDR_TIEBREAK_W", "0.5"))  # name/ctx rerank key = exact cos + w x exact addr cos
ADDR_TIEBREAK_CHANNELS = ("name", "ctx")
COUNTRY_EQUAL_MIN = 0.995
META_PATH = INTERIM_DIR / "block_meta.json"
TEXT_COLUMNS = ["name_core", "postcode", "city_token", "addr_norm"]
SOURCES = ("S2", "S3")

# Budget model, measured with tracemalloc on the full train data (US partition, name K'=6xK, duplicate-name channel on):
# a pool shard holds ~217 B/doc/channel of matrices (exact + pruned transposed), x~3 for build transients -> 600;
# a query costs ~11.4 KB live (both sources' top-K' states + query matrices) and ~19.8 KB at the merge peak -> 20000.
SHARD_BYTES_PER_DOC_PER_CHANNEL = 600
QUERY_BYTES_PER_QUERY = 20000
SHARD_BUDGET_SHARE = 0.15
QUERY_BUDGET_SHARE = 0.45  # the rest (~40%) is headroom: IDF vectors, the partition's query frame, pool id arrays, OS

CANDIDATE_SCHEMA = pa.schema(
    [
        ("s1_id", pa.string()), ("cand_id", pa.string()), ("country", pa.string()), ("ch_name", pa.bool_()),
        ("ch_ctx", pa.bool_()), ("ch_addr", pa.bool_()), ("rank_name", pa.float32()), ("rank_ctx", pa.float32()),
        ("rank_addr", pa.float32()), ("ch_dup", pa.bool_()), ("rank_dup", pa.float32()), ("cos_name", pa.float32()), ("cos_ctx", pa.float32()), ("cos_addr", pa.float32()),
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


def blocking_config() -> Dict[str, object]:
    """Every setting that changes WHICH candidates are produced (memory/parallelism knobs excluded).

    Stored with the train-side blocking decision; the test side must match it, otherwise the model would score
    candidates generated differently from the ones it was trained on.

    Returns:
        JSON-serialisable settings dict.
    """
    from .normalize import ROMANIZE

    return {
        "channels": list(ENABLED), "k": dict(CHANNELS), "retrieve_mult": RETRIEVE_MULT,
        "retrieve_mult_by_channel": dict(RETRIEVE_MULT_BY_CHANNEL), "addr_tiebreak_weight": ADDR_TIEBREAK_WEIGHT,
        "rare_fallback_n": RARE_FALLBACK_N, "rare_fallback_channels": list(RARE_FALLBACK_CHANNELS),
        "dup_n": DUP_N if DUP_ON else 0, "k_dup": K_DUP, "max_df_frac": MAX_DF_FRAC, "max_df_floor": MAX_DF_FLOOR,
        "romanize": ROMANIZE,
    }


def check_blocking_config(meta: Dict[str, object]) -> None:
    """Fail if the current blocking/normalize settings differ from those stored by the train side.

    Args:
        meta: Contents of ``block_meta.json``.

    Raises:
        ValueError: Listing every differing setting (meta files from before this check carry no config and pass).
    """
    stored = meta.get("config")
    if stored is None:
        return
    current = json.loads(json.dumps(blocking_config()))  # same JSON round-trip as the stored copy
    diff = {k: (stored.get(k), current.get(k)) for k in set(stored) | set(current) if stored.get(k) != current.get(k)}
    if diff:
        raise ValueError(f"blocking config differs from the train side (stored vs current): {diff} - "
                         "use the same ER_* settings as run_train_side.sh or retrain")


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
    meta = {"country_equal_share": share, "n_true_pairs": int(total), "partition_by_country": share >= COUNTRY_EQUAL_MIN,
            "config": blocking_config()}
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
        dup_hashes: Sorted hashes of the core names shared by more than ``DUP_N`` pool documents (empty when off).
    """

    df: Dict[str, np.ndarray]
    idf: Dict[str, np.ndarray]
    n_docs: int
    max_df: int
    n_pool: Dict[str, int]
    dup_hashes: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint64))


def name_hash(names: pd.Series) -> np.ndarray:
    """64-bit hash of each core name (0 for an empty name, which never forms a duplicate cluster).

    Args:
        names: ``name_core`` strings.

    Returns:
        uint64 array aligned with ``names``.
    """
    values = names.to_numpy(dtype=object)
    h = pd.util.hash_array(values, categorize=False)
    h[values == ""] = 0
    return h


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
    pool_hashes: List[np.ndarray] = []
    columns = ["source", *TEXT_COLUMNS]
    for batch in iter_partition_batches(path, country, ("S1", "S2", "S3"), columns):
        frame = to_frame(batch)
        n_docs += len(frame)
        for s in SOURCES:
            n_pool[s] += int((frame["source"] == s).sum())
        for ch in ENABLED:
            df[ch] += document_frequency(hashed_counts(channel_text(frame, ch)))
        if DUP_ON:
            pool_hashes.append(name_hash(frame.loc[(frame["source"] != "S1").to_numpy(), "name_core"]))
    dup_hashes = np.zeros(0, np.uint64)
    if DUP_ON and pool_hashes:
        values, counts = np.unique(np.concatenate(pool_hashes), return_counts=True)
        dup_hashes = values[(counts > DUP_N) & (values != 0)]
    return Idf(df=df, idf={ch: idf_vector(df[ch], n_docs) for ch in ENABLED}, n_docs=n_docs,
               max_df=max(MAX_DF_FLOOR, int(MAX_DF_FRAC * n_docs)), n_pool=n_pool, dup_hashes=dup_hashes)


@dataclass
class QueryBlock:
    """A block of S1 queries with their TF-IDF matrices (exact and search-pruned) per channel.

    Attributes:
        ids: S1 ids.
        country: Country label of each query.
        split_code: Index of the output split (0 = first requested split) of each query.
        exact: Per channel full-vector matrix (for exact cosines).
        search: Per channel matrix pruned of very common n-grams (for the sparse product), plus the rare-n-gram
            fallback entries of rows that lost every n-gram to the cap.
        exempt: Per channel bool mask of the buckets the pool side must NOT prune (the fallback buckets), if any.
        n_fallback: Per channel number of query rows that needed the fallback.
    """

    ids: np.ndarray
    country: np.ndarray
    split_code: np.ndarray
    exact: Dict[str, sp.csr_matrix] = field(default_factory=dict)
    search: Dict[str, sp.csr_matrix] = field(default_factory=dict)
    exempt: Dict[str, np.ndarray] = field(default_factory=dict)
    n_fallback: Dict[str, int] = field(default_factory=dict)
    name_hash: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint64))
    dup_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))


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
        pruned = prune_common(x, idf.df[ch], idf.max_df)
        if ch in RARE_FALLBACK_CHANNELS:  # rows with NO surviving n-gram get their rarest few back; other rows are untouched
            pruned, block.exempt[ch], block.n_fallback[ch] = rare_fallback(x, pruned, idf.df[ch], RARE_FALLBACK_N)
        block.exact[ch] = x
        block.search[ch] = pruned
    if DUP_ON:
        block.name_hash = name_hash(frame["name_core"])
        block.dup_mask = isin_sorted(block.name_hash, idf.dup_hashes)
    return block


@dataclass
class Shard:
    """One slice of the S2 or S3 pool of a partition (float32 sparse matrices per channel).

    Attributes:
        ids: Arrow string array of the pool entity ids of this shard.
        offset: Index (within the partition's pool of this source) of the shard's first document.
        exact: Per channel full-vector matrix ``(n, N_FEATURES)``.
        search_t: Per channel pruned matrix, TRANSPOSED to ``(N_FEATURES, n)`` CSR for ``query @ shard``.
        name_hash: Core-name hash per document (only when the duplicate-name channel is on).
    """

    ids: pa.Array
    offset: int
    exact: Dict[str, sp.csr_matrix]
    search_t: Dict[str, sp.csr_matrix]
    name_hash: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint64))


def iter_shards(path, country: Optional[str], source: str, shard_docs: int, idf: Idf, exempt: Optional[Dict[str, np.ndarray]] = None) -> Iterator[Shard]:
    """Stream the pool of one source into shards of about ``shard_docs`` documents.

    Args:
        path: Records parquet.
        country: Partition label (None = everything).
        source: ``"S2"`` or ``"S3"``.
        shard_docs: Target documents per shard (a shard is closed at the first batch boundary past this size).
        idf: Fitted IDF statistics.
        exempt: Optional per channel mask of buckets that stay in the pruned pool matrices although above the cap
            (the query block's fallback buckets).

    Returns:
        Iterator of ``Shard`` (only the current one should be kept alive by the caller).
    """
    parts: Dict[str, List[sp.csr_matrix]] = {ch: [] for ch in ENABLED}
    ids: List[pa.Array] = []
    hashes: List[np.ndarray] = []
    n = offset = 0

    def close() -> Shard:
        """Assemble the shard from the accumulated batches."""
        exact = {ch: sp.vstack(parts[ch], format="csr", dtype=np.float32) if len(parts[ch]) > 1 else parts[ch][0] for ch in ENABLED}
        search_t = {ch: prune_common(exact[ch], idf.df[ch], idf.max_df, (exempt or {}).get(ch)).T.tocsr() for ch in ENABLED}
        return Shard(ids=pa.concat_arrays(ids), offset=offset, exact=exact, search_t=search_t,
                     name_hash=np.concatenate(hashes) if hashes else np.zeros(0, np.uint64))

    for batch in iter_partition_batches(path, country, (source,), ["entity_id", "source", *TEXT_COLUMNS]):
        frame = to_frame(batch)
        for ch in ENABLED:
            parts[ch].append(tfidf_weight(hashed_counts(channel_text(frame, ch)), idf.idf[ch]))
        ids.append(batch.column("entity_id").cast(pa.string()))
        if DUP_ON:
            hashes.append(name_hash(frame["name_core"]))
        n += len(frame)
        if n >= shard_docs:
            yield close()
            offset += n
            parts, ids, hashes, n = {ch: [] for ch in ENABLED}, [], [], 0
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


def final_k(channel: str) -> int:
    """Number of candidates finally kept per (channel, source).

    Args:
        channel: Channel name (``dup`` included).

    Returns:
        K.
    """
    return K_DUP if channel == "dup" else CHANNELS[channel]


def retrieve_k(channel: str) -> int:
    """Number of candidates kept per (channel, source) during the shard merge.

    ``K' = multiplier x K`` (``ER_RETRIEVE_MULT``, overridable per channel, e.g. ``ER_RETRIEVE_MULT_NAME``); the
    duplicate-name channel is already scored exactly, so it keeps ``K_DUP``.

    Args:
        channel: Channel name.

    Returns:
        K'.
    """
    if channel == "dup":
        return K_DUP
    return RETRIEVE_MULT_BY_CHANNEL.get(channel, RETRIEVE_MULT) * CHANNELS[channel]


def rerank_topk(state: TopK, channel: str) -> TopK:
    """Rerank the K' retrieved entries of one (channel, source) by EXACT similarity and keep the top K.

    Key: the channel's exact cosine (full, unpruned vectors); for the ``name`` and ``ctx`` channels plus
    ``ADDR_TIEBREAK_WEIGHT`` x the exact address cosine (0 when either address is empty or the address channel is off),
    so that among many identical names the one at the matching address ranks first. Ties keep the retrieval order.

    Args:
        state: Running top-K' after all shards (every entry already has exact cosines).
        channel: Channel name.

    Returns:
        ``TopK`` with K slots per query, sorted by the rerank key (empty slots, idx -1, last).
    """
    k = final_k(channel)
    own = "addr" if channel == "dup" else channel  # the duplicate-name channel ranks by address similarity
    key = np.nan_to_num(state.cos[:, :, ENABLED.index(own)], nan=0.0).astype(np.float32)
    if channel in ADDR_TIEBREAK_CHANNELS and "addr" in ENABLED and ADDR_TIEBREAK_WEIGHT:
        key = key + ADDR_TIEBREAK_WEIGHT * np.nan_to_num(state.cos[:, :, ENABLED.index("addr")], nan=0.0)
    key = np.where(state.idx >= 0, key, -np.inf)
    order = np.argsort(-key, axis=1, kind="stable")[:, :k]
    return TopK(np.take_along_axis(state.idx, order, axis=1), np.take_along_axis(state.score, order, axis=1),
                np.take_along_axis(state.cos, order[:, :, None], axis=1))


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
        k = retrieve_k(ch)
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
    out = {ch: (np.full((n, retrieve_k(ch)), -1, np.int32), np.full((n, retrieve_k(ch)), -1.0, np.float32)) for ch in ENABLED}
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


def search_dup(block: QueryBlock, shard: Shard, pair_chunk: int = 2_000_000) -> Tuple[np.ndarray, np.ndarray]:
    """Duplicate-name channel for one shard: for queries whose core name is shared by > ``DUP_N`` pool docs, score
    every shard document with EXACTLY the same core name by exact address cosine and keep the top ``K_DUP``.

    Resolves crowding among identical names ('Ridgeline Inc' x 412) by locality instead of by n-gram luck.

    Args:
        block: Query block (with ``name_hash`` and ``dup_mask``).
        shard: Pool shard (with ``name_hash``).
        pair_chunk: Pairs per exact-cosine computation.

    Returns:
        ``(shard-local idx [n, K_DUP] (-1 empty), score [n, K_DUP] (-1 empty))``.
    """
    n = len(block.ids)
    idx = np.full((n, K_DUP), -1, np.int32)
    score = np.full((n, K_DUP), -1.0, np.float32)
    qm = np.flatnonzero(block.dup_mask)
    if len(qm) == 0:
        return idx, score
    wanted = np.unique(block.name_hash[qm])
    sel = np.flatnonzero(isin_sorted(shard.name_hash, wanted))
    if len(sel) == 0:
        return idx, score
    pairs = pd.DataFrame({"h": block.name_hash[qm], "q": qm}).merge(pd.DataFrame({"h": shard.name_hash[sel], "l": sel}), on="h")
    q, l = pairs["q"].to_numpy(np.int64), pairs["l"].to_numpy(np.int64)
    s = np.empty(len(q), np.float32)
    for lo in range(0, len(q), pair_chunk):
        s[lo : lo + pair_chunk] = pair_cosine(block.exact["addr"], q[lo : lo + pair_chunk], shard.exact["addr"], l[lo : lo + pair_chunk])
    order = np.lexsort((l, -s, q))
    q, l, s = q[order], l[order], s[order]
    rank = np.arange(len(q)) - np.searchsorted(q, q, side="left")
    keep = rank < K_DUP
    idx[q[keep], rank[keep]] = l[keep]
    score[q[keep], rank[keep]] = s[keep]
    return idx, score


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
    rank_cols = [f"rank_{ch}" for ch in ACTIVE]
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
    for ch in RANK_CHANNELS:
        rank = pairs[f"rank_{ch}"].to_numpy(dtype=np.float32) if ch in ACTIVE else np.full(len(pairs), np.nan, np.float32)
        out[f"ch_{ch}"] = ~np.isnan(rank)
        out[f"rank_{ch}"] = rank
    for ch in CHANNELS:
        out[f"cos_{ch}"] = pairs[f"cos_{ch}"].to_numpy(dtype=np.float32) if ch in ENABLED else np.full(len(pairs), np.nan, np.float32)
    frame = pd.DataFrame(out)
    frame["block_score"] = frame[[f"cos_{ch}" for ch in ENABLED]].max(axis=1).astype(np.float32)
    frame = frame.sort_values(["s1_id", "block_score", "cand_id"], ascending=[True, False, True], kind="mergesort")
    return pa.Table.from_pandas(frame[CANDIDATE_SCHEMA.names], schema=CANDIDATE_SCHEMA, preserve_index=False)


def block_query_block(path, country: Optional[str], block: QueryBlock, idf: Idf, budget: Budget, n_jobs: int, label: str,
                      rerank: bool = True) -> Tuple[Dict, Dict]:
    """Run all pool shards of both sources against one query block and return the final top-K states.

    Args:
        path: Records parquet.
        country: Partition label.
        block: Query block.
        idf: Fitted IDF statistics.
        budget: Shard sizing.
        n_jobs: Forked workers for the sparse products.
        label: Text used in progress lines.
        rerank: Rerank the retrieved K' to the final K by exact similarity (``rerank_topk``); False returns the raw
            top-K' by pruned score (used by tests).

    Returns:
        ``(states {(channel, source): TopK}, pool_ids {source: Arrow id array of the whole pool})``.
    """
    n = len(block.ids)
    states = {(ch, s): empty_topk(n, retrieve_k(ch)) for ch in ACTIVE for s in SOURCES}
    pool_ids: Dict[str, List[pa.Array]] = {s: [] for s in SOURCES}
    for src in SOURCES:
        for i, shard in enumerate(iter_shards(path, country, src, budget.shard_docs, idf, block.exempt)):
            t0 = time.perf_counter()
            res = search_shard(block, shard, n_jobs)
            for ch in ENABLED:
                states[(ch, src)] = merge_shard_result(states[(ch, src)], res[ch][0], res[ch][1], shard, block)
            if DUP_ON:
                d_idx, d_score = search_dup(block, shard)
                states[("dup", src)] = merge_shard_result(states[("dup", src)], d_idx, d_score, shard, block)
            pool_ids[src].append(shard.ids)
            print(f"block: {label} {src} shard {i} ({len(shard.ids)} docs, offset {shard.offset}) searched by {n} queries in "
                  f"{time.perf_counter() - t0:.1f}s, peak RSS {peak_rss_mb():.0f} MB", flush=True)
            del shard, res
    if rerank:
        states = {(ch, s): rerank_topk(st, ch) for (ch, s), st in states.items()}
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
    check_blocking_config(meta)
    print(f"block: true pairs with equal country label = {meta['country_equal_share']:.4%} "
          f"({meta['n_true_pairs']} pairs) -> partition by country: {meta['partition_by_country']}")
    print(f"block: memory budget {max_mem_gb():.1f} GB -> shards of {budget.shard_docs} docs, query blocks of {budget.query_block}, channels {ENABLED}, "
          f"K'={RETRIEVE_MULT}xK (name {RETRIEVE_MULT_BY_CHANNEL['name']}xK) reranked by exact cosine (+{ADDR_TIEBREAK_WEIGHT} x addr cosine for {list(ADDR_TIEBREAK_CHANNELS)}), "
          f"duplicate-name channel {'N>' + str(DUP_N) + ', K=' + str(K_DUP) if DUP_ON else 'off'}")
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
                print(f"block: {label} rare-n-gram fallback rows (of {len(sub)}): {block.n_fallback}"
                      + (f"; duplicate-name queries (cluster > {DUP_N}): {int(block.dup_mask.sum())}" if DUP_ON else ""), flush=True)
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
