"""Stage 2: candidate generation -> data/interim/candidates_{split}.parquet.

Run as ``python -m src.block --split <train|val|test>`` from ``code/business_entity_resolution/``.

Method (per country partition when true pairs share the country label >= 99.5% of the time, else one global
partition):

* channel ``name``: char-3-gram TF-IDF cosine on ``name_core``; top ``K_NAME`` from S2 and from S3.
* channel ``ctx``:  same on ``name_core + ' ' + postcode + ' ' + city_token``; top ``K_CTX`` from S2 and S3.
* the union of the enabled channels is kept with per-channel flags, ranks and EXACT cosines.

Scaling (never a dense all-pairs matrix):

* S1 queries are processed in chunks of ``CHUNK_ROWS`` rows: ``Q_chunk @ pool.T`` is a sparse product whose
  size is bounded by the (pruned) posting lists, top-K is extracted per row, then the chunk is discarded.
* n-grams occurring in more than ``max_df`` documents of the partition are removed from the SEARCH matrices only
  (cost bound); the stored cosines are exact and use the full vectors.
* chunks are independent, so ``ER_N_JOBS=<n>`` runs them in forked worker processes sharing the matrices copy-on-write.
* one Parquet row group is written per (partition, chunk); a row group never splits an S1's candidates.

Validation S1s are searched against the FULL train S2/S3 pool (only the query rows are restricted).
"""
from __future__ import annotations

import json
import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

from .cli import parse_split
from .config import INTERIM_DIR, TRAIN_DIR, ensure_dirs
from .io_utils import GROUND_TRUTH_SUFFIX, load_ground_truth
from .normalize import records_path
from .perf import SAMPLE_CAVEAT, save_metrics, stage_timer
from .split import load_split_ids
from .tfidf import (
    N_FEATURES, document_frequency, hashed_counts, idf_vector, pair_cosine, prune_common, tfidf_weight,
    topk_per_row,
)

K_NAME = 15  # top-K per source on the name channel
K_CTX = 10  # top-K per source on the name+postcode+city channel
K_ADDR = 10  # top-K per source on the address channel (recovers pairs whose NAMES differ: scripts, DBA names)
CHANNELS: Dict[str, int] = {"name": K_NAME, "ctx": K_CTX, "addr": K_ADDR}
ENABLED = [c for c in os.environ.get("ER_CHANNELS", "name,ctx,addr").split(",") if c in CHANNELS]
CHUNK_ROWS = int(os.environ.get("ER_BLOCK_CHUNK", "500"))
MAX_DF_FRAC = 0.002  # n-gram pruned from the search matrix when in > this share of the partition's documents ...
MAX_DF_FLOOR = 500  # ... but never below this many documents
COUNTRY_EQUAL_MIN = 0.995
META_PATH = INTERIM_DIR / "block_meta.json"
RECORD_COLUMNS = ["entity_id", "source", "country", "name_core", "postcode", "city_token", "addr_norm"]

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


def true_pairs_frame(truth: Dict[str, set], s1_ids: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Ground truth as a two-column frame ``(s1_id, cand_id)`` with one row per true pair.

    Args:
        truth: ``{s1_id: {matched ids}}``.
        s1_ids: Optional restriction to these S1 ids.

    Returns:
        DataFrame with string columns ``s1_id`` and ``cand_id``.
    """
    keys = truth.keys() if s1_ids is None else s1_ids
    s1s = [k for k in keys for _ in truth[k]]
    cands = [c for k in keys for c in truth[k]]
    return pd.DataFrame({"s1_id": s1s, "cand_id": cands}, dtype=object)


def country_equality_share() -> Dict[str, object]:
    """Share of true train pairs whose S1 and S2/S3 country labels are identical strings.

    Uses ALL train ground truth and ``records_train.parquet``. Pairs with an id missing from the records are
    ignored. The result is cached in ``data/interim/block_meta.json`` so val/test runs reuse the train decision.

    Returns:
        ``{"country_equal_share", "n_true_pairs", "partition_by_country"}``.
    """
    truth = load_ground_truth(TRAIN_DIR / f"train_{GROUND_TRUTH_SUFFIX}")
    pairs = true_pairs_frame(truth)
    rec = pq.read_table(records_path("train"), columns=["entity_id", "country"]).to_pandas()
    index = pd.Index(rec["entity_id"])
    country = rec["country"].to_numpy(dtype=object)
    i1, i2 = index.get_indexer(pairs["s1_id"]), index.get_indexer(pairs["cand_id"])
    ok = (i1 >= 0) & (i2 >= 0)
    share = float((country[i1[ok]] == country[i2[ok]]).mean()) if ok.any() else 1.0
    meta = {"country_equal_share": share, "n_true_pairs": int(ok.sum()), "partition_by_country": share >= COUNTRY_EQUAL_MIN}
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


@dataclass
class Partition:
    """Everything needed to search one country partition (matrices are shared read-only between chunks).

    Attributes:
        country: Partition label (``"*"`` when not partitioned).
        s1_ids: Query S1 ids of this partition (rows of the query matrices).
        s1_country: Country label written for each query row.
        pool_ids: S2 ids followed by S3 ids (rows of the pool matrices).
        n_s2: Number of S2 rows at the start of the pool.
        exact: Per channel ``(query matrix, pool matrix)``, full TF-IDF vectors for the exact cosines.
        search: Per channel ``(pruned query matrix, pruned S2 matrix^T, pruned S3 matrix^T)``.
    """

    country: str
    s1_ids: np.ndarray
    s1_country: np.ndarray
    pool_ids: np.ndarray
    n_s2: int
    exact: Dict[str, Tuple[sp.csr_matrix, sp.csr_matrix]] = field(default_factory=dict)
    search: Dict[str, Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]] = field(default_factory=dict)


def channel_text(rec: pd.DataFrame, channel: str) -> pd.Series:
    """The text that is vectorised for one channel: ``name`` (name_core), ``ctx`` (name + postcode + city) or ``addr``.

    Built one channel at a time so only one set of strings is alive at once.

    Args:
        rec: Records with ``name_core``, ``postcode``, ``city_token``, ``addr_norm``.
        channel: ``"name"``, ``"ctx"`` or ``"addr"``.

    Returns:
        String Series aligned with ``rec``.
    """
    if channel == "name":
        return rec["name_core"]
    if channel == "ctx":
        return (rec["name_core"] + " " + rec["postcode"] + " " + rec["city_token"]).str.strip()
    return rec["addr_norm"]


def build_partition(rec: pd.DataFrame, s1_keep: Optional[set], country: str) -> Optional[Partition]:
    """Build the TF-IDF matrices of one partition.

    IDF is fitted on ALL S1+S2+S3 names of the partition (unsupervised, so allowed on test too); the query
    rows are then restricted to ``s1_keep`` (val/train S1s), while the pool is always the full S2/S3 set.
    Memory: one channel is vectorised at a time and intermediate count matrices are released immediately.

    Args:
        rec: Records of the partition (all sources).
        s1_keep: S1 ids to query, or None for every S1.
        country: Partition label.

    Returns:
        A ``Partition``, or None if there is no query S1 or no pool record.
    """
    src = rec["source"].to_numpy()
    all_s1 = rec[src == "S1"]
    query_rows = np.flatnonzero(all_s1["entity_id"].isin(s1_keep).to_numpy()) if s1_keep is not None else np.arange(len(all_s1))
    s1 = all_s1.iloc[query_rows]
    s2, s3 = rec[src == "S2"], rec[src == "S3"]
    if len(s1) == 0 or len(s2) + len(s3) == 0:
        return None
    n_docs = len(all_s1) + len(s2) + len(s3)
    max_df = max(MAX_DF_FLOOR, int(MAX_DF_FRAC * n_docs))
    part = Partition(
        country=country, s1_ids=s1["entity_id"].to_numpy(dtype=object), s1_country=s1["country"].to_numpy(dtype=object),
        pool_ids=np.concatenate([s2["entity_id"].to_numpy(dtype=object), s3["entity_id"].to_numpy(dtype=object)]),
        n_s2=len(s2),
    )
    for ch in ENABLED:
        c_all = hashed_counts(channel_text(all_s1, ch))
        c2, c3 = hashed_counts(channel_text(s2, ch)), hashed_counts(channel_text(s3, ch))
        df = document_frequency(c_all, c2, c3)
        idf = idf_vector(df, n_docs)
        xq, x2, x3 = tfidf_weight(c_all[query_rows], idf), tfidf_weight(c2, idf), tfidf_weight(c3, idf)
        del c_all, c2, c3
        part.search[ch] = (prune_common(xq, df, max_df), prune_common(x2, df, max_df).T.tocsr(), prune_common(x3, df, max_df).T.tocsr())
        part.exact[ch] = (xq, sp.vstack([x2, x3], format="csr"))
        del x2, x3
    return part


def search_chunk(part: Partition, a: int, b: int) -> pa.Table:
    """Generate the candidates of query rows ``[a, b)`` of a partition.

    Args:
        part: Partition matrices.
        a: First query row.
        b: One past the last query row.

    Returns:
        Arrow table with ``CANDIDATE_SCHEMA``: union of the enabled channels, sorted by S1 then descending score.
    """
    pairs: Optional[pd.DataFrame] = None
    for ch in ENABLED:
        qs, pt2, pt3 = part.search[ch]
        chunk = qs[a:b]
        frames = []
        for offset, pt in ((0, pt2), (part.n_s2, pt3)):
            row, col, _, rank = topk_per_row((chunk @ pt).tocsr(), CHANNELS[ch])
            frames.append(pd.DataFrame({"q": row, "p": col + offset, f"rank_{ch}": rank.astype(np.float32)}))
        found = pd.concat(frames, ignore_index=True)
        pairs = found if pairs is None else pairs.merge(found, on=["q", "p"], how="outer")
    q, p = pairs["q"].to_numpy(dtype=np.int64), pairs["p"].to_numpy(dtype=np.int64)
    out = {"s1_id": part.s1_ids[a + q], "cand_id": part.pool_ids[p], "country": part.s1_country[a + q]}
    nan = np.full(len(q), np.nan, dtype=np.float32)
    for ch in CHANNELS:  # channels that are switched off are all-False / NaN so the schema is fixed
        rank = pairs[f"rank_{ch}"].to_numpy(dtype=np.float32) if ch in ENABLED else nan
        out[f"ch_{ch}"] = ~np.isnan(rank)
        out[f"rank_{ch}"] = rank
    for ch in CHANNELS:  # exact cosine on the full vectors, for every union pair and every enabled channel
        if ch in ENABLED:
            xq, xp = part.exact[ch]
            out[f"cos_{ch}"] = pair_cosine(xq[a:b], q, xp, p)
        else:
            out[f"cos_{ch}"] = nan
    frame = pd.DataFrame(out)
    frame["block_score"] = frame[[f"cos_{ch}" for ch in ENABLED]].max(axis=1).astype(np.float32)
    frame = frame.sort_values(["s1_id", "block_score", "cand_id"], ascending=[True, False, True], kind="mergesort")
    return pa.Table.from_pandas(frame[CANDIDATE_SCHEMA.names], schema=CANDIDATE_SCHEMA, preserve_index=False)


_SHARED: Optional[Partition] = None  # set before forking so workers inherit the matrices copy-on-write (no pickling)


def _search_shared_range(bounds: Tuple[int, int]) -> pa.Table:
    """Worker entry point: ``search_chunk`` on the partition inherited through ``fork``.

    Args:
        bounds: ``(a, b)`` query row range.

    Returns:
        The candidate table of that range.
    """
    return search_chunk(_SHARED, bounds[0], bounds[1])


def iter_chunk_tables(part: Partition, n_jobs: int) -> Iterator[pa.Table]:
    """Yield the candidate tables of all query chunks of a partition, in order (optionally in parallel).

    With ``n_jobs > 1`` the chunks are processed by forked worker processes that share the (read-only) partition
    matrices copy-on-write; only the small result tables are pickled back. Results are identical to ``n_jobs=1``.

    Args:
        part: Partition matrices.
        n_jobs: Worker processes (1 = in-process).

    Returns:
        Iterator of Arrow tables.
    """
    global _SHARED
    ranges = [(a, min(a + CHUNK_ROWS, len(part.s1_ids))) for a in range(0, len(part.s1_ids), CHUNK_ROWS)]
    if n_jobs <= 1 or len(ranges) < 2:
        for a, b in ranges:
            yield search_chunk(part, a, b)
        return
    _SHARED = part
    try:
        with multiprocessing.get_context("fork").Pool(n_jobs) as pool:
            yield from pool.imap(_search_shared_range, ranges)
    finally:
        _SHARED = None


def run_block(split: str, n_jobs: int = 1) -> Dict[str, object]:
    """Generate candidates for a split and write ``candidates_<split>.parquet``.

    Args:
        split: ``"train"``, ``"val"`` or ``"test"``.
        n_jobs: Parallel worker processes for the chunk loop.

    Returns:
        Dict with ``n_pairs``, ``n_queries`` and the partition decision.
    """
    meta = partition_decision(split)
    print(f"block: true pairs with equal country label = {meta['country_equal_share']:.4%} "
          f"({meta['n_true_pairs']} pairs) -> partition by country: {meta['partition_by_country']}")
    path = records_path(split)
    keep_ids = load_split_ids(split)
    s1_keep = set(keep_ids) if keep_ids is not None else None
    partitions = list_countries(path) if meta["partition_by_country"] else ["*"]
    ensure_dirs()
    n_pairs = n_queries = 0
    with pq.ParquetWriter(candidates_path(split), CANDIDATE_SCHEMA, compression="zstd") as writer:
        for country in partitions:
            filters = [("country", "=", country)] if country != "*" else None
            # Arrow-backed strings use ~3x less memory than Python str objects (matters for multi-million-row partitions)
            rec = pq.read_table(path, columns=RECORD_COLUMNS, filters=filters).to_pandas(types_mapper=pd.ArrowDtype)
            part = build_partition(rec, s1_keep, country)
            del rec
            if part is None:
                continue
            n_queries += len(part.s1_ids)
            for table in iter_chunk_tables(part, n_jobs):
                writer.write_table(table)
                n_pairs += table.num_rows
            print(f"block: partition {country!r}: {len(part.s1_ids)} S1 queries vs {len(part.pool_ids)} pool records", flush=True)
            del part
    return {"n_pairs": n_pairs, "n_queries": n_queries, **meta}


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point: run blocking for a split and, for train/val, print the blocking-recall report.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    split = parse_split(__doc__.splitlines()[0], argv)
    n_jobs = int(os.environ.get("ER_N_JOBS", "1"))
    with stage_timer("block", split) as info:
        result = run_block(split, n_jobs)
        info["pairs"] = result["n_pairs"]
        print(f"block: {result['n_pairs']} candidate pairs for {result['n_queries']} S1 -> {candidates_path(split)}")
    if split != "test":
        from .block_report import report_blocking  # imported here: block_report needs candidates_path from this module

        print(SAMPLE_CAVEAT)
        report_blocking(split)
    else:
        save_metrics(f"block_{split}", {"n_pairs": result["n_pairs"], "n_queries": result["n_queries"],
                                        "avg_candidates": result["n_pairs"] / max(result["n_queries"], 1)})


if __name__ == "__main__":
    main()
