"""Stage 3: pair features -> data/interim/features_{split}.parquet (one row per candidate pair).

Run as ``python -m src.features --split <train|val|test>`` from ``code/business_entity_resolution/``.

Scaling: candidates are read one Parquet row group at a time (row groups are written by ``src.block`` and are
per country), consecutive row groups of the same country are merged into batches of ~``BATCH_PAIRS`` pairs, the
records of that country are looked up by id, all fuzzy scores are computed with rapidfuzz ``process.cpdist``
(multi-threaded C++, no Python loop per pair) and the float32 result batch is appended to the output file. Only the
records of ONE country are in memory at a time (when blocking was partitioned by country).

All scores are pairwise (no cross-pair context) and scaled to [0, 1] unless noted.
"""
from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from .block import candidates_path, partition_decision
from .cli import parse_split
from .config import INTERIM_DIR, TRAIN_DIR, ensure_dirs, max_mem_gb
from .io_utils import GROUND_TRUTH_SUFFIX, load_ground_truth_subset
from .normalize import records_path
from .perf import stage_timer
from .split import load_split_ids

BATCH_PAIRS = 100_000
OBJECT_BYTES_PER_RECORD = 450  # measured: Python-object string frame of REC_COLUMNS
OBJECT_LAYOUT_BUDGET_SHARE = 0.35  # use Python-object strings (faster) only if the country's records fit in this share
RF_WORKERS = int(os.environ.get("ER_RF_WORKERS", "-1"))  # rapidfuzz threads (-1 = all cores)
REC_COLUMNS = ["entity_id", "name_norm", "name_core", "addr_norm", "numbers", "postcode"]
SCORERS = {"ratio": fuzz.ratio, "partial": fuzz.partial_ratio, "tsort": fuzz.token_sort_ratio, "tset": fuzz.token_set_ratio}

FUZZY_COLUMNS = [f"{p}_{k}" for p in ("nn", "nc", "ad") for k in (*SCORERS, "jw")]  # nn=name_norm, nc=name_core, ad=addr_norm
PAIR_COLUMNS = [
    "name_jaccard", "core_exact", "name_len_diff", "addr_empty_s1", "addr_empty_cand", "num_shared",
    "num_conflict", "pc_state",
]
BLOCK_COLUMNS = [
    "cand_is_s3", "ch_name", "ch_ctx", "ch_addr", "n_channels", "rank_name", "rank_ctx", "rank_addr", "cos_name",
    "cos_ctx", "cos_addr", "block_score",
]
FEATURE_COLUMNS = FUZZY_COLUMNS + PAIR_COLUMNS + BLOCK_COLUMNS


def features_path(split: str):
    """Path of the feature parquet of a split.

    Args:
        split: ``"train"``, ``"val"`` or ``"test"``.

    Returns:
        ``data/interim/features_<split>.parquet``.
    """
    return INTERIM_DIR / f"features_{split}.parquet"


def feature_schema(with_label: bool) -> pa.Schema:
    """Arrow schema of the feature file (ids as strings, features float32, label int8).

    Args:
        with_label: Include the ``label`` column (train/val).

    Returns:
        The schema.
    """
    fields = [("s1_id", pa.string()), ("cand_id", pa.string())] + [(c, pa.float32()) for c in FEATURE_COLUMNS]
    if with_label:
        fields.append(("label", pa.int8()))
    return pa.schema(fields)


def pairwise_scores(a: Sequence[str], b: Sequence[str], scorer, scale: float) -> np.ndarray:
    """Score corresponding elements of two string lists with a rapidfuzz scorer (0 when either is empty).

    Args:
        a: Left strings.
        b: Right strings (same length).
        scorer: rapidfuzz scorer returning a similarity.
        scale: Divisor bringing the score to [0, 1] (100 for ``fuzz`` scorers, 1 for Jaro-Winkler).

    Returns:
        float32 vector of scores.
    """
    if len(a) == 0:
        return np.zeros(0, dtype=np.float32)
    scores = process.cpdist(a, b, scorer=scorer, dtype=np.float32, workers=RF_WORKERS) / scale
    empty = np.fromiter((not x or not y for x, y in zip(a, b)), dtype=bool, count=len(a))
    scores[empty] = 0.0
    return scores.astype(np.float32)


def fuzzy_block(prefix: str, a: Sequence[str], b: Sequence[str]) -> Dict[str, np.ndarray]:
    """The five fuzzy similarities (ratio, partial_ratio, token_sort, token_set, Jaro-Winkler) for one text field.

    Args:
        prefix: Column prefix (``nn``, ``nc`` or ``ad``).
        a: S1-side strings.
        b: Candidate-side strings.

    Returns:
        ``{"<prefix>_<scorer>": scores}``.
    """
    out = {f"{prefix}_{k}": pairwise_scores(a, b, fn, 100.0) for k, fn in SCORERS.items()}
    out[f"{prefix}_jw"] = pairwise_scores(a, b, JaroWinkler.similarity, 1.0)
    return out


def _number_set(arr) -> frozenset:
    """Digit runs of an address as a set with leading zeros stripped (``"0401"`` == ``"401"``).

    Args:
        arr: Sequence of digit strings.

    Returns:
        frozenset of normalised digit strings.
    """
    return frozenset((x.lstrip("0") or "0") for x in arr)


def pair_features(a: pd.DataFrame, b: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Non-fuzzy pair features: token Jaccard, exact core match, length gap, address-empty flags, numbers, postcode.

    Args:
        a: S1-side records aligned with the pairs (columns name_core, addr_norm, numbers, postcode).
        b: Candidate-side records, same alignment.

    Returns:
        ``{column: float32 vector}`` for ``PAIR_COLUMNS``.
    """
    n = len(a)
    core_a, core_b = a["name_core"].tolist(), b["name_core"].tolist()
    jac = np.zeros(n, dtype=np.float32)
    for i, (x, y) in enumerate(zip(core_a, core_b)):
        sx, sy = set(x.split()), set(y.split())
        union = len(sx | sy)
        jac[i] = len(sx & sy) / union if union else 0.0
    exact = np.fromiter((x == y and bool(x) for x, y in zip(core_a, core_b)), dtype=np.float32, count=n)
    len_diff = np.abs(a["name_core"].str.len().to_numpy() - b["name_core"].str.len().to_numpy()).astype(np.float32)
    num_a, num_b = [_number_set(x) for x in a["numbers"]], [_number_set(x) for x in b["numbers"]]
    shared = np.fromiter((len(x & y) for x, y in zip(num_a, num_b)), dtype=np.float32, count=n)
    both_have = np.fromiter((bool(x) and bool(y) for x, y in zip(num_a, num_b)), dtype=bool, count=n)
    pa_, pb_ = a["postcode"].to_numpy(dtype=object), b["postcode"].to_numpy(dtype=object)
    both_pc = (pa_ != "") & (pb_ != "")
    pc_state = np.where(both_pc, np.where(pa_ == pb_, 1.0, -1.0), 0.0).astype(np.float32)  # 1 match, -1 conflict, 0 missing
    return {
        "name_jaccard": jac,
        "core_exact": exact,
        "name_len_diff": len_diff,
        "addr_empty_s1": (a["addr_norm"].to_numpy(dtype=object) == "").astype(np.float32),
        "addr_empty_cand": (b["addr_norm"].to_numpy(dtype=object) == "").astype(np.float32),
        "num_shared": shared,
        "num_conflict": (both_have & (shared == 0)).astype(np.float32),
        "pc_state": pc_state,
    }


def block_features(cands: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Features carried over from blocking: source, channel flags, ranks, exact cosines and score.

    Args:
        cands: Candidate rows (``src.block`` schema).

    Returns:
        ``{column: float32 vector}`` for ``BLOCK_COLUMNS``.
    """
    out = {"cand_is_s3": cands["cand_id"].str.startswith("S3-").to_numpy(dtype=np.float32)}
    for ch in ("name", "ctx", "addr"):
        out[f"ch_{ch}"] = cands[f"ch_{ch}"].to_numpy(dtype=np.float32)
        out[f"rank_{ch}"] = cands[f"rank_{ch}"].to_numpy(dtype=np.float32)
        out[f"cos_{ch}"] = cands[f"cos_{ch}"].to_numpy(dtype=np.float32)
    out["n_channels"] = out["ch_name"] + out["ch_ctx"] + out["ch_addr"]
    out["block_score"] = cands["block_score"].to_numpy(dtype=np.float32)
    return out


def compute_batch(cands: pd.DataFrame, rec: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Compute all features for a batch of candidate pairs.

    Args:
        cands: Candidate rows for the batch.
        rec: Records (``REC_COLUMNS``) covering every id in the batch.
        index: ``pd.Index`` over ``rec['entity_id']`` used for lookups.

    Returns:
        DataFrame with ``s1_id``, ``cand_id`` and every ``FEATURE_COLUMNS`` (float32).
    """
    ia, ib = index.get_indexer(cands["s1_id"]), index.get_indexer(cands["cand_id"])
    if (ia < 0).any() or (ib < 0).any():
        raise KeyError("candidate ids missing from the records of this country partition")
    a, b = rec.iloc[ia].reset_index(drop=True), rec.iloc[ib].reset_index(drop=True)
    feats: Dict[str, np.ndarray] = {}
    feats.update(fuzzy_block("nn", a["name_norm"].tolist(), b["name_norm"].tolist()))
    feats.update(fuzzy_block("nc", a["name_core"].tolist(), b["name_core"].tolist()))
    feats.update(fuzzy_block("ad", a["addr_norm"].tolist(), b["addr_norm"].tolist()))
    feats.update(pair_features(a, b))
    feats.update(block_features(cands))
    out = pd.DataFrame({"s1_id": cands["s1_id"].to_numpy(), "cand_id": cands["cand_id"].to_numpy()})
    return pd.concat([out, pd.DataFrame({c: feats[c] for c in FEATURE_COLUMNS})], axis=1)


def iter_batches(cand_path, batch_pairs: int = BATCH_PAIRS) -> Iterator[Tuple[str, pd.DataFrame]]:
    """Yield ``(country, candidate batch)``: consecutive row groups of one country merged up to ``batch_pairs`` rows.

    Args:
        cand_path: Candidate parquet written by ``src.block``.
        batch_pairs: Target rows per batch.

    Returns:
        Iterator of ``(country label, DataFrame)``.
    """
    pf = pq.ParquetFile(cand_path)
    pending: List[pd.DataFrame] = []
    country: Optional[str] = None
    size = 0
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg).to_pandas()
        if len(df) == 0:
            continue
        c = df["country"].iloc[0]
        if pending and (c != country or size >= batch_pairs):
            yield country, pd.concat(pending, ignore_index=True)
            pending, size = [], 0
        country = c
        pending.append(df)
        size += len(df)
    if pending:
        yield country, pd.concat(pending, ignore_index=True)


def load_country_records(path, country: Optional[str], mem_gb: Optional[float] = None) -> pd.DataFrame:
    """Load the feature-relevant columns of one country's records, in a layout chosen from the memory budget.

    Python-object strings are ~2x faster to score but ~6x bigger than Arrow-backed strings, so the Arrow layout is used
    when the country's records would take more than ``OBJECT_LAYOUT_BUDGET_SHARE`` of ``ER_MAX_MEM_GB``.

    Args:
        path: Records parquet.
        country: Country label (None = all records).
        mem_gb: Budget in GB (defaults to ``config.max_mem_gb()``).

    Returns:
        DataFrame with ``REC_COLUMNS``.
    """
    mem_gb = max_mem_gb() if mem_gb is None else mem_gb
    expr = (ds.field("country") == country) if country is not None else None
    n_rows = ds.dataset(str(path), format="parquet").count_rows(filter=expr)
    table = ds.dataset(str(path), format="parquet").to_table(columns=REC_COLUMNS, filter=expr)
    if n_rows * OBJECT_BYTES_PER_RECORD > OBJECT_LAYOUT_BUDGET_SHARE * mem_gb * 1024 ** 3:
        return table.to_pandas(types_mapper=pd.ArrowDtype)
    return table.to_pandas()


def run_features(split: str) -> int:
    """Compute and write features for every candidate pair of a split.

    Args:
        split: ``"train"``, ``"val"`` or ``"test"``.

    Returns:
        Number of feature rows written.
    """
    with_label = split != "test"
    truth = load_ground_truth_subset(TRAIN_DIR / f"train_{GROUND_TRUTH_SUFFIX}", load_split_ids(split)) if with_label else None
    partitioned = bool(partition_decision(split)["partition_by_country"])
    schema = feature_schema(with_label)
    ensure_dirs()
    rec_key, rec, index = None, None, None
    n_rows = 0
    with pq.ParquetWriter(features_path(split), schema, compression="zstd") as writer:
        for country, cands in iter_batches(candidates_path(split)):
            key = country if partitioned else "*"
            if key != rec_key:  # load the records of one country at a time
                rec = None  # release the previous country before loading the next one
                rec = load_country_records(records_path(split), country if partitioned else None)
                index = pd.Index(rec["entity_id"].to_numpy(dtype=object))
                rec_key = key
            frame = compute_batch(cands, rec, index)
            if with_label:
                frame["label"] = np.fromiter(
                    (c in truth.get(s, ()) for s, c in zip(frame["s1_id"], frame["cand_id"])), dtype=np.int8, count=len(frame)
                )
            writer.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False))
            n_rows += len(frame)
    return n_rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point: build the feature file of a split.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    split = parse_split(__doc__.splitlines()[0], argv)
    with stage_timer("features", split) as info:
        n = run_features(split)
        info["pairs"] = n
        print(f"features: {n} pairs x {len(FEATURE_COLUMNS)} features -> {features_path(split)}")


if __name__ == "__main__":
    main()
