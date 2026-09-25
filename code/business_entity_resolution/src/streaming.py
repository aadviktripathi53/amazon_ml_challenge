"""Memory-safe helpers for the (possibly multi-GB) competition TSVs.

Nothing here loads a whole source file: files are read in chunks of ``CHUNKSIZE``
rows (``dtype=str``, ``sep="\\t"``, ``keep_default_na=False``, ``quoting=QUOTE_NONE``) and IDs
are kept as 64-bit hashes (8 bytes per ID) instead of Python strings. Comparing IDs by hash
is exact up to 64-bit collisions (~1e-6 probability for 1e7 IDs); IDs are stripped of
surrounding whitespace before hashing.

Why ``QUOTE_NONE``: the files are one record per line with tabs as the only delimiter. With
pandas' default quoting, a field that merely STARTS with ``"`` (e.g. ``"Joe's Diner``) opens a
quoted field that swallows the following lines, silently dropping rows (and their IDs). Use
``raw_line_stats`` to cross-check parsed row counts against the physical line count.
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .io_utils import ADDRESS_COL, COUNTRY_COL, ID_COL, NAME_COL

CHUNKSIZE = 200_000


def human_size(n_bytes: float) -> str:
    """Format a byte count as a human-readable string (e.g. ``"3.2 GB"``).

    Args:
        n_bytes: Size in bytes.

    Returns:
        String with a binary-scaled unit.
    """
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.0f} {unit}" if unit == "B" else f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} GB"


def read_header(path) -> List[str]:
    """Read only the column names of a TSV.

    Args:
        path: TSV file path.

    Returns:
        List of column names.
    """
    return pd.read_csv(path, sep="\t", nrows=0, dtype=str, quoting=csv.QUOTE_NONE).columns.tolist()


def iter_chunks(path, usecols: Optional[Sequence[str]] = None, chunksize: Optional[int] = None) -> Iterator[pd.DataFrame]:
    """Yield a TSV in string-typed chunks.

    Args:
        path: TSV file path.
        usecols: Columns to load (None = all).
        chunksize: Rows per chunk; defaults to the module-level ``CHUNKSIZE``
            (looked up at call time so tests can shrink it).

    Yields:
        DataFrames of at most ``chunksize`` rows, every column ``str``, no NaN.
    """
    yield from pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, usecols=usecols, chunksize=chunksize or CHUNKSIZE,
        quoting=csv.QUOTE_NONE,
    )


def hash_ids(ids: pd.Series, seed: int = 0, salt: str = "") -> np.ndarray:
    """Deterministically hash IDs to uint64 (stable across runs and chunk sizes).

    Args:
        ids: Series of ID strings (surrounding whitespace is ignored).
        seed: Global seed mixed into the hash.
        salt: Extra label so different uses of the same seed give independent hashes.

    Returns:
        ``uint64`` array aligned with ``ids``.
    """
    return pd.util.hash_pandas_object(f"{seed}:{salt}:" + ids.str.strip(), index=False).to_numpy()


def isin_sorted(values: np.ndarray, sorted_ref: np.ndarray) -> np.ndarray:
    """Membership test of ``values`` in an already-sorted reference array.

    Args:
        values: Array to test.
        sorted_ref: Sorted array of allowed values.

    Returns:
        Boolean array, True where ``values`` occurs in ``sorted_ref``.
    """
    if len(sorted_ref) == 0:
        return np.zeros(len(values), dtype=bool)
    idx = np.searchsorted(sorted_ref, values)
    idx[idx == len(sorted_ref)] = 0
    return sorted_ref[idx] == values


def raw_line_stats(path, block_bytes: int = 16 * 1024 * 1024) -> Tuple[int, int]:
    """Count physical data lines and double-quote characters of a file, without parsing it.

    Independent of the CSV parser, so it exposes rows lost (or merged) by parsing.

    Args:
        path: File path.
        block_bytes: Read block size.

    Returns:
        ``(data_lines, quote_chars)`` where ``data_lines`` excludes the header line.
    """
    lines = quotes = 0
    last = b"\n"
    with open(path, "rb") as f:
        while block := f.read(block_bytes):
            lines += block.count(b"\n")
            quotes += block.count(b'"')
            last = block[-1:]
    if last != b"\n":
        lines += 1  # final line without trailing newline
    return max(lines - 1, 0), quotes


@dataclass
class ScanResult:
    """Streaming statistics of one source TSV."""

    path: Path
    size_bytes: int
    columns: List[str]
    rows: int = 0
    country_counts: Counter = field(default_factory=Counter)
    empty_name: int = 0
    empty_address: int = 0
    bad_prefix: int = 0
    ids_with_whitespace: int = 0
    id_hashes: Optional[np.ndarray] = None  # only when keep_ids=True

    @property
    def duplicate_ids(self) -> Optional[int]:
        """Number of rows whose entity_id repeats an earlier one (None if IDs were not kept)."""
        if self.id_hashes is None:
            return None
        return int(len(self.id_hashes) - len(np.unique(self.id_hashes)))


def scan_source(path, expect_prefix: Optional[str] = None, keep_ids: bool = False) -> ScanResult:
    """Stream a source file (entity_id/business_name/business_address/country) and collect stats.

    Args:
        path: Source TSV path.
        expect_prefix: If given (e.g. ``"S2-"``), count IDs not starting with it.
        keep_ids: Keep 64-bit ID hashes (for duplicate/existence checks).

    Returns:
        ScanResult with row count, country counts, empty-field counts, etc.
    """
    path = Path(path)
    columns = read_header(path)
    wanted = [c for c in (ID_COL, NAME_COL, ADDRESS_COL, COUNTRY_COL) if c in columns]
    res = ScanResult(path=path, size_bytes=path.stat().st_size, columns=columns)
    hashes: List[np.ndarray] = []
    for chunk in iter_chunks(path, usecols=wanted):
        res.rows += len(chunk)
        if COUNTRY_COL in chunk:
            res.country_counts.update(chunk[COUNTRY_COL].value_counts().to_dict())
        if NAME_COL in chunk:
            res.empty_name += int((chunk[NAME_COL].str.strip() == "").sum())
        if ADDRESS_COL in chunk:
            res.empty_address += int((chunk[ADDRESS_COL].str.strip() == "").sum())
        if ID_COL in chunk:
            res.ids_with_whitespace += int((chunk[ID_COL] != chunk[ID_COL].str.strip()).sum())
            if expect_prefix:
                res.bad_prefix += int((~chunk[ID_COL].str.strip().str.startswith(expect_prefix)).sum())
            if keep_ids:
                hashes.append(hash_ids(chunk[ID_COL]))
    if keep_ids:
        res.id_hashes = np.concatenate(hashes) if hashes else np.empty(0, dtype=np.uint64)
    return res
