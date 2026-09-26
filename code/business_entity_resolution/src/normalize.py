"""Stage 1: normalise raw records -> data/interim/records_{train,test}.parquet.

Run as ``python -m src.normalize --split <train|val|test>`` from ``code/business_entity_resolution/``.
``val`` is carved out of the train files, so it re-uses ``records_train.parquet`` (nothing to do).

Scaling: each source TSV is read in chunks (``src.streaming.iter_chunks``), normalised row by row in
plain Python (~50 us/record) and appended to one Parquet file via ``ParquetWriter``; memory is bounded
by the chunk size, not by the file size. Schema: ``docs/contracts.md``.

Language-agnostic by construction: text is transliterated with ``unidecode`` and country labels are never
inspected. The abbreviation / legal-suffix lists below are word lists (English, Indian-English, French), not
country switches.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from unidecode import unidecode

from .cli import parse_split
from .config import INTERIM_DIR, ensure_dirs, split_dir
from .io_utils import ADDRESS_COL, COUNTRY_COL, ID_COL, NAME_COL, SOURCE_SUFFIXES
from .perf import stage_timer
from .streaming import CHUNKSIZE, iter_chunks

# Legal-form words and generic connector words removed to build ``name_core``.
LEGAL_SUFFIXES: FrozenSet[str] = frozenset(
    "pvt private ltd limited llc inc incorporated corp corporation co company llp plc sarl sas sa eurl gmbh "
    "lp pllc ag spa srl bv nv pte opc public sci snc sasu scop sca selarl".split()
)
GENERIC_WORDS: FrozenSet[str] = frozenset("the and of de du des la le les et".split())
CORE_STOPWORDS: FrozenSet[str] = LEGAL_SUFFIXES | GENERIC_WORDS

# Address abbreviations expanded token by token (English/Indian-English plus French av/bd).
ADDRESS_ABBREVIATIONS: Dict[str, str] = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "bd": "boulevard",
    "nr": "near", "opp": "opposite", "hwy": "highway", "bldg": "building", "apt": "apartment", "ste": "suite",
    "pkwy": "parkway", "ln": "lane",
}
# Address tokens that mark a component as street / building / landmark (used only to find the city component).
NON_CITY_WORDS: FrozenSet[str] = frozenset(
    "road rd street st avenue ave av boulevard blvd bd lane ln drive dr court ct place pl way highway hwy parkway "
    "pkwy circle cir trail terrace square plaza loop marg floor plot block phase flat unit suite apartment "
    "building bldg tower near opposite opp nr behind beside ward gali colony society complex layout cross main "
    "chowk market industrial area estate rue chemin chm impasse allee cours cour cite quai route residence "
    "appartement batiment bat zone lieu passage villa hameau".split()
)

_DROP_INSIDE = re.compile(r"[.'`]")  # deleted without a space: "S.G." -> "sg", "Joe's" -> "joes"
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGIT_RUN = re.compile(r"\d+")
POSTCODE_LENGTHS = (5, 6)  # US/France zip = 5 digits, India PIN = 6 digits

OUT_COLUMNS = [
    "entity_id", "source", "country", "name_raw", "addr_raw", "name_norm", "name_core", "addr_norm",
    "numbers", "postcode", "city_token", "name_tokens",
]
RECORD_SCHEMA = pa.schema(
    [
        ("entity_id", pa.string()), ("source", pa.string()), ("country", pa.string()), ("name_raw", pa.string()),
        ("addr_raw", pa.string()), ("name_norm", pa.string()), ("name_core", pa.string()),
        ("addr_norm", pa.string()), ("numbers", pa.list_(pa.string())), ("postcode", pa.string()),
        ("city_token", pa.string()), ("name_tokens", pa.list_(pa.string())),
    ]
)


def records_path(split: str) -> Path:
    """Path of the records parquet for a split (``val`` shares the train file).

    Args:
        split: ``"train"``, ``"val"`` or ``"test"``.

    Returns:
        ``data/interim/records_train.parquet`` or ``records_test.parquet``.
    """
    return INTERIM_DIR / f"records_{'test' if split == 'test' else 'train'}.parquet"


def ascii_fold(text: str) -> str:
    """Transliterate to ASCII with ``unidecode`` (skipped for text that is already ASCII).

    Args:
        text: Any string.

    Returns:
        ASCII string.
    """
    return text if text.isascii() else unidecode(text)


def _clean_text(text: str) -> str:
    """Lowercase + ASCII-fold + ``&`` -> ``and`` + drop ``. ' ``` + other punctuation -> space + collapse spaces.

    Args:
        text: Raw name or address.

    Returns:
        Normalised string (empty for empty input).
    """
    if not text:
        return ""
    text = ascii_fold(text).lower().replace("&", " and ")
    text = _DROP_INSIDE.sub("", text)
    return _NON_ALNUM.sub(" ", text).strip()


def normalize_name(raw: str) -> str:
    """Normalise a business name: lowercase, unidecode, ``&`` -> ``and``, punctuation removed, spaces collapsed.

    Args:
        raw: Raw business name (may be empty).

    Returns:
        ``name_norm``.
    """
    return _clean_text(raw)


def name_core(name_norm: str) -> str:
    """Remove legal suffixes and generic words from a normalised name (wherever they appear).

    Legal forms are removed anywhere in the string because the noise model moves them
    (``"Inc Holy Ministries"``). If everything would be removed, ``name_norm`` is returned unchanged.

    Args:
        name_norm: Output of ``normalize_name``.

    Returns:
        ``name_core``.
    """
    kept = [t for t in name_norm.split() if t not in CORE_STOPWORDS]
    return " ".join(kept) if kept else name_norm


def normalize_address(raw: str) -> str:
    """Normalise an address like a name, then expand abbreviations (rd->road, nr->near, av->avenue, ...).

    The literal token ``null`` (a data artefact in some sources) is dropped.

    Args:
        raw: Raw address (may be empty).

    Returns:
        ``addr_norm``.
    """
    tokens = _clean_text(raw).split()
    return " ".join(ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens if t != "null")


def digit_runs(raw_address: str) -> List[str]:
    """All digit runs of the address, in order (non-ASCII digits are folded to ASCII first).

    Args:
        raw_address: Raw address.

    Returns:
        List of digit strings, e.g. ``["532", "1", "2"]`` for ``"532 1/2 Kestrel Ln"``.
    """
    return _DIGIT_RUN.findall(ascii_fold(raw_address)) if raw_address else []


def extract_postcode(numbers: Sequence[str]) -> str:
    """The last digit run that is 5 or 6 digits long (a zip / PIN code), else ``""``.

    Args:
        numbers: Output of ``digit_runs``.

    Returns:
        Postcode string keeping leading zeros, or ``""``.
    """
    for run in reversed(numbers):
        if len(run) in POSTCODE_LENGTHS:
            return run
    return ""


def city_token(raw_address: str) -> str:
    """Best-effort city component of an address, without any country-specific rule.

    Split on commas; drop components that contain digits, are <= 2 letters (state/department codes), are
    ``null``, or contain a street/building/landmark word. Of the remainder the second-to-last is the city
    (the last is usually the state / region) or, if only one remains, that one.

    Args:
        raw_address: Raw address.

    Returns:
        Lowercase city string, or ``""`` when nothing plausible is left.
    """
    if not raw_address:
        return ""
    kept: List[str] = []
    for part in ascii_fold(raw_address).lower().split(","):
        if any(ch.isdigit() for ch in part):
            continue
        clean = _clean_text(part)
        if not clean or clean == "null" or len(clean.replace(" ", "")) <= 2:
            continue
        if any(tok in NON_CITY_WORDS for tok in clean.split()):
            continue
        kept.append(clean)
    if not kept:
        return ""
    return kept[-2] if len(kept) >= 2 else kept[0]


def normalize_record(entity_id: str, name: str, address: str, country: str) -> Tuple:
    """Normalise one record into the ``OUT_COLUMNS`` tuple.

    Args:
        entity_id: ``S1-``/``S2-``/``S3-`` id.
        name: Raw business name.
        address: Raw business address.
        country: Raw country label (kept as an opaque string, whitespace-stripped).

    Returns:
        Tuple aligned with ``OUT_COLUMNS``.
    """
    entity_id = entity_id.strip()
    name_norm = normalize_name(name)
    core = name_core(name_norm)
    numbers = digit_runs(address)
    return (
        entity_id, entity_id[:2], country.strip(), name, address, name_norm, core, normalize_address(address),
        numbers, extract_postcode(numbers), city_token(address), core.split(),
    )


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a chunk of raw records (columns entity_id, business_name, business_address, country).

    Args:
        df: Raw chunk; empty names/addresses are fine.

    Returns:
        DataFrame with ``OUT_COLUMNS``.
    """
    rows = [
        normalize_record(i, n, a, c)
        for i, n, a, c in zip(df[ID_COL], df[NAME_COL], df[ADDRESS_COL], df[COUNTRY_COL])
    ]
    return pd.DataFrame(rows, columns=OUT_COLUMNS)


def normalize_split(split: str, chunksize: int = CHUNKSIZE) -> Tuple[Path, int]:
    """Stream all three sources of a split through the normaliser into one parquet file.

    Args:
        split: ``"train"`` or ``"test"``.
        chunksize: Rows per chunk.

    Returns:
        ``(path, n_records)``.
    """
    out = records_path(split)
    directory = split_dir(split)
    prefix = "test" if split == "test" else "train"
    ensure_dirs()
    n = 0
    with pq.ParquetWriter(out, RECORD_SCHEMA, compression="zstd") as writer:
        for suffix in SOURCE_SUFFIXES:
            for chunk in iter_chunks(directory / f"{prefix}_{suffix}", chunksize=chunksize):
                writer.write_table(pa.Table.from_pandas(normalize_frame(chunk), schema=RECORD_SCHEMA, preserve_index=False))
                n += len(chunk)
    return out, n


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point: normalise a split's records to parquet (``val`` is a no-op that reuses the train file).

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    split = parse_split(__doc__.splitlines()[0], argv)
    if split == "val":
        print("normalize: 'val' is carved out of the train files -> reusing records_train.parquet")
        return
    with stage_timer("normalize", split) as info:
        path, n = normalize_split(split)
        info["records"] = n
        print(f"normalize: {n} records -> {path}")


if __name__ == "__main__":
    main()
