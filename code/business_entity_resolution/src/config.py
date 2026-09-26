"""Single source of truth for paths and the global seed.

Every module imports its paths from here; nothing else hard-codes a path.

* ``REPO_ROOT``   auto-detected (works in the git repo and inside the submission zip).
* ``DATA_DIR``    env var ``ER_DATA_DIR`` (relative paths resolve against ``REPO_ROOT``),
                  default ``<repo_root>/dataset``.
* ``INTERIM_DIR`` ``<repo_root>/data/interim``.
* ``OUTPUT_DIR``  ``<repo_root>/output``.
* ``SEED``        42.
* ``ER_MAX_MEM_GB``      memory budget in GB for the pipeline's big stages (default 8), see ``max_mem_gb``.
* ``ER_TRAIN_S1_FRAC``   fraction (0-1] of TRAIN S1 ids used for train/val (default 1.0), see ``train_s1_frac``.
* ``REPORTS_DIR`` ``<repo_root>/reports``; ``FAKE_DATA_DIR`` ``<repo_root>/dataset_fake``;
  ``SAMPLE_DIR`` ``<repo_root>/dataset_sample`` (generated, git-ignored).
"""
from __future__ import annotations

import os
from pathlib import Path

SEED = 42
DATA_DIR_ENV = "ER_DATA_DIR"
_PROJECT_SUBDIR = Path("code") / "business_entity_resolution"


def find_repo_root(start: Path = Path(__file__)) -> Path:
    """Locate the repo root by walking up from ``start``.

    The root is the first ancestor containing ``code/business_entity_resolution``
    (true both for the git repo and for the unzipped submission package). Falls
    back to three levels above ``src/`` (``<root>/code/business_entity_resolution/src``).

    Args:
        start: File or directory to start searching from.

    Returns:
        Absolute path of the repo root.
    """
    start = start.resolve()
    for parent in start.parents:
        if (parent / _PROJECT_SUBDIR).is_dir():
            return parent
    return start.parents[3]


def resolve_data_dir(repo_root: Path, env: "os._Environ[str] | dict" = os.environ) -> Path:
    """Resolve the dataset directory from ``ER_DATA_DIR`` or the default.

    Args:
        repo_root: Repo root used for the default and for relative env values.
        env: Environment mapping to read ``ER_DATA_DIR`` from.

    Returns:
        Absolute dataset directory path.
    """
    value = env.get(DATA_DIR_ENV, "").strip()
    if not value:
        return repo_root / "dataset"
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


REPO_ROOT = find_repo_root()
DATA_DIR = resolve_data_dir(REPO_ROOT)
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
INTERIM_DIR = REPO_ROOT / "data" / "interim"
OUTPUT_DIR = REPO_ROOT / "output"
REPORTS_DIR = REPO_ROOT / "reports"
FAKE_DATA_DIR = REPO_ROOT / "dataset_fake"
SAMPLE_DIR = REPO_ROOT / "dataset_sample"


def split_dir(split: str) -> Path:
    """Return the raw-data directory for a split name.

    The pipeline's ``val`` split is carved out of the training data, so it
    reads from the ``train`` directory.

    Args:
        split: One of ``"train"``, ``"val"``, ``"test"``.

    Returns:
        ``DATA_DIR/test`` for ``"test"``, otherwise ``DATA_DIR/train``.
    """
    return TEST_DIR if split == "test" else TRAIN_DIR


def max_mem_gb(env: "os._Environ[str] | dict" = os.environ) -> float:
    """Memory budget (GB) for the memory-bounded stages, from ``ER_MAX_MEM_GB`` (default 8).

    Blocking sizes its pool shards and query blocks from it, features chooses its record layout from it and
    training derives its default row cap from it.

    Args:
        env: Environment mapping to read from.

    Returns:
        Budget in gigabytes (> 0).
    """
    value = float(env.get("ER_MAX_MEM_GB", "").strip() or 8.0)
    if value <= 0:
        raise ValueError("ER_MAX_MEM_GB must be positive")
    return value


def train_s1_frac(env: "os._Environ[str] | dict" = os.environ) -> float:
    """Fraction of TRAIN S1 ids used for train/val, from ``ER_TRAIN_S1_FRAC`` (default 1.0).

    Args:
        env: Environment mapping to read from.

    Returns:
        A value in (0, 1].
    """
    value = float(env.get("ER_TRAIN_S1_FRAC", "").strip() or 1.0)
    if not 0 < value <= 1:
        raise ValueError("ER_TRAIN_S1_FRAC must be in (0, 1]")
    return value


def ensure_dirs() -> None:
    """Create ``INTERIM_DIR`` and ``OUTPUT_DIR`` if they do not exist."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
