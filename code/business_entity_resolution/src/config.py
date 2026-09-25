"""Single source of truth for paths and the global seed.

Every module imports its paths from here; nothing else hard-codes a path.

* ``REPO_ROOT``   auto-detected (works in the git repo and inside the submission zip).
* ``DATA_DIR``    env var ``ER_DATA_DIR`` (relative paths resolve against ``REPO_ROOT``),
                  default ``<repo_root>/dataset``.
* ``INTERIM_DIR`` ``<repo_root>/data/interim``.
* ``OUTPUT_DIR``  ``<repo_root>/output``.
* ``SEED``        42.
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


def ensure_dirs() -> None:
    """Create ``INTERIM_DIR`` and ``OUTPUT_DIR`` if they do not exist."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
