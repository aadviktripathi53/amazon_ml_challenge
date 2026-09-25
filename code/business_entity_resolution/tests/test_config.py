"""Tests for src.config path resolution."""
from pathlib import Path

from src import config


def test_repo_root_contains_project_dir():
    """Auto-detected repo root is the ancestor holding code/business_entity_resolution."""
    assert (config.REPO_ROOT / "code" / "business_entity_resolution" / "src" / "config.py").is_file()


def test_data_dir_default_and_env_override(tmp_path):
    """Default is <root>/dataset; absolute env paths are kept; relative ones resolve against the root."""
    root = Path("/repo")
    assert config.resolve_data_dir(root, {}) == root / "dataset"
    assert config.resolve_data_dir(root, {"ER_DATA_DIR": "  "}) == root / "dataset"
    assert config.resolve_data_dir(root, {"ER_DATA_DIR": str(tmp_path)}) == tmp_path
    assert config.resolve_data_dir(root, {"ER_DATA_DIR": "mnt/data"}) == root / "mnt" / "data"


def test_constants():
    """Interim/output dirs hang off the repo root; seed is 42; val reads train files."""
    assert config.SEED == 42
    assert config.INTERIM_DIR == config.REPO_ROOT / "data" / "interim"
    assert config.OUTPUT_DIR == config.REPO_ROOT / "output"
    assert config.split_dir("test") == config.TEST_DIR
    assert config.split_dir("val") == config.split_dir("train") == config.TRAIN_DIR
