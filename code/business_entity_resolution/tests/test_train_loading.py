"""The train loader reads features into a float32 matrix and samples whole S1 groups above the row cap."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyarrow")


@pytest.fixture
def feature_file(tmp_path):
    """A small feature parquet: 40 S1 with 3-7 candidate rows each, written as several row groups."""
    from src.features import FEATURE_COLUMNS, feature_schema

    rng = np.random.RandomState(0)
    rows = [(f"S1-{i}", f"S2-{i}-{j}") for i in range(40) for j in range(rng.randint(3, 8))]
    n = len(rows)
    cols = {"s1_id": [r[0] for r in rows], "cand_id": [r[1] for r in rows]}
    cols.update({c: rng.rand(n).astype(np.float32) for c in FEATURE_COLUMNS})
    cols["label"] = (rng.rand(n) < 0.3).astype(np.int8)
    table = pa.table(cols, schema=feature_schema(True))
    path = tmp_path / "features.parquet"
    pq.write_table(table, path, row_group_size=25)
    return path, n


def test_loader_without_sampling(feature_file):
    """Below the cap every row is kept and the arrays line up."""
    try:
        from src.train import load_train_arrays
    except OSError as exc:  # LightGBM without libomp on macOS
        pytest.skip(str(exc))
    from src.features import FEATURE_COLUMNS

    path, n = feature_file
    x, y, groups, ids = load_train_arrays(path, max_pairs=10 ** 9)
    assert x.shape == (n, len(FEATURE_COLUMNS)) and x.dtype == np.float32
    assert len(y) == len(groups) == ids.num_rows == n and y.dtype == np.int8
    assert len(set(groups)) == 40


def test_loader_samples_whole_s1_groups_reproducibly(feature_file):
    """Above the cap whole S1 groups are kept (never split), within the cap, and the choice is seeded."""
    try:
        from src.train import load_train_arrays
    except OSError as exc:
        pytest.skip(str(exc))

    path, n = feature_file
    _, _, groups, ids = load_train_arrays(path, max_pairs=n // 2)
    _, _, groups2, ids2 = load_train_arrays(path, max_pairs=n // 2)
    kept = ids.column("s1_id").to_pylist()
    full = pq.read_table(path, columns=["s1_id"]).column("s1_id").to_pylist()
    assert 0 < len(kept) <= n // 2
    for s in set(kept):
        assert kept.count(s) == full.count(s)  # a sampled S1 keeps all of its pairs
    assert ids.equals(ids2) and np.array_equal(groups, groups2)
