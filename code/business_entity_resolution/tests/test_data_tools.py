"""Tests for make_fake_data, profile_data and make_sample invariants (on generated fake data)."""
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src import make_fake_data, make_sample, profile_data, streaming
from src.io_utils import parse_id_list, read_tsv


@pytest.fixture(scope="module")
def fake(tmp_path_factory) -> Path:
    """Generate the fake dataset once per module."""
    root = tmp_path_factory.mktemp("fake")
    make_fake_data.write_fake_dataset(root, n_train_s1=200, n_test_s1=100, seed=42)
    return root


def digest(root: Path) -> dict:
    """Map every file under ``root`` to its SHA-256 (to compare runs / detect modification)."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def by_id(path: Path) -> pd.DataFrame:
    """Load a TSV indexed by its first column."""
    return read_tsv(path).set_index(read_tsv(path).columns[0], drop=False)


# ---------------------------------------------------------------- make_fake_data
def test_fake_data_format_and_countries(fake):
    """All competition files exist; France is test-only; ids carry the right prefix."""
    for split in ("train", "test"):
        for n in (1, 2, 3):
            df = read_tsv(fake / split / f"{split}_source{n}.tsv")
            assert list(df.columns) == ["entity_id", "business_name", "business_address", "country"]
            assert df["entity_id"].str.startswith(f"S{n}-").all()
    assert (fake / "train" / "train_ground_truth.tsv").exists() and not (fake / "test" / "test_ground_truth.tsv").exists()
    train_c = set(read_tsv(fake / "train" / "train_source1.tsv")["country"])
    test_c = set(read_tsv(fake / "test" / "test_source1.tsv")["country"])
    assert train_c == {"US", "India"} and test_c == {"US", "India", "France"}


def test_fake_ground_truth_shape(fake):
    """Singletons exist, some S1s match several S2 and S3 records, every GT id exists."""
    gt = read_tsv(fake / "train" / "train_ground_truth.tsv")
    ids = {f"S{n}": set(read_tsv(fake / "train" / f"train_source{n}.tsv")["entity_id"]) for n in (1, 2, 3)}
    sets = [parse_id_list(m) for m in gt["matched_entity_ids"]]
    assert any(not s for s in sets)
    assert any(sum(i.startswith("S2-") for i in s) > 1 and sum(i.startswith("S3-") for i in s) > 1 for s in sets)
    assert set(gt["source1_entity_id"]) == ids["S1"]
    assert all(i in ids[i[:2]] for s in sets for i in s)


def test_fake_data_is_deterministic(tmp_path):
    """Same seed -> byte-identical files."""
    make_fake_data.write_fake_dataset(tmp_path / "a", 30, 20, seed=7)
    make_fake_data.write_fake_dataset(tmp_path / "b", 30, 20, seed=7)
    assert digest(tmp_path / "a") == digest(tmp_path / "b")


# ---------------------------------------------------------------- profile_data
def test_profile_matches_pandas(fake):
    """Streaming profile equals a direct (in-memory) pandas computation, with tiny chunks."""
    old = streaming.CHUNKSIZE
    streaming.CHUNKSIZE = 13
    try:
        res = profile_data.profile(fake)
    finally:
        streaming.CHUNKSIZE = old
    s2 = read_tsv(fake / "train" / "train_source2.tsv")
    scan = res["scans"][fake / "train" / "train_source2.tsv"]
    assert scan.rows == len(s2) and scan.empty_address == int((s2["business_address"] == "").sum())
    assert dict(scan.country_counts) == s2["country"].value_counts().to_dict()
    gt = read_tsv(fake / "train" / "train_ground_truth.tsv")
    sets = [parse_id_list(m) for m in gt["matched_entity_ids"]]
    t = res["truth"]
    assert t["rows"] == len(gt) and t["singletons"] == sum(not s for s in sets)
    assert t["n_s2"] == sum(i.startswith("S2-") for s in sets for i in s)
    assert t["shared_match_ids"] == 0 and t["missing"] == {"S1": 0, "S2": 0, "S3": 0}
    assert sum(t["hist"].values()) == len(gt)
    assert "YES" in profile_data.render_markdown(res)


def test_profile_flags_shared_and_missing_ids(tmp_path):
    """A shared S2 id, a dangling S3 id and a duplicated S1 row are all reported."""
    (tmp_path / "train").mkdir()
    hdr = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    for n, ids in ((1, ["S1-1", "S1-2", "S1-3"]), (2, ["S2-1", "S2-2"]), (3, ["S3-1"])):
        (tmp_path / "train" / f"train_source{n}.tsv").write_text(hdr + "".join(f"{i}\tName\t\tUS\n" for i in ids))
    (tmp_path / "train" / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\tS2-1,S3-99\nS1-2\t\n"
    )
    t = profile_data.profile(tmp_path)["truth"]
    assert t["shared_match_ids"] == 1 and t["missing"]["S3"] == 1 and t["dup_gt_s1_rows"] == 1
    assert t["s1_without_gt"] == 1 and t["s1_with_both"] == 2
    assert "NO" in profile_data.render_markdown(profile_data.profile(tmp_path))


# ---------------------------------------------------------------- make_sample
@pytest.fixture(scope="module")
def sample(fake, tmp_path_factory):
    """Run make_sample once (n_s1=25, ratio 3, test_frac 0.5) and return (out_dir, result)."""
    out = tmp_path_factory.mktemp("sample")
    return out, make_sample.make_sample(fake, out, n_s1=25, distractor_ratio=3, test_frac=0.5, seed=42)


def test_sample_verification_passes(sample):
    """The built-in verification reports PASS for every check."""
    _, result = sample
    assert result["checks"] and all(ok for _, ok, _ in result["checks"]), result["checks"]


def test_sample_train_invariants(fake, sample):
    """S1 count, filtered GT, retained matches, distractor counts and row fidelity."""
    out, result = sample
    s1 = read_tsv(out / "train" / "train_source1.tsv")
    gt = read_tsv(out / "train" / "train_ground_truth.tsv")
    orig_gt = by_id(fake / "train" / "train_ground_truth.tsv")
    assert len(s1) == 25 and set(gt["source1_entity_id"]) == set(s1["entity_id"])
    for _, row in gt.iterrows():  # ground truth rows are copied verbatim
        assert row["matched_entity_ids"] == orig_gt.loc[row["source1_entity_id"], "matched_entity_ids"]
    matched = set().union(*(parse_id_list(m) for m in gt["matched_entity_ids"]))
    for n in (2, 3):
        orig = read_tsv(fake / "train" / f"train_source{n}.tsv")
        samp = read_tsv(out / "train" / f"train_source{n}.tsv")
        m_n = {i for i in matched if i.startswith(f"S{n}-")}
        assert m_n <= set(samp["entity_id"])  # ALL matches kept
        assert len(samp) == len(m_n) + 3 * len(m_n) and len(orig) - len(m_n) >= 3 * len(m_n)
        assert samp.merge(orig, how="left", indicator=True)["_merge"].eq("both").all()  # rows unchanged
    assert result["counts"]["train_source1.tsv"] == (200, 25)


def test_sample_is_stratified(fake, sample):
    """Sampled S1 per (country, singleton) stratum is within 1 of the proportional share."""
    out, _ = sample
    gt = read_tsv(fake / "train" / "train_ground_truth.tsv").set_index("source1_entity_id")["matched_entity_ids"]
    def strata(s1):
        return (s1["country"] + "|" + s1["entity_id"].map(lambda e: str(gt[e] == ""))).value_counts()
    full, samp = strata(read_tsv(fake / "train" / "train_source1.tsv")), strata(read_tsv(out / "train" / "train_source1.tsv"))
    for k, v in full.items():
        assert abs(samp.get(k, 0) - 25 * v / 200) <= 1


def test_sample_test_files_are_hash_subsets(fake, sample):
    """Test samples are subsets of the originals with unchanged rows, roughly test_frac large."""
    out, _ = sample
    for n in (1, 2, 3):
        orig = read_tsv(fake / "test" / f"test_source{n}.tsv")
        samp = read_tsv(out / "test" / f"test_source{n}.tsv")
        assert samp.merge(orig, how="left", indicator=True)["_merge"].eq("both").all()
        assert 0.25 * len(orig) < len(samp) < 0.75 * len(orig)


def test_sample_deterministic_and_chunk_size_independent(fake, sample, tmp_path, monkeypatch):
    """A rerun, and a run with tiny streaming chunks, give byte-identical output."""
    out, _ = sample
    make_sample.make_sample(fake, tmp_path / "again", 25, 3, 0.5, 42)
    assert digest(tmp_path / "again") == digest(out)
    monkeypatch.setattr(streaming, "CHUNKSIZE", 17)
    make_sample.make_sample(fake, tmp_path / "tiny", 25, 3, 0.5, 42)
    assert digest(tmp_path / "tiny") == digest(out)


def test_sample_does_not_touch_source_and_refuses_inside_it(fake, tmp_path):
    """The source dataset is unchanged; an output dir inside it is rejected."""
    before = digest(fake)
    make_sample.make_sample(fake, tmp_path / "o", 10, 2, 0.2, 1)
    assert digest(fake) == before
    with pytest.raises(ValueError):
        make_sample.make_sample(fake, fake / "nested", 10, 2, 0.2, 1)
    with pytest.raises(ValueError):
        make_sample.make_sample(fake, fake, 10, 2, 0.2, 1)


def test_verification_detects_corruption(fake, tmp_path):
    """FAIL is reported for a dangling ground-truth id, a wrong prefix and a duplicate id."""
    def run(mutate):
        out = tmp_path / f"o{len(list(tmp_path.iterdir()))}"
        make_sample.make_sample(fake, out, 25, 3, 0.5, 42)
        mutate(out)
        s1_orig = streaming.scan_source(fake / "train" / "train_source1.tsv").country_counts
        test_orig = {n: streaming.scan_source(fake / "test" / f"test_source{n}.tsv").country_counts for n in (1, 2, 3)}
        return {name: ok for name, ok, _ in make_sample.verify(out, s1_orig, test_orig)}

    def drop_matched(out):
        gt = read_tsv(out / "train" / "train_ground_truth.tsv")
        victim = next(i for m in gt["matched_entity_ids"] for i in parse_id_list(m) if i.startswith("S2-"))
        p = out / "train" / "train_source2.tsv"
        df = read_tsv(p)
        df[df["entity_id"] != victim].to_csv(p, sep="\t", index=False)

    def bad_prefix(out):
        p = out / "test" / "test_source3.tsv"
        p.write_text(p.read_text().replace("S3-", "S2-", 1))

    def duplicate(out):
        p = out / "train" / "train_source2.tsv"
        lines = p.read_text().splitlines(keepends=True)
        p.write_text("".join(lines) + lines[1])

    assert not run(drop_matched)["every ground-truth id exists in the sampled sources"]
    assert not run(bad_prefix)["id prefixes match their files"]
    assert not run(duplicate)["no duplicate entity_ids"]
    assert all(run(lambda o: None).values())


# ---------------------------------------------------------------- existence-check regressions
HDR = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def write_dataset(root: Path, s1_names=None, n=7, gt_rows=None, source_ids=None) -> Path:
    """Write a tiny train dataset: n rows per source, ids S{k}-1..n, GT S1-i -> S2-i,S3-i (all overridable)."""
    d = root / "train"
    d.mkdir(parents=True)
    for k in (1, 2, 3):
        ids = (source_ids or {}).get(k, [f"S{k}-{i}" for i in range(1, n + 1)])
        names = s1_names if (k == 1 and s1_names) else [f"Name {i}" for i in range(len(ids))]
        (d / f"train_source{k}.tsv").write_text(HDR + "".join(f"{i}\t{nm}\t1 Main St\tUS\n" for i, nm in zip(ids, names)))
    rows = gt_rows or [f"S1-{i}\tS2-{i},S3-{i}" for i in range(1, n + 1)]
    (d / "train_ground_truth.tsv").write_text("source1_entity_id\tmatched_entity_ids\n" + "".join(r + "\n" for r in rows))
    return root


@pytest.mark.parametrize("chunksize", [1, 2, 3, 4, 200_000])
def test_existence_check_across_chunk_boundaries(tmp_path, monkeypatch, chunksize):
    """IDs on the last row of one chunk / first row of the next are found (7 rows, chunks of 3: rows 3|4)."""
    monkeypatch.setattr(streaming, "CHUNKSIZE", chunksize)
    root = write_dataset(tmp_path, n=7)
    res = profile_data.profile(root)
    t = res["truth"]
    assert t["missing"] == {"S1": 0, "S2": 0, "S3": 0} and t["missing_distinct"] == {"S1": 0, "S2": 0, "S3": 0}
    assert t["s1_without_gt"] == 0 and profile_data.integrity_warnings(res) == []
    assert "YES" in profile_data.render_markdown(res)


def test_existence_check_boundary_ids_only_present_in_boundary_rows(tmp_path, monkeypatch):
    """A real gap next to a chunk boundary is still reported exactly (no over- or under-counting)."""
    monkeypatch.setattr(streaming, "CHUNKSIZE", 3)
    # GT references S2-3 (last row of chunk 1) and S2-4 (first of chunk 2); the S2 file lacks S2-4 only.
    ids = {2: ["S2-1", "S2-2", "S2-3", "S2-5", "S2-6", "S2-7"]}
    t = profile_data.profile(write_dataset(tmp_path, n=7, source_ids=ids))["truth"]
    assert t["missing"]["S2"] == 1 and t["missing_examples"]["S2"] == ["S2-4"]
    assert t["missing"]["S1"] == 0 and t["missing"]["S3"] == 0


def test_stray_leading_quote_does_not_drop_rows(tmp_path):
    """A name starting with an unclosed `"` used to swallow following rows -> IDs falsely 'missing'."""
    names = ["Alpha Ltd", "Beta Inc", '"Gamma Traders', "Delta Co", 'Epsilon" Sons', "Zeta LLC", "Eta Corp", "Theta Ltd"]
    root = write_dataset(tmp_path, s1_names=names, n=8)
    src = root / "train" / "train_source1.tsv"
    assert len(pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False)) < 8  # the failure mode: default quoting loses rows
    assert len(read_tsv(src)) == 8  # our reader keeps them
    res = profile_data.profile(root)
    assert res["scans"][src].rows == 8 and res["truth"]["missing"]["S1"] == 0
    assert profile_data.integrity_warnings(res) == []


def test_parsed_vs_raw_line_mismatch_is_flagged(tmp_path):
    """If the parser ever loses rows again, the report says so explicitly."""
    res = profile_data.profile(write_dataset(tmp_path, n=4))
    src = tmp_path / "train" / "train_source2.tsv"
    res["raw"][src] = (9, 3)  # pretend the file has 9 lines
    warns = profile_data.integrity_warnings(res)
    assert len(warns) == 1 and "train_source2.tsv" in warns[0] and "9" in warns[0]
    assert "MISMATCH" in profile_data.render_markdown(res)


def test_raw_line_stats(tmp_path):
    """Header excluded; last line counted even without trailing newline; quotes counted."""
    p = tmp_path / "f.tsv"
    p.write_text('h\n"a\nb"\nc')
    assert streaming.raw_line_stats(p) == (3, 2)
    p.write_text("h\na\nb\n")
    assert streaming.raw_line_stats(p) == (2, 0)
    p.write_text("h\n")
    assert streaming.raw_line_stats(p) == (0, 0)


def test_whitespace_in_ids_is_not_a_false_missing_but_is_flagged(tmp_path):
    """IDs differing only by surrounding whitespace match; the report warns about the whitespace."""
    ids = {2: [f"S2-{i} " if i == 2 else f"S2-{i}" for i in range(1, 6)]}
    res = profile_data.profile(write_dataset(tmp_path, n=5, source_ids=ids, gt_rows=[f"S1-{i}\tS2-{i}, S3-{i}" for i in range(1, 6)]))
    t = res["truth"]
    assert t["missing"] == {"S1": 0, "S2": 0, "S3": 0}
    assert any("whitespace" in w for w in profile_data.integrity_warnings(res))


def test_missing_reports_mentions_and_distinct(tmp_path):
    """The same dangling id in two GT rows = 2 mentions but 1 distinct id."""
    rows = ["S1-1\tS2-1,S3-1", "S1-2\tS2-99,S3-2", "S1-3\tS2-99,S3-3"]
    t = profile_data.profile(write_dataset(tmp_path, n=3, gt_rows=rows))["truth"]
    assert t["missing"]["S2"] == 2 and t["missing_distinct"]["S2"] == 1 and t["shared_match_ids"] == 1


def test_gt_s1_ids_beyond_source1_are_reported_as_distinct_anomaly(tmp_path):
    """GT rows for S1 ids that are not in train_source1 are counted as distinct missing S1 ids."""
    rows = [f"S1-{i}\tS2-{i},S3-{i}" for i in range(1, 4)] + ["S1-77\t", "S1-78\t"]
    res = profile_data.profile(write_dataset(tmp_path, n=3, gt_rows=rows))
    t = res["truth"]
    assert t["rows"] == 5 and t["distinct_gt_s1"] == 5 and t["missing_distinct"]["S1"] == 2
    assert "distinct ground-truth S1 IDs absent from train_source1.tsv: 2" in profile_data.render_markdown(res)


def test_make_sample_preserves_stray_quotes_byte_for_byte(tmp_path):
    """Sampling a file containing stray quotes neither drops rows nor re-quotes them."""
    names = ["Alpha Ltd", "Beta Inc", '"Gamma Traders', "Delta Co", 'Epsilon" Sons', "Zeta LLC", "Eta Corp", "Theta Ltd"]
    root = write_dataset(tmp_path / "src", s1_names=names, n=8)
    (root / "test").mkdir()
    for k in (1, 2, 3):
        (root / "test" / f"test_source{k}.tsv").write_text(HDR + f"S{k}-1\t\"Q Name\tAddr\tFrance\n")
    out = tmp_path / "out"
    make_sample.make_sample(root, out, n_s1=8, distractor_ratio=1, test_frac=1.0, seed=1)
    got = (out / "train" / "train_source1.tsv").read_text().splitlines()
    assert len(got) == 9 and any('"Gamma Traders' in line for line in got) and not any('""' in line for line in got)
    assert (out / "test" / "test_source1.tsv").read_text() == (root / "test" / "test_source1.tsv").read_text()
