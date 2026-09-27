"""exp-v2: phonetic key, phonetic and exact-key blocking channels, and the v2 feature groups."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import src.block as B
import src.features as F
from src.normalize import consonant_skeleton, name_core, normalize_frame, normalize_name, phonetic_key


@pytest.mark.parametrize(
    "raw, phon, skel",
    [
        ("Phoenix Khanna Bhawan", "foenix kana bavan", "fnx kn bvn"),
        ("Shree Zenith Woods", "sri jenith vuds", "sr jnth vds"),  # no th->t rule in the spec
        ("Sri Kanna Bhavan", "sri kana bavan", "sr kn bvn"),
        ("अल आईटी प्राइवेट लिमिटेड", "al aiti", "l t"),
        ("Al It Private Limited", "al it", "l t"),
        ("", "", ""),
    ],
)
def test_phonetic_key_and_skeleton(raw, phon, skel):
    """ph->f, kh->k, bh->b, sh->s, aa->a, ee->i, oo->u, w->v, z->j, collapsed doubles, legal residue dropped."""
    got = phonetic_key(name_core(normalize_name(raw)))
    assert got == phon and consonant_skeleton(got) == skel


def test_normalize_emits_phonetic_columns():
    """Records carry name_phon and name_skel."""
    out = normalize_frame(pd.DataFrame({"entity_id": ["S1-1"], "business_name": ["Shree Traders"], "business_address": [""], "country": ["India"]}))
    assert out.loc[0, "name_phon"] == "sri traders" and out.loc[0, "name_skel"] == "sr trdrs"


def _set_channels(monkeypatch, phon=False, key=False):
    """Switch the optional blocking channels on for a test (module flags are read at import time)."""
    enabled = ["name", "ctx", "addr"] + (["phon"] if phon else [])
    monkeypatch.setattr(B, "ENABLED", enabled)
    monkeypatch.setattr(B, "CH_PHON", phon)
    monkeypatch.setattr(B, "CH_KEY", key)
    monkeypatch.setattr(B, "KEY_ON", key)
    monkeypatch.setattr(B, "ACTIVE", enabled + (["dup"] if B.DUP_ON else []) + (["key"] if key else []))
    monkeypatch.setattr(B, "TEXT_COLUMNS", ["name_core", "postcode", "city_token", "addr_norm"] + (["name_phon"] if phon else []) + (["numbers"] if key else []))


def _write_records(tmp_path, raw_rows):
    """Normalise raw (id, name, address, country) rows and write them as a records parquet."""
    df = pd.DataFrame(raw_rows, columns=["entity_id", "business_name", "business_address", "country"])
    rec = normalize_frame(df)
    path = tmp_path / "records.parquet"
    pq.write_table(pa.Table.from_pandas(rec, preserve_index=False), path, row_group_size=17)
    return path


def _run(path, country, shard_docs=40):
    """Run the sharded blocking of one partition and return the candidate frame."""
    idf = B.fit_idf(path, country)
    q = pd.concat([B.to_frame(b) for b in B.iter_partition_batches(path, country, ("S1",), ["entity_id", "country", "source", *B.TEXT_COLUMNS])], ignore_index=True)
    block = B.build_query_block(q, np.zeros(len(q), np.int8), idf)
    states, pool_ids = B.block_query_block(path, country, block, idf, B.Budget(shard_docs, 10 ** 6), 1, "t")
    return B.emit_candidates(block, states, pool_ids, 0, len(block.ids)).to_pandas()


def _filler(n, prefix, country="India"):
    """Unrelated pool records so top-K is competitive."""
    rng = np.random.RandomState(0)
    words = ["global", "tech", "traders", "foods", "enterprises", "solutions", "industries", "services", "exports", "agro"]
    return [(f"{prefix}-f{i}", f"{rng.choice(words)} {rng.choice(words)} {i}", f"{i} main road pune", country) for i in range(n)]


def test_phonetic_channel_finds_transliterated_name(tmp_path, monkeypatch):
    """'Eastern Raj Global' vs its Bengali-script copy: found by the phonetic channel."""
    rows = [("S1-q", "Eastern Raj Global Private Limited", "12 park street kolkata", "India"),
            ("S2-t", "ইস্টার্ন রাজ গ্লোবাল প্রাইভেট লিমিটেড", "", "India"),
            ("S3-x", "other", "", "India")] + _filler(200, "S2")
    path = _write_records(tmp_path, rows)
    _set_channels(monkeypatch, phon=True)
    cands = _run(path, "India")
    hit = cands[cands.cand_id == "S2-t"]
    assert len(hit) == 1 and bool(hit.ch_phon.iloc[0]) and not bool(hit.ch_name.iloc[0])  # only the phonetic channel finds it
    assert hit.cos_phon.iloc[0] > hit.cos_name.iloc[0]


def test_key_channel_links_same_postcode_and_house_number(tmp_path, monkeypatch):
    """Totally different names, same (postcode, house number): the key channel links them; the name channel does not."""
    rows = [("S1-q", "Secure Investment Works LLC", "32371 Clarkston Street, San Tan Valley, AZ 85140", "US"),
            ("S2-t", "Pyravantage", "032371 CLARKSTON ST, QUEEN CREEK, AZ 85140", "US"),
            ("S3-x", "unrelated", "", "US")] + _filler(200, "S2", "US")
    path = _write_records(tmp_path, rows)
    _set_channels(monkeypatch, key=True)
    cands = _run(path, "US")
    hit = cands[cands.cand_id == "S2-t"]
    assert len(hit) == 1 and bool(hit.ch_key.iloc[0]) and hit.rank_key.iloc[0] == 1


def test_key_channel_skips_keys_shared_by_more_than_key_max(tmp_path, monkeypatch):
    """A (postcode, first-token) key shared by > KEY_MAX pool records is not used."""
    rows = [("S1-q", "Metro Care", "5 Elm St, Austin, TX 73301", "US")]
    rows += [(f"S2-{i}", f"Metro {i}", f"{100 + i} Oak Ave, Austin, TX 73301", "US") for i in range(60)]  # key 'metro' x 60
    rows += [("S3-x", "zzz", "", "US")]
    path = _write_records(tmp_path, rows)
    _set_channels(monkeypatch, key=True)
    monkeypatch.setattr(B, "KEY_MAX", 50)
    assert not _run(path, "US").ch_key.any()
    monkeypatch.setattr(B, "KEY_MAX", 100)
    assert _run(path, "US").ch_key.sum() == B.K_KEY


@pytest.mark.parametrize("shard_docs", [7, 23, 10 ** 6])
def test_new_channels_are_shard_size_invariant(tmp_path, monkeypatch, shard_docs):
    """With phon + key on, results do not depend on the shard size."""
    rows = [("S1-a", "Sri Kanna Bhavan", "4 Hill Rd, Pune 411001", "India"), ("S1-b", "Shree Khanna Bhawan", "9 Lake Rd, Pune 411001", "India"),
            ("S2-a", "Shri Kana Bavan", "4 Hill Road, Pune 411001", "India"), ("S3-a", "Kanna Bhavan", "4 Hill Rd 411001", "India")] + _filler(80, "S3")
    path = _write_records(tmp_path, rows)
    _set_channels(monkeypatch, phon=True, key=True)
    monkeypatch.setattr(B, "BATCH_ROWS", 7)
    ref = _run(path, "India", 10 ** 6).reset_index(drop=True)
    pd.testing.assert_frame_equal(ref, _run(path, "India", shard_docs).reset_index(drop=True))
    assert list(ref.columns) == B.CANDIDATE_SCHEMA.names


@pytest.mark.parametrize("x, y, near", [("1670", "670", True), ("188", "18", True), ("42", "42", True), ("123", "124", True),
                                        ("123", "456", False), ("5", "15", False), ("", "5", False), ("2503", "2879", False)])
def test_numbers_near(x, y, near):
    """House-number noise ('1670'~'670', '188'~'18', one differing digit) is near; other numbers are not."""
    assert F.numbers_near(x, y) == near


def test_v2_and_phonetic_features():
    """House-number state, IDF mismatch, name rarity, S1 context and phonetic similarities on a small example."""
    raw = pd.DataFrame(
        {"entity_id": ["S1-1", "S2-1", "S2-2", "S3-1", "S3-2"],
         "business_name": ["Gabelli LLC", "Gabelli", "Gabelli", "Gabelli", "Page Clemons Metal"],
         "business_address": ["1670 Duncans Chapel Road, Cookeville, TN", "670 Duncans Chapel Rd, Cookeville", "999 Other St, Cookeville", "", "1670 Duncans Chapel Road"],
         "country": ["US"] * 5})
    rec = normalize_frame(raw)
    stats = F.CountryStats(rec)
    a = rec.iloc[[0, 0, 0, 0]].reset_index(drop=True)
    b = rec.iloc[[1, 2, 3, 4]].reset_index(drop=True)
    v = F.v2_pair_features(a, b, stats)
    assert list(v["house_state"]) == [0.5, -1.0, 0.0, 1.0]  # near / conflict / missing / equal
    assert v["name_idf_mismatch"][0] == 0 and v["name_idf_mismatch"][3] == 1.0  # same name vs no shared token
    assert v["core_freq_cand"][0] == pytest.approx(np.log1p(3))  # 'gabelli' is shared by 3 pool records
    ctx = F.s1_context_features(np.array(["s"] * 4), np.array([1.0, 1.0, 1.0, 0.2]), np.array([0.9, 0.3, 0.0, 1.0]), b["name_core"].to_numpy(dtype=object))
    assert list(ctx["s1_rank_ad"]) == [2, 3, 4, 1] and list(ctx["s1_same_core"]) == [3, 3, 3, 1] and ctx["s1_gap_ad"][3] == 0
    ph = F.phonetic_features(pd.DataFrame({"name_phon": ["sri kana bavan"], "name_skel": ["sr kn bvn"]}),
                             pd.DataFrame({"name_phon": ["sri kana bavan"], "name_skel": ["sr kn bvn"]}))
    assert ph["sk_exact"][0] == 1 and ph["ph_ratio"][0] == 1
