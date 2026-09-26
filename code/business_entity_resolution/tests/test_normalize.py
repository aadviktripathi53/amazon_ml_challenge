"""Tests for the normaliser: suffix stripping, number/postcode extraction, empty fields, abbreviations."""
import pandas as pd
import pytest

from src.normalize import (
    OUT_COLUMNS, city_token, digit_runs, extract_postcode, name_core, normalize_address, normalize_frame,
    normalize_name, normalize_record,
)


@pytest.mark.parametrize(
    "raw, core",
    [
        ("Acme Pvt. Ltd.", "acme"),
        ("ACME PRIVATE LIMITED", "acme"),
        ("Acme Corp", "acme"),
        ("Acme Corporation", "acme"),
        ("Inc Holy Ministries", "holy ministries"),  # the legal form was moved to the front by the noise model
        ("Europ & Frères Distribution S.A.", "europ freres distribution"),
        ("The Sharma & Sons Private Limited", "sharma sons"),
        ("Dupont SARL", "dupont"),
        ("Durand SAS", "durand"),
        ("Smith Holdings LLC", "smith holdings"),
    ],
)
def test_name_core_strips_legal_suffixes(raw, core):
    """Legal suffixes and generic words are removed wherever they occur."""
    assert name_core(normalize_name(raw)) == core


def test_name_core_falls_back_when_everything_is_stripped():
    """A name made only of stop-words keeps its normalised form instead of becoming empty."""
    assert name_core(normalize_name("The Company Ltd")) == "the company ltd"


def test_name_norm_rules():
    """Lowercase, transliteration, & -> and, punctuation removed, spaces collapsed."""
    assert normalize_name("  Café   Müller & Co.,  ") == "cafe muller and co"
    assert normalize_name("S.G. Joe's-Diner") == "sg joes diner"
    assert normalize_name("डिजिटल टेक") != ""  # Devanagari is transliterated, not dropped


@pytest.mark.parametrize(
    "raw, norm",
    [
        ("12 MG Rd, Nr SBI ATM", "12 mg road near sbi atm"),
        ("Opp. City Mall, Blvd 5", "opposite city mall boulevard 5"),
        ("5 Av des Hêtres", "5 avenue des hetres"),
        ("10 Bd Victor Hugo", "10 boulevard victor hugo"),
        ("Pune, null, A-68", "pune a 68"),
    ],
)
def test_address_abbreviation_expansion(raw, norm):
    """Abbreviations are expanded token by token; the data artefact 'null' is dropped."""
    assert normalize_address(raw) == norm


def test_number_and_postcode_extraction():
    """All digit runs are kept in order; the postcode is the last 5-6 digit run (leading zeros kept)."""
    assert digit_runs("532 1/2 Kestrel Ln, Pune 411001") == ["532", "1", "2", "411001"]
    assert extract_postcode(digit_runs("532 1/2 Kestrel Ln, Pune 411001")) == "411001"
    assert extract_postcode(digit_runs("Boston MA 02134")) == "02134"
    assert extract_postcode(digit_runs("Call 9876543210 at 12 Main St")) == ""  # 10 digits is a phone number
    assert extract_postcode(digit_runs("12 Main St")) == ""
    assert extract_postcode(["12345", "560001"]) == "560001"  # the LAST candidate wins
    assert digit_runs("१२ सड़क") == ["12"]  # Devanagari digits are folded to ASCII


@pytest.mark.parametrize(
    "addr, city",
    [
        ("1130 COLUMBIA AVE, BRIDGEPORT, WA", "bridgeport"),
        ("9336 W Haven Dr, Portland, Oregon", "portland"),
        ("Columbia City, Indiana, 91 Tidewater Trail", "columbia city"),
        ("MA, 500 Quincy Street, Abington, Unit 3-C1", "abington"),
        ("17 Rue Gustave Delory, Lille, Hauts-de-France", "lille"),
        ("", ""),
    ],
)
def test_city_token(addr, city):
    """The city component is found without any country-specific rule."""
    assert city_token(addr) == city


def test_empty_fields_are_safe():
    """Empty names, addresses and countries produce empty (not failing) outputs."""
    row = dict(zip(OUT_COLUMNS, normalize_record("S2-1", "", "", "")))
    assert row["name_norm"] == "" and row["name_core"] == "" and row["addr_norm"] == ""
    assert row["numbers"] == [] and row["postcode"] == "" and row["city_token"] == "" and row["name_tokens"] == []
    assert row["source"] == "S2" and row["country"] == ""


def test_country_is_an_opaque_string_and_ids_are_stripped():
    """Unknown country labels pass through untouched; whitespace around ids/countries is removed."""
    df = pd.DataFrame(
        {"entity_id": [" S1-1 "], "business_name": ["Café SARL"], "business_address": ["3 Rue X, Lyon"], "country": [" Atlantis "]}
    )
    out = normalize_frame(df)
    assert list(out.columns) == OUT_COLUMNS
    assert out.loc[0, "entity_id"] == "S1-1" and out.loc[0, "country"] == "Atlantis"
    assert out.loc[0, "name_raw"] == "Café SARL"  # raw text is preserved


def test_romanize_non_latin_tokens_only(monkeypatch):
    """Non-Latin-script tokens are romanized (repeated letters collapsed, legal forms mapped); Latin tokens untouched."""
    import src.normalize as N

    assert N.romanize_non_latin("\u0905\u0932 \u0906\u0908\u091f\u0940 \u092a\u094d\u0930\u093e\u0907\u0935\u0947\u091f \u0932\u093f\u092e\u093f\u091f\u0947\u0921") == "al aiti private limited"
    assert N.romanize_non_latin("Golden Solutions L\u00edmited") == "Golden Solutions L\u00edmited"  # accented Latin: unchanged
    assert N.romanize_non_latin("Li Wei Trading") == "Li Wei Trading"  # 'li' is only mapped when it came from a non-Latin token
    monkeypatch.setattr(N, "ROMANIZE", True)
    assert N.name_core(N.normalize_name("\u0905\u0932 \u0906\u0908\u091f\u0940 \u092a\u094d\u0930\u093e\u0907\u0935\u0947\u091f \u0932\u093f\u092e\u093f\u091f\u0947\u0921")) == "al aiti"
    monkeypatch.setattr(N, "ROMANIZE", False)
    assert "aa" in N.normalize_name("\u0905\u0932 \u0906\u0908\u091f\u0940")  # off: plain unidecode
