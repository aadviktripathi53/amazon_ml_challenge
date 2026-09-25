"""Generate a small FAKE dataset in the exact competition TSV format (for testing our code only).

Writes ``<out>/train/train_source{1,2,3}.tsv``, ``train_ground_truth.tsv`` and
``<out>/test/test_source{1,2,3}.tsv``. Train covers US and India; test adds France
(never seen in train). Records carry the noise patterns from the problem statement:
abbreviations, typos, legal-suffix changes, word swaps, transliterations, missing
postcodes/states, "Near X" landmarks, plus singletons, S1s with several matches from
both S2 and S3, distractors and same-name/different-branch hard negatives.

Run from ``code/business_entity_resolution/``::

    python -m src.make_fake_data [--out ../../dataset_fake] [--n-train-s1 200] [--n-test-s1 100]
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from unidecode import unidecode

from .config import FAKE_DATA_DIR, SEED
from .io_utils import ADDRESS_COL, COUNTRY_COL, GROUND_TRUTH_SUFFIX, ID_COL, MATCHED_COL, NAME_COL, SOURCE_SUFFIXES, TRUTH_ID_COL

TRAIN_COUNTRIES = {"US": 0.5, "India": 0.5}
TEST_COUNTRIES = {"US": 0.35, "India": 0.35, "France": 0.30}

# ---- vocabulary (all hand-written; no external data) -------------------------------
LEGAL = {  # (short form, long form)
    "US": [("LLC", "Limited Liability Company"), ("Inc", "Incorporated"), ("Corp", "Corporation"), ("Co", "Company")],
    "India": [("Pvt Ltd", "Private Limited"), ("Ltd", "Limited"), ("LLP", "LLP")],
    "France": [("SARL", "S.A.R.L."), ("SAS", "S.A.S."), ("EURL", "E.U.R.L.")],
}
US_A = ["Maple", "Ridge", "Summit", "Lakeside", "Pioneer", "Eagle", "Cedar", "Liberty", "Harbor", "Sunrise",
        "Oak", "Pine", "Granite", "Redwood", "Prairie", "Silver", "Northstar", "Bluebird", "Heritage", "Evergreen"]
US_B = ["Plumbing", "Logistics", "Bakery", "Dental Care", "Auto Repair", "Consulting", "Electric", "Hardware",
        "Pharmacy", "Catering", "Fitness", "Printing", "Roofing", "Landscaping", "Insurance"]
IN_A = ["Sri Lakshmi", "Shree Ganesh", "Om Sai", "Bharat", "Krishna", "Anand", "Royal", "National", "Tirumala",
        "Annapurna", "Vijay", "Kaveri", "Mahalakshmi", "Balaji", "Gayatri", "Venkateshwara"]
IN_B = ["Traders", "Textiles", "Enterprises", "Pharma", "Foods", "Electricals", "Constructions", "Agencies",
        "Motors", "Sweets", "Steels", "Agro Products"]
FR_A = ["Boulangerie", "Café", "Pharmacie", "Garage", "Atelier", "Librairie", "Maison", "Restaurant", "Fromagerie"]
FR_B = ["Dupont", "Martin", "Bernard", "Lefèvre", "Moreau", "Girard", "Renaud", "Fournier", "Château", "Étoile", "Bélanger"]
TRANSLIT = [("Lakshmi", "Laxmi"), ("Shree", "Sri"), ("Krishna", "Krushna"), ("Vijay", "Vijaya"),
            ("Kaveri", "Cauvery"), ("Venkateshwara", "Venkateswara"), ("Ganesh", "Ganesha")]

# (base name, long type, short type)
US_STREETS = [("Main", "Street", "St"), ("Oak", "Avenue", "Ave"), ("Elm", "Road", "Rd"), ("Sunset", "Boulevard", "Blvd"),
              ("Lake", "Drive", "Dr"), ("Washington", "Street", "St"), ("Highland", "Avenue", "Ave")]
IN_STREETS = [("MG", "Road", "Rd"), ("Gandhi", "Road", "Rd"), ("Nehru", "Street", "St"), ("Temple", "Road", "Rd"),
              ("Station", "Road", "Rd"), ("Residency", "Cross", "Cr"), ("Church", "Street", "St")]
FR_STREETS = [("de la République", "Rue", "R."), ("Victor Hugo", "Avenue", "Av."), ("Saint-Michel", "Boulevard", "Bd"),
              ("Pasteur", "Rue", "R."), ("de la Gare", "Rue", "R."), ("Jean Jaurès", "Avenue", "Av.")]
US_CITIES = [("Springfield", "Illinois", "IL", "627"), ("Portland", "Oregon", "OR", "972"), ("Austin", "Texas", "TX", "787"),
             ("Columbus", "Ohio", "OH", "432"), ("Denver", "Colorado", "CO", "802")]
IN_CITIES = [("Bengaluru", "Karnataka", "560"), ("Chennai", "Tamil Nadu", "600"), ("Mumbai", "Maharashtra", "400"),
             ("Hyderabad", "Telangana", "500"), ("Pune", "Maharashtra", "411"), ("Delhi", "Delhi", "110")]
IN_AREAS = ["Indiranagar", "Koramangala", "T Nagar", "Andheri", "Banjara Hills", "Kothrud", "Karol Bagh", "Jayanagar"]
FR_CITIES = [("Paris", "750"), ("Lyon", "690"), ("Marseille", "130"), ("Toulouse", "310"), ("Bordeaux", "330")]
LANDMARKS = {
    "US": ["Next to Walgreens", "Near City Hall", "Across from the Post Office"],
    "India": ["Near SBI ATM", "Opp. City Mall", "Behind Bus Stand", "Near Hanuman Temple", "Near Metro Station"],
    "France": ["Près de la Poste", "Face à la Mairie", "Près de la Gare"],
}


@dataclass
class Biz:
    """Structured description of one real-world business (before rendering to noisy text)."""

    country: str
    core: str
    legal: Tuple[str, str]
    number: int
    street: Tuple[str, str, str]
    city: str
    state: str
    state_abbr: str
    postcode: str
    area: str
    landmark: str


def _typo(text: str, rng: random.Random) -> str:
    """Apply one random character-level typo (delete/swap/replace/duplicate) to a word of length >= 4.

    Args:
        text: Input string.
        rng: Random source.

    Returns:
        String with one typo, or unchanged if no word is long enough.
    """
    words = text.split(" ")
    cands = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not cands:
        return text
    i = rng.choice(cands)
    w, j = words[i], rng.randrange(1, len(words[i]) - 1)
    op = rng.choice(["del", "swap", "rep", "dup"])
    if op == "del":
        w = w[:j] + w[j + 1:]
    elif op == "swap":
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    elif op == "rep":
        w = w[:j] + rng.choice("aeiourstnl") + w[j + 1:]
    else:
        w = w[:j] + w[j] + w[j:]
    words[i] = w
    return " ".join(words)


def make_biz(rng: random.Random, country: str, used_cores: Optional[set] = None) -> Biz:
    """Sample a random business for a country, with a unique core name if ``used_cores`` is given.

    Args:
        rng: Random source.
        country: One of US / India / France.
        used_cores: Set of already-used core names (updated in place) to keep S1 deduplicated.

    Returns:
        A ``Biz``.
    """
    for _ in range(50):
        if country == "US":
            core = f"{rng.choice(US_A)} {rng.choice(US_B)}" if rng.random() > 0.15 else f"{rng.choice(US_A)} & Sons {rng.choice(US_B)}"
        elif country == "India":
            core = f"{rng.choice(IN_A)} {rng.choice(IN_B)}"
        else:
            core = f"{rng.choice(FR_A)} {rng.choice(FR_B)}"
        if used_cores is None or core not in used_cores:
            break
    else:
        core = f"{core} {rng.choice(['Group', 'Bros', 'Global', 'Systems'])}"
    if used_cores is not None:
        used_cores.add(core)
    legal = rng.choice(LEGAL[country])
    number = rng.randint(1, 999)
    if country == "US":
        city, state, abbr, zp = rng.choice(US_CITIES)
        return Biz(country, core, legal, number, rng.choice(US_STREETS), city, state, abbr, f"{zp}{rng.randint(0, 99):02d}", "", rng.choice(LANDMARKS[country]))
    if country == "India":
        city, state, pin = rng.choice(IN_CITIES)
        return Biz(country, core, legal, number, rng.choice(IN_STREETS), city, state, "", f"{pin}{rng.randint(0, 999):03d}", rng.choice(IN_AREAS), rng.choice(LANDMARKS[country]))
    city, pc = rng.choice(FR_CITIES)
    return Biz(country, core, legal, number, rng.choice(FR_STREETS), city, "", "", f"{pc}{rng.randint(0, 99):02d}", "", rng.choice(LANDMARKS[country]))


def render_name(b: Biz, rng: random.Random, noise: float) -> str:
    """Render a business name; ``noise`` in [0,1] scales the probability of each corruption.

    Corruptions: legal suffix short/long/dropped, ``&`` <-> ``and``, adjacent word swap,
    transliteration (India), accent stripping (France), typo, upper-casing.

    Args:
        b: Business.
        rng: Random source.
        noise: 0 gives a clean name; 1 applies every corruption often.

    Returns:
        Name string.
    """
    core = b.core
    if noise and rng.random() < noise * 0.5:
        core = core.replace("&", "and") if "&" in core else core.replace(" and ", " & ")
    if noise and b.country == "India" and rng.random() < noise * 0.7:
        for a, c in TRANSLIT:
            if a in core:
                core = core.replace(a, c)
                break
            if c in core:
                core = core.replace(c, a)
                break
    if noise and rng.random() < noise * 0.3 and len(core.split()) > 1:
        w = core.split()
        i = rng.randrange(len(w) - 1)
        w[i], w[i + 1] = w[i + 1], w[i]
        core = " ".join(w)
    if noise and rng.random() < noise * 0.6:
        core = _typo(core, rng)
    if noise and b.country == "France" and rng.random() < noise:
        core = unidecode(core)
    short, long_ = b.legal
    if noise and rng.random() < noise * 0.25:
        name = core  # legal suffix dropped
    elif rng.random() < 0.5:
        name = f"{core} {long_}"
    else:
        name = f"{core} {short}"
    if noise and rng.random() < noise * 0.25:
        name = name.upper()
    return name


def render_address(b: Biz, rng: random.Random, noise: float) -> str:
    """Render a business address, optionally noisy (abbreviations, missing parts, landmarks, reordering).

    Args:
        b: Business.
        rng: Random source.
        noise: 0 gives the full clean address; 1 applies corruptions often.

    Returns:
        Address string (empty string occasionally when ``noise`` > 0: missing address).
    """
    if noise and rng.random() < noise * 0.05:
        return ""
    base, long_t, short_t = b.street
    typ = short_t if (noise and rng.random() < noise * 0.7) or (not noise and rng.random() < 0.3) else long_t
    street = f"{typ} {base}" if b.country == "France" else f"{base} {typ}"
    if noise and rng.random() < noise * 0.3:
        street = _typo(street, rng)
    num = str(b.number)
    if noise and rng.random() < noise * 0.15:
        num = f"No. {num}" if b.country == "India" else ""
    keep_pc = not (noise and rng.random() < noise * 0.5)
    keep_state = not (noise and rng.random() < noise * 0.5)
    if b.country == "US":
        state = (b.state_abbr if rng.random() < 0.7 else b.state) if keep_state else ""
        tail = " ".join(x for x in [state, b.postcode if keep_pc else ""] if x)
        parts = [f"{num} {street}".strip(), b.city, tail]
    elif b.country == "India":
        state = b.state if keep_state and rng.random() < 0.4 else ""
        parts = [num, street, b.area, b.city, " ".join(x for x in [state, b.postcode if keep_pc else ""] if x)]
    else:
        city = f"{b.postcode} {b.city}" if keep_pc else b.city
        parts = [f"{num} {street}".strip(), city]
    if noise and rng.random() < noise * 0.35:
        parts.insert(rng.randrange(1, max(2, len(parts))), b.landmark)
    if noise and rng.random() < noise * 0.15 and len(parts) > 2:
        parts[0], parts[1] = parts[1], parts[0]  # component reordering
    addr = ", ".join(p for p in parts if p)
    if b.country == "France" and noise and rng.random() < noise:
        addr = unidecode(addr)
    return addr


def _pick_count(rng: random.Random) -> Tuple[int, int]:
    """Draw the number of (S2, S3) matches for a non-singleton; guarantees at least one match.

    Args:
        rng: Random source.

    Returns:
        Tuple ``(n_s2, n_s3)`` with ``n_s2 + n_s3 >= 1``; about 25% have several from both.
    """
    while True:
        n2 = rng.choices([0, 1, 2, 3], [0.25, 0.45, 0.22, 0.08])[0]
        n3 = rng.choices([0, 1, 2, 3], [0.25, 0.45, 0.22, 0.08])[0]
        if n2 + n3 >= 1:
            return n2, n3


def generate_split(rng: random.Random, n_s1: int, countries: Dict[str, float], with_truth: bool) -> Dict[str, pd.DataFrame]:
    """Generate one split (train or test) as DataFrames in competition format.

    Args:
        rng: Random source.
        n_s1: Number of Source 1 entities.
        countries: Country -> sampling weight.
        with_truth: Also return the ground-truth frame.

    Returns:
        Dict with keys ``s1``, ``s2``, ``s3`` (and ``gt`` when ``with_truth``).
    """
    recs: Dict[str, List[dict]] = {"s1": [], "s2": [], "s3": []}
    gt: Dict[str, List[str]] = {}
    used: Dict[str, set] = {c: set() for c in countries}
    counter = 0

    def add(src: str, b: Biz, noise: float) -> str:
        """Append a rendered record to a source; return its temporary key."""
        nonlocal counter
        counter += 1
        key = f"k{counter}"
        recs[src].append({"key": key, NAME_COL: render_name(b, rng, noise), ADDRESS_COL: render_address(b, rng, noise), COUNTRY_COL: b.country})
        return key

    names, weights = list(countries), list(countries.values())
    for _ in range(n_s1):
        country = rng.choices(names, weights)[0]
        b = make_biz(rng, country, used[country])
        s1_key = add("s1", b, 0.0)
        matches: List[str] = []
        if rng.random() >= 0.25:  # non-singleton
            n2, n3 = _pick_count(rng)
            matches += [add("s2", b, rng.uniform(0.3, 0.7)) for _ in range(n2)]
            matches += [add("s3", b, rng.uniform(0.5, 0.9)) for _ in range(n3)]
        gt[s1_key] = matches
        if rng.random() < 0.10:  # hard negative: same business name, different branch address
            other = make_biz(rng, country)
            other.core, other.legal = b.core, b.legal
            add(rng.choice(["s2", "s3"]), other, rng.uniform(0.2, 0.6))
        for src in ("s2", "s3"):  # distractors: unrelated businesses
            if rng.random() < 0.9:
                add(src, make_biz(rng, country), rng.uniform(0.2, 0.8))

    ids: Dict[str, str] = {}
    frames: Dict[str, pd.DataFrame] = {}
    for src, prefix in (("s1", "S1"), ("s2", "S2"), ("s3", "S3")):
        rows = recs[src]
        if src != "s1":
            rng.shuffle(rows)
        for i, r in enumerate(rows, start=1):
            ids[r["key"]] = f"{prefix}-{i:05d}"
        frames[src] = pd.DataFrame([{ID_COL: ids[r["key"]], NAME_COL: r[NAME_COL], ADDRESS_COL: r[ADDRESS_COL], COUNTRY_COL: r[COUNTRY_COL]} for r in rows])
    if with_truth:
        frames["gt"] = pd.DataFrame(
            [{TRUTH_ID_COL: ids[k], MATCHED_COL: ",".join(sorted(ids[m] for m in ms))} for k, ms in gt.items()]
        )
    return frames


def write_fake_dataset(out_dir, n_train_s1: int = 200, n_test_s1: int = 100, seed: int = SEED) -> Path:
    """Generate and write the full fake dataset.

    Args:
        out_dir: Output root; ``train/`` and ``test/`` subfolders are created.
        n_train_s1: Number of training Source 1 entities.
        n_test_s1: Number of test Source 1 entities.
        seed: Random seed.

    Returns:
        The output root path.
    """
    out_dir = Path(out_dir)
    rng = random.Random(seed)
    for split, n, countries, truth in (("train", n_train_s1, TRAIN_COUNTRIES, True), ("test", n_test_s1, TEST_COUNTRIES, False)):
        d = out_dir / split
        d.mkdir(parents=True, exist_ok=True)
        frames = generate_split(rng, n, countries, truth)
        for src, suffix in zip(("s1", "s2", "s3"), SOURCE_SUFFIXES):
            frames[src].to_csv(d / f"{split}_{suffix}", sep="\t", index=False, lineterminator="\n")
        if truth:
            frames["gt"].to_csv(d / f"{split}_{GROUND_TRUTH_SUFFIX}", sep="\t", index=False, lineterminator="\n")
    return out_dir


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point: write the fake dataset and print a short summary.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=FAKE_DATA_DIR)
    ap.add_argument("--n-train-s1", type=int, default=200)
    ap.add_argument("--n-test-s1", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    out = write_fake_dataset(args.out, args.n_train_s1, args.n_test_s1, args.seed)
    for f in sorted(out.rglob("*.tsv")):
        print(f"{f.relative_to(out)}: {sum(1 for _ in open(f, encoding='utf-8')) - 1} rows")


if __name__ == "__main__":
    main()
