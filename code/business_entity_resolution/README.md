# Business Entity Resolution — Amazon ML Challenge 2026

Given noisy business records from three independent sources, this pipeline finds, for every
Source 1 (reference) record, all Source 2 and Source 3 records that refer to the same real-world
business. It is optimised for the challenge metric: **macro F0.5 per Source 1 entity**.

> Status: work in progress during the challenge window (25–27 Sep 2026). Sections marked
> **TBD** are filled in as the pipeline is finalised.

---

## 1. Problem in one paragraph

Source 1 is a deduplicated reference list. Sources 2 and 3 contain partial, noisy copies of the
same businesses — abbreviations (Pvt/Private, Rd/Road), typos, legal-suffix differences,
word-order swaps, landmark-based addresses, missing postcodes and transliterations — with no
shared identifiers. A Source 1 record may match zero, one or many Source 2/3 records. Training
data covers the US and India; the test set also contains France, which never appears in training.
Precision is weighted twice as heavily as recall, and correctly predicting "no match" for a
singleton earns full credit.

## 2. Approach overview

The pipeline has five stages that communicate only through files, so each can be developed,
tested and replaced independently.

| # | Stage | What it does | Output |
|---|-------|--------------|--------|
| 1 | **Normalise** | Unicode/accent folding, abbreviation expansion using a synonym map **mined from training matches**, a hand-written French legal-form/street-word layer, name split into core / business-type / legal suffix, number and postcode extraction | `data/interim/records.parquet` |
| 2 | **Block** | Multi-channel candidate generation: char n-gram TF-IDF on names and addresses, number/postcode keys, multilingual sentence-embedding nearest neighbours; channels are unioned | `data/interim/candidates_{split}.parquet` |
| 3 | **Features** | Pairwise string similarities (Jaro-Winkler, token set/sort ratios, TF-IDF cosine, embedding cosine), number agreement/conflict, plus **context features**: candidate rank and score margins within the S1 entity, competition from other S1 entities, name and address frequency | `data/interim/features_{split}.parquet` |
| 4 | **Score** | LightGBM binary classifier trained with GroupKFold (grouped by S1 entity), isotonic calibration; optional small cross-encoder for uncertain pairs | `data/interim/preds_{split}.parquet` |
| 5 | **Decide + write** | Consistency layer (exclusivity across S1 entities, cross-source support), then a **metric-aware decision**: per S1 entity, choose the candidate set (including the empty set) that maximises expected F0.5 | `output/*.tsv` |

Full methodology, experiments and ablations: see `Documentation_template.md` and `experiments.md`.

## 3. Repository structure

```
.
├── dataset/                     # competition data (not committed)
│   ├── train/                   # train_source1..3.tsv, train_ground_truth.tsv
│   └── test/                    # test_source1..3.tsv
├── data/interim/                # intermediate parquet files (not committed)
├── output/                      # matching_results.tsv, candidate_pairs.tsv (not committed)
├── code/business_entity_resolution/
│   ├── src/
│   │   ├── config.py            # all paths and seeds in one place
│   │   ├── io_utils.py          # TSV loaders
│   │   ├── normalize.py         # stage 1
│   │   ├── block.py             # stage 2
│   │   ├── features.py          # stage 3
│   │   ├── train.py             # stage 4 (training)
│   │   ├── predict.py           # stage 4 (inference)
│   │   ├── decide.py            # stage 5 (consistency + expected-F0.5 decision)
│   │   ├── write_submission.py  # writes both output TSVs
│   │   └── evaluate.py          # macro F0.5, blocking recall, validation split
│   ├── tests/                   # unit tests (metric, writer)
│   ├── run_all.sh               # end-to-end pipeline
│   ├── README.md                # this file
│   └── requirements.txt         # pinned dependencies
├── notebooks/                   # EDA and exploration
├── reports/                     # EDA outputs
├── docs/contracts.md            # schemas of files passed between stages
├── utils/validate_submission.py # official validator (from organisers)
├── experiments.md               # experiment log
└── CLAUDE.md                    # context file for the coding assistant
```

## 4. Setup

Requirements: Python 3.10+, about 8 GB RAM. A GPU is optional (see section 7).

```bash
git clone <repo-url>
cd <repo>/code/business_entity_resolution
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the competition data so the layout matches `dataset/train/` and `dataset/test/` at the repo
root. To keep it elsewhere, set:

```bash
export ER_DATA_DIR=/path/to/dataset      # Windows (PowerShell): $env:ER_DATA_DIR="C:\path\to\dataset"
```

## 5. Reproduce end to end

All commands run from `code/business_entity_resolution/`.

**One command:**

```bash
bash run_all.sh
```

This runs every stage for the validation split (prints local F0.5 and blocking recall) and for
the test split, writes both output files to `output/`, and runs the official validator.

**Stage by stage:**

```bash
python -m src.normalize
python -m src.block     --split train
python -m src.block     --split val
python -m src.block     --split test
python -m src.features  --split train
python -m src.features  --split val
python -m src.features  --split test
python -m src.train
python -m src.predict   --split val
python -m src.predict   --split test
python -m src.decide    --split val      # prints validation F0.5
python -m src.decide    --split test
python -m src.write_submission
```

**Validate the output:**

```bash
python3 ../../utils/validate_submission.py \
  --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv \
  --test-dir ../../dataset/test
```

It must print `PASS` before any file is uploaded.

## 6. Local evaluation

```bash
python -m src.evaluate --split val            # macro F0.5 + blocking report
python -m src.evaluate --loco                 # leave-one-country-out (US→India, India→US)
pytest tests/                                 # metric and writer unit tests
```

- The validation split holds out 20% of Source 1 training ids (seed 42), stratified by country
  and singleton flag. Validation entities are matched against the full Source 2/3 training pool.
- Leave-one-country-out is a proxy for the unseen country (France) in the test set.

## 7. GPU stages on AWS (optional)

The core pipeline runs on CPU. Two optional stages use a GPU on Amazon SageMaker Studio
(JupyterLab space, `ml.g4dn.xlarge`, region `ap-south-1`):

1. **Embeddings:** encode normalised names and addresses; saved as `data/interim/emb_*.npy`.
2. **Cross-encoder (if used):** fine-tuned on hard training pairs; scores uncertain candidates.

Artifacts are shared through S3:

```bash
aws s3 sync data/interim/ s3://<bucket-name>/interim/     # upload
aws s3 sync s3://<bucket-name>/interim/ data/interim/     # download
```

To reproduce **without AWS**, the same scripts run on CPU (slower):

```bash
python -m src.embed --device cpu        # TBD: exact script name once implemented
```

## 8. Models and licenses

Only models under MIT or Apache 2.0 with at most 8B parameters are used (challenge rule).

| Model | Purpose | Params | License | Verified |
|-------|---------|--------|---------|----------|
| LightGBM | Pair classifier | — | MIT | ☐ |
| TBD (e.g. multilingual-e5-small) | Embeddings for blocking/features | TBD | TBD | ☐ |
| TBD (optional cross-encoder) | Re-scoring uncertain pairs | TBD | TBD | ☐ |

## 9. Compliance

- No external databases, APIs, geocoding, web lookups or data augmentation from the internet.
- All normalisation rules are either mined from the provided training data or written by hand
  in `src/normalize.py`.
- TF-IDF statistics are fitted on the provided train and test records only.

## 10. Reproducibility

- All random seeds fixed to 42 (`src/config.py`).
- Dependency versions pinned in `requirements.txt`.
- Every leaderboard submission is tagged in git (`sub-01`, `sub-02`, …) so any submission can be
  rebuilt exactly.
- Expected runtime on a laptop CPU: **TBD**.

## 11. Results

| Version | Change | Val F0.5 | Blocking recall | Public LB |
|---------|--------|----------|-----------------|-----------|
| sub-01 | All-empty probe | — | — | TBD |
| sub-02 | Baseline | TBD | TBD | TBD |

Full log: `experiments.md`.

## 12. Team

| Role | Member | Responsibility |
|------|--------|----------------|
| A — Lead / integrator | TBD | Pipeline integration, decision layer, submissions |
| B — Data & blocking | TBD | Normalisation, synonym mining, blocking |
| C — Modelling | TBD | Features, LightGBM, calibration |
| D — Evaluation, AWS & docs | TBD | Validation, SageMaker stages, methodology |

## 13. Working on this repo

- One branch per person; merge small changes into `main` often. `main` must always run end to end.
- Never commit `dataset/`, `data/`, `output/`, credentials or API keys.
- After any meaningful change, add a row to `experiments.md` with validation F0.5 and blocking recall.
- Keep every function documented with a docstring.
