# File contracts between pipeline stages

Stages talk **only** through these files (see `CLAUDE.md`). Do not change a schema
without updating this file and telling the team. Columns and stage order come from
`CLAUDE.md`; dtypes marked *(proposed)* are not yet agreed and should be confirmed by
the stage owner. All interim files are Parquet under `data/interim/` (gitignored);
paths come from `src/config.py`.

Run every stage from `code/business_entity_resolution/`:
`python -m src.<stage> --split <train|val|test>`.

## Raw inputs (read-only, `dataset/`)

| File | Columns |
|---|---|
| `{split}_source{1,2,3}.tsv` | `entity_id`, `business_name`, `business_address`, `country` |
| `train_ground_truth.tsv` | `source1_entity_id`, `matched_entity_ids` (comma-separated S2-/S3- IDs, empty = singleton) |

Read with `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)` (`src.io_utils.read_tsv`).

## 0. Split: `data/interim/split.json`

80/20 split of train S1 IDs, seed 42, stratified by (country, singleton flag).
`{"train": [S1 ids...], "val": [S1 ids...]}`. Produced by `src.split.make_split`.

## 1. normalize -> `data/interim/records.parquet`

One row per record from all sources and splits.

| Column | Type | Notes |
|---|---|---|
| `entity_id` | str | `S1-`/`S2-`/`S3-` prefixed |
| `source` | str *(proposed)* | `S1`/`S2`/`S3`, from the ID prefix |
| `country` | str | open set of labels; never enumerate |
| `name_raw`, `addr_raw` | str | unchanged input |
| `name_norm` | str | normalised name |
| `name_core` | str | name without legal suffix / generic words |
| `name_type` | str *(proposed)* | TBD by owner B |
| `name_legal` | str *(proposed)* | legal suffix, if any (TBD by B) |
| `addr_norm` | str | normalised address |
| `numbers` | list[str] *(proposed)* | numbers extracted from the address |
| `postcode` | str | empty if missing |

## 2. block -> `data/interim/candidates_{split}.parquet`

One row per (S1, candidate) pair; the final candidate set the model scores.

| Column | Type | Notes |
|---|---|---|
| `s1_id` | str | S1 entity ID |
| `cand_id` | str | S2 or S3 entity ID |
| `<channel>` | bool | one column per blocking channel (names TBD by B) |
| `block_score` | float *(proposed)* | |

## 3. features -> `data/interim/features_{split}.parquet`

| Column | Type | Notes |
|---|---|---|
| `s1_id`, `cand_id` | str | |
| feature columns | float | pair + context features (owner C) |
| `label` | int (0/1) | train/val only; absent for test |

## 4. train/predict -> `data/interim/preds_{split}.parquet`

| Column | Type | Notes |
|---|---|---|
| `s1_id`, `cand_id` | str | |
| `prob` | float | match probability |

## 5. decide + write -> `output/`

| File | Columns |
|---|---|
| `matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` |
| `candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` |

One row per test S1 ID; IDs comma-separated (no spaces/quotes), sorted, deduplicated, empty
string when none; only S2-/S3- IDs present in the test set; every matched ID is also a
candidate. Written by `src.write_submission.write_results`, then check with
`python3 utils/validate_submission.py ...` (from the repo root).
