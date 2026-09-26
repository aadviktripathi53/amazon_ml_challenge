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
`{"train": [S1 ids...], "val": [S1 ids...]}`. Produced by `python -m src.split` (no `--split` argument).
Validation S1s are always searched against the FULL train S2/S3 pool.

## 1. normalize -> `data/interim/records_{train,test}.parquet`

One row per record of all three sources of that split (`val` re-uses `records_train.parquet`). Written in chunks,
so it never needs to fit in memory.

| Column | Type | Notes |
|---|---|---|
| `entity_id` | str | `S1-`/`S2-`/`S3-` prefixed, whitespace-stripped |
| `source` | str | `S1`/`S2`/`S3`, from the ID prefix |
| `country` | str | raw label (stripped); open set, never enumerated |
| `name_raw`, `addr_raw` | str | unchanged input |
| `name_norm` | str | lowercase, unidecode, `&`->`and`, punctuation removed, spaces collapsed |
| `name_core` | str | `name_norm` without legal suffixes / generic words (falls back to `name_norm` if that would be empty) |
| `addr_norm` | str | like `name_norm` plus abbreviation expansion (rd->road, st->street, av->avenue, nr->near, ...); `null` dropped |
| `numbers` | list[str] | ALL digit runs of the raw address, in order, leading zeros kept |
| `postcode` | str | last 5-6 digit run of the address, else `""` |
| `city_token` | str | best-effort city component (see `src/normalize.py::city_token`), else `""` |
| `name_tokens` | list[str] | `name_core.split()` |

`name_type` / `name_legal` from the first draft of this contract are not produced in baseline v0.

## 2. block -> `data/interim/candidates_{split}.parquet`

One row per (S1, candidate) pair; the final candidate set the model scores. One Parquet row group per (country
partition, S1 chunk); a row group never splits an S1's candidates. `val` S1s are searched against the full train pool.

| Column | Type | Notes |
|---|---|---|
| `s1_id` | str | S1 entity ID |
| `cand_id` | str | S2 or S3 entity ID |
| `country` | str | partition label (= S1 country; `"*"` semantics not used, the S1's own label is written) |
| `ch_name`, `ch_ctx`, `ch_addr` | bool | pair found by that channel (name_core / name_core+postcode+city / addr_norm) |
| `rank_name`, `rank_ctx`, `rank_addr` | float32 | rank (1 = best) within that channel and source; NaN if not found by it |
| `cos_name`, `cos_ctx`, `cos_addr` | float32 | EXACT char-3-gram TF-IDF cosine of the pair on that channel (for every union pair) |
| `block_score` | float32 | max of the exact cosines |

`data/interim/block_meta.json`: `{country_equal_share, n_true_pairs, partition_by_country}` measured on train
ground truth (partition by country iff share >= 99.5%); val/test reuse it.
Env knobs: `ER_CHANNELS` (default `name,ctx,addr`), `ER_N_JOBS`, `ER_BLOCK_CHUNK`.

## 3. features -> `data/interim/features_{split}.parquet`

Same rows and order as `candidates_{split}.parquet`.

| Column | Type | Notes |
|---|---|---|
| `s1_id`, `cand_id` | str | |
| 35 feature columns (`src/features.py::FEATURE_COLUMNS`) | float32 | fuzzy scores on name_norm (`nn_*`), name_core (`nc_*`), addr_norm (`ad_*`): ratio/partial/tsort/tset/jw in [0,1]; `name_jaccard`, `core_exact`, `name_len_diff`, `addr_empty_s1/cand`, `num_shared`, `num_conflict`, `pc_state` (1 match / 0 missing / -1 conflict); blocking: `cand_is_s3`, `ch_*`, `n_channels`, `rank_*`, `cos_*`, `block_score` |
| `label` | int8 (0/1) | train/val only; absent for test |

## 4. train/predict -> `data/interim/preds_{split}.parquet`, `model.txt`

| Column | Type | Notes |
|---|---|---|
| `s1_id`, `cand_id` | str | same order as the feature file |
| `prob` | float32 | match probability (`preds_train`: OUT-OF-FOLD, 5-fold GroupKFold by `s1_id`) |
| `label` | int8 | `preds_train` only |

`model.txt` = final LightGBM model (`save_model`), refit on all train rows.

## 5. decide -> `threshold.json`, `matches_{split}.parquet`

`threshold.json` (from `decide --split train`): `{threshold, oof_macro_f05, sweep, n_s1}`, threshold chosen on the
out-of-fold train predictions. `matches_{val,test}.parquet`: `s1_id, cand_id, prob` for pairs with `prob >= threshold`.

## 6. write_submission -> `output/`

| File | Columns |
|---|---|
| `matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` |
| `candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` |

One row per test S1 ID (S1s without candidates are appended with empty lists); IDs comma-separated (no spaces/quotes),
sorted, deduplicated, empty string when none; only S2-/S3- IDs present in the test set; every matched ID is also a
candidate; `candidate_pairs.tsv` is exactly `candidates_test.parquet` (row counts of candidates, features and preds are
asserted equal). Written by `src.write_submission.write_results_streaming`, then check with
`python3 utils/validate_submission.py ...` (from the repo root).

## Side files

`data/interim/metrics/*.json` (headline metrics per stage), `data/interim/stage_metrics.jsonl` (wall time and peak
memory per stage), both read by `python -m src.summary`.
