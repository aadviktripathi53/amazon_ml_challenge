# Experiment plan v2 (run AFTER the full test run has finished)

Branch `exp-v2`, worktree `~/er-dev`. Nothing here has been run on the full data yet. All estimates are extrapolated
from the measured realistic-5% baseline (see `experiments.md`, "A+C" row) and are marked as estimates.

## Baseline being improved (measured, 5% setup, val)

| | value |
|---|---|
| blocking recall | 0.8386 (India 0.808, US 0.859) |
| avg candidates / S1 | 63.5 |
| val macro F0.5 (hybrid + exclusivity) | 0.8717 |
| stage times | normalize 130 s, block 565 s (6.6 GB), features 94 s (6.1 GB), train 505 s (4.9 GB), predict 11 s, decide 8 s |

False-negative analysis (val): model FN = 6.6% of true pairs, blocking misses = 16.1%; FN causes: empty address 22%,
same name + conflicting house numbers 17%, same name low score 15%, trade/DBA name 14%, decision layer 12%,
non-Latin script 10%. False positives: 2.0% of predictions, mostly identical names of different businesses (50%)
and co-located businesses (30%).

## Common settings

```bash
cd ~/er-dev/code/business_entity_resolution
export PYTHON=<venv python> ER_DATA_DIR=<repo>/dataset ER_TRAIN_S1_FRAC=0.05 ER_MAX_MEM_GB=8 ER_N_JOBS=4
```

Step 0 (once): `split` + `normalize --split train` in the worktree (records need the new `name_phon`/`name_skel`
columns): ~2.5 min, ~1.2 GB.

Blocking experiments run `python -m src.block --split train` only (writes train + val candidates and prints recall,
per-channel recall, candidates per S1) and need no retrain. Keep rule (organisers rank smaller candidate sets higher):
**keep a blocking change only if it gains >= +0.02 recall per +5 avg candidates per S1.**

Feature experiments run `features` (train, val) -> `train` -> `predict val` -> `decide train/val` -> `evaluate`.

## Experiments

| # | Flags (on top of the current config) | Measures | Time (est.) | Peak memory (est.) |
|---|---|---|---|---|
| B0 | none (reproduce the baseline in the worktree) | recall, cands/S1 must equal 0.8386 / 63.5 | ~10 min | ~6.6 GB |
| B1 | `ER_CH_PHON=1` | recall (overall, per country, non-Latin subset), cands/S1, block time | ~12-14 min (+1 TF-IDF channel of 4: IDF fit, shard build and search each ~+25-35%) | ~6.8 GB (query blocks shrink to ~131k via the memory model: ~29.6 KB/query) |
| B2 | `ER_CH_KEY=1` | same | ~10-11 min (postcode coverage is low: ~12% US, ~1% India, so few keys) | ~6.6 GB |
| B3 | best of B1/B2 combined (if both pass) | same | ~13-15 min | ~6.8 GB |
| F1 | chosen blocking + `ER_FEAT_PHON=1` | val F0.5 overall / singletons / non-singletons / per country, non-Latin subset | ~18-22 min (features +15%, train ~+10%) | features ~6.2 GB, train ~5 GB |
| F2 | chosen blocking + `ER_FEAT_V2=1` | same, plus FN-by-cause table again (does it rescue the number-conflict / empty-address groups?) | ~22-28 min (features ~3-4 min: per-pair Python for house numbers + IDF sets, plus country stats; train ~+30%) | features ~6.8 GB (country core-name counts + token IDF dicts ~0.6 GB) |
| F3 | chosen blocking + both feature groups (if both help) | same | ~25-30 min | ~6.8 GB |
| R20 | best config at `ER_TRAIN_S1_FRAC=0.2` (full train side) | everything above; compare to 5% | see below | see below |

Blocking experiment total ~45-55 min; feature experiments ~65-80 min.

### R20: retrain at ER_TRAIN_S1_FRAC=0.2 (est.)

441k S1 (353k train, 88k val) instead of 110k, same 10.3M pool.

| stage | time (est.) | peak memory (est.) | notes |
|---|---|---|---|
| split + normalize | ~2.5 min | ~1.5 GB | |
| block (train+val) | ~30-40 min (~40-50 with phon) | ~6.8-7.2 GB | IDF fit ~3 min is fixed; search scales ~4x; India 177k / US 265k queries -> 1-2 passes per pool (3 with phon) |
| features (train+val) | ~8-12 min (~20-25 with v2) | ~6.8 GB | ~28M pairs |
| train | ~25-40 min | ~7 GB | ~22M train pairs > the default cap of 10M rows (1.25M x ER_MAX_MEM_GB): whole S1 groups are sampled to 10M (~45% of train S1s), still ~1.8x the 5% training rows; OOF predictions exist only for those |
| predict / decide / evaluate | ~3 min | ~2 GB | |
| **total** | **~1.2-1.8 h** | **~7.2 GB** | |

Why 0.2: more training S1s for the model (the 5% run trains on ~88k S1), and a validation set 4x larger
(88k S1, ~305k true pairs), so F0.5 differences of ~0.002 become measurable instead of noise.

## Reporting per experiment

recall (overall, per country, per channel), full-true-set captured, **avg candidates per S1** (flag any change that
gains < +0.02 recall per +5 candidates), block/features/train time and peak RSS, val F0.5 (overall, singleton,
non-singleton, per country, precision, recall), and a row in `experiments.md`.

## Risks

- Everything must finish before the next full test run; R20 alone is 1.2-1.8 h.
- The test side must be re-run with the same flags as the chosen training run (the blocking config fingerprint in
  `block_meta.json` enforces the blocking/normalize part; predict checks the feature list against the model).
- Sample-only note: on very small partitions the n-gram cap is set by its 500-document floor, so sparse products are
  much denser than on the full data; sample runs use `ER_BLOCK_CHUNK=200` to stay under 2 GB.
