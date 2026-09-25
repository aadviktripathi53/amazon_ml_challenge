# Amazon ML Challenge 2026: Business Entity Resolution

## What this project is
We are a team of 4 in a 72-hour hackathon (deadline: Sun 27 Sep 2026, 11:59 PM IST).
Task: we get business records from 3 sources. Source 1 (S1) is a clean, deduplicated reference list.
For EVERY S1 record, find ALL records in Source 2 (S2) and Source 3 (S3) that refer to the same
real-world business. An S1 record can match zero, one, or many S2/S3 records.
Each record has: entity_id (prefix S1-/S2-/S3-), business_name, business_address, country.
Names and addresses are noisy: abbreviations (Pvt/Private, Rd/Road), typos, legal suffixes,
word-order swaps, landmark addresses ("Near SBI ATM"), missing pincodes, transliterations.

## How we are scored
- Macro F0.5: F-beta (beta=0.5) computed PER S1 entity, then averaged over all S1 entities.
- Precision matters 2x more than recall. A wrong merge hurts more than a missed match.
- Singletons (S1 with no true matches): an empty prediction scores 1.0; any prediction scores 0.0.
- So when unsure, predicting fewer matches (or none) is usually better.

## Hard rules (breaking these = disqualification or rejection)
- NO external data, APIs, web lookups, geocoding, or business registries. Only the provided data.
- Final models must be MIT or Apache 2.0 licensed and at most 8B parameters. Check the license before using any pretrained model.
- Train data has US and India only; TEST ALSO HAS FRANCE. Treat country as an open set of strings.
  Never hard-code, filter, or one-hot countries to {US, India}. Prefer language-agnostic features.
- All files are tab-separated. Always read with:
  pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
- Never modify anything in dataset/. Never commit dataset/ or output/ to git.

## Required outputs (in output/)
1. matching_results.tsv: columns source1_entity_id, matched_entity_ids
2. candidate_pairs.tsv: columns source1_entity_id, candidate_entity_ids
Rules for both: exactly one row per test S1 id; IDs comma-separated with no spaces or quotes;
empty string when none; only S2-/S3- ids that exist in the test set; no duplicates in a list.
Every matched id MUST also appear in candidate_pairs.tsv (the exact set the model scored).
Always run the official validator before calling a submission ready:
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test

## Repo structure
```
dataset/train/        train_source1..3.tsv, train_ground_truth.tsv   (provided, read-only)
dataset/test/         test_source1..3.tsv                            (provided, read-only)
data/interim/         intermediate parquet files between stages (gitignored)
output/               final TSVs (gitignored)
code/business_entity_resolution/src/   all pipeline source code
code/business_entity_resolution/README.md, requirements.txt
notebooks/            EDA and exploration scripts
reports/              EDA outputs and analysis
docs/contracts.md     schemas of the files passed between stages
experiments.md        log of every experiment and score
```

## Pipeline: stages talk ONLY through files (so 4 people can work in parallel)
1. normalize  -> data/interim/records.parquet
   (entity_id, source, country, name_raw, addr_raw, name_norm, name_core, name_type, name_legal, addr_norm, numbers, postcode)
2. block      -> data/interim/candidates_{split}.parquet   (s1_id, cand_id, one bool column per blocking channel, block_score)
3. features   -> data/interim/features_{split}.parquet     (s1_id, cand_id, feature columns..., label for train/val)
4. train/predict -> data/interim/preds_{split}.parquet     (s1_id, cand_id, prob)
5. decide + write -> output/matching_results.tsv, output/candidate_pairs.tsv
Splits: train, val, test. Each stage runs from code/business_entity_resolution/ as:
`python -m src.<stage> --split <split>` (stages: normalize, block, features, train, decide).
All paths come from src/config.py (DATA_DIR from env var ER_DATA_DIR, default <repo_root>/dataset).
Do NOT change a file schema without updating docs/contracts.md and telling the team.

## Validation (our "practice exam")
- S1 train ids are split 80/20 (seed 42), stratified by country and singleton flag; saved in data/interim/split.json.
- Validation S1s are matched against the FULL S2/S3 train pool (mimics test conditions).
- Leave-one-country-out check: train on US -> evaluate on India and vice versa (proxy for unseen France).
- Model training uses GroupKFold grouped by s1_id (no pair leakage across folds).
- A change is kept ONLY if it improves validation F0.5. Always report after a change:
  validation F0.5, blocking recall (share of true pairs inside candidates), average candidates per S1.

## Experiment log
Append one row to experiments.md after every meaningful run:
| date-time | who | what changed | val F0.5 | blocking recall | avg cands | public LB (if submitted) |

## Coding standards
- Python 3.10+. Every function has a docstring explaining what it does (organisers require commented code).
- Pin all versions in requirements.txt. Core libs: pandas, numpy, scikit-learn, rapidfuzz, lightgbm, unidecode, pyarrow, pytest.
- Set random seeds everywhere (seed 42) so results are reproducible.
- Pipeline commands run from code/business_entity_resolution/ as `python -m src.<module>` (the validator and run_all.sh run from the repo root).
- Every module imports paths from src/config.py. No hard-coded paths.
- Prefer small, testable functions. Add unit tests for the metric and the output writer.
- No API keys or AWS credentials in code, ever.

## Team roles
- A (lead/integrator): repo, run_all.sh, decision layer, submissions, final zip.
- B (data/blocking): normalisation, abbreviation mining, French handling, number extraction, blocking.
- C (modelling): pair and context features, LightGBM, calibration.
- D (eval/AWS/docs): validation split, metric, leave-one-country-out, SageMaker embeddings, methodology doc.

## AWS usage
- Core work runs locally. SageMaker (GPU, ml.g4dn.xlarge) only for embeddings and the optional cross-encoder.
- Shared artifacts live in the S3 bucket (name contains "sagemaker"). Region: ap-south-1.
- Stop GPU spaces immediately after each job.

## CURRENT PHASE: Phase 0 (setup, dataset not yet received)
Goals for this phase:
- [x] Create the repo structure above, .gitignore (dataset/, data/, output/, __pycache__, .venv), requirements.txt
- [x] src/config.py (paths + seed)
- [x] src/io_utils.py: loaders for sources and ground truth (parse matched ids into sets; empty -> empty set)
- [ ] src/evaluate.py: macro F0.5 exactly as specified, blocking recall report, 80/20 split, unit tests
      (worked example: predicted 3, true 2, 2 correct -> 0.714; singleton edge cases)
- [ ] src/write_submission.py: writes both TSVs per the rules; asserts matches are a subset of candidates
- [ ] docs/contracts.md with the schemas above
- [ ] Stub versions of every pipeline stage + run_all.sh that runs end to end and then the validator
- [ ] A fake mini dataset generator (US/India/France records in the exact TSV format) to test the whole pipeline
- [ ] Script to write an all-empty submission (for the singleton-rate probe)
When the real dataset arrives, the next phase starts with notebooks/eda_assumptions.py.
