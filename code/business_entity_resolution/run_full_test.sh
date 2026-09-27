#!/usr/bin/env bash
# TEST SIDE on the FULL test set, using the model, threshold and calibration trained by run_train_side.sh:
#   normalize -> block -> features -> predict -> decide (calibration + hybrid rule + exclusivity) -> write_submission
#   -> official validator.
#   ER_DATA_DIR=dataset bash run_full_test.sh
# Requires data/interim/{model.txt, threshold.json, calibration.json, block_meta.json} from the train side; blocking
# refuses to run if the ER_* blocking/normalize settings differ from the train side's (block_meta.json "config").
# Decision: ER_DECISION=hybrid (default) and ER_EXCLUSIVE=1 (default) - exclusivity is resolved in ONE pass over the
# decisions of ALL test S1s (decide streams preds_test.parquet in S1-complete chunks, then resolves duplicate claims).
# Knobs: ER_MAX_MEM_GB (default 8), ER_N_JOBS (blocking workers, default 4), VALIDATE_IDS=1 (validator ID-existence
# check, ~2 GB more memory).
#
# ESTIMATE (NOT measured) for 1.73M test S1 vs 10.0M S2+S3, exp-v2 final config (blocking unchanged: name K'=6xK, rerank +
# address tiebreak, duplicate-name channel; features + ER_FEAT_V2 + ER_FEAT_PHON, both on by default; hybrid decision),
# ER_MAX_MEM_GB=8, ER_N_JOBS=4, extrapolated from the exp-v2 0.2-fraction train side (441k S1 vs 10.3M pool: block 1,282 s,
# features ~36 us/pair = 1,017 s for 28M pairs, predict 48 s for 5.6M pairs):
#   normalize        ~3-4 min    ~1.2 GB    (+ phonetic key columns)
#   block            ~45-80 min  ~6.5-7 GB  India 5 / US 4 / France 2 passes over their pools; ~110M candidate pairs
#   features         ~55-80 min  ~6.5 GB    (was ~25-40: house-number / IDF / context features are per-pair Python)
#   predict          ~15-20 min  ~1.5 GB    (batches sized from the budget)
#   decide           ~3-5 min    ~2-3 GB
#   write_submission ~5-10 min   ~3 GB
#   => total ~2.1-3.3 h, peak ~7 GB (block). Needs the exp-v2 train-side artifacts (model trained with the same flags).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
[ -x ../../.venv/bin/python ] && [ -z "${PYTHON:-}" ] && PY=../../.venv/bin/python
export ER_MAX_MEM_GB="${ER_MAX_MEM_GB:-8}"
export ER_N_JOBS="${ER_N_JOBS:-4}"
export ER_DECISION="${ER_DECISION:-hybrid}"
export ER_EXCLUSIVE="${ER_EXCLUSIVE:-1}"
INTERIM=../../data/interim
for f in model.txt threshold.json calibration.json block_meta.json; do
  [ -e "$INTERIM/$f" ] || { echo "missing $INTERIM/$f - run run_train_side.sh first" >&2; exit 1; }
done

if [ "$(uname)" = "Darwin" ] && [ ! -e /opt/homebrew/opt/libomp/lib/libomp.dylib ] && [ ! -e /usr/local/opt/libomp/lib/libomp.dylib ]; then
  SK_DYLIBS="$("$PY" -c 'import os, sklearn; print(os.path.join(os.path.dirname(sklearn.__file__), ".dylibs"))')"
  export DYLD_FALLBACK_LIBRARY_PATH="$SK_DYLIBS${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

TEST_DIR="$("$PY" -c 'from src.config import TEST_DIR; print(TEST_DIR)')"
echo "test dir: $TEST_DIR   memory budget: ${ER_MAX_MEM_GB} GB   blocking workers: ${ER_N_JOBS}"
echo "decision: ${ER_DECISION}, exclusive=${ER_EXCLUSIVE}, threshold $("$PY" -c 'import json; print(json.load(open("'$INTERIM'/threshold.json"))["threshold"])'), calibration $INTERIM/calibration.json"
"$PY" -c 'import json; m = json.load(open("'$INTERIM'/block_meta.json")); print("train-side blocking config:", m.get("config", "MISSING (created before the config check) - settings are NOT verified"))'


"$PY" -m src.normalize --split test
"$PY" -m src.block --split test
"$PY" -m src.features --split test
"$PY" -m src.predict --split test
"$PY" -m src.decide --split test
"$PY" -m src.write_submission --split test

EXTRA=""
[ "${VALIDATE_IDS:-0}" = "1" ] && EXTRA="--check-ids"
python3 ../../utils/validate_submission.py --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv --test-dir "$TEST_DIR" $EXTRA
