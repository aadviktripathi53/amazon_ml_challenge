#!/usr/bin/env bash
# TEST SIDE on the FULL test set, using the model and threshold already trained by run_train_side.sh:
#   normalize -> block -> features -> predict -> decide -> write_submission -> official validator.
#   ER_DATA_DIR=dataset bash run_full_test.sh
# Requires data/interim/{model.txt, threshold.json, block_meta.json} from the train side. Knobs: ER_MAX_MEM_GB (default 8),
# ER_N_JOBS (blocking workers), VALIDATE_IDS=1 (also run the validator's ID-existence check, ~2 GB more memory).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
[ -x ../../.venv/bin/python ] && [ -z "${PYTHON:-}" ] && PY=../../.venv/bin/python
export ER_MAX_MEM_GB="${ER_MAX_MEM_GB:-8}"
INTERIM=../../data/interim
for f in model.txt threshold.json block_meta.json; do
  [ -e "$INTERIM/$f" ] || { echo "missing $INTERIM/$f - run run_train_side.sh first" >&2; exit 1; }
done

if [ "$(uname)" = "Darwin" ] && [ ! -e /opt/homebrew/opt/libomp/lib/libomp.dylib ] && [ ! -e /usr/local/opt/libomp/lib/libomp.dylib ]; then
  SK_DYLIBS="$("$PY" -c 'import os, sklearn; print(os.path.join(os.path.dirname(sklearn.__file__), ".dylibs"))')"
  export DYLD_FALLBACK_LIBRARY_PATH="$SK_DYLIBS${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

TEST_DIR="$("$PY" -c 'from src.config import TEST_DIR; print(TEST_DIR)')"
echo "test dir: $TEST_DIR   memory budget: ${ER_MAX_MEM_GB} GB   threshold: $("$PY" -c 'import json; print(json.load(open("'$INTERIM'/threshold.json"))["threshold"])')"

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
