#!/usr/bin/env bash
# Run the whole pipeline for train / val / test, print the summary table, then run the official validator.
#   ER_DATA_DIR=dataset_sample bash run_all.sh     (development sample)
#   bash run_all.sh                                 (full data in <repo>/dataset)
# Optional: ER_N_JOBS=<n> parallel blocking workers, ER_CHANNELS=name,ctx,addr, ER_MAX_TRAIN_PAIRS=<rows>.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
[ -x ../../.venv/bin/python ] && [ -z "${PYTHON:-}" ] && PY=../../.venv/bin/python

# macOS: LightGBM needs libomp. If Homebrew's is missing, fall back to the copy bundled with scikit-learn
# (the clean fix is `brew install libomp`).
if [ "$(uname)" = "Darwin" ] && [ ! -e /opt/homebrew/opt/libomp/lib/libomp.dylib ] && [ ! -e /usr/local/opt/libomp/lib/libomp.dylib ]; then
  SK_DYLIBS="$("$PY" -c 'import os, sklearn; print(os.path.join(os.path.dirname(sklearn.__file__), ".dylibs"))')"
  export DYLD_FALLBACK_LIBRARY_PATH="$SK_DYLIBS${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

rm -f ../../data/interim/stage_metrics.jsonl          # start a fresh time / memory log
"$PY" -m src.split
for s in train val test; do "$PY" -m src.normalize --split "$s"; done
for s in train val test; do "$PY" -m src.block --split "$s"; done
for s in train val test; do "$PY" -m src.features --split "$s"; done
"$PY" -m src.train --split train
for s in val test; do "$PY" -m src.predict --split "$s"; done
for s in train val test; do "$PY" -m src.decide --split "$s"; done
"$PY" -m src.evaluate --split val
"$PY" -m src.write_submission --split test
"$PY" -m src.summary

TEST_DIR="$("$PY" -c 'from src.config import TEST_DIR; print(TEST_DIR)')"
python3 ../../utils/validate_submission.py --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv --test-dir "$TEST_DIR"
