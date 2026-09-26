#!/usr/bin/env bash
# TRAIN SIDE only (no test files are touched): split -> normalize -> block -> features -> train -> predict(val) -> decide -> evaluate.
#   ER_DATA_DIR=dataset ER_TRAIN_S1_FRAC=0.05 bash run_train_side.sh
# Knobs: ER_MAX_MEM_GB (default 8), ER_TRAIN_S1_FRAC (default 1.0), ER_N_JOBS (blocking workers), ER_MAX_TRAIN_PAIRS.
# `block --split train` also produces the val candidates (same pass over the pool); `block --split val` is then a no-op.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
[ -x ../../.venv/bin/python ] && [ -z "${PYTHON:-}" ] && PY=../../.venv/bin/python
export ER_MAX_MEM_GB="${ER_MAX_MEM_GB:-8}"

# macOS: LightGBM needs libomp; fall back to the copy bundled with scikit-learn if Homebrew's is missing.
if [ "$(uname)" = "Darwin" ] && [ ! -e /opt/homebrew/opt/libomp/lib/libomp.dylib ] && [ ! -e /usr/local/opt/libomp/lib/libomp.dylib ]; then
  SK_DYLIBS="$("$PY" -c 'import os, sklearn; print(os.path.join(os.path.dirname(sklearn.__file__), ".dylibs"))')"
  export DYLD_FALLBACK_LIBRARY_PATH="$SK_DYLIBS${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

rm -f ../../data/interim/stage_metrics.jsonl
"$PY" -m src.split
"$PY" -m src.normalize --split train
"$PY" -m src.block --split train
"$PY" -m src.block --split val
for s in train val; do "$PY" -m src.features --split "$s"; done
"$PY" -m src.train --split train
"$PY" -m src.predict --split val
for s in train val; do "$PY" -m src.decide --split "$s"; done
"$PY" -m src.evaluate --split val
"$PY" -m src.summary
