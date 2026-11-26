#!/bin/bash
# Robustness of trained models to photometric perturbations of the test images (python -m hynet.cli.robustness); resumable.
#
#   scripts/robustness.sh <group> <runs_root> <models...>
#   e.g. FOLDS_FILE=folds/lodo_ours.json scripts/robustness.sh lodo_Wr_ours outputs/lodo_Wr/ours/runs hynet_res
#
# Results: outputs/robustness/<group>/<model>/fold<k>/per_well.csv ; compare with scripts/summarize_robustness.py.
# Environment: FOLDS_FILE (required, e.g. folds/lodo_ours.json), DATASET=data/pool, FOLDS=0, OUT=outputs/robustness, WORKERS=3, SETTINGS (default: clean and the whole sweep; e.g. "clean noise:0.05")
set -euo pipefail
GROUP=$1
RUNS=$2
shift 2
FOLDS_FILE=${FOLDS_FILE:?set FOLDS_FILE, e.g. folds/lodo_ours.json}
DATASET=${DATASET:-data/pool}
FOLDS=${FOLDS:-0}
OUT=${OUT:-outputs/robustness}
WORKERS=${WORKERS:-3}
export PYTHONUNBUFFERED=1

for MODEL in "$@"; do
  for FOLD in $FOLDS; do
    DIR=$OUT/$GROUP/$MODEL/fold$FOLD
    mkdir -p "$DIR"
    if [ ! -f "$DIR/per_well.csv" ]; then
      uv run --no-sync python -m hynet.cli.robustness --model $MODEL --folds_file $FOLDS_FILE --fold $FOLD --dataset_path $DATASET \
        --runs_root $RUNS --out_dir "$OUT/$GROUP" --workers $WORKERS ${SETTINGS:+--settings $SETTINGS}
    fi
  done
done
