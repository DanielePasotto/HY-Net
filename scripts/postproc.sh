#!/bin/bash
# Threshold chosen on the validation wells and mirroring at inference (python -m hynet.cli.postproc) for the cross-validation models; resumable.
#
#   scripts/postproc.sh <group> <runs_root> <models...>      e.g. scripts/postproc.sh cv_s42 outputs/runs hynet ynet_raw unet hynet_nogate
#
# Results: outputs/postproc/<group>/<model>/fold<k>/per_well.csv ; compare with scripts/summarize_postproc.py.
# Environment: FOLDS_FILE=folds/k5_seed42.json, DATASET=data/DatasetOrganoidsSplitted_70_15_15, FOLDS="0 1 2 3 4", OUT=outputs/postproc
set -euo pipefail
GROUP=$1
RUNS=$2
shift 2
FOLDS_FILE=${FOLDS_FILE:-folds/k5_seed42.json}
DATASET=${DATASET:-data/DatasetOrganoidsSplitted_70_15_15}
FOLDS=${FOLDS:-0 1 2 3 4}
OUT=${OUT:-outputs/postproc}
export PYTHONUNBUFFERED=1

for MODEL in "$@"; do
  for FOLD in $FOLDS; do
    DIR=$OUT/$GROUP/$MODEL/fold$FOLD
    mkdir -p "$DIR"
    if [ ! -f "$DIR/per_well.csv" ]; then
      uv run --no-sync python -m hynet.cli.postproc --model $MODEL --folds_file $FOLDS_FILE --fold $FOLD --dataset_path $DATASET \
        --runs_root $RUNS --out_dir "$OUT/$GROUP"
    fi
  done
done
