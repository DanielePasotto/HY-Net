#!/bin/bash
# Leave-one-dataset-out: train on the training images of two datasets (ours, orgaseg, organoid), test on the third.
#
#   scripts/lodo.sh <test_domain> <model>        e.g. SCALE=2 AUGMENT=true AUGMENT_RANGE=wide scripts/lodo.sh organoid hynet_res
#
# Data: hynet-prepare-pool writes data/pool and folds/lodo_<domain>.json.
# The training samples are drawn with the same probability from each dataset (--domain_balance), SPE samples per epoch.
# Results: $ROOT/<test_domain>/eval/<model>/fold0 (summary.json, per_well.csv).
# Environment: EPOCHS=100, SPE=400, ROOT=outputs/lodo, SEED=42 (use a different ROOT for each seed), DATA=data/pool,
#              MICRO (gradient-accumulation chunks per batch: 2 for the models with HarmNet, 1 otherwise; 4 for CAMPEOD in the paper),
#              AUGMENT=false (true: random photometric augmentation of the training images), AUGMENT_RANGE=default (wide: ranges that contain the
#              whole robustness sweep), SCALE=0 (F > 1: random zoom of the training images by a factor in [1/F, F]), FOLDS_FILE=folds/lodo_<domain>.json
set -euo pipefail
DOMAIN=$1
MODEL=$2
EPOCHS=${EPOCHS:-100}
SPE=${SPE:-400}
ROOT=${ROOT:-outputs/lodo}
SEED=${SEED:-42}
DATA=${DATA:-data/pool}
AUGMENT=${AUGMENT:-false}
AUGMENT_RANGE=${AUGMENT_RANGE:-default}
SCALE=${SCALE:-0}
# the dual-stream models do not fit a batch of 4 in 12 GB at the very peak: same batch, accumulated over MICRO chunks
case $MODEL in
  hynet_res|hynet_fast|hynet_res_nogate) MICRO=${MICRO:-2};;
  *) MICRO=${MICRO:-1};;
esac
FOLDS=${FOLDS_FILE:-folds/lodo_$DOMAIN.json}
RUNS=$ROOT/$DOMAIN/runs
EVAL=$ROOT/$DOMAIN/eval
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
POOL="--folds_file $FOLDS --fold 0 --dataset_path $DATA --runs_root $RUNS"
TRAIN="$POOL --domain_balance true --samples_per_epoch $SPE --epochs $EPOCHS --amp true --seed $SEED"
RUN="uv run --no-sync"

mkdir -p "$RUNS/$MODEL/fold0"
if [ ! -f "$RUNS/$MODEL/fold0/DONE" ]; then
  $RUN hynet-train --model $MODEL $TRAIN --micro_batches $MICRO --augment $AUGMENT --augment_range $AUGMENT_RANGE --scale_aug $SCALE >"$RUNS/$MODEL/fold0/train.log"
  touch "$RUNS/$MODEL/fold0/DONE"
fi
$RUN hynet-evaluate --model $MODEL $POOL --out_dir "$EVAL"
