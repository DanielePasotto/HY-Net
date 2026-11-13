#!/bin/bash
# Leave-one-dataset-out: train on the training images of two datasets (ours, orgaseg, organoid), test on the third.
#
#   scripts/lodo.sh <test_domain> <model>        e.g. scripts/lodo.sh organoid hynet
#
# Data: hynet-prepare-pool writes data/pool and folds/lodo_<domain>.json, hynet-make-targets (--input_root data/pool --output_root data/pool_processed)
# the HarmNet targets. The models that start from HarmNet first train HarmNet on the training images of the two datasets. The training samples are
# drawn with the same probability from each dataset (--domain_balance), SPE samples per epoch.
# Results: $ROOT/<test_domain>/eval/<model>/fold0 ; compare with
#   hynet-compare --folds_file folds/lodo_<domain>.json --dataset_path data/pool --models hynet ynet_raw unet --eval_dir outputs/lodo/<domain>/eval
# Environment: EPOCHS=100, SPE=400, ROOT=outputs/lodo, MICRO (gradient-accumulation chunks per batch of the segmentation model: 2 for the models with HarmNet, 1 otherwise),
#              SEED=42 (training seed: use a different ROOT for each seed), DATA=data/pool (data/pool_processed trains and tests the model on the output of the classical pipeline instead of the raw images),
#              AUGMENT=false (true: random photometric augmentation of the training images of the segmentation model), SCALE=0 (F > 1: random zoom of the training images by a factor in [1/F, F]), AUGMENT_RANGE=default (wide: ranges that contain the whole robustness sweep),
#              CONSIST=0, CONSIST_HARM=0 (weights of the unsupervised consistency between two photometric views of the same image: prediction and harmonized image),
#              CONSIST_TARGET=false (true: half of those images are the unlabeled test images of the held-out dataset, masks never read), CONSIST_PAIRS=2,
#              HARM_ADV=0 and HARM_MATCH=0 (weights of the domain-adversarial loss and of the domain-statistics penalty of the HarmNet training; MODEL=harmnet trains and stops there; use a different ROOT),
#              NORM=batch (group: GroupNorm in the segmentation model; use a different ROOT), ADABN=false (true: re-estimate the BatchNorm statistics on the test images before predicting; EVAL_ROOT=<dir> for the output)
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
NORM=${NORM:-batch}
CONSIST=${CONSIST:-0}
CONSIST_HARM=${CONSIST_HARM:-0}
CONSIST_TARGET=${CONSIST_TARGET:-false}
CONSIST_PAIRS=${CONSIST_PAIRS:-2}
HARM_ADV=${HARM_ADV:-0}
HARM_MATCH=${HARM_MATCH:-0}
ADABN=${ADABN:-false}
# the dual-stream models do not fit a batch of 4 in 12 GB at the very peak: same batch, accumulated over MICRO chunks
case $MODEL in
  hynet|hynet_res|hynet_frozen|hynet_nogate|harm_unet|harmnet) MICRO=${MICRO:-2};;
  *) MICRO=${MICRO:-1};;
esac
FOLDS=folds/lodo_$DOMAIN.json
RUNS=$ROOT/$DOMAIN/runs
EVAL=${EVAL_ROOT:-$ROOT}/$DOMAIN/eval
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
POOL="--folds_file $FOLDS --fold 0 --dataset_path $DATA --processed_dataset_path data/pool_processed --runs_root $RUNS"
TRAIN="$POOL --domain_balance true --samples_per_epoch $SPE --epochs $EPOCHS --amp true --seed $SEED"
RUN="uv run --no-sync"

mkdir -p "$RUNS/harmnet/fold0" "$RUNS/$MODEL/fold0"
case $MODEL in
  hynet|hynet_frozen|hynet_nogate|harm_unet|harmnet)
    if [ ! -f "$RUNS/harmnet/fold0/DONE" ]; then
      $RUN hynet-train --harmnet true --hynet false $TRAIN --harm_domain_adv $HARM_ADV --harm_moment_match $HARM_MATCH > "$RUNS/harmnet/fold0/train.log"
      touch "$RUNS/harmnet/fold0/DONE"
    fi;;
esac
[ "$MODEL" = harmnet ] && exit 0
if [ ! -f "$RUNS/$MODEL/fold0/DONE" ]; then
  $RUN hynet-train --model $MODEL $TRAIN --micro_batches $MICRO --augment $AUGMENT --augment_range $AUGMENT_RANGE --scale_aug $SCALE --norm $NORM \
    --consist $CONSIST --consist_harm $CONSIST_HARM --consist_target $CONSIST_TARGET --consist_pairs $CONSIST_PAIRS >"$RUNS/$MODEL/fold0/train.log"
  touch "$RUNS/$MODEL/fold0/DONE"
fi
$RUN hynet-evaluate --model $MODEL $POOL --norm $NORM --adabn $ADABN --out_dir "$EVAL"
