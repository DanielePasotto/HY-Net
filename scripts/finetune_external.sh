#!/bin/bash
# Few-shot adaptation to an external dataset: how many annotated images of a new dataset does a model need?
#
#   scripts/finetune_external.sh <dataset> <model> [pretrained|scratch]
#
# <dataset> is organoid or organosegment (see hynet-prepare-external). The images of the dataset are split in 4 folds
# (folds/external_<dataset>_cv4.json, made with hynet-make-folds). For every external fold j, and for every number of annotated images
# N and every random draw R of those images, the model is
#   pretrained: initialised from the model trained on fold j of OUR data (outputs/runs/<model>/fold<j>/model.pth), then trained on N images
#               of the external training wells (N = 0 is the model as it is, without adaptation);
#   scratch:    trained on the same N images starting from random weights (reference: does the pre-training on our data help at all?).
# Each run is evaluated on the external test fold j. Results: outputs/ft/<dataset>/<init>/n<N>_r<R>/<model>/fold<j>/ ; summarize them
# with hynet-compare --folds_file folds/external_<dataset>_cv4.json --dataset_path data/external/<dataset> --eval_dir outputs/ft/<dataset>/<init>/n<N>_r0 \
#   --extra_eval_dirs outputs/ft/<dataset>/<init>/n<N>_r1 ...
# Environment: NS="2 4 8 16" (images), REPEATS="0 1", EPOCHS=60, NVAL=4,
# SOURCE_RUNS=outputs/runs, SOURCE_MODEL=<model>, FT_ROOT=outputs/ft/<dataset>/<init>,
# MICRO=1 (gradient-accumulation chunks; use 4 for CAMPEOD on 12 GB GPUs)
set -euo pipefail
DS=$1
MODEL=$2
INIT=${3:-pretrained}
NS=${NS:-"2 4 8 16"}
REPEATS=${REPEATS:-"0 1"}
EPOCHS=${EPOCHS:-60}
NVAL=${NVAL:-4}
MICRO=${MICRO:-1}
SOURCE_RUNS=${SOURCE_RUNS:-outputs/runs}
SOURCE_MODEL=${SOURCE_MODEL:-$MODEL}
FOLDS=folds/external_${DS}_cv4.json
DATA=data/external/$DS
ROOT=${FT_ROOT:-outputs/ft/$DS/$INIT}
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN="uv run --no-sync"

for j in 0 1 2 3; do
  SRC=$SOURCE_RUNS/$SOURCE_MODEL/fold$j/model.pth
  if [ "$INIT" = "pretrained" ]; then
    out=$ROOT/n0_r0
    if [ ! -f "$out/$MODEL/fold$j/summary.json" ]; then
      $RUN hynet-evaluate --model $MODEL --folds_file $FOLDS --fold $j --dataset_path $DATA --weights_path $SRC --out_dir $out > /dev/null
    fi
  fi
  for n in $NS; do
    for r in $REPEATS; do
      tag=n${n}_r${r}
      out=$ROOT/$tag
      [ -f "$out/$MODEL/fold$j/summary.json" ] && continue
      INIT_ARGS=""
      [ "$INIT" = "pretrained" ] && INIT_ARGS="--init_weights $SRC"
      OPTS="--model $MODEL --folds_file $FOLDS --fold $j --dataset_path $DATA --runs_root $ROOT/runs/$tag --n_train $n --n_val $NVAL --subset_seed $r"
      mkdir -p "$ROOT/runs/$tag/$MODEL/fold$j"
      $RUN hynet-train $OPTS $INIT_ARGS --micro_batches $MICRO --epochs $EPOCHS --amp true > "$ROOT/runs/$tag/$MODEL/fold$j/train.log"
      # On the cluster's shared filesystem a newly created parent may be briefly invisible to the compute node.
      for attempt in 1 2 3 4 5; do
        mkdir -p "$out/$MODEL/fold$j/predictions" && break
        sleep 2
      done
      [ -d "$out/$MODEL/fold$j/predictions" ] || { echo "cannot create $out/$MODEL/fold$j/predictions" >&2; exit 1; }
      $RUN hynet-evaluate $OPTS --out_dir $out > /dev/null
      echo "$DS $INIT $MODEL fold$j $tag done"
    done
  done
done
