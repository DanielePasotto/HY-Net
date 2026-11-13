#!/bin/bash
# Adaptive BatchNorm on trained LODO models: the BatchNorm statistics are re-estimated on the unlabeled test images, the weights are not changed.
#
#   scripts/adabn.sh <root> [<root> ...]        e.g. scripts/adabn.sh outputs/lodo outputs/lodo_s43
#
# Every <root>/<domain>/runs/<model> must already be trained (scripts/lodo.sh); results go to outputs/adabn/<root name>/<domain>/eval/<model>/fold0,
# next to the unadapted ones in <root>/<domain>/eval/<model>/fold0. MODELS and DOMAINS can be overridden.
set -euo pipefail
MODELS=${MODELS:-"hynet ynet_raw"}
DOMAINS=${DOMAINS:-"organoid orgaseg ours"}
for ROOT in "$@"; do
  for DOMAIN in $DOMAINS; do
    for MODEL in $MODELS; do
      RUNS=$ROOT/$DOMAIN/runs
      if [ ! -f "$RUNS/$MODEL/fold0/DONE" ] || { [ "$MODEL" = hynet ] && [ ! -f "$RUNS/harmnet/fold0/DONE" ]; }; then
        echo "skip $ROOT $DOMAIN $MODEL: not trained"; continue
      fi
      ROOT=$ROOT EVAL_ROOT=outputs/adabn/$(basename "$ROOT") ADABN=true scripts/lodo.sh "$DOMAIN" "$MODEL"
    done
  done
done
