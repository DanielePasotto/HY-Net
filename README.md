# HY-Net

HY-Net segments organoids in bright-field microscopy images. A residual harmonization network (**HarmNet**) transforms the raw image, and an
**Asymmetric Gated Y-Net** segments the harmonized and the raw image together; both are trained end to end with a segmentation loss.
This repository contains the code to use the released model and to reproduce the results of the paper.

## Installation

The environment is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra cu128   # NVIDIA GPU (CUDA 12.8 build of PyTorch)
uv sync --extra cpu     # CPU only
```

Python 3.13 is pinned in `.python-version`. Commands are run with `uv run --extra <cpu|cu128> ...`. If the project lives in a cloud-synced folder
(e.g. OneDrive), keep the environment outside it: `UV_PROJECT_ENVIRONMENT=~/.venvs/hynet uv sync --extra cu128`. Unit tests: `uv run --extra cpu pytest`.

## Using the model

The released model (`checkpoints/final/hynet_res_pool_all.pth`, trained on the three datasets of the paper) segments a folder of images:

```bash
uv run --extra cu128 hynet-predict --dataset_path path/to/folder        # reads path/to/folder/images/, writes path/to/folder/results/
```

The masks are binary PNGs at the size of the input images. `--hynet_weights_path` selects other weights and `--model` the architecture (`hynet_res` by default).

## Repository layout

```text
src/hynet/
├── config.py          # command-line options shared by all commands
├── data.py            # datasets and transforms (images resized to 480x640)
├── engine.py          # training / validation loops, early stopping
├── losses.py          # Tversky-BCE loss; region-aware composite loss (HarmNet pre-training)
├── metrics.py         # training-time metrics
├── evaluation.py      # per-well foreground, boundary and instance metrics, bootstrap CIs
├── folds.py, splits.py  # well-grouped folds and the datasets / weight paths of a run
├── scoring.py         # scoring of saved prediction masks
├── targets.py         # classical pipeline = targets of the pre-trained HarmNet variant
├── perturb.py, scale.py # photometric perturbations (robustness sweep, augmentation) and random zoom
├── models/            # HarmNet, HY-Net, U-Net / Attention U-Net, wrappers of CAMPEOD and MACPnet, factory (model names)
└── cli/               # hynet-train, -evaluate, -predict, -robustness, -benchmark, -compare, -make-folds, -make-targets, -prepare-pool
scripts/               # lodo.sh, robustness.sh, summaries (summarize_plan, summarize_robustness, compare_wells), HarmNet analyses
folds/                 # leave-one-dataset-out folds, five-fold CV of our dataset, folds of the released model
tests/                 # unit tests (pytest)
```

Not versioned: `data/` (datasets), `outputs/` (runs and evaluations), `checkpoints/`, and the code of the compared methods
(`baselines/` for nnU-Net, OrganoID and OrgaSegment, the clones `CAMPEOD/`, `MACPnet/`, `OrganoID/`, `OrgaSegment/`, and `src/hynet/models/campeod.py`, `macpnet.py`).

## Reproducing the paper

All commands run from the repository root. The models are named as in `src/hynet/models/factory.py`: `hynet_res` (HY-Net), `ynet_raw`
(Y-Net without HarmNet), `hynet` (pre-trained HarmNet variant), `hynet_nogate` (concatenation instead of attention and gates), `hynet_fast`
(HarmNet at half resolution), `unet`, `attunet`, `campeod`, `macpnet`. Every run needs its own `ROOT` (one per recipe and seed: `_s43`, `_s44`);
`scripts/lodo.sh` documents its environment variables and skips a training whose `fold0/DONE` exists.

```bash
# 1. data: pooled dataset (ours, OrgaSegment, OrganoID) and folds/lodo_<domain>.json; targets for the pre-trained variant
uv run --extra cu128 hynet-prepare-pool                     # see --help for the paths of the original datasets
uv run --extra cu128 hynet-make-targets --input_root data/pool --output_root data/pool_processed

# 2. leave-one-dataset-out (test domain: ours | orgaseg | organoid), Table "lodo"; "wide" = photometric augmentation with the wide ranges
SCALE=2 AUGMENT=true AUGMENT_RANGE=wide ROOT=outputs/lodo_Wr scripts/lodo.sh ours hynet_res              # HY-Net
SCALE=2 AUGMENT=true AUGMENT_RANGE=wide ROOT=outputs/lodo_Wr_s43 SEED=43 scripts/lodo.sh ours hynet_res  # second seed
SCALE=2 AUGMENT=true AUGMENT_RANGE=wide ROOT=outputs/lodo_Wy scripts/lodo.sh ours ynet_raw               # Y-Net without HarmNet
SCALE=2 AUGMENT=true AUGMENT_RANGE=wide ROOT=outputs/lodo_Wg scripts/lodo.sh ours hynet_res_nogate       # concatenation instead of attention and gates
SCALE=2 AUGMENT=true AUGMENT_RANGE=wide ROOT=outputs/lodo_Uw scripts/lodo.sh ours unet                   # U-Net (wide)
SCALE=2 AUGMENT=true ROOT=outputs/lodo_Ua scripts/lodo.sh ours unet                                      # U-Net (default ranges)
SCALE=2 AUGMENT=true AUGMENT_RANGE=wide MICRO=4 ROOT=outputs/lodo_Kw scripts/lodo.sh ours campeod        # CAMPEOD (12 GB GPU: MICRO=4)
SCALE=2 AUGMENT=true ROOT=outputs/lodo_At scripts/lodo.sh ours attunet                                   # likewise macpnet (lodo_Mp), campeod default (lodo_Ka)

# 3. contribution of HarmNet (default ranges): Y-Net without HarmNet, pre-trained variant, residual HarmNet
SCALE=2 AUGMENT=true ROOT=outputs/lodo_Ya scripts/lodo.sh ours ynet_raw
SCALE=2 AUGMENT=true ROOT=outputs/lodo_Ha scripts/lodo.sh ours hynet
SCALE=2 AUGMENT=true ROOT=outputs/lodo_Ra scripts/lodo.sh ours hynet_res

# 4. robustness to photometric perturbations (ours and orgaseg) of the trained models
FOLDS_FILE=folds/lodo_ours.json DATASET=data/pool FOLDS=0 scripts/robustness.sh lodo_Wr_ours outputs/lodo_Wr/ours/runs hynet_res

# 5. summaries: Dice per recipe and seed, paired per-well differences and robustness with bootstrap CIs over wells
python scripts/summarize_plan.py --seeds 42 43 44
python scripts/compare_wells.py Wr-Uw Wr-Kw Wr-Nn Wr-Wy Wr-Wg --seeds 42 43 44
python scripts/summarize_robustness.py --root outputs/robustness --reference campeod_Kw \
    --model hynet_res_Wr=hynet_res:lodo_Wr_ours,lodo_Wr_ours_s43,lodo_Wr_ours_s44 --model campeod_Kw=campeod:lodo_Kw_ours,lodo_Kw_ours_s43,lodo_Kw_ours_s44

# 6. size and inference time; HarmNet at half resolution
uv run --extra cu128 hynet-benchmark --inference true --models hynet_res hynet_fast ynet_raw unet attunet campeod macpnet
SCALE=2 AUGMENT=true AUGMENT_RANGE=wide ROOT=outputs/lodo_Wh scripts/lodo.sh ours hynet_fast
```

The other baselines are run with the code in `baselines/` (Slurm scripts in `slurm/`): nnU-Net v2 (`baselines/nnunet/prepare_pool.py`, then
`nnUNetv2_train` 2D for 100 epochs; robustness with `baselines/nnunet/robustness.py`, results in `outputs/lodo_Nn`), the OrganoID U-Net and released
model (`baselines/organoid/run.py`, `outputs/lodo_Oi` and `lodo_Zo`) and the released OrgaSegment model (`baselines/orgasegment/zeroshot.py`, `outputs/lodo_Zs`).
The gate/attention ablation is a five-fold cross-validation of our dataset (`folds/k5_seed42.json`, models `hynet` and `hynet_nogate`, compared with `hynet-compare`).
`scripts/analyze_residual_harm.py` and `scripts/measure_harm_delta.py` measure what the trained HarmNet does to the image and how much the Y-Net depends on it.

The released model is trained on the training images of all three datasets:

```bash
SCALE=2 AUGMENT=true AUGMENT_RANGE=wide ROOT=outputs/final_all FOLDS_FILE=folds/pool_all.json scripts/lodo.sh all hynet_res
# -> outputs/final_all/all/runs/hynet_res/fold0/model.pth (its test numbers are on the official test images, not comparable with the LODO ones)
```

The 100-epoch runs take 1-3 hours on an RTX 3060 (CAMPEOD about 5 hours). The results of the paper use seeds 42, 43 and 44 on RTX 3060 and RTX 5070 Ti GPUs.

## Evaluation

`hynet-evaluate` (called by `scripts/lodo.sh`) treats the **well** as the unit of analysis: organoid Dice, IoU, precision and recall from the pixel counts
of the frames of a well, boundary metrics (HD95, boundary Dice), instance metrics on connected components by size, and 95% bootstrap confidence
intervals over wells. It writes `per_image.csv`, `per_well.csv`, `summary.json` and the binary `predictions/` (480x640); `--reuse_predictions`
scores masks of any other model placed in `predictions/`. Touching organoids are one object in the semantic ground truth, so instance metrics evaluate organoid clusters.
