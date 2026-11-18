# HY-Net

## Overview
HY-Net is a deep learning project for semantic segmentation. It features a dual-network architecture primarily comprising **HarmNet** (for image harmonization/processing) and **Asymmetric Gated Y-Net** (for the main segmentation task). This repository contains the source code for training, evaluating, and running inference with these models.

## Installation

The environment is managed with [uv](https://docs.astral.sh/uv/). Pick the extra matching your hardware:

```bash
uv sync --extra cu128   # NVIDIA GPU (CUDA 12.8 build of PyTorch)
uv sync --extra cpu     # CPU only
```

Python 3.13 is pinned in `.python-version` (uv downloads it if needed). Commands below are run with `uv run --extra <cpu|cu128> ...`
(or activate the environment and call the `hynet-*` commands directly).

> **Note:** if the project lives in a cloud-synced folder (e.g. OneDrive), keep the virtual environment outside it:
> `UV_PROJECT_ENVIRONMENT=~/.venvs/hynet uv sync --extra cu128`.

### Main Dependencies
- PyTorch & Torchvision
- NumPy, SciPy, scikit-image, Pillow, OpenCV (headless, for the HarmNet targets), tqdm
- pytorch-msssim

## Repository Layout

```text
src/hynet/
├── config.py          # argparse configuration shared by all commands
├── data.py            # OrganoidDataset and transforms
├── engine.py          # training / validation loops, EarlyStopping
├── losses.py          # Tversky-BCE and Region-Aware Composite loss
├── metrics.py         # SemanticEvaluator (training-time metrics), harmonization metrics
├── evaluation.py      # foreground/boundary/instance metrics, per-well aggregation, bootstrap CIs
├── folds.py, splits.py
│                      # well-grouped cross-validation folds and the datasets/paths of a run
├── scoring.py         # score saved prediction masks
├── targets.py         # classical pre-processing pipeline = HarmNet targets
├── perturb.py         # photometric perturbations: robustness sweep and training augmentation
├── models/
│   ├── harmnet.py     # HarmNet (image harmonization)
│   ├── hynet.py       # HY-Net: HarmNet + Asymmetric Gated Y-Net (with ablation options)
│   ├── baselines.py   # U-Net, Attention U-Net, HarmNet + U-Net
│   └── factory.py     # build_model(name): the models compared in the paper
└── cli/               # train, test, evaluate, compare, make_folds, make_targets, predict, benchmark, prepare_external, prepare_pool, robustness, postproc
scripts/               # experiment drivers (lodo, robustness, postproc), summaries, HarmNet visualization
folds/                 # cross-validation folds (versioned, so that results can be reproduced)
tests/                 # unit tests (pytest)
outputs/               # runs and evaluations (not versioned)
data/                  # datasets (not versioned)
checkpoints/           # model weights (not versioned)
```

## Dataset Structure

Two folders are used (both under `data/`, not versioned). They share the same train/val/test partition (split at well level).

- **Segmentation dataset (`--dataset_path`, default `data/DatasetOrganoidsSplitted_70_15_15`)**: raw `images` and binary ground-truth `masks` for each split. Used to train/test HY-Net and by `hynet-predict`.
- **Processed dataset (`--processed_dataset_path`, default `data/ProcessedDatasetOrganoidsSplitted_70_15_15`)**: used to train/test HarmNet. For each split it contains the raw inputs, the binary masks and the harmonization targets (the output of the classical pre-processing pipeline: CLAHE, illumination correction, histogram equalization, bilateral filter).

```text
data/
├── DatasetOrganoidsSplitted_70_15_15/
│   └── train|val|test/{images,masks}/
└── ProcessedDatasetOrganoidsSplitted_70_15_15/
    └── train|val|test/
        ├── true_images/   # raw images (HarmNet input)        --harm_input_folder
        ├── images/        # processed images (HarmNet target) --harm_target_folder
        └── masks/         # binary organoid masks
```

Files are paired by name (`IMG_B11_Interval1.tif` ↔ `MASK_B11_Interval1.tif`); `hynet-predict` only needs `images/` in the folder passed with `--dataset_path`.

The processed dataset can be regenerated from the segmentation dataset (or built for any other dataset with the same `<split>/images` and `<split>/masks` layout):
```bash
uv run --extra cu128 hynet-make-targets --input_root data/DatasetOrganoidsSplitted_70_15_15 --output_root data/ProcessedDatasetOrganoidsSplitted_70_15_15
```
The targets are the output of the classical pipeline in `src/hynet/targets.py`. Its filters are sized in pixels for our 1280x960 images, so the images of other
datasets are first brought to that size (`--pipeline_size`); on our dataset the command reproduces the released targets to within one grey level.

## Usage

All scripts share a unified configuration via `argparse` defined in `src/hynet/config.py`.

### 1. Training (`hynet-train`)
You can define which network you want to train by passing the appropriate flags. Check `src/hynet/config.py` for a full list of hyperparameters such as learning rate, batch size, epochs, and loss weights.

- **Train HY-Net (Default)**
  ```bash
  uv run --extra cu128 hynet-train
  ```
- **Train HarmNet**
  ```bash
  uv run --extra cu128 hynet-train --harmnet True --hynet False
  ```
- **Train Both**
  ```bash
  uv run --extra cu128 hynet-train --both True
  ```
- **Fine-Tune HY-Net**
  ```bash
  uv run --extra cu128 hynet-train --fine_tune True
  ```

### 2. Testing / Evaluation (`hynet-test`)
Evaluate the trained models on your separate test set. Metrics such as mIoU, Dice Score, Precision, and Recall are calculated automatically.

- **Test HY-Net**
  ```bash
  uv run --extra cu128 hynet-test --hynet True
  ```

### 3. Detailed evaluation (`hynet-evaluate`)
`hynet-test` reports pixel metrics macro-averaged over organoid and background. `hynet-evaluate` reports the organoid class on its own
and treats the **well** as the unit of analysis (the frames of a well are not independent):

- organoid Dice, IoU, precision, recall (per well and pooled over pixels), plus the background Dice and the `(organoid+background)/2` score used for the manuscript's mDice;
- boundary metrics: HD95 and boundary Dice (`--boundary_tolerance`, pixels);
- instance metrics on connected components matched at `--iou_threshold` (precision, recall, F1, PQ), stratified by component size (`--size_edges`, pixels);
- Dice by organoid density (wells below/above the median foreground fraction);
- 95% bootstrap confidence intervals obtained by resampling wells (`--n_boot`).

```bash
uv run --extra cu128 hynet-evaluate                       # runs HY-Net on the test split
uv run --extra cu128 hynet-evaluate --reuse_predictions   # re-scores the masks already in outputs/eval/test/predictions
```
Outputs go to `outputs/eval/<split>/`: `per_image.csv`, `per_well.csv`, `summary.json` and the binary `predictions/`.
Masks from any other model can be scored by placing them in `predictions/` (PNG, 480x640, named like the images) and using `--reuse_predictions`.
The metrics live in `src/hynet/evaluation.py` (numpy/scipy only); unit tests: `uv run --extra cpu pytest`.
Touching organoids are a single object in the semantic ground truth, so instance metrics evaluate organoid clusters.

### 4. Reproducing the main results
The main comparison is leave-one-dataset-out (LODO) on three pooled datasets: ours (`ours`), OrgaSegment (`orgaseg`) and OrganoID (`organoid`). Each model is trained on the training images of two of them and tested on the third; the metric is the per-well foreground Dice. All commands run from the repository root; the datasets and `outputs/` are not versioned.

```bash
# 1. data: pooled dataset, folds/lodo_<domain>.json, HarmNet targets (needs the original datasets in data/, see --help for the paths)
uv run --extra cu128 hynet-prepare-pool
uv run --extra cu128 hynet-make-targets --input_root data/pool --output_root data/pool_processed

# 2. final model: residual HarmNet + Y-Net, random zoom (SCALE=2), photometric augmentation with the wide ranges
#    (one root per seed: _s43 for SEED=43; test domains: ours | orgaseg | organoid)
SCALE=2 AUGMENT=true AUGMENT_RANGE=wide ROOT=outputs/lodo_Wr scripts/lodo.sh ours hynet_res
SCALE=2 AUGMENT=true AUGMENT_RANGE=wide ROOT=outputs/lodo_Wr_s43 SEED=43 scripts/lodo.sh ours hynet_res
#    on OrganoID the default augmentation ranges are better (omit AUGMENT_RANGE=wide)
SCALE=2 AUGMENT=true ROOT=outputs/lodo_Ra scripts/lodo.sh organoid hynet_res

# 3. controls and baselines with the same recipe
SCALE=2 AUGMENT=true AUGMENT_RANGE=wide ROOT=outputs/lodo_Wy scripts/lodo.sh ours ynet_raw    # same Y-Net without HarmNet: isolates the harmonizer
SCALE=2 AUGMENT=true ROOT=outputs/lodo_Ua scripts/lodo.sh ours unet                           # U-Net
SCALE=2 AUGMENT=true MICRO=4 ROOT=outputs/lodo_Ka scripts/lodo.sh ours campeod                # CAMPEOD (third-party code cloned into CAMPEOD/, wrapper src/hynet/models/campeod.py; 12 GB GPU needs MICRO=4)
SCALE=2 AUGMENT=true AUGMENT_RANGE=wide ROOT=outputs/lodo_Wh scripts/lodo.sh ours hynet_fast  # harmonizer at half resolution: 25% faster inference

# 4. robustness to photometric perturbations (gamma, noise, blur, uneven illumination, JPEG) of the models trained above
FOLDS_FILE=folds/lodo_ours.json DATASET=data/pool FOLDS=0 scripts/robustness.sh lodo_Wr_ours outputs/lodo_Wr/ours/runs hynet_res
FOLDS_FILE=folds/lodo_ours.json DATASET=data/pool FOLDS=0 scripts/robustness.sh lodo_Ka_ours outputs/lodo_Ka/ours/runs campeod

# 5. summaries: per-well Dice per group and seed, paired differences, robustness with bootstrap CIs over wells
python scripts/summarize_plan.py --domains ours orgaseg organoid --seeds 42 43
python scripts/summarize_robustness.py --root outputs/robustness --reference campeod_Ka \
    --model hynet_res_Wr=hynet_res:lodo_Wr_ours,lodo_Wr_ours_s43 --model campeod_Ka=campeod:lodo_Ka_ours,lodo_Ka_ours_s43

# 6. cost: parameters, latency and training memory
uv run --extra cu128 hynet-benchmark --inference true --models hynet_res hynet_fast ynet_raw unet campeod
```
`scripts/lodo.sh` documents all its environment variables (`EPOCHS`, `SPE`, `SEED`, `MICRO`, ...) and skips a training whose `fold0/DONE` exists, so every experiment needs its own `ROOT`. The model names are listed in `src/hynet/models/factory.py`.
The 100-epoch runs take 1-3 hours on an RTX 3060 (CAMPEOD about 5 hours); the reported results use seeds 42 and 43, on RTX 3060 and RTX 5070 Ti GPUs.
The cross-validation protocols, external datasets and few-shot adaptation of the earlier work are no longer documented here; the commands (`hynet-make-folds`, `hynet-compare`, `hynet-prepare-external`) remain in the code and the old guide is in the git history (`docs/experiments.md`).

### 5. Inference (`hynet-predict`)
Run inference on a set of images to extract the predicted segmentation masks. The outputs are saved as `.png` images matching the original inputs.

```bash
uv run --extra cu128 hynet-predict --dataset_path path/to/inference_dataset
```
The predicted masks will be saved in `path/to/inference_dataset/results/`.

## Key Configuration Arguments
You can customize the execution by passing arguments in the command line (refer to `src/hynet/config.py` for all available options):

- `--dataset_path`: Path to the raw dataset (Default: `data/DatasetOrganoidsSplitted_70_15_15`).
- `--processed_dataset_path`: Path to the processed dataset (Default: `data/ProcessedDatasetOrganoidsSplitted_70_15_15`).
- `--hynet_weights_path`: Path to load/save HY-Net weights.
- `--harmnet_weights_path`: Path to load/save HarmNet weights.
- `--epochs`: Number of training epochs (Default: 100).
- `--batch_size`: Batch size for the dataloaders (Default: 4).
- `--learning_rate`: Optimizer learning rate (Default: 1e-4).
