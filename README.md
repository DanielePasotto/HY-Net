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
- NumPy, scikit-image, Pillow, tqdm
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
├── models/
│   ├── harmnet.py     # HarmNet (image harmonization)
│   ├── hynet.py       # HY-Net: HarmNet + Asymmetric Gated Y-Net (with ablation options)
│   ├── baselines.py   # U-Net, Attention U-Net, HarmNet + U-Net
│   └── factory.py     # build_model(name): the models compared in the paper
└── cli/               # train, test, evaluate, compare, make_folds, predict, benchmark, prepare_external
scripts/               # analysis scripts (what HarmNet does to the images)
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

### 4. Cross-validated comparison with baselines and ablations
All models are trained and scored with the same code, folds and metrics. Folds are **grouped by well** (optionally by donor/plate with `--groups_csv`),
every well is in the test set of exactly one fold, so each model gets out-of-fold predictions for all the wells.

```bash
uv run --extra cu128 hynet-make-folds                      # -> folds/k5_seed42.json (5 folds, 66 wells, balanced by organoid density)

# for each fold k = 0..4
F="--folds_file folds/k5_seed42.json --fold $k"
uv run --extra cu128 hynet-train --harmnet true --hynet false --amp true $F      # HarmNet of the fold (trained on the fold's training wells only)
for m in hynet hynet_frozen hynet_nogate harm_unet ynet_raw unet attunet; do
  uv run --extra cu128 hynet-train    --model $m --amp true $F         # weights -> outputs/runs/<model>/fold<k>/model.pth
  uv run --extra cu128 hynet-evaluate --model $m $F                    # masks and metrics -> outputs/eval/<model>/fold<k>/
done

uv run --extra cu128 hynet-compare --folds_file folds/k5_seed42.json --models hynet unet attunet ynet_raw
```
`hynet-compare` prints per-well mean [95% bootstrap CI] of each metric and the paired per-well differences with respect to the reference model (`--reference`, default the first).
Models (`--model`): `hynet` (full), `hynet_frozen` (HarmNet not fine-tuned), `hynet_nogate` (concatenation instead of attention+gating),
`ynet_raw` (dual-stream Y-Net without harmonization), `harm_unet` (HarmNet + single-stream U-Net), `unet`, `attunet`.

`--amp true` trains in fp16 mixed precision (loss in fp32). The dual-stream models need it on a 12 GB GPU (about 11.5 GB peak for `hynet` with batch size 4);
`hynet-benchmark --amp true` reports memory and speed per model, and `hynet-benchmark --inference true` the number of parameters and the latency of a forward pass.

**Training seeds.** `--seed` sets the seed; to keep the runs of other seeds apart use `--runs_root outputs/runs_s43` for `hynet-train` and `hynet-evaluate`
and `--out_dir outputs/eval_s43` for `hynet-evaluate`. `hynet-compare --extra_eval_dirs outputs/eval_s43 outputs/eval_s44` then also reports, for the models
available in them, the per-seed Dice, its standard deviation between seeds and the paired differences of the seed-averaged per-well values.

Masks produced by any other model can be scored by writing them as binary PNGs (480x640) in `outputs/eval/<model>/fold<k>/predictions/` and using `--reuse_predictions true`.

### 4b. External datasets
Models trained on the folds can be applied, without adaptation, to public datasets with masks. `hynet-prepare-external` converts them to the layout of this
repository (each image is its own "well"; `--letterbox` keeps the aspect ratio instead of stretching the image to 480x640):

```bash
uv run --extra cu128 hynet-prepare-external --dataset organosegment    # OrgaSegment OrganoidBasic val+eval -> data/external/organosegment, folds/external_organosegment.json
uv run --extra cu128 hynet-prepare-external --dataset organoid         # OrganoID testing sets          -> data/external/organoid, folds/external_organoid.json

# the model of every fold k on the external images (all the images are "fold 0" of the external folds file)
uv run --extra cu128 hynet-evaluate --model hynet --folds_file folds/external_organoid.json --fold 0 --dataset_path data/external/organoid \
    --weights_path outputs/runs/hynet/fold$k/model.pth --out_dir outputs/eval_ext/organoid/m$k

# score and average the five fold models
uv run --extra cu128 hynet-compare --dataset_path data/external/organoid --folds_file folds/external_organoid.json --models hynet unet \
    --eval_dir outputs/eval_ext/organoid/m0 --extra_eval_dirs outputs/eval_ext/organoid/m{1,2,3,4} --replicates_label "fold models"
```
`scripts/visualize_harmonization.py` saves panels (input, HarmNet before and after the end-to-end training) and statistics of what HarmNet does to the images of each dataset.

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
