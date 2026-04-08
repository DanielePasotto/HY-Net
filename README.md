# HY-Net

## Overview
HAGY-Net is a deep learning project for semantic segmentation. It features a dual-network architecture primarily comprising **HarmNet** (for image harmonization/processing) and **HAGY-Net** (for the main segmentation task). This repository contains the source code for training, evaluating, and running inference with these models.

## Installation

A predefined `environment.yml` is provided for easy setup using Conda. This will install Python 3.10 along with PyTorch and all necessary libraries.

```bash
# Create the environment from the yml file
conda env create -f environment.yml

# Activate the environment
conda activate hagynet
```

### Main Dependencies
- Python 3.10
- PyTorch & Torchvision
- NumPy, scikit-image, Pillow, tqdm
- pytorch-msssim

## Dataset Structure

It is highly recommended to use two separate folders for the dataset: one for the raw original images and one for the processed images. 
- **Raw Dataset (`--dataset_path`)**: Used by HAGY-Net (or for fine-tuning). This folder contains the unprocessed original images and masks. Check the configuration argument `--dataset_path` (default `dataset/`).
- **Processed Dataset (`--processed_dataset_path`)**: Used by HarmNet. It contains images that might have been processed, normalized, or harmonized, along with targets, and masks. Check the configuration argument `--processed_dataset_path` (default `processed_dataset/`).
Keeping these datasets strictly separated prevents accidental overwriting of your original ground-truth data, maintains a clear data lineage, and easily allows training the two networks (HarmNet and HAGY-Net) with their respective appropriate data distributions without conflicts.

Both datasets should generally be organized into split folders (`train`, `val`, `test`), each containing the input `images` and corresponding ground-truth `masks`. If training HarmNet, a `targets` directory may also be required.

Example:
```text
dataset/ (or processed_dataset/)
├── train/
│   ├── images/
│   ├── masks/
│   └── targets/ (Optional, mainly for HarmNet)
├── val/
...
```

## Usage

All scripts share a unified configuration via `argparse` defined in `config.py`.

### 1. Training (`train.py`)
You can define which network you want to train by passing the appropriate flags. Check `config.py` for a full list of hyperparameters such as learning rate, batch size, epochs, and loss weights.

- **Train HAGY-Net (Default)**
  ```bash
  python train.py
  ```
- **Train HarmNet**
  ```bash
  python train.py --harmnet True --hagynet False
  ```
- **Train Both**
  ```bash
  python train.py --both True
  ```
- **Fine-Tune HAGY-Net**
  ```bash
  python train.py --fine_tune True
  ```

### 2. Testing / Evaluation (`test.py`)
Evaluate the trained models on your separate test set. Metrics such as mIoU, Dice Score, Precision, and Recall are calculated automatically.

- **Test HAGY-Net**
  ```bash
  python test.py --hagynet True
  ```

### 3. Inference (`predict.py`)
Run inference on a set of images to extract the predicted segmentation masks. The outputs are saved as `.png` images matching the original inputs.

```bash
python predict.py --dataset_path path/to/inference_dataset
```
The predicted masks will be saved in `path/to/inference_dataset/results/`.

## Key Configuration Arguments
You can customize the execution by passing arguments in the command line (refer to `config.py` for all available options):

- `--dataset_path`: Path to the raw dataset (Default: `dataset`).
- `--processed_dataset_path`: Path to the processed dataset (Default: `processed_dataset`).
- `--weights_dir`: Directory where weights are saved during training (Default: `weights`).
- `--hagynet_weights_path`: Path to load/save HAGY-Net weights.
- `--harmnet_weights_path`: Path to load/save HarmNet weights.
- `--epochs`: Number of training epochs (Default: 100).
- `--batch_size`: Batch size for the dataloaders (Default: 4).
- `--learning_rate`: Optimizer learning rate (Default: 1e-4).
