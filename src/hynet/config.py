import argparse

def str2bool(value):
    """Parse boolean CLI values; `type=bool` would turn any non-empty string (even "False") into True."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if value.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")

def get_args(add_arguments=None):
    """Parse the shared CLI; `add_arguments(parser)` lets a command register its own extra options."""
    parser = argparse.ArgumentParser(description="Unified Configuration for HY-Net and HarmNet.")
    flag = dict(type=str2bool, nargs="?", const=True)

    # General Execution
    parser.add_argument("--harmnet", default=False, help="Train/Test HarmNet", **flag)
    parser.add_argument("--hynet", default=True, help="Train/Test the segmentation model selected with --model", **flag)
    parser.add_argument("--model", type=str, default="hynet", help="Segmentation model (hynet, hynet_frozen, hynet_decoder, hynet_sum, hynet_nogate, hynet_res, ynet_raw, harm_unet, unet, attunet)")
    parser.add_argument("--both", default=False, help="Train/Test both HarmNet and HY-Net", **flag)
    parser.add_argument("--fine_tune", default=False, help="Fine tune HY-Net", **flag)

    # Cross-validation (see hynet-make-folds)
    parser.add_argument("--folds_file", type=str, default=None, help="JSON with the well-grouped folds; if set, --fold selects the train/val/test wells instead of the train/val/test folders")
    parser.add_argument("--fold", type=int, default=0, help="Fold index used with --folds_file")
    parser.add_argument("--runs_root", type=str, default="outputs/runs", help="Where the weights of the cross-validation runs are saved")

    # Few-shot adaptation to another dataset (see docs/experiments.md, section 3)
    parser.add_argument("--init_weights", type=str, default=None, help="Start the segmentation model from these weights (e.g. a model trained on another dataset) instead of from scratch")
    parser.add_argument("--n_train", type=int, default=None, help="With --folds_file: train on a random subset of this many wells of the fold's training set")
    parser.add_argument("--n_val", type=int, default=None, help="With --folds_file: validate on a random subset of this many wells of the fold's validation set")
    parser.add_argument("--subset_seed", type=int, default=0, help="Seed of the subsets drawn by --n_train/--n_val (same seed: nested subsets)")

    # Training on datasets pooled from several sources (see docs/experiments.md, section 4)
    parser.add_argument("--domain_balance", default=False, help="Draw the training samples with the same probability from every domain (the IMG_<domain>-<id> prefix of the file names)", **flag)
    parser.add_argument("--samples_per_epoch", type=int, default=None, help="Samples drawn per epoch with --domain_balance (default: the size of the training set)")

    # Photometric perturbations (see hynet.perturb)
    parser.add_argument("--augment", default=False, help="Random photometric augmentation (illumination ramp, gamma, blur, noise, JPEG) of the segmentation training images", **flag)
    parser.add_argument("--scale_aug", type=float, default=0.0, help="Random zoom of the segmentation training images and masks by a log-uniform factor in [1/F, F], applied with probability 1/2 (F in (1, 2]; 0 = off)")
    parser.add_argument("--augment_range", type=str, default="default", choices=("default", "wide"), help="Ranges of the --augment levels: 'wide' contains every level of the robustness sweep")
    parser.add_argument("--perturb", type=str, default=None, help="Perturb every image of the segmentation datasets, as 'name:level' (gamma, noise, blur, shading, jpeg), e.g. noise:0.05")

    # Unsupervised consistency of the segmentation model (see hynet.consistency)
    parser.add_argument("--consist", type=float, default=0.0, help="Weight of the loss that makes the prediction identical on two random photometric views of the same image (0 = off)")
    parser.add_argument("--consist_harm", type=float, default=0.0, help="Weight of the same consistency on the harmonized image of the models with HarmNet (0 = off)")
    parser.add_argument("--consist_target", default=False, help="Draw half of the consistency images from the unlabeled images of the test wells (the masks of the test domain are never read: transductive use)", **flag)
    parser.add_argument("--consist_pairs", type=int, default=2, help="Image pairs of the consistency loss per optimizer step")

    # Paths & Directories
    parser.add_argument("--hynet_weights_path", type=str, default="checkpoints/hynet_weights.pth", help="Path to load/save HY-Net weights")
    parser.add_argument("--harmnet_weights_path", type=str, default="checkpoints/harmnet_weights.pth", help="Path to load/save HarmNet weights")
    parser.add_argument("--dataset_path", type=str, default="data/DatasetOrganoidsSplitted_70_15_15", help="Path where is located dataset")
    parser.add_argument("--processed_dataset_path", type=str, default="data/ProcessedDatasetOrganoidsSplitted_70_15_15", help="Path where is located the processed dataset used to train HarmNet")
    parser.add_argument("--harm_input_folder", type=str, default="true_images", help="Sub-folder of each processed split with the raw HarmNet inputs")
    parser.add_argument("--harm_target_folder", type=str, default="images", help="Sub-folder of each processed split with the HarmNet reference targets")

    # Model Architecture
    parser.add_argument("--input_shape", type=int, default=1, help="Input shape")
    parser.add_argument("--norm", type=str, default="batch", choices=("batch", "group"), help="Normalization layers of the segmentation Y-Net/U-Net-like blocks: BatchNorm (published model) or GroupNorm (per-image statistics, independent of the training domain)")
    parser.add_argument("--hidden_units", type=int, default=64, help="Number of hidden units")
    parser.add_argument("--output_shape", type=int, default=1, help="Output shape")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--micro_batches", type=int, default=1, help="Split each batch in this many chunks and accumulate their gradients (same batch size, lower peak memory)")
    parser.add_argument("--amp", default=False, help="Train with fp16 mixed precision (needed to fit the dual-stream models in 12 GB with batch size 4)", **flag)

    # Losses
    parser.add_argument("--alpha", type=float, default=0.8, help="Alpha for Tversky loss")
    parser.add_argument("--beta", type=float, default=0.2, help="Beta for Tversky loss")
    parser.add_argument("--lambda_mae", type=float, default=1.0, help="Lambda for MAE loss")
    parser.add_argument("--lambda_ssim", type=float, default=0.5, help="Lambda for SSIM loss")
    parser.add_argument("--lambda_tv", type=float, default=0.1, help="Lambda for TV loss")
    parser.add_argument("--lambda_edge", type=float, default=0.1, help="Lambda for edge loss")
    parser.add_argument("--lambda_bg", type=float, default=0.5, help="Lambda for background loss")

    # Callbacks
    parser.add_argument("--scheduler_mode", type=str, default="max", help="Scheduler mode")
    parser.add_argument("--scheduler_patience", type=int, default=5, help="Scheduler patience")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--early_stopping_mode", type=str, default="max", help="Early stopping mode")

    # Environment
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    if add_arguments is not None:
        add_arguments(parser)

    args = parser.parse_args()
    return args
