"""Datasets and file locations for a run, either on the original train/val/test folders or on a fold of --folds_file."""
import functools
import os

from hynet import data, perturb
from hynet.evaluation import well_id
from hynet.folds import load_folds, fold_wells

def run_tag(args, split="test"):
    return f"fold{args.fold}" if args.folds_file else split

def weights_path(args):
    """Weights of the segmentation model: <runs_root>/<model>/fold<k>/model.pth with --folds_file, else --hynet_weights_path."""
    if args.folds_file:
        return os.path.join(args.runs_root, args.model, f"fold{args.fold}", "model.pth")
    return args.hynet_weights_path

def _select(pool, wells, what):
    wanted = set(wells)
    chosen = [key for key in sorted(pool) if well_id(key) in wanted]
    missing = wanted - {well_id(key) for key in chosen}
    if missing:
        raise ValueError(f"Wells of the {what} fold not found in the dataset: {sorted(missing)}")
    return [pool[key] for key in chosen]

def segmentation_dataset(args, role, train=False):
    """Dataset of the wells of `role` ('train' | 'val' | 'test'): from --folds_file/--fold if given, else from <dataset_path>/<role>."""
    kwargs = dict(transform=data.IMG_TRANSFORM, target_transform=data.MASK_TRANSFORM, train=train)
    if train:
        kwargs["scale_aug"] = getattr(args, "scale_aug", 0.0)
    if train and getattr(args, "augment", False):  # --augment: random photometric augmentation of the training images
        kwargs["image_fn"] = functools.partial(perturb.random_augment, ranges=getattr(args, "augment_range", "default"))
    elif getattr(args, "perturb", None):  # --perturb name:level: the same perturbation of every image (robustness test)
        kwargs["image_fn"] = perturb.fixed(args.perturb)
    if not args.folds_file:
        return data.OrganoidDataset(os.path.join(args.dataset_path, role), **kwargs)
    wells = fold_wells(load_folds(args.folds_file), args.fold, role)
    return data.OrganoidDataset(samples=_select(data.pool_segmentation_samples(args.dataset_path), wells, role), **kwargs)
