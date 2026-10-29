"""Datasets and file locations for a run, either on the original train/val/test folders or on a cross-validation fold."""
import os

from hynet import data
from hynet.evaluation import well_id
from hynet.folds import load_folds, fold_wells

def run_tag(args, split="test"):
    return f"fold{args.fold}" if args.folds_file else split

def weights_paths(args):
    """(HarmNet weights, segmentation-model weights). With --folds_file they live under <runs_root>/ and HarmNet is shared by all the models of a fold."""
    if args.folds_file:
        return (os.path.join(args.runs_root, "harmnet", f"fold{args.fold}", "harmnet.pth"),
                os.path.join(args.runs_root, args.model, f"fold{args.fold}", "model.pth"))
    return args.harmnet_weights_path, args.hynet_weights_path

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
    if not args.folds_file:
        return data.OrganoidDataset(os.path.join(args.dataset_path, role), **kwargs)
    wells = fold_wells(load_folds(args.folds_file), args.fold, role)
    return data.OrganoidDataset(samples=_select(data.pool_segmentation_samples(args.dataset_path), wells, role), **kwargs)

def harmonization_dataset(args, role):
    kwargs = dict(transform=data.IMG_TRANSFORM, target_transform=data.MASK_TRANSFORM)
    if not args.folds_file:
        return data.HarmonizationDataset(os.path.join(args.processed_dataset_path, role), args.harm_input_folder,
                                         args.harm_target_folder, **kwargs)
    wells = fold_wells(load_folds(args.folds_file), args.fold, role)
    pool = data.pool_harmonization_samples(args.processed_dataset_path, args.harm_input_folder, args.harm_target_folder)
    return data.HarmonizationDataset(samples=_select(pool, wells, role), **kwargs)
