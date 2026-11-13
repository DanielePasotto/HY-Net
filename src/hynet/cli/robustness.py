"""Robustness of a trained segmentation model to photometric perturbations of the test images.

The model is loaded once and the test wells of --fold (or of the --split folder) are segmented clean and with each perturbation of hynet.perturb.SWEEP
(gamma, noise, blur, uneven illumination, JPEG). Written to <out_dir>/<model>/<fold>/per_well.csv: the organoid Dice of every well
(pixel counts summed over the well's images) for each setting. scripts/summarize_robustness.py compares the models.
"""
import csv
import os
import torch
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from tqdm import tqdm
from hynet import data, evaluation, perturb
from hynet.config import get_args
from hynet.engine import seed_everything
from hynet.models.factory import build_model
from hynet.splits import segmentation_dataset, weights_paths, run_tag

def add_arguments(parser):
    group = parser.add_argument_group("robustness")
    group.add_argument("--split", type=str, default="test", help="Split of --dataset_path to use (ignored with --folds_file: the test wells of --fold)")
    group.add_argument("--weights_path", type=str, default=None, help="Weights of the segmentation model, instead of the ones of --runs_root/--hynet_weights_path")
    group.add_argument("--out_dir", type=str, default="outputs/robustness", help="Results are written to <out_dir>/<model>/<split or fold>/per_well.csv")
    group.add_argument("--settings", type=str, nargs="+", default=["clean", *perturb.SWEEP], help="'clean' and/or perturbations as name:level")
    group.add_argument("--workers", type=int, default=0, help="Data-loading workers (the perturbations are computed on the CPU)")

def well_scores(model, dataset, masks, device, workers):
    """{well: (number of images, organoid Dice)} of the model on `dataset`; `masks` are the ground-truth masks in dataset order."""
    preds = []
    with torch.no_grad():
        for image, _ in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers):
            preds.append((torch.sigmoid(model(image.to(device))) > 0.5).squeeze().cpu().numpy())
    return evaluation.well_dice(preds, masks, dataset.images)

def main():
    args = get_args(add_arguments)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    _, weights_path = weights_paths(args)
    weights_path = args.weights_path or weights_path

    model = build_model(args.model, args, device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    role = "test" if args.folds_file else args.split
    masks = [data.load_mask_480x640(p) for p in segmentation_dataset(args, role).mask_paths]
    out_dir = os.path.join(args.out_dir, args.model, run_tag(args, args.split))
    os.makedirs(out_dir, exist_ok=True)

    start_time = timer()
    rows = []
    for setting in tqdm(args.settings, desc="Settings"):
        args.perturb = None if setting == "clean" else setting
        scores = well_scores(model, segmentation_dataset(args, role), masks, device, args.workers)
        rows += [{"setting": setting, "well": well, "n_images": n, "dice": dice} for well, (n, dice) in sorted(scores.items())]
    with open(os.path.join(out_dir, "per_well.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["setting", "well", "n_images", "dice"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})
    print(f"{args.model}, {run_tag(args, args.split)}: {len(args.settings)} settings written to {out_dir} ({timer() - start_time:.1f} s)")

if __name__ == "__main__":
    main()
