"""Threshold chosen on the validation wells and mirroring at inference: do they change the test Dice of a trained model?

For the --fold of --folds_file the model is run on the validation and the test wells of the fold, either on the image alone or averaging the probabilities of
the image and of its horizontal, vertical and double flips (test-time augmentation). For each of the two inferences the threshold is the one of THRESHOLDS that
maximizes the mean per-well Dice of the validation wells (ties: the one closest to 0.5); the test wells are scored with 0.5 and with that threshold.
Written to <out_dir>/<model>/fold<k>/per_well.csv (variants plain_0.5, plain_tuned, tta_0.5, tta_tuned); scripts/summarize_postproc.py compares them.
"""
import csv
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from hynet import data, evaluation
from hynet.config import get_args
from hynet.engine import seed_everything
from hynet.models.factory import build_model
from hynet.splits import segmentation_dataset, weights_paths

THRESHOLDS = [round(0.1 + 0.05 * i, 2) for i in range(17)]  # 0.1 ... 0.9, includes 0.5
FLIPS = [(), (3,), (2,), (2, 3)]  # tensor dims of the batch x channel x height x width input

def add_arguments(parser):
    group = parser.add_argument_group("post-processing")
    group.add_argument("--weights_path", type=str, default=None, help="Weights of the segmentation model, instead of the ones of --runs_root")
    group.add_argument("--out_dir", type=str, default="outputs/postproc", help="Results are written to <out_dir>/<model>/fold<k>/per_well.csv")

def probabilities(model, dataset, device, tta):
    """Foreground probability maps (numpy, in dataset order); with `tta` the mean over the flips of the input, each flipped back."""
    flips = FLIPS if tta else FLIPS[:1]
    out = []
    with torch.no_grad():
        for image, _ in DataLoader(dataset, batch_size=1, shuffle=False):
            image = image.to(device)
            out.append((sum(torch.sigmoid(model(image.flip(f))).flip(f) for f in flips) / len(flips)).squeeze().cpu().numpy())
    return out

def main():
    args = get_args(add_arguments)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    _, weights_path = weights_paths(args)
    model = build_model(args.model, args, device)
    model.load_state_dict(torch.load(args.weights_path or weights_path, map_location=device, weights_only=True))
    model.eval()

    sets = {role: segmentation_dataset(args, role) for role in ("val", "test")}
    masks = {role: [data.load_mask_480x640(p) for p in ds.mask_paths] for role, ds in sets.items()}
    rows = []
    for tta in (False, True):
        probs = {role: probabilities(model, ds, device, tta) for role, ds in sets.items()}
        val_dice = {t: np.mean([d for _, d in evaluation.well_dice([p > t for p in probs["val"]], masks["val"], sets["val"].images).values()]) for t in THRESHOLDS}
        tuned = max(THRESHOLDS, key=lambda t: (round(val_dice[t], 6), -abs(t - 0.5)))
        for name, t in (("0.5", 0.5), ("tuned", tuned)):
            scores = evaluation.well_dice([p > t for p in probs["test"]], masks["test"], sets["test"].images)
            rows += [{"variant": f"{'tta' if tta else 'plain'}_{name}", "threshold": t, "well": w, "n_images": n, "dice": d} for w, (n, d) in sorted(scores.items())]

    out_dir = os.path.join(args.out_dir, args.model, f"fold{args.fold}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "per_well.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "threshold", "well", "n_images", "dice"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{args.model}, fold{args.fold}: thresholds " + ", ".join(f"{v} {t}" for v, t in sorted({(r['variant'], r['threshold']) for r in rows})) + f" -> {out_dir}")

if __name__ == "__main__":
    main()
