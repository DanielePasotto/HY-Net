import csv
import json
import os
import numpy as np
import torch
from PIL import Image
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from tqdm import tqdm
from hynet import data
from hynet import evaluation
from hynet.config import get_args
from hynet.engine import seed_everything
from hynet.models.factory import build_model
from hynet.scoring import score_predictions
from hynet.splits import segmentation_dataset, weights_paths, run_tag

def str2bool(v):
    return str(v).lower() in ("true", "1", "yes")

def add_arguments(parser):
    group = parser.add_argument_group("evaluation")
    group.add_argument("--split", type=str, default="test", help="Split of --dataset_path to evaluate (ignored with --folds_file: the test wells of --fold are used)")
    group.add_argument("--weights_path", type=str, default=None, help="Weights of the segmentation model, instead of the ones of --runs_root/--hynet_weights_path (e.g. the model of another fold)")
    group.add_argument("--out_dir", type=str, default="outputs/eval", help="Where predictions, tables and the summary are written: <out_dir>/<model>/<split or fold>/")
    group.add_argument("--reuse_predictions", default=False, type=str2bool, nargs="?", const=True,
                       help="Score the masks already saved in <out_dir>/<model>/<split or fold>/predictions instead of running the model (masks of any other model, as binary PNGs named like the images, can be scored too)")
    group.add_argument("--size_edges", type=int, nargs="+", default=[200, 1000], help="Component area edges in pixels (480x640 images) for the size strata")
    group.add_argument("--iou_threshold", type=float, default=0.5, help="IoU needed to match a predicted and a ground-truth component")
    group.add_argument("--min_instance_area", type=int, default=0, help="Components smaller than this (pixels) are ignored in the instance metrics")
    group.add_argument("--boundary_tolerance", type=float, default=2.0, help="Tolerance in pixels of the boundary Dice")
    group.add_argument("--n_boot", type=int, default=10000, help="Number of bootstrap resamples (wells)")

def predict_masks(args, dataset, pred_dir, device, weights_path):
    model = build_model(args.model, args, device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for i, (image, _) in enumerate(tqdm(loader, desc="Predicting")):
            pred = torch.sigmoid(model(image.to(device))) > 0.5
            stem = os.path.splitext(dataset.images[i])[0]
            Image.fromarray(pred.squeeze().cpu().numpy().astype(np.uint8) * 255).save(os.path.join(pred_dir, stem + ".png"))

def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([f"{row[c]:.6f}" if isinstance(row[c], float) else row[c] for c in columns])

def main():
    args = get_args(add_arguments)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    _, weights_path = weights_paths(args)
    weights_path = args.weights_path or weights_path

    dataset = segmentation_dataset(args, "test" if args.folds_file else args.split)
    tag = run_tag(args, args.split)
    out_dir = os.path.join(args.out_dir, args.model, tag)
    pred_dir = os.path.join(out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    start_time = timer()
    if not args.reuse_predictions:
        predict_masks(args, dataset, pred_dir, device, weights_path)

    size_edges = tuple(args.size_edges)
    entries = [(data.sample_key(i), i, m) for i, m in zip(dataset.image_paths, dataset.mask_paths)]
    records = score_predictions(entries, pred_dir, args.boundary_tolerance, args.iou_threshold, size_edges, args.min_instance_area)

    rows = evaluation.aggregate_wells(records)
    summary = evaluation.summarize(rows, size_edges, args.n_boot, args.seed)
    summary["settings"] = {"model": args.model, "run": tag, "size_edges": list(size_edges), "iou_threshold": args.iou_threshold,
                           "min_instance_area": args.min_instance_area, "boundary_tolerance": args.boundary_tolerance, "seed": args.seed}

    image_rows = [{"image": r["image"], "well": r["well"], **evaluation.overlap_metrics(r["counts"]),
                   "hd95": r["boundary"]["hd95"], "boundary_dice": r["boundary"]["boundary_dice"],
                   **evaluation.instance_summary(r["instances"]["tp"], r["instances"]["n_gt"], r["instances"]["n_pred"], r["instances"]["iou_sum"])}
                  for r in records]
    metric_columns = list(evaluation.WELL_METRICS)
    write_csv(os.path.join(out_dir, "per_image.csv"), image_rows, ["image", "well"] + metric_columns)
    write_csv(os.path.join(out_dir, "per_well.csv"), rows, ["well", "n_images", "gt_fraction"] + metric_columns)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(evaluation.format_summary(summary, title=f"{args.model}, {tag}"))
    print(f"\nResults written to {out_dir} ({timer() - start_time:.1f} s)")

if __name__ == "__main__":
    main()
