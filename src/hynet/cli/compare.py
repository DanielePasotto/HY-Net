import csv
import json
import os
import numpy as np
from hynet import data
from hynet import evaluation
from hynet.config import get_args
from hynet.evaluation import well_id
from hynet.folds import load_folds, fold_wells
from hynet.scoring import score_predictions

def add_arguments(parser):
    group = parser.add_argument_group("comparison")
    group.add_argument("--models", type=str, nargs="+", required=True, help="Models to compare; their predictions must be in <eval_dir>/<model>/fold<k>/predictions")
    group.add_argument("--reference", type=str, default=None, help="Reference model for the paired differences (default: the first of --models)")
    group.add_argument("--eval_dir", type=str, default="outputs/eval", help="Directory written by hynet-evaluate (the main seed)")
    group.add_argument("--extra_eval_dirs", type=str, nargs="*", default=[],
                       help="Evaluation directories of other training seeds (e.g. outputs/eval_s43 outputs/eval_s44). Models found in them are also reported averaged over seeds")
    group.add_argument("--replicates_label", type=str, default="training seeds", help="What the eval dirs are replicates of, for the printed text (e.g. \"fold models\")")
    group.add_argument("--compare_out", type=str, default="outputs/eval/comparison", help="Prefix of the comparison files (.csv, .json)")
    group.add_argument("--size_edges", type=int, nargs="+", default=[200, 1000])
    group.add_argument("--iou_threshold", type=float, default=0.5)
    group.add_argument("--boundary_tolerance", type=float, default=2.0)
    group.add_argument("--n_boot", type=int, default=10000)

METRICS = (("dice", "Dice"), ("iou", "IoU"), ("boundary_dice", "Boundary Dice"), ("hd95", "HD95 (px)"), ("inst_f1", "Instance F1"), ("macro_dice", "mDice (org+bg)/2"))

def collect_rows(args, model, pool, eval_dir):
    """Out-of-fold per-well rows of `model`: every well is scored with the predictions of the fold where it was in the test set."""
    doc = load_folds(args.folds_file)
    records = []
    for fold in range(len(doc["folds"])):
        wells = set(fold_wells(doc, fold, "test"))
        entries = [(key, *pool[key]) for key in sorted(pool) if well_id(key) in wells]
        pred_dir = os.path.join(eval_dir, model, f"fold{fold}", "predictions")
        records += score_predictions(entries, pred_dir, args.boundary_tolerance, args.iou_threshold, tuple(args.size_edges))
    return evaluation.aggregate_wells(records)

def average_rows(per_seed_rows):
    """Per-well metrics averaged over training seeds."""
    by_well = {}
    for rows in per_seed_rows:
        for r in rows:
            by_well.setdefault(r["well"], []).append(r)
    averaged = []
    for well, rs in sorted(by_well.items()):
        row = {"well": well, "n_images": rs[0]["n_images"], "gt_fraction": rs[0]["gt_fraction"]}
        for key in evaluation.WELL_METRICS:
            row[key] = float(np.nanmean([r[key] for r in rs]))
        averaged.append(row)
    return averaged

def cell(values_or_entry, key, ci):
    scale = 1 if key == "hd95" else 100
    mean = float(np.nanmean(values_or_entry))
    return f"{scale * mean:.2f} [{scale * ci[0]:.2f}, {scale * ci[1]:.2f}]"

def print_paired(title, rows, models, reference, args):
    out = {}
    print(f"\n{title} (reference - model; positive = {reference} better; HD95: negative = {reference} better)")
    for m in models:
        if m == reference:
            continue
        out[m] = {}
        cells = []
        for key, _ in METRICS:
            diff = evaluation.compare_models(rows[reference], rows[m], key, args.n_boot, args.seed)
            out[m][key] = diff
            scale = 1 if key == "hd95" else 100
            lo, hi = diff["ci"]
            sig = "*" if (lo > 0 or hi < 0) else " "
            cells.append(f"{scale * diff['mean_difference']:+.2f} [{scale * lo:+.2f}, {scale * hi:+.2f}]{sig}")
        print(f"{m:<22}" + "".join(f"{c:>26}" for c in cells))
    if out:
        print("* = 95% CI of the paired difference excludes 0")
    return out

def main():
    args = get_args(add_arguments)
    if not args.folds_file:
        raise SystemExit("hynet-compare needs --folds_file (the out-of-fold predictions of every model)")
    reference = args.reference or args.models[0]
    pool = data.pool_segmentation_samples(args.dataset_path)

    rows = {m: collect_rows(args, m, pool, args.eval_dir) for m in args.models}
    summaries = {m: evaluation.summarize(rows[m], tuple(args.size_edges), args.n_boot, args.seed) for m in args.models}

    n_wells = len(rows[args.models[0]])
    print(f"Out-of-fold results of {args.eval_dir}, {n_wells} wells (each tested once), mean over wells [95% bootstrap CI over wells]\n")
    print(f"{'model':<22}" + "".join(f"{label:>26}" for _, label in METRICS))
    for m in args.models:
        cells = []
        for key, _ in METRICS:
            d = summaries[m]["per_well"][key]
            cells.append(cell([d["mean"]], key, (d["ci_low"], d["ci_high"])))
        print(f"{m:<22}" + "".join(f"{c:>26}" for c in cells))
    comparisons = print_paired("Paired per-well difference", rows, args.models, reference, args)

    seed_report = None
    if args.extra_eval_dirs:
        per_seed = {}
        for m in args.models:
            seeds = [rows[m]]
            for d in args.extra_eval_dirs:
                if os.path.isdir(os.path.join(d, m)):
                    try:
                        seeds.append(collect_rows(args, m, pool, d))
                    except FileNotFoundError:
                        pass  # incomplete seed for this model: skipped
            per_seed[m] = seeds
        multi = [m for m in args.models if len(per_seed[m]) > 1]
        if multi:
            averaged = {m: average_rows(per_seed[m]) for m in multi}
            print(f"\nAveraged over training seeds (models with more than one seed), mean over wells [95% bootstrap CI]; SD = standard deviation of the overall Dice between seeds\n")
            print(f"{'model':<22}{'seeds':>6}{'Dice per seed':>30}{'SD':>7}" + "".join(f"{label:>26}" for _, label in METRICS[:5]))
            report = {}
            for m in multi:
                seed_dice = [float(np.nanmean([r["dice"] for r in rs])) for rs in per_seed[m]]
                sd = float(np.std(seed_dice, ddof=1)) if len(seed_dice) > 1 else float("nan")
                cells = []
                for key, _ in METRICS[:5]:
                    values = np.array([r[key] for r in averaged[m]], dtype=float)
                    cells.append(cell(values, key, evaluation.bootstrap_mean_ci(values, args.n_boot, args.seed)))
                print(f"{m:<22}{len(seed_dice):>6}{'  '.join(f'{100 * d:.2f}' for d in seed_dice):>30}{100 * sd:>7.2f}" + "".join(f"{c:>26}" for c in cells))
                report[m] = {"n_seeds": len(seed_dice), "dice_per_seed": seed_dice, "dice_seed_sd": sd}
            if reference in averaged:
                report["paired_seed_averaged"] = print_paired("Paired per-well difference of the seed-averaged values", averaged, multi, reference, args)
            seed_report = report

    os.makedirs(os.path.dirname(args.compare_out) or ".", exist_ok=True)
    with open(args.compare_out + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "well", "n_images", "gt_fraction"] + [k for k, _ in METRICS])
        for m in args.models:
            for r in rows[m]:
                writer.writerow([m, r["well"], r["n_images"], f"{r['gt_fraction']:.6f}"] + [f"{r[k]:.6f}" for k, _ in METRICS])
    with open(args.compare_out + ".json", "w") as f:
        json.dump({"reference": reference, "summaries": summaries, "paired_differences": comparisons, "seeds": seed_report}, f, indent=2)
    print(f"\nWritten {args.compare_out}.csv and .json")

if __name__ == "__main__":
    main()
