"""Dice versus number of annotated images of the external dataset (outputs of scripts/finetune_external.sh).

For every dataset, number of images N and model: per-image Dice averaged over the random draws, mean over the images of the external test
folds with a 95% bootstrap CI over images, and the paired difference of HY-Net with the other models (a * marks a CI that excludes 0).
"""
import argparse
import os
import numpy as np
from hynet import data, evaluation
from hynet.cli.compare import average_rows, collect_rows

def rows_of(args, ds, init, n, model, pool):
    dirs = [os.path.join(args.root, ds, init, f"n{n}_r{r}") for r in range(args.repeats if n else 1)]
    return average_rows([collect_rows(args, model, pool, d) for d in dirs])

def cell(values, n_boot, seed):
    lo, hi = evaluation.bootstrap_mean_ci(np.asarray(values, float), n_boot, seed)
    return f"{100 * np.nanmean(values):5.2f} [{100 * lo:5.2f},{100 * hi:5.2f}]"

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="outputs/ft")
    ap.add_argument("--datasets", nargs="+", default=["organoid", "organosegment"])
    ap.add_argument("--models", nargs="+", default=["hynet", "ynet_raw", "unet"])
    ap.add_argument("--ns", type=int, nargs="+", default=[0, 2, 4, 8, 16])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--reference", default="hynet")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.boundary_tolerance, args.iou_threshold, args.size_edges = 2.0, 0.5, [200, 1000]

    for ds in args.datasets:
        args.folds_file = f"folds/external_{ds}_cv4.json"
        pool = data.pool_segmentation_samples(f"data/external/{ds}")
        print(f"\n=== {ds}: Dice of the organoid class (mean over test images [95% CI]), N = annotated images used to adapt ===")
        print(f"{'N':>3}" + "".join(f"{m:>26}" for m in args.models) + f"{'unet from scratch':>26}")
        table = {}
        for n in args.ns:
            line = f"{n:>3}"
            for m in args.models:
                table[(n, m)] = rows_of(args, ds, "pretrained", n, m, pool)
                line += f"{cell([r['dice'] for r in table[(n, m)]], args.n_boot, args.seed):>26}"
            if n:
                scratch = rows_of(args, ds, "scratch", n, "unet", pool)
                line += f"{cell([r['dice'] for r in scratch], args.n_boot, args.seed):>26}"
            print(line)
        print(f"\npaired difference {args.reference} - model (points of Dice; * = 95% CI excludes 0)")
        print(f"{'N':>3}" + "".join(f"{m:>26}" for m in args.models if m != args.reference))
        for n in args.ns:
            line = f"{n:>3}"
            for m in args.models:
                if m == args.reference:
                    continue
                d = evaluation.compare_models(table[(n, args.reference)], table[(n, m)], "dice", args.n_boot, args.seed)
                lo, hi = d["ci"]
                line += f"{100 * d['mean_difference']:+8.2f} [{100 * lo:+6.2f},{100 * hi:+6.2f}]{'*' if lo > 0 or hi < 0 else ' '}".rjust(26)
            print(line)

if __name__ == "__main__":
    main()
