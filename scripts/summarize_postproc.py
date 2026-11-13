"""Compare the post-processing variants from the per_well.csv files written by `python -m hynet.cli.postproc`.

    python scripts/summarize_postproc.py --root outputs/postproc --groups cv_s42 cv_s43 cv_s44 --models hynet ynet_raw unet hynet_nogate --reference hynet

For every model: mean per-well Dice of each variant [95% bootstrap CI over wells], the paired difference from plain_0.5 (the current protocol), and the mean threshold
chosen on the validation wells. Then, for every variant, the paired difference of the reference model from each other model. The Dice of a well is averaged over the groups (seeds).
"""
import argparse
import csv
import glob
import os
import numpy as np

def load(root, groups, model):
    """({variant: {well: Dice}}, {variant: mean threshold}); Dice in percent, averaged over the groups."""
    dice, thr = {}, {}
    for group in groups:
        paths = sorted(glob.glob(os.path.join(root, group, model, "fold*", "per_well.csv")))
        if not paths:
            raise SystemExit(f"No results under {os.path.join(root, group, model)}")
        for path in paths:
            with open(path) as f:
                for row in csv.DictReader(f):
                    dice.setdefault(row["variant"], {}).setdefault(row["well"], []).append(100 * float(row["dice"]))
                    thr.setdefault(row["variant"], []).append(float(row["threshold"]))
    return {v: {w: float(np.mean(x)) for w, x in ws.items()} for v, ws in dice.items()}, {v: float(np.mean(t)) for v, t in thr.items()}

def fmt(values, rng, signed=False):
    values = np.asarray(values)
    lo, hi = np.percentile(values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1), [2.5, 97.5])
    f = "+.2f" if signed else ".2f"
    return f"{values.mean():{f}} [{lo:{f}}, {hi:{f}}]"

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/postproc")
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--reference", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    loaded = {m: load(args.root, args.groups, m) for m in args.models}
    wells = sorted(set.intersection(*[set(d["plain_0.5"]) for d, _ in loaded.values()]))
    vec = {m: {v: np.array([d[v][w] for w in wells]) for v in d} for m, (d, _) in loaded.items()}
    variants = list(next(iter(vec.values())))
    print(f"{len(wells)} wells, groups {', '.join(args.groups)}\n")
    for m in args.models:
        print(m)
        for v in variants:
            diff = "" if v == "plain_0.5" else f"   vs plain_0.5 {fmt(vec[m][v] - vec[m]['plain_0.5'], rng, True)}"
            print(f"  {v:12s} {fmt(vec[m][v], rng):24s} threshold {loaded[m][1][v]:.2f}{diff}")
    for v in variants:
        print(f"\n{v}: {args.reference} minus the other models (paired, positive = {args.reference} better)")
        for m in args.models:
            if m != args.reference:
                print(f"  {m:14s} {fmt(vec[args.reference][v] - vec[m][v], rng, True)}")

if __name__ == "__main__":
    main()
