"""Compare the robustness of models from the per_well.csv files written by `python -m hynet.cli.robustness`.

    python scripts/summarize_robustness.py --root outputs/robustness --reference ynet_raw \\
        --model hynet=hynet:cv_s42,cv_s43,cv_s44 --model ynet_raw=ynet_raw:cv_s42,cv_s43,cv_s44

Every --model is <label>=<model directory>:<group>[,<group>...]; the files are <root>/<group>/<model directory>/fold*/per_well.csv.
The Dice of a well is averaged over the groups (e.g. the training seeds), then compared over wells: mean Dice per setting, the drop from the clean
images, and for every model against the reference the paired difference in Dice and in drop (positive = more robust than the reference) with a
95% bootstrap CI over wells.
"""
import argparse
import csv
import glob
import os
import numpy as np

def load(root, directory, groups):
    """{setting: {well: mean Dice over the groups}}"""
    runs = []
    for group in groups:
        scores = {}
        for path in sorted(glob.glob(os.path.join(root, group, directory, "fold*", "per_well.csv"))):
            with open(path) as f:
                for row in csv.DictReader(f):
                    scores.setdefault(row["setting"], {})[row["well"]] = 100 * float(row["dice"])
        if not scores:
            raise SystemExit(f"No results under {os.path.join(root, group, directory)}")
        runs.append(scores)
    return {s: {w: float(np.mean([r[s][w] for r in runs])) for w in runs[0][s]} for s in runs[0]}

def ci(values, rng, n_boot=10000):
    values = np.asarray(values)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])

def fmt(values, rng):
    lo, hi = ci(values, rng)
    return f"{np.mean(values):+6.2f} [{lo:+6.2f}, {hi:+6.2f}]"

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/robustness")
    ap.add_argument("--model", action="append", required=True, help="label=model_dir:group1,group2")
    ap.add_argument("--reference", required=True, help="label of the reference model")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    models = {}
    for spec in args.model:
        label, rest = spec.split("=", 1)
        directory, groups = rest.split(":", 1)
        models[label] = load(args.root, directory, groups.split(","))
    settings = [s for s in next(iter(models.values())) if s != "clean"]
    wells = sorted(set.intersection(*[set(m["clean"]) for m in models.values()]))
    vec = {label: {s: np.array([m[s][w] for w in wells]) for s in m} for label, m in models.items()}
    print(f"{len(wells)} wells; Dice of a well averaged over the groups of each model\n")

    labels = list(models)
    print(f"{'setting':14s}" + "".join(f"{label:>12s}" for label in labels))
    for s in ["clean"] + settings:
        print(f"{s:14s}" + "".join(f"{vec[label][s].mean():12.2f}" for label in labels))
    print(f"{'mean perturbed':14s}" + "".join(f"{np.mean([vec[label][s].mean() for s in settings]):12.2f}" for label in labels))

    ref = vec[args.reference]
    for label in labels:
        if label == args.reference:
            continue
        print(f"\n{label} minus {args.reference}: Dice, and Dice lost from the clean images (positive = {label} better / more robust), paired over wells")
        print(f"{'setting':14s}{'Dice difference':>30s}{'robustness (less drop)':>30s}")
        drop_all, diff_all = [], []
        for s in ["clean"] + settings:
            diff = vec[label][s] - ref[s]
            if s == "clean":
                print(f"{s:14s}{fmt(diff, rng):>30s}")
                continue
            drop = (vec[label][s] - vec[label]["clean"]) - (ref[s] - ref["clean"])
            diff_all.append(diff), drop_all.append(drop)
            print(f"{s:14s}{fmt(diff, rng):>30s}{fmt(drop, rng):>30s}")
        print(f"{'mean perturbed':14s}{fmt(np.mean(diff_all, axis=0), rng):>30s}{fmt(np.mean(drop_all, axis=0), rng):>30s}")

if __name__ == "__main__":
    main()
