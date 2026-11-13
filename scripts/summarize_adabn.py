"""Compare the LODO models with and without adaptive BatchNorm (scripts/adabn.sh): organoid Dice per well, seeds averaged per well first,
paired difference AdaBN minus standard evaluation with a bootstrap 95% CI over the wells.

  python scripts/summarize_adabn.py [--outputs outputs]
"""
import argparse
import csv
import os
import numpy as np

GROUPS = {"no augmentation": ["lodo", "lodo_s43", "lodo_s44"], "augmentation": ["lodo_aug_s42", "lodo_aug_s43", "lodo_aug_s44"]}
DOMAINS = {"ours": "Nostro", "orgaseg": "OrgaSegment", "organoid": "OrganoID"}

def read(path):
    with open(path) as f:
        return {r["well"]: float(r["dice"]) for r in csv.DictReader(f)}

def averaged(paths):
    runs = [read(p) for p in paths if os.path.exists(p)]
    if not runs:
        return None
    return {w: float(np.mean([r[w] for r in runs])) for w in runs[0]}, len(runs)

def bootstrap(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()
    print("| Training | Test su | Modello | Seed | Standard | AdaBN | AdaBN - standard [IC 95%] |\n|---|---|---|---:|---:|---:|---:|")
    for group, roots in GROUPS.items():
        for domain, label in DOMAINS.items():
            for model in ("hynet", "ynet_raw"):
                std = averaged([f"{args.outputs}/{r}/{domain}/eval/{model}/fold0/per_well.csv" for r in roots])
                ada = averaged([f"{args.outputs}/adabn/{r}/{domain}/eval/{model}/fold0/per_well.csv" for r in roots])
                if std is None or ada is None:
                    continue
                (s, ns), (a, na) = std, ada
                wells = sorted(set(s) & set(a))
                diff = np.array([a[w] - s[w] for w in wells]) * 100
                lo, hi = bootstrap(diff)
                print(f"| {group} | {label} | {model} | {na} | {np.mean([s[w] for w in wells]) * 100:.2f} | {np.mean([a[w] for w in wells]) * 100:.2f} | {diff.mean():+.2f} [{lo:+.2f}; {hi:+.2f}] |")

if __name__ == "__main__":
    main()
