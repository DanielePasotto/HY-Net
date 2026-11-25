"""Paired comparison of two recipes of scripts/summarize_plan.py on the per-well Dice of the test domain, with a 95% bootstrap CI over wells.

    python scripts/compare_wells.py Wr-Uw Wr-Kw Wr-Nn --seeds 42 43 44

The Dice of a well is averaged over the seeds, then the two recipes are compared well by well (positive = the first is better).
"""
import argparse
import csv
import os
import numpy as np
from summarize_plan import GROUPS

def wells(tag, domain, seeds):
    model = GROUPS[tag][0]
    runs = []
    for seed in seeds:
        suffix = "" if seed == 42 else f"_s{seed}"
        path = f"outputs/lodo_{tag}{suffix}/{domain}/eval/{model}/fold0/per_well.csv"
        if not os.path.exists(path):
            return None
        with open(path) as f:
            runs.append({r["well"]: 100 * float(r["dice"]) for r in csv.DictReader(f)})
    return {w: float(np.mean([r[w] for r in runs])) for w in runs[0]}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", help="A-B, e.g. Hw-Uw")
    ap.add_argument("--domains", nargs="+", default=["ours", "orgaseg", "organoid"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for domain in args.domains:
        print(f"=== test domain: {domain}")
        for pair in args.pairs:
            a, b = pair.split("-")
            va, vb = wells(a, domain, args.seeds), wells(b, domain, args.seeds)
            if va is None or vb is None:
                print(f"  {a} - {b}: missing results")
                continue
            common = sorted(set(va) & set(vb))
            d = np.array([va[w] - vb[w] for w in common])
            means = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
            lo, hi = np.percentile(means, [2.5, 97.5])
            sig = "*" if lo > 0 or hi < 0 else " "
            print(f"  {a} - {b}: {d.mean():+6.2f} [{lo:+6.2f}, {hi:+6.2f}] {sig} ({len(d)} wells, wins {int((d > 0).sum())})")

if __name__ == "__main__":
    main()
