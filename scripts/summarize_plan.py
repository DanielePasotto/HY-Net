"""Per-well organoid Dice of the leave-one-dataset-out runs of the paper, per test domain and seed.

Run from the repository root on the machine with the outputs:  python scripts/summarize_plan.py [--domains ours orgaseg organoid] [--seeds 42 43 44]
Reads outputs/lodo_<tag>[_s43|_s44]/<domain>/eval/<model>/fold0/summary.json (ROOT of scripts/lodo.sh). A cell that is not finished is shown as '-'.
Also prints the paired difference of every group with respect to the reference of its family (per seed and mean), and the instance recall/precision of the smallest size bin.
"""
import argparse
import json
import os

# tag -> (model, label, reference tag of its family). "a" = default photometric augmentation, "w" = wide; all with the random zoom (SCALE=2).
GROUPS = {
    "Ya": ("ynet_raw", "Y-Net without HarmNet, augment", None),
    "Ra": ("hynet_res", "HY-Net, augment", "Ya"),
    "Ua": ("unet", "U-Net, augment", "Ya"),
    "Ka": ("campeod", "CAMPEOD, augment", "Ya"),
    "At": ("attunet", "Attention U-Net, augment", "Ua"),
    "Mp": ("macpnet", "MACPnet, augment", "Ua"),
    "Wy": ("ynet_raw", "Y-Net without HarmNet, wide augment", "Ya"),
    "Wr": ("hynet_res", "HY-Net, wide augment", "Wy"),
    "Wg": ("hynet_res_nogate", "HY-Net concat, no attention/gates, wide", "Wr"),
    "Wh": ("hynet_fast", "HY-Net half-resolution HarmNet, wide", "Wr"),
    "Uw": ("unet", "U-Net, wide augment", None),
    "Kw": ("campeod", "CAMPEOD, wide augment", "Ka"),
    "Nn": ("nnunet", "nnU-Net 2D, 100 epochs (not balanced)", None),
    "Oi": ("organoid", "OrganoID U-Net, own recipe", None),
    "Zs": ("orgasegment_zeroshot", "OrgaSegment released weights", None),
    "Zo": ("organoid_zeroshot", "OrganoID released weights", None),
}


def load(tag, model, domain, seed):
    suffix = "" if seed == 42 else f"_s{seed}"
    path = f"outputs/lodo_{tag}{suffix}/{domain}/eval/{model}/fold0/summary.json"
    if not os.path.exists(path):
        return None
    s = json.load(open(path))
    small = next((b for b in s.get("by_size", []) if b["bin"].startswith("<")), None)
    return 100 * s["per_well"]["dice"]["mean"], (100 * small["recall"], 100 * small["precision"]) if small else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["ours", "orgaseg", "organoid"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = ap.parse_args()
    for domain in args.domains:
        print(f"\n=== test domain: {domain} ===")
        print(f"{'group':5s}{'model':10s}{'recipe':38s}" + "".join(f"{'s' + str(s):>8s}" for s in args.seeds) + f"{'mean':>8s}{'n':>3s}   small-bin rec/prec (mean)")
        cells = {}
        for tag, (model, label, ref) in GROUPS.items():
            vals = {s: load(tag, model, domain, s) for s in args.seeds}
            cells[tag] = {s: v[0] for s, v in vals.items() if v}
            ok = [v for v in vals.values() if v]
            mean = sum(v[0] for v in ok) / len(ok) if ok else float("nan")
            small = [v[1] for v in ok if v[1]]
            sm = f"{sum(x[0] for x in small) / len(small):.0f}/{sum(x[1] for x in small) / len(small):.0f}" if small else ""
            print(f"{tag:5s}{model:10s}{label:38s}" + "".join(f"{vals[s][0]:8.2f}" if vals[s] else f"{'-':>8s}" for s in args.seeds) + f"{mean:8.2f}{len(ok):3d}   {sm}")
        print("paired difference to the reference of the family (same seeds only):")
        pairs = [(tag, ref) for tag, (_, _, ref) in GROUPS.items() if ref is not None] + [("Wr", "Uw"), ("Wr", "Kw"), ("Wr", "Nn"), ("Wr", "Ra"), ("Kw", "Uw"),
                                                                                                          ("Wr", "At"), ("Wr", "Mp"), ("Wr", "Oi"), ("Wr", "Zs"), ("Wr", "Zo")]
        for tag, ref in pairs:
            common = sorted(set(cells[tag]) & set(cells[ref]))
            if common:
                d = [cells[tag][s] - cells[ref][s] for s in common]
                print(f"  {tag} - {ref}: " + " ".join(f"s{s} {x:+.2f}" for s, x in zip(common, d)) + f"  | mean {sum(d) / len(d):+.2f} (n={len(d)})")


if __name__ == "__main__":
    main()
