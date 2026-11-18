"""Per-well organoid Dice of the groups of the 2026-11-15 plan, per test domain and seed.

Run from the repository root on the machine with the outputs:  python scripts/summarize_plan.py [--domains ours orgaseg] [--seeds 42 43 44]
Reads outputs/lodo_<tag>[_s43|_s44]/<domain>/eval/<model>/fold0/summary.json. A cell that is not finished is shown as '-'.
Also prints the paired difference of every group with respect to the reference of its family (per seed and mean), and the instance recall/precision of the smallest size bin.
"""
import argparse
import json
import os

# tag -> (model, label, reference tag of its family). Rc = the Rx recipe run again on claudio (RTX 5070 Ti), the reference of the Nx/Sx jobs that ran there;
# "Rc - Rx" is the effect of the GPU alone.
GROUPS = {
    "Rx": ("ynet_raw", "scale", None),
    "Rc": ("ynet_raw", "scale (repeated on claudio)", None),
    "Cx": ("ynet_raw", "scale + consistency (test images)", "Rx"),
    "Nx": ("ynet_raw", "scale, trained without OrganoID", "Rx"),
    "Sx": ("ynet_raw", "scale + small-object weight 4", "Rx"),
    "Ya": ("ynet_raw", "scale + augment", None),
    "Ha": ("hynet", "scale + augment", "Ya"),
    "Ra": ("hynet_res", "scale + augment", "Ya"),
    "Pa": ("hynet_par", "scale + augment + harm consistency", "Ya"),
    "augsc": ("ynet_raw", "scale + augment (E, organoid test)", None),
    "Ua": ("unet", "U-Net, scale + augment", "Ya"),
    "Ka": ("campeod", "CAMPEOD, scale + augment", "Ya"),
    "Wy": ("ynet_raw", "scale + wide augment", "Ya"),
    "Wr": ("hynet_res", "scale + wide augment", "Wy"),
    "Wh": ("hynet_fast", "half-resolution HarmNet, wide augment", "Wr"),
    "Kw": ("campeod", "CAMPEOD, scale + wide augment", "Ka"),
    "Nn": ("nnunet", "nnU-Net 2D, 100 epochs (not balanced)", None),
    "Zs": ("orgasegment_zeroshot", "OrgaSegment released weights", None),
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
    ap.add_argument("--domains", nargs="+", default=["ours", "orgaseg"])
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
        pairs = [(tag, ref) for tag, (_, _, ref) in GROUPS.items() if ref is not None] + [("Nx", "Rc"), ("Sx", "Rc"), ("Rc", "Rx"), ("Ua", "augsc"), ("Ka", "augsc"), ("Ra", "augsc"),
                                                                                                          ("Kw", "Wr"), ("Nn", "Wr"), ("Zs", "Wr"), ("Wr", "Ra"), ("Nn", "Ra"), ("Zs", "Ra"), ("Nn", "Ka")]
        for tag, ref in pairs:
            common = sorted(set(cells[tag]) & set(cells[ref]))
            if common:
                d = [cells[tag][s] - cells[ref][s] for s in common]
                print(f"  {tag} - {ref}: " + " ".join(f"s{s} {x:+.2f}" for s, x in zip(common, d)) + f"  | mean {sum(d) / len(d):+.2f} (n={len(d)})")


if __name__ == "__main__":
    main()
