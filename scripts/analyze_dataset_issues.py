"""What makes the images of the pooled datasets hard to segment.

For every image of data/pool it measures, on the grey levels and on the ground-truth mask: contrast and dynamic range, illumination ramp, noise,
sharpness, JPEG blocking, foreground/background contrast, edge sharpness at the organoid boundary, organoid size, number, shape and contact with the
image border. It then compares the three datasets and, where per-image Dice files of trained models are given, relates every measure to the Dice.

  python scripts/analyze_dataset_issues.py --out outputs/dataset_analysis --dice ref=<per_image.csv> ref=<...> alt=<...>

`--dice <group>=<per_image.csv>`: files of the same group are averaged over seeds (the file name must tell the domain: `..__<domain>.csv`).
Writes features.csv, domain_summary.txt, dice_relation.txt in --out.
"""
import argparse
import csv
import os
import re
from collections import defaultdict

import cv2
import numpy as np
from scipy import stats
from skimage.measure import regionprops, label

EVAL_SIZE = (640, 480)  # (width, height) at which the instance metrics bin the organoid sizes
SMALL, MEDIUM = 200, 1000  # area edges in pixels at EVAL_SIZE, as in hynet-evaluate
BAND = 3  # half width in pixels of the boundary band


MORPHOLOGY = ["fg_frac", "frac_small_area", "perim_per_area", "median_diam", "border_touch", "solidity", "n_comp"]
PHOTOMETRIC = ["range_p1_p99", "std", "abs_contrast_io", "snr_io", "edge_strength", "edge_to_bg_grad", "ramp_amp", "ramp_r2", "lowfreq_cv",
               "bg_cv", "texture_in", "noise_sigma", "lap_var_norm", "blockiness", "levels_used", "mean"]


def domain_of(name):
    return re.match(r"IMG_([a-z]+)-", name).group(1)


def noise_sigma(g):
    """Immerkaer's estimate of the standard deviation of white noise, from a Laplacian-like kernel that cancels smooth structure."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = cv2.filter2D(g, cv2.CV_32F, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(r).mean() / 6)


def plane_ramp(g, bg):
    """Illumination ramp: amplitude of the plane fitted on the background pixels (range across the image / mean) and the share of variance it explains."""
    small = cv2.resize(g, (64, 48), interpolation=cv2.INTER_AREA)
    sbg = cv2.resize(bg.astype(np.uint8), (64, 48), interpolation=cv2.INTER_NEAREST) > 0
    if sbg.sum() < 50:
        sbg = np.ones_like(sbg)
    yy, xx = np.mgrid[0:48, 0:64]
    A = np.c_[np.ones(sbg.sum()), xx[sbg] / 64.0, yy[sbg] / 48.0]
    y = small[sbg]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    amp = (abs(coef[1]) + abs(coef[2])) / max(float(y.mean()), 1e-6)
    r2 = 1 - ((y - A @ coef) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-6)
    return float(amp), float(r2), float(small.std() / max(small.mean(), 1e-6))


def blockiness(g):
    """Ratio of the mean horizontal+vertical grey-level jump across the 8-pixel grid to the jump elsewhere (1 = no JPEG blocking)."""
    out = []
    for d in (np.abs(np.diff(g, axis=1)).mean(0), np.abs(np.diff(g, axis=0)).mean(1)):
        on = d[7::8].mean()
        off = np.delete(d, np.arange(7, len(d), 8)).mean()
        out.append(on / max(off, 1e-6))
    return float(np.mean(out))


def features(path, mask_path):
    g = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if g.ndim == 3:
        g = cv2.cvtColor(g[..., :3], cv2.COLOR_BGR2GRAY)
    depth = 16 if g.dtype == np.uint16 else 8
    g = g.astype(np.float32)
    if depth == 16:
        g = 255 * (g - g.min()) / max(float(g.max() - g.min()), 1.0)
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    m = (m > 0).astype(np.uint8) if m.ndim == 2 else (m.max(-1) > 0).astype(np.uint8)
    if m.shape != g.shape:
        m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    h, w = g.shape
    fg = m > 0
    f = {"width": w, "height": h}
    p1, p99 = np.percentile(g, [1, 99])
    f["mean"], f["std"] = float(g.mean()), float(g.std())
    f["range_p1_p99"] = float((p99 - p1) / 255)
    f["levels_used"] = int(len(np.unique(g.astype(np.uint8))))
    f["sat_low"], f["sat_high"] = float((g <= 2).mean()), float((g >= 253).mean())
    f["ramp_amp"], f["ramp_r2"], f["lowfreq_cv"] = plane_ramp(g, ~fg)
    f["noise_sigma"] = noise_sigma(g)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    f["lap_var_norm"] = float(lap.var() / max(g.var(), 1e-6))
    f["blockiness"] = blockiness(g)

    f["fg_frac"] = float(fg.mean())
    if fg.any() and (~fg).any():
        mu_in, mu_out = float(g[fg].mean()), float(g[~fg].mean())
        f["contrast_io"] = (mu_in - mu_out) / max(float(g.std()), 1e-6)  # signed: > 0 organoids brighter than the background
        f["abs_contrast_io"] = abs(f["contrast_io"])
        f["snr_io"] = abs(mu_in - mu_out) / max(f["noise_sigma"], 1e-6)
        f["texture_in"] = float(g[fg].std() / max(mu_in, 1e-6))
        f["bg_cv"] = float(g[~fg].std() / max(mu_out, 1e-6))
        k = np.ones((2 * BAND + 1,) * 2, np.uint8)
        band = (cv2.dilate(m, k) - cv2.erode(m, k)) > 0
        gs = cv2.GaussianBlur(g, (0, 0), 1.0)
        grad = np.hypot(cv2.Sobel(gs, cv2.CV_32F, 1, 0), cv2.Sobel(gs, cv2.CV_32F, 0, 1)) / 8
        f["edge_strength"] = float(grad[band].mean() / max(abs(mu_in - mu_out), 1e-6)) if band.any() else np.nan  # 1/width of the transition
        f["edge_to_bg_grad"] = float(grad[band].mean() / max(grad[~fg & ~band].mean(), 1e-6)) if band.any() else np.nan
    else:
        for k_ in ("contrast_io", "abs_contrast_io", "snr_io", "texture_in", "bg_cv", "edge_strength", "edge_to_bg_grad"):
            f[k_] = np.nan

    # organoids at the evaluation scale
    me = cv2.resize(m, EVAL_SIZE, interpolation=cv2.INTER_NEAREST)
    lab = label(me, connectivity=2)
    props = regionprops(lab)
    areas = np.array([p.area for p in props], float)
    f["n_comp"] = len(props)
    f["frac_small_comp"] = float((areas < SMALL).mean()) if len(areas) else np.nan
    f["frac_small_area"] = float(areas[areas < SMALL].sum() / areas.sum()) if len(areas) else np.nan
    f["median_diam"] = float(np.median([p.equivalent_diameter_area for p in props])) if props else np.nan
    f["max_area_frac"] = float(areas.max() / areas.sum()) if len(areas) else np.nan
    f["solidity"] = float(np.mean([p.solidity for p in props])) if props else np.nan
    f["border_touch"] = float(np.mean([p.bbox[0] == 0 or p.bbox[1] == 0 or p.bbox[2] == EVAL_SIZE[1] or p.bbox[3] == EVAL_SIZE[0] for p in props])) if props else np.nan
    f["perim_per_area"] = float(sum(p.perimeter for p in props) / areas.sum()) if len(areas) else np.nan
    f["empty_mask"] = int(not fg.any())
    return f


def read_dice(specs):
    """{group: {image: dice averaged over the files of the group}}"""
    acc = defaultdict(lambda: defaultdict(list))
    for spec in specs:
        group, path = spec.split("=", 1)
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                acc[group][r["image"]].append(float(r["dice"]))
    return {g: {k: float(np.mean(v)) for k, v in d.items()} for g, d in acc.items()}


def cohen_d(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    sd = np.sqrt((a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="data/pool")
    ap.add_argument("--out", default="outputs/dataset_analysis")
    ap.add_argument("--dice", nargs="*", default=[], help="<group>=<per_image.csv> (see above)")
    ap.add_argument("--limit", type=int, default=0, help="debug: images per domain")
    ap.add_argument("--reuse", action="store_true", help="read <out>/features.csv instead of measuring the images again")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    names = [] if args.reuse else sorted(os.listdir(os.path.join(args.pool, "train", "images")))
    if args.limit:
        keep = defaultdict(int)
        sel = []
        for n in names:
            d = domain_of(n)
            if keep[d] < args.limit:
                keep[d] += 1
                sel.append(n)
        names = sel
    rows = []
    for i, n in enumerate(names):
        mp = next((os.path.join(args.pool, "train", "masks", c) for c in (n.replace("IMG_", "MASK_", 1), n) if os.path.exists(os.path.join(args.pool, "train", "masks", c))), None)
        if mp is None:
            continue
        r = {"image": n, "domain": domain_of(n)}
        r.update(features(os.path.join(args.pool, "train", "images", n), mp))
        rows.append(r)
        if i % 500 == 0:
            print(i, len(names), flush=True)
    if args.reuse:
        with open(os.path.join(args.out, "features.csv"), newline="") as fh:
            rows = [{k: (v if k in ("image", "domain") else float(v)) for k, v in r.items()} for r in csv.DictReader(fh)]
        cols = list(rows[0])
    else:
        cols = list(rows[0])
        with open(os.path.join(args.out, "features.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, cols)
            wr.writeheader()
            wr.writerows(rows)

    feats = [c for c in cols if c not in ("image", "domain")]
    X = {c: np.array([r[c] for r in rows], float) for c in feats}
    dom = np.array([r["domain"] for r in rows])
    domains = sorted(set(dom))

    with open(os.path.join(args.out, "domain_summary.txt"), "w") as fh:
        fh.write("median [p10, p90] of every measure, per dataset\n")
        fh.write(f"{'measure':18s}" + "".join(f"{d:>30s}" for d in domains) + "\n")
        for c in feats:
            cells = []
            for d in domains:
                v = X[c][dom == d]
                v = v[~np.isnan(v)]
                cells.append(f"{np.median(v):9.3g} [{np.percentile(v, 10):8.3g},{np.percentile(v, 90):8.3g}]" if len(v) else "-")
            fh.write(f"{c:18s}" + "".join(f"{x:>30s}" for x in cells) + "\n")
        fh.write("\nstandardized difference (Cohen d) between datasets; |d| > 0.8 = large shift\n")
        pairs = [(a, b) for i, a in enumerate(domains) for b in domains[i + 1:]]
        fh.write(f"{'measure':18s}" + "".join(f"{a + '-' + b:>22s}" for a, b in pairs) + "\n")
        for c in feats:
            fh.write(f"{c:18s}" + "".join(f"{cohen_d(X[c][dom == a], X[c][dom == b]):22.2f}" for a, b in pairs) + "\n")

    if args.dice:
        dice = read_dice(args.dice)
        idx = {r["image"]: i for i, r in enumerate(rows)}
        with open(os.path.join(args.out, "dice_relation.txt"), "w") as fh:
            for group, dd in dice.items():
                for d in domains:
                    ims = [k for k in dd if k in idx and rows[idx[k]]["domain"] == d]
                    if len(ims) < 10:
                        continue
                    y = np.array([dd[k] for k in ims])
                    ii = np.array([idx[k] for k in ims])
                    fh.write(f"\n== {group} on {d}: {len(ims)} test images, Dice mean {100 * y.mean():.1f}, median {100 * np.median(y):.1f}, "
                             f"p10 {100 * np.percentile(y, 10):.1f}, share < 0.5: {(y < 0.5).mean():.2f}\n")
                    fh.write(f"{'measure':18s}{'spearman':>10s}{'p':>9s}{'Q1 dice':>9s}{'Q4 dice':>9s}{'Q1 value':>11s}{'Q4 value':>11s}\n")
                    res = []
                    for c in feats:
                        v = X[c][ii]
                        ok = ~np.isnan(v)
                        if ok.sum() < 10 or np.ptp(v[ok]) == 0:
                            continue
                        rho, p = stats.spearmanr(v[ok], y[ok])
                        lo, hi = np.percentile(v[ok], [25, 75])
                        res.append((abs(rho), c, rho, p, 100 * y[ok][v[ok] <= lo].mean(), 100 * y[ok][v[ok] >= hi].mean(), lo, hi))
                    for _, c, rho, p, q1, q4, lo, hi in sorted(res, reverse=True):
                        fh.write(f"{c:18s}{rho:10.2f}{p:9.3f}{q1:9.1f}{q4:9.1f}{lo:11.3g}{hi:11.3g}\n")
                    worst = ii[y <= np.percentile(y, 10)]
                    fh.write("worst decile vs the rest of the dataset (z of the median, in units of the dataset's std):\n")
                    allv = np.where(dom == d)[0]
                    zs = []
                    for c in feats:
                        a, b = X[c][worst], X[c][allv]
                        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
                        if len(a) and b.std() > 0:
                            zs.append((abs(np.median(a) - np.median(b)) / b.std(), c, (np.median(a) - np.median(b)) / b.std()))
                    fh.write("  " + ", ".join(f"{c} {z:+.2f}" for _, c, z in sorted(zs, reverse=True)[:8]) + "\n")
                    # photometric measures against the Dice left over once the organoid morphology is regressed out (on ranks)
                    M = np.c_[[stats.rankdata(np.nan_to_num(X[c][ii], nan=np.nanmedian(X[c][ii]))) for c in MORPHOLOGY]].T
                    M = np.c_[np.ones(len(ii)), M]
                    ry = stats.rankdata(y)
                    resid = ry - M @ np.linalg.lstsq(M, ry, rcond=None)[0]
                    fh.write(f"morphology ({', '.join(MORPHOLOGY)}) explains R2 = {1 - resid.var() / ry.var():.2f} of the Dice ranks; photometric measures vs the Dice left over:\n  ")
                    part = []
                    for c in PHOTOMETRIC:
                        v = X[c][ii]
                        ok = ~np.isnan(v)
                        if ok.sum() > 10 and np.ptp(v[ok]) > 0:
                            rho, p = stats.spearmanr(v[ok], resid[ok])
                            part.append((abs(rho), c, rho, p))
                    fh.write(", ".join(f"{c} {rho:+.2f}{'*' if p < 0.01 else ''}" for _, c, rho, p in sorted(part, reverse=True)) + "  (* p < 0.01)\n")
                    cio = X["contrast_io"][ii]
                    if (cio > 0).sum() >= 3 and (cio < 0).sum() >= 3:
                        fh.write(f"polarity: organoids brighter than the background in {(cio > 0).sum()} images (Dice {100 * y[cio > 0].mean():.1f}), "
                                 f"darker in {(cio < 0).sum()} (Dice {100 * y[cio < 0].mean():.1f})\n")
            if "ref" in dice and "alt" in dice:
                fh.write("\n== per-image gain of alt over ref (Dice points) ==\n")
                for d in domains:
                    ims = [k for k in dice["ref"] if k in dice["alt"] and k in idx and rows[idx[k]]["domain"] == d]
                    if len(ims) < 10:
                        continue
                    gain = np.array([dice["alt"][k] - dice["ref"][k] for k in ims]) * 100
                    ii = np.array([idx[k] for k in ims])
                    fh.write(f"{d}: mean {gain.mean():+.2f}, share improved {(gain > 0).mean():.2f}; correlation of gain with ref Dice (low-Dice images gain more if < 0): "
                             f"{stats.spearmanr([dice['ref'][k] for k in ims], gain)[0]:+.2f}\n")
                    top = []
                    for c in feats:
                        v = X[c][ii]
                        ok = ~np.isnan(v)
                        if ok.sum() > 10 and np.ptp(v[ok]) > 0:
                            top.append((abs(stats.spearmanr(v[ok], gain[ok])[0]), c, stats.spearmanr(v[ok], gain[ok])[0]))
                    fh.write("  measures that predict the gain: " + ", ".join(f"{c} {r:+.2f}" for _, c, r in sorted(top, reverse=True)[:6]) + "\n")
    print("done", len(rows))


if __name__ == "__main__":
    main()
