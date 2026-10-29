"""Evaluation of binary organoid segmentations.

Pure numpy/scipy (no torch), so it can score masks produced by any model. The unit of analysis is the *well*:
the images of a well are time-lapse frames of the same culture and are not independent, so per-well values are
computed first and uncertainty is estimated by bootstrapping wells.

Metrics
- pixel overlap of the organoid class (Dice, IoU, precision, recall), plus background Dice and the macro average
  (organoid + background) / 2 that was used for the "mDice" of the manuscript, to make the two comparable;
- boundary quality: HD95 and boundary Dice (surface Dice at a pixel tolerance);
- instance level: connected components matched one-to-one at IoU >= threshold (precision, recall, F1, PQ),
  also stratified by component size.
"""
import math
import os
import re

import numpy as np
from scipy import ndimage as ndi

NAN = float("nan")
_WELL_RE = re.compile(r"^(?:IMG_|MASK_)?(.+?)_Interval\d+")

def well_id(filename):
    """'IMG_B11_Interval3.tif' -> 'B11' (falls back to the file stem when the pattern does not match)."""
    name = os.path.basename(filename)
    match = _WELL_RE.match(name)
    return match.group(1) if match else os.path.splitext(name)[0]

def _ratio(num, den, empty=NAN):
    return num / den if den > 0 else empty

# ----------------------------------------------------------------------------------------------------------------
# Pixel overlap
# ----------------------------------------------------------------------------------------------------------------
def confusion_counts(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": int(pred.size - tp - fp - fn)}

def overlap_metrics(c):
    """Metrics from pixel counts. Dice/IoU of an empty-vs-empty comparison is 1; precision/recall are NaN if undefined."""
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    dice = _ratio(2 * tp, 2 * tp + fp + fn, empty=1.0)
    iou = _ratio(tp, tp + fp + fn, empty=1.0)
    dice_bg = _ratio(2 * tn, 2 * tn + fp + fn, empty=1.0)
    iou_bg = _ratio(tn, tn + fp + fn, empty=1.0)
    return {
        "dice": dice,
        "iou": iou,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "dice_bg": dice_bg,
        "iou_bg": iou_bg,
        "macro_dice": (dice + dice_bg) / 2,
        "macro_iou": (iou + iou_bg) / 2,
    }

# ----------------------------------------------------------------------------------------------------------------
# Boundary
# ----------------------------------------------------------------------------------------------------------------
def _boundary(mask):
    # border_value=1: the image frame is not an object edge
    return mask & ~ndi.binary_erosion(mask, border_value=1)

def boundary_metrics(pred, gt, tolerance=2.0):
    """HD95 (pixels) and boundary Dice: share of boundary pixels (of both masks) lying within `tolerance` px of the other boundary."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    b_pred, b_gt = _boundary(pred), _boundary(gt)
    if not b_pred.any() and not b_gt.any():
        return {"hd95": 0.0, "boundary_dice": 1.0}
    if not b_pred.any() or not b_gt.any():
        return {"hd95": NAN, "boundary_dice": 0.0}
    d_pred_to_gt = ndi.distance_transform_edt(~b_gt)[b_pred]
    d_gt_to_pred = ndi.distance_transform_edt(~b_pred)[b_gt]
    hd95 = max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95))
    close = np.sum(d_pred_to_gt <= tolerance) + np.sum(d_gt_to_pred <= tolerance)
    return {"hd95": float(hd95), "boundary_dice": float(close / (b_pred.sum() + b_gt.sum()))}

# ----------------------------------------------------------------------------------------------------------------
# Instances (connected components)
# ----------------------------------------------------------------------------------------------------------------
def _label(mask, min_area):
    labels, n = ndi.label(mask)
    areas = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    if min_area > 0 and n > 0:
        keep = np.concatenate([[0], np.where(areas >= min_area, np.cumsum(areas >= min_area), 0)])
        labels = keep[labels]
        areas = areas[areas >= min_area]
        n = len(areas)
    return labels, n, areas

def instance_metrics(pred, gt, iou_threshold=0.5, size_edges=(200, 1000), min_area=0):
    """Match predicted and ground-truth connected components one-to-one (greedy by IoU, IoU >= threshold).

    Components are 4-connected, so organoids that touch are a single instance in both masks (the ground truth is a
    semantic mask): this evaluates detection of organoid *clusters*, not of individual touching organoids.
    Returns raw counts (so that they can be summed across images/wells), also split into size bins by area in pixels
    (`size_edges` -> len(size_edges) + 1 bins, GT components binned by GT area, predicted ones by predicted area).
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    lab_g, n_g, area_g = _label(gt, min_area)
    lab_p, n_p, area_p = _label(pred, min_area)
    n_bins = len(size_edges) + 1
    bin_g = np.digitize(area_g, size_edges)
    bin_p = np.digitize(area_p, size_edges)

    matched_g = np.zeros(n_g, dtype=bool)
    matched_p = np.zeros(n_p, dtype=bool)
    iou_g = np.zeros(n_g)
    if n_g and n_p:
        both = (lab_g > 0) & (lab_p > 0)
        keys, inter = np.unique(lab_g[both].astype(np.int64) * (n_p + 1) + lab_p[both], return_counts=True)
        gi, pi = keys // (n_p + 1) - 1, keys % (n_p + 1) - 1
        iou = inter / (area_g[gi] + area_p[pi] - inter)
        for k in np.argsort(-iou, kind="stable"):
            if iou[k] < iou_threshold:
                break
            if not matched_g[gi[k]] and not matched_p[pi[k]]:
                matched_g[gi[k]] = matched_p[pi[k]] = True
                iou_g[gi[k]] = iou[k]

    def per_bin(bins, flags):
        return np.bincount(bins, weights=flags.astype(float), minlength=n_bins).tolist()

    return {
        "n_gt": int(n_g), "n_pred": int(n_p), "tp": int(matched_g.sum()), "iou_sum": float(iou_g.sum()),
        "bin_gt": per_bin(bin_g, np.ones(n_g)), "bin_gt_tp": per_bin(bin_g, matched_g), "bin_gt_iou_sum": per_bin(bin_g, iou_g),
        "bin_pred": per_bin(bin_p, np.ones(n_p)), "bin_pred_tp": per_bin(bin_p, matched_p),
    }

def instance_summary(tp, n_gt, n_pred, iou_sum):
    precision = _ratio(tp, n_pred)
    recall = _ratio(tp, n_gt)
    f1 = _ratio(2 * tp, n_gt + n_pred, empty=1.0)
    sq = _ratio(iou_sum, tp)
    return {"inst_precision": precision, "inst_recall": recall, "inst_f1": f1, "inst_sq": sq,
            "inst_pq": f1 * sq if not math.isnan(sq) else 0.0}

# ----------------------------------------------------------------------------------------------------------------
# Per-image record
# ----------------------------------------------------------------------------------------------------------------
def evaluate_image(pred, gt, name, tolerance=2.0, iou_threshold=0.5, size_edges=(200, 1000), min_area=0):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    return {
        "image": name,
        "well": well_id(name),
        "gt_fraction": float(gt.mean()),
        "counts": confusion_counts(pred, gt),
        "boundary": boundary_metrics(pred, gt, tolerance),
        "instances": instance_metrics(pred, gt, iou_threshold, size_edges, min_area),
    }

# ----------------------------------------------------------------------------------------------------------------
# Aggregation over wells
# ----------------------------------------------------------------------------------------------------------------
_COUNT_KEYS = ("tp", "fp", "fn", "tn")
_INST_KEYS = ("n_gt", "n_pred", "tp", "iou_sum")
_BIN_KEYS = ("bin_gt", "bin_gt_tp", "bin_gt_iou_sum", "bin_pred", "bin_pred_tp")

def aggregate_wells(records):
    """One row per well: pixel/instance counts are summed over the well's images, boundary metrics averaged."""
    wells = {}
    for r in records:
        wells.setdefault(r["well"], []).append(r)
    rows = []
    for well, recs in sorted(wells.items()):
        counts = {k: sum(r["counts"][k] for r in recs) for k in _COUNT_KEYS}
        inst = {k: sum(r["instances"][k] for r in recs) for k in _INST_KEYS}
        bins = {k: np.sum([r["instances"][k] for r in recs], axis=0) for k in _BIN_KEYS}
        row = {"well": well, "n_images": len(recs), "gt_fraction": float(np.mean([r["gt_fraction"] for r in recs]))}
        row.update(overlap_metrics(counts))
        row["hd95"] = float(np.nanmean([r["boundary"]["hd95"] for r in recs])) if any(
            not math.isnan(r["boundary"]["hd95"]) for r in recs) else NAN
        row["boundary_dice"] = float(np.mean([r["boundary"]["boundary_dice"] for r in recs]))
        row.update(instance_summary(inst["tp"], inst["n_gt"], inst["n_pred"], inst["iou_sum"]))
        row["_counts"] = counts
        row["_inst"] = inst
        row["_bins"] = bins
        rows.append(row)
    return rows

# ----------------------------------------------------------------------------------------------------------------
# Bootstrap over wells
# ----------------------------------------------------------------------------------------------------------------
def _resample(n, n_boot, seed):
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))

def bootstrap_mean_ci(values, n_boot=10000, seed=0, alpha=0.05):
    """Percentile CI of the mean of `values` (NaN ignored), resampling the entries (= wells) with replacement."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.all(np.isnan(values)):
        return NAN, NAN
    boot = np.nanmean(values[_resample(len(values), n_boot, seed)], axis=1)
    return tuple(np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)]))

def bootstrap_pooled_ci(counts, stat_fn, n_boot=10000, seed=0, alpha=0.05):
    """CI of a statistic of the *pooled* counts. `counts` is (n_wells, k); `stat_fn` maps summed counts (n_boot, k) -> (n_boot,)."""
    counts = np.asarray(counts, dtype=float)
    if len(counts) < 2:
        return NAN, NAN
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = stat_fn(counts[_resample(len(counts), n_boot, seed)].sum(axis=1))
    if np.all(np.isnan(boot)):
        return NAN, NAN
    return tuple(np.nanpercentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)]))

def _paired_difference_ci(a, b, n_boot, seed, alpha):
    return bootstrap_mean_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), n_boot, seed, alpha)

def compare_models(rows_a, rows_b, metric="dice", n_boot=10000, seed=0, alpha=0.05):
    """Paired bootstrap CI of the per-well difference `metric(A) - metric(B)` (same wells, same resamples)."""
    a = {r["well"]: r[metric] for r in rows_a}
    b = {r["well"]: r[metric] for r in rows_b}
    common = sorted(set(a) & set(b))
    diff = np.array([a[w] - b[w] for w in common])
    return {"mean_difference": float(np.nanmean(diff)), "ci": bootstrap_mean_ci(diff, n_boot, seed, alpha), "n_wells": len(common)}

# ----------------------------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------------------------
WELL_METRICS = ("dice", "iou", "precision", "recall", "hd95", "boundary_dice",
                "inst_precision", "inst_recall", "inst_f1", "inst_pq", "dice_bg", "macro_dice", "macro_iou")

def _dice_from(s):  # columns: tp, fp, fn, tn
    return 2 * s[:, 0] / (2 * s[:, 0] + s[:, 1] + s[:, 2])

def _iou_from(s):
    return s[:, 0] / (s[:, 0] + s[:, 1] + s[:, 2])

def _macro_dice_from(s):
    return (_dice_from(s) + 2 * s[:, 3] / (2 * s[:, 3] + s[:, 1] + s[:, 2])) / 2

def _macro_iou_from(s):
    return (_iou_from(s) + s[:, 3] / (s[:, 3] + s[:, 1] + s[:, 2])) / 2

def summarize(rows, size_edges=(200, 1000), n_boot=10000, seed=0, alpha=0.05):
    """Summary of per-well rows: mean/SD/median with bootstrap CI, pooled-pixel values, size and density strata."""
    summary = {"n_wells": len(rows), "n_images": int(sum(r["n_images"] for r in rows)), "n_boot": n_boot, "alpha": alpha}

    per_well = {}
    for m in WELL_METRICS:
        v = np.array([r[m] for r in rows], dtype=float)
        lo, hi = bootstrap_mean_ci(v, n_boot, seed, alpha)
        per_well[m] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else NAN,
                       "median": float(np.nanmedian(v)), "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
                       "ci_low": float(lo), "ci_high": float(hi)}
    summary["per_well"] = per_well

    counts = np.array([[r["_counts"][k] for k in _COUNT_KEYS] for r in rows], dtype=float)
    pooled_counts = dict(zip(_COUNT_KEYS, counts.sum(axis=0).astype(np.int64).tolist()))
    pooled = overlap_metrics(pooled_counts)
    fns = {"dice": _dice_from, "iou": _iou_from, "macro_dice": _macro_dice_from, "macro_iou": _macro_iou_from}
    summary["pooled"] = {}
    for m in ("dice", "iou", "precision", "recall", "dice_bg", "macro_dice", "macro_iou"):
        entry = {"value": pooled[m]}
        stat = fns.get(m)
        if m == "precision":
            stat = lambda s: s[:, 0] / (s[:, 0] + s[:, 1])
        elif m == "recall":
            stat = lambda s: s[:, 0] / (s[:, 0] + s[:, 2])
        elif m == "dice_bg":
            stat = lambda s: 2 * s[:, 3] / (2 * s[:, 3] + s[:, 1] + s[:, 2])
        lo, hi = bootstrap_pooled_ci(counts, stat, n_boot, seed, alpha)
        entry.update(ci_low=float(lo), ci_high=float(hi))
        summary["pooled"][m] = entry

    # instance metrics by component size (pooled over wells)
    n_bins = len(size_edges) + 1
    labels = []
    for i in range(n_bins):
        lo_e = 0 if i == 0 else size_edges[i - 1]
        labels.append(f"<{size_edges[0]}px" if i == 0 else (f">={size_edges[-1]}px" if i == n_bins - 1 else f"{lo_e}-{size_edges[i]}px"))
    by_size = []
    for i in range(n_bins):
        cols = np.array([[r["_bins"]["bin_gt"][i], r["_bins"]["bin_gt_tp"][i], r["_bins"]["bin_gt_iou_sum"][i],
                          r["_bins"]["bin_pred"][i], r["_bins"]["bin_pred_tp"][i]] for r in rows], dtype=float)
        s = cols.sum(axis=0)
        rec_ci = bootstrap_pooled_ci(cols, lambda x: x[:, 1] / x[:, 0], n_boot, seed, alpha)
        prec_ci = bootstrap_pooled_ci(cols, lambda x: x[:, 4] / x[:, 3], n_boot, seed, alpha)
        by_size.append({
            "bin": labels[i], "n_gt": int(s[0]), "n_pred": int(s[3]),
            "recall": _ratio(s[1], s[0]), "recall_ci": [float(x) for x in rec_ci],
            "precision": _ratio(s[4], s[3]), "precision_ci": [float(x) for x in prec_ci],
            "mean_matched_iou": _ratio(s[2], s[1]),
        })
    summary["by_size"] = by_size

    # density strata: wells below / above the median foreground fraction of the ground truth
    fraction = np.array([r["gt_fraction"] for r in rows])
    median = float(np.median(fraction))
    summary["by_density"] = {"median_gt_fraction": median}
    for name, sel in (("low", fraction <= median), ("high", fraction > median)):
        if sel.any():
            vals = np.array([r["dice"] for r in rows])[sel]
            lo, hi = bootstrap_mean_ci(vals, n_boot, seed, alpha)
            summary["by_density"][name] = {"n_wells": int(sel.sum()), "dice_mean": float(vals.mean()),
                                           "ci_low": float(lo), "ci_high": float(hi)}
    return summary

def format_summary(s, title="Evaluation"):
    pct = lambda x: f"{100 * x:6.2f}"
    ci = 100 * (1 - s["alpha"])
    lines = [f"{title}: {s['n_images']} images, {s['n_wells']} wells (bootstrap over wells, {s['n_boot']} resamples, {ci:.0f}% CI)", ""]
    lines.append("Per-well, organoid class: mean [CI of the mean] (SD, min-max)")
    names = {"dice": "Dice", "iou": "IoU", "precision": "Precision", "recall": "Recall", "boundary_dice": "Boundary Dice",
             "inst_precision": "Instance precision", "inst_recall": "Instance recall", "inst_f1": "Instance F1", "inst_pq": "Instance PQ"}
    for m, label in names.items():
        d = s["per_well"][m]
        lines.append(f"  {label:<20}{pct(d['mean'])} [{pct(d['ci_low'])}, {pct(d['ci_high'])}]  (SD {100 * d['sd']:.2f}, {pct(d['min'])}-{pct(d['max'])})")
    d = s["per_well"]["hd95"]
    lines.append(f"  {'HD95 (px)':<20}{d['mean']:6.2f} [{d['ci_low']:6.2f}, {d['ci_high']:6.2f}]  (SD {d['sd']:.2f}, {d['min']:.2f}-{d['max']:.2f})")
    lines += ["", "Pooled over all pixels (comparable to the manuscript): value [CI over wells]"]
    pooled_names = {"dice": "Dice (organoid)", "iou": "IoU (organoid)", "precision": "Precision (organoid)", "recall": "Recall (organoid)",
                    "dice_bg": "Dice (background)", "macro_dice": "mDice (org+bg)/2", "macro_iou": "mIoU (org+bg)/2"}
    for m, label in pooled_names.items():
        d = s["pooled"][m]
        lines.append(f"  {label:<20}{pct(d['value'])} [{pct(d['ci_low'])}, {pct(d['ci_high'])}]")
    lines += ["", "Instances by component size (4-connected components, matched at IoU>=threshold)"]
    lines.append(f"  {'size':<14}{'n GT':>7}{'n pred':>8}{'recall':>9}{'precision':>11}{'matched IoU':>13}")
    for b in s["by_size"]:
        lines.append(f"  {b['bin']:<14}{b['n_gt']:>7}{b['n_pred']:>8}{pct(b['recall']):>9}{pct(b['precision']):>11}{pct(b['mean_matched_iou']):>13}")
    dens = s["by_density"]
    lines += ["", f"Per-well Dice by organoid density (median GT foreground fraction {100 * dens['median_gt_fraction']:.1f}%)"]
    for name in ("low", "high"):
        if name in dens:
            d = dens[name]
            lines.append(f"  {name + ' density':<20}{pct(d['dice_mean'])} [{pct(d['ci_low'])}, {pct(d['ci_high'])}]  ({d['n_wells']} wells)")
    return "\n".join(lines)
