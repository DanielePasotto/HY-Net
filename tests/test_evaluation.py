import math

import numpy as np
import pytest

from hynet import evaluation as ev

def square(shape=(100, 100), top=10, left=10, size=20):
    m = np.zeros(shape, dtype=bool)
    m[top:top + size, left:left + size] = True
    return m

def test_well_id():
    assert ev.well_id("IMG_B11_Interval3.tif") == "B11"
    assert ev.well_id("MASK_A1_Interval9.tif") == "A1"
    assert ev.well_id("/some/dir/IMG_B14_Interval1.tif") == "B14"
    assert ev.well_id("weird_name.png") == "weird_name"

def test_perfect_prediction():
    gt = square()
    r = ev.evaluate_image(gt, gt, "IMG_B1_Interval1.tif")
    m = ev.overlap_metrics(r["counts"])
    assert m["dice"] == m["iou"] == m["precision"] == m["recall"] == 1.0
    assert r["boundary"] == {"hd95": 0.0, "boundary_dice": 1.0}
    assert r["instances"]["tp"] == r["instances"]["n_gt"] == r["instances"]["n_pred"] == 1

def test_overlap_numbers():
    gt, pred = square(), square(left=11)  # shifted by one pixel: 380 of 400 pixels overlap
    c = ev.confusion_counts(pred, gt)
    assert c == {"tp": 380, "fp": 20, "fn": 20, "tn": 100 * 100 - 420}
    m = ev.overlap_metrics(c)
    assert m["dice"] == pytest.approx(2 * 380 / 800)
    assert m["iou"] == pytest.approx(380 / 420)
    assert m["macro_dice"] == pytest.approx((m["dice"] + m["dice_bg"]) / 2)
    # the background term inflates the macro score: this is the point of reporting the organoid class on its own
    assert m["macro_dice"] > m["dice"]

def test_empty_prediction():
    gt = square()
    pred = np.zeros_like(gt)
    r = ev.evaluate_image(pred, gt, "IMG_B1_Interval1.tif")
    m = ev.overlap_metrics(r["counts"])
    assert m["dice"] == 0.0 and m["recall"] == 0.0 and math.isnan(m["precision"])
    assert math.isnan(r["boundary"]["hd95"]) and r["boundary"]["boundary_dice"] == 0.0
    assert r["instances"]["tp"] == 0 and r["instances"]["n_gt"] == 1 and r["instances"]["n_pred"] == 0

def test_boundary_shift():
    gt, pred = square(), square(left=13)  # 3 px shift
    b = ev.boundary_metrics(pred, gt, tolerance=2.0)
    assert 0 < b["boundary_dice"] < 1
    assert b["hd95"] >= 3.0 - 1e-9
    assert ev.boundary_metrics(pred, gt, tolerance=5.0)["boundary_dice"] == 1.0

def test_merged_prediction_matches_one_instance_only():
    gt = square(size=10, left=10) | square(size=10, left=40)
    pred = np.zeros_like(gt)
    pred[10:20, 10:50] = True  # one blob covering both organoids
    inst = ev.instance_metrics(pred, gt, iou_threshold=0.2)
    assert inst["n_gt"] == 2 and inst["n_pred"] == 1
    assert inst["tp"] == 1  # one-to-one matching: the blob cannot explain both
    assert ev.instance_metrics(pred, gt, iou_threshold=0.5)["tp"] == 0

def test_instance_size_bins():
    gt = square(size=5, top=2, left=2) | square(size=20, top=50, left=50)
    inst = ev.instance_metrics(gt, gt, size_edges=(100,))
    assert inst["bin_gt"] == [1.0, 1.0] and inst["bin_gt_tp"] == [1.0, 1.0]
    assert inst["bin_pred"] == [1.0, 1.0] and inst["bin_pred_tp"] == [1.0, 1.0]

def test_min_instance_area_ignores_specks():
    gt = square()
    pred = gt.copy()
    pred[80, 80] = True  # one-pixel false positive
    assert ev.instance_metrics(pred, gt)["n_pred"] == 2
    assert ev.instance_metrics(pred, gt, min_area=5)["n_pred"] == 1

def test_bootstrap_is_reproducible_and_bounded():
    values = np.array([0.90, 0.92, 0.95, 0.97, 0.99])
    lo, hi = ev.bootstrap_mean_ci(values, n_boot=2000, seed=1)
    assert (lo, hi) == ev.bootstrap_mean_ci(values, n_boot=2000, seed=1)
    assert values.min() <= lo <= values.mean() <= hi <= values.max()
    assert ev.bootstrap_mean_ci(np.full(5, 0.8), n_boot=500) == (pytest.approx(0.8), pytest.approx(0.8))
    assert all(math.isnan(x) for x in ev.bootstrap_mean_ci([0.5]))

def make_records(rng, n_wells=6, frames=3):
    records = []
    for w in range(n_wells):
        for t in range(frames):
            gt = square(top=10 + w, left=10 + 3 * t)
            pred = square(top=10 + w, left=11 + 3 * t, size=19)
            records.append(ev.evaluate_image(pred, gt, f"IMG_W{w}_Interval{2 * t + 1}.tif"))
    return records

def test_aggregation_and_summary():
    rows = ev.aggregate_wells(make_records(np.random.default_rng(0)))
    assert len(rows) == 6 and all(r["n_images"] == 3 for r in rows)
    summary = ev.summarize(rows, n_boot=500)
    assert summary["n_wells"] == 6 and summary["n_images"] == 18
    d = summary["per_well"]["dice"]
    assert d["ci_low"] <= d["mean"] <= d["ci_high"]
    p = summary["pooled"]["dice"]
    assert p["ci_low"] <= p["value"] <= p["ci_high"]
    assert [b["bin"] for b in summary["by_size"]] == ["<200px", "200-1000px", ">=1000px"]
    assert sum(b["n_gt"] for b in summary["by_size"]) == 18
    assert "Dice" in ev.format_summary(summary)

def test_pooled_dice_matches_pixel_totals():
    rows = ev.aggregate_wells(make_records(np.random.default_rng(0)))
    pooled = ev.summarize(rows, n_boot=100)["pooled"]["dice"]["value"]
    tp = sum(r["_counts"]["tp"] for r in rows)
    fp = sum(r["_counts"]["fp"] for r in rows)
    fn = sum(r["_counts"]["fn"] for r in rows)
    assert pooled == pytest.approx(2 * tp / (2 * tp + fp + fn))

def test_paired_model_comparison():
    rows_a = [{"well": f"W{i}", "dice": 0.95 + 0.001 * i} for i in range(8)]
    rows_b = [{"well": f"W{i}", "dice": 0.90 + 0.001 * i} for i in range(8)]
    out = ev.compare_models(rows_a, rows_b, n_boot=500)
    assert out["mean_difference"] == pytest.approx(0.05)
    assert out["ci"][0] > 0 and out["n_wells"] == 8
