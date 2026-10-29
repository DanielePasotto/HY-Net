"""Score saved binary prediction masks against the ground truth (works for any model that wrote PNG masks)."""
import os
import numpy as np
from PIL import Image
from hynet import data
from hynet import evaluation

def score_predictions(entries, pred_dir, tolerance=2.0, iou_threshold=0.5, size_edges=(200, 1000), min_area=0):
    """`entries` is a list of (key, image_path, mask_path); the prediction of an image is <pred_dir>/<image stem>.png.

    Predictions and ground truth are compared on the 480x640 grid used for training (ground truth resized with nearest neighbour).
    """
    records = []
    for key, image_path, mask_path in entries:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        pred_path = os.path.join(pred_dir, stem + ".png")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing prediction {pred_path}")
        pred = np.array(Image.open(pred_path)) > 0
        gt = data.load_mask_480x640(mask_path)
        if pred.shape != gt.shape:
            raise ValueError(f"{pred_path}: prediction is {pred.shape}, expected {gt.shape}")
        records.append(evaluation.evaluate_image(pred, gt, os.path.basename(image_path), tolerance, iou_threshold, size_edges, min_area))
    return records
