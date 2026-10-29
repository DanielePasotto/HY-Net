"""Convert an external organoid dataset to the layout of this project so that the models trained on our folds can be evaluated on it.

  organosegment  OrgaSegment "OrganoidBasic" (1024x1024 JPEG + instance-labelled PNG masks): the `val` and `eval` folders
  organoid       OrganoID (https://osf.io/xmes4/): OriginalData/testing (28 human organoid images) and MouseOrganoids/testing (4 large images)

Images become 8-bit grayscale PNG and masks binary (0/255) PNG, kept at their native size (the evaluation resizes both to 480x640 exactly as
for our data). Every image is its own "well" (IMG_<id>_Interval0.png), so that the per-well statistics of hynet-compare are per image.
A one-fold folds file listing all the images as test is written next to it (folds/external_<name>.json).

With --letterbox the images are not stretched to 480x640: they are scaled to fit (aspect ratio kept) and padded to 480x640 (image: median
grey level, mask: background), in data/external/<name>_lb. This tells whether a drop in performance comes from the change of aspect ratio.
"""
import argparse
import json
import os
import numpy as np
from PIL import Image

def to_gray(path):
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    a = a.astype(np.float32)
    lo, hi = a.min(), a.max()
    return np.round(255 * (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)).astype(np.uint8)

def to_mask(path):
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    return (a > 0).astype(np.uint8) * 255

def letterbox(a, is_mask, height=480, width=640):
    """Scale a uint8 array to fit height x width keeping the aspect ratio and pad the rest."""
    h, w = a.shape
    s = min(height / h, width / w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = np.array(Image.fromarray(a).resize((nw, nh), Image.NEAREST if is_mask else Image.BILINEAR))
    canvas = np.full((height, width), 0 if is_mask else int(np.median(small)), dtype=np.uint8)
    top, left = (height - nh) // 2, (width - nw) // 2
    canvas[top:top + nh, left:left + nw] = small
    return canvas

def organosegment_items(root):
    for split in ("val", "eval"):
        d = os.path.join(root, split)
        for f in sorted(os.listdir(d)):
            if f.endswith("_img.jpg"):
                stem = f[:-len("_img.jpg")]
                yield stem, os.path.join(d, f), os.path.join(d, stem + "_masks_organoid.png")

def organoid_items(root):
    d = os.path.join(root, "OriginalData", "testing")
    for f in sorted(os.listdir(os.path.join(d, "images"))):
        yield os.path.splitext(f)[0], os.path.join(d, "images", f), os.path.join(d, "segmentations", f)
    d = os.path.join(root, "MouseOrganoids", "testing")
    for f in sorted(os.listdir(os.path.join(d, "images"))):
        yield "mouse_" + os.path.splitext(f)[0], os.path.join(d, "images", f), os.path.join(d, "segmentations", f)

DATASETS = {"organosegment": ("data/OrganoidBasic_v20211206", organosegment_items), "organoid": ("data/Dataset", organoid_items)}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--source", default=None, help="folder of the original dataset (default: data/OrganoidBasic_v20211206 or data/Dataset)")
    ap.add_argument("--out_root", default="data/external")
    ap.add_argument("--folds_dir", default="folds")
    ap.add_argument("--letterbox", action="store_true", help="keep the aspect ratio (scale + pad to 480x640) instead of stretching; writes <name>_lb")
    args = ap.parse_args()

    default_source, items = DATASETS[args.dataset]
    name = args.dataset + ("_lb" if args.letterbox else "")
    out = os.path.join(args.out_root, name, "test")
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "masks"), exist_ok=True)
    wells = []
    for well, image_path, mask_path in items(args.source or default_source):
        image, mask = to_gray(image_path), to_mask(mask_path)
        if args.letterbox:
            image, mask = letterbox(image, False), letterbox(mask, True)
        Image.fromarray(image).save(os.path.join(out, "images", f"IMG_{well}_Interval0.png"))
        Image.fromarray(mask).save(os.path.join(out, "masks", f"MASK_{well}_Interval0.png"))
        wells.append(well)
    assert len(set(wells)) == len(wells), "duplicate image ids"
    os.makedirs(args.folds_dir, exist_ok=True)
    doc = {"k": 1, "seed": 0, "grouped_by": "well", "groups": {w: w for w in wells}, "folds": [{"train": [], "val": [], "test": wells}]}
    with open(os.path.join(args.folds_dir, f"external_{name}.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print(f"{len(wells)} images of {args.dataset} written to {out}")

if __name__ == "__main__":
    main()
