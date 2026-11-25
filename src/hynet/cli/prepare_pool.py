"""Pool our dataset with OrgaSegment and OrganoID into one dataset, and write the leave-one-dataset-out folds.

Everything is written to <out_root>/train/{images,masks} (a single container folder: the folds files decide which image is used for what), as
IMG_<domain>-<id>_Interval<n>.<ext> / MASK_... with domain `ours`, `orgaseg` or `organoid`. Our images are copied unchanged; the external ones become
8-bit grayscale PNG and binary masks (instance masks are binarized), at their native size.

  ours       DatasetOrganoidsSplitted_70_15_15 (train / val / test folders)
  orgaseg    OrgaSegment "OrganoidBasic" (train / val / eval folders)
  organoid   OrganoID OriginalData and MouseOrganoids (training / validation / testing folders; the training images are 512x512 crops)

folds/lodo_<domain>.json tests on <domain> and trains on the training images of the other two (validation: their validation images):
the test images of `ours` and `orgaseg` are all of their images (none is used for training in that direction), those of `organoid` are its official testing images.
folds/pool_all.json is for the model released for use: it trains on the training images of all three datasets and tests on their official test images.
"""
import argparse
import csv
import json
import os
import shutil
import numpy as np
from PIL import Image
from hynet import data
from hynet.evaluation import well_id
from hynet.folds import check_folds

DOMAINS = ("ours", "orgaseg", "organoid")


def to_gray(path):
    """Any image as 8-bit grayscale (channel mean, min-max stretched)."""
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    a = a.astype(np.float32)
    lo, hi = a.min(), a.max()
    return np.round(255 * (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)).astype(np.uint8)

def to_mask(path):
    """Any (instance) mask as a binary 0/255 mask."""
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    return (a > 0).astype(np.uint8) * 255

def ours_items(root):
    for split in data.SPLITS:
        image_dir, mask_dir = os.path.join(root, split, "images"), os.path.join(root, split, "masks")
        masks = {data._pair_key(f): f for f in data._list_files(mask_dir)}
        for f in data._list_files(image_dir):
            key = data._pair_key(f)
            yield dict(domain="ours", role=split, name=f"ours-{key}", image=os.path.join(image_dir, f), mask=os.path.join(mask_dir, masks[key]), convert=False)

def orgaseg_items(root):
    for split, role in (("train", "train"), ("val", "val"), ("eval", "test")):
        d = os.path.join(root, split)
        for f in sorted(os.listdir(d)):
            if f.endswith("_img.jpg"):
                stem = f[:-len("_img.jpg")]
                yield dict(domain="orgaseg", role=role, name=f"orgaseg-{stem}_Interval0.png", image=os.path.join(d, f),
                           mask=os.path.join(d, stem + "_masks_organoid.png"), convert=True)

def organoid_items(root):
    for sub, tag in (("OriginalData", ""), ("MouseOrganoids", "mouse-")):
        for split, role in (("training", "train"), ("validation", "val"), ("testing", "test")):
            d = os.path.join(root, sub, split)
            for f in sorted(os.listdir(os.path.join(d, "images"))):
                yield dict(domain="organoid", role=role, name=f"organoid-{tag}{os.path.splitext(f)[0]}_Interval0.png",
                           image=os.path.join(d, "images", f), mask=os.path.join(d, "segmentations", f), convert=True)

def lodo_folds(items):
    """{test domain: folds document}: trains on the training images of the other domains, validates on their validation images."""
    wells = {d: {r: sorted({well_id(i["name"]) for i in items if i["domain"] == d and i["role"] == r}) for r in ("train", "val", "test")} for d in DOMAINS}
    docs = {}
    for held_out in DOMAINS:
        others = [d for d in DOMAINS if d != held_out]
        test = wells[held_out]["test"] if held_out == "organoid" else sorted(w for r in wells[held_out].values() for w in r)
        fold = {"train": sorted(w for d in others for w in wells[d]["train"]), "val": sorted(w for d in others for w in wells[d]["val"]), "test": test}
        doc = {"k": 1, "seed": 0, "grouped_by": "well", "holdout": True, "folds": [fold]}
        check_folds(doc)
        docs[held_out] = doc
    return docs

def pool_folds(items):
    """Folds document of the model released for use: trains and validates on the training and validation images of all the datasets and tests on their official test images."""
    wells = {r: sorted({well_id(i["name"]) for i in items if i["role"] == r}) for r in ("train", "val", "test")}
    doc = {"k": 1, "seed": 0, "grouped_by": "well", "holdout": True, "folds": [wells]}
    check_folds(doc)
    return doc

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", default="data/DatasetOrganoidsSplitted_70_15_15")
    ap.add_argument("--orgasegment", default="data/OrganoidBasic_v20211206")
    ap.add_argument("--organoid", default="data/Dataset")
    ap.add_argument("--out_root", default="data/pool")
    ap.add_argument("--folds_dir", default="folds")
    args = ap.parse_args()

    items = [*ours_items(args.ours), *orgaseg_items(args.orgasegment), *organoid_items(args.organoid)]
    names = [i["name"] for i in items]
    assert len(set(names)) == len(names), "duplicate image names"
    image_dir, mask_dir = (os.path.join(args.out_root, "train", d) for d in ("images", "masks"))
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    for i in items:
        if i["convert"]:
            Image.fromarray(to_gray(i["image"])).save(os.path.join(image_dir, "IMG_" + i["name"]))
            Image.fromarray(to_mask(i["mask"])).save(os.path.join(mask_dir, "MASK_" + i["name"]))
        else:
            shutil.copyfile(i["image"], os.path.join(image_dir, "IMG_" + i["name"]))
            shutil.copyfile(i["mask"], os.path.join(mask_dir, "MASK_" + i["name"]))
    with open(os.path.join(args.out_root, "manifest.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "domain", "role", "source"])
        for i in items:
            writer.writerow(["IMG_" + i["name"], i["domain"], i["role"], i["image"]])

    os.makedirs(args.folds_dir, exist_ok=True)
    for held_out, doc in lodo_folds(items).items():
        with open(os.path.join(args.folds_dir, f"lodo_{held_out}.json"), "w") as f:
            json.dump(doc, f, indent=2)
        fold = doc["folds"][0]
        print(f"lodo_{held_out}: train {len(fold['train'])} wells, val {len(fold['val'])}, test {len(fold['test'])}")
    doc = pool_folds(items)
    with open(os.path.join(args.folds_dir, "pool_all.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("pool_all: " + ", ".join(f"{k} {len(v)} wells" for k, v in doc["folds"][0].items()))
    print(f"{len(items)} images written to {args.out_root}")

if __name__ == "__main__":
    main()
