"""Write a copy of the pool whose images went through the deterministic spectral harmonization (hynet.spectral); the masks are copied unchanged.

The reference spectrum is estimated only on the datasets in --reference (the training datasets of a LODO fold), so that nothing of the test dataset enters
the preprocessing. --reference none writes the control (intensity normalization only).

  python scripts/spectral_pool.py --reference ours orgaseg --out data/pool_spec_organoid     # LODO test = organoid
  python scripts/spectral_pool.py --reference none --out data/pool_robust                    # control
Then: DATA=data/pool_spec_organoid scripts/lodo.sh organoid ynet_raw   (use a different ROOT)
"""
import argparse
import os
import shutil
import numpy as np
from PIL import Image
from hynet import data, spectral

def domain_of(name):
    return name.split("_", 1)[1].split("-", 1)[0]

def load(path):
    return data.IMG_TRANSFORM(Image.open(path).convert("L"))[0].numpy().astype(np.float64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="data/pool")
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference", nargs="+", required=True, help="Datasets that define the reference spectrum, or `none`")
    parser.add_argument("--n_reference", type=int, default=100, help="Images per reference dataset")
    args = parser.parse_args()

    images = os.path.join(args.pool, "train", "images")
    masks = os.path.join(args.pool, "train", "masks")
    names = sorted(os.listdir(images))
    grid = spectral.Grid()
    reference = None
    if args.reference != ["none"]:
        rng = np.random.default_rng(0)
        by_domain = {}
        for d in args.reference:
            pool = [n for n in names if domain_of(n) == d]
            by_domain[d] = [load(os.path.join(images, pool[i])) for i in sorted(rng.choice(len(pool), min(args.n_reference, len(pool)), replace=False))]
        reference = spectral.reference_profile(by_domain, grid)
    out_images, out_masks = os.path.join(args.out, "train", "images"), os.path.join(args.out, "train", "masks")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)
    for n in names:
        y = spectral.harmonize(load(os.path.join(images, n)), reference, grid)
        Image.fromarray(np.round(255 * y).astype(np.uint8)).save(os.path.join(out_images, os.path.splitext(n)[0] + ".png"))
    for n in os.listdir(masks):
        shutil.copy(os.path.join(masks, n), os.path.join(out_masks, n))
    print(f"{len(names)} images -> {args.out}")

if __name__ == "__main__":
    main()
