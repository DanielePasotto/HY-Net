"""Build the processed dataset used to train HarmNet from a segmentation dataset (<split>/images and <split>/masks).

For every split it writes, with the layout read by HarmonizationDataset:
  true_images/  the raw images (HarmNet input), copied unchanged
  images/       the harmonization targets: output of the classical pipeline (hynet.targets)
  masks/        the binary masks, copied unchanged
The targets are computed at --pipeline_size (default 1280x960, the size of our images), so for an image of another size the targets have that size
and not the size of the raw image; the training code resizes both to the network input.
"""
import argparse
import os
import shutil
import cv2
from hynet import data
from hynet.targets import PIPELINE_SIZE, make_target

def build_processed_dataset(input_root, output_root, splits=None, size=PIPELINE_SIZE):
    """Returns {split: number of images}."""
    splits = splits or [s for s in data.SPLITS if os.path.isdir(os.path.join(input_root, s, "images"))]
    counts = {}
    for split in splits:
        image_dir, mask_dir = (os.path.join(input_root, split, d) for d in ("images", "masks"))
        masks = {data._pair_key(f): f for f in data._list_files(mask_dir)}
        out = {name: os.path.join(output_root, split, name) for name in ("true_images", "images", "masks")}
        for path in out.values():
            os.makedirs(path, exist_ok=True)
        files = data._list_files(image_dir)
        for f in files:
            if data._pair_key(f) not in masks:
                raise FileNotFoundError(f"No mask for {os.path.join(image_dir, f)} in {mask_dir}")
            image = cv2.imread(os.path.join(image_dir, f), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Cannot read {os.path.join(image_dir, f)}")
            shutil.copyfile(os.path.join(image_dir, f), os.path.join(out["true_images"], f))
            shutil.copyfile(os.path.join(mask_dir, masks[data._pair_key(f)]), os.path.join(out["masks"], masks[data._pair_key(f)]))
            if not cv2.imwrite(os.path.join(out["images"], f), make_target(image, size)):
                raise OSError(f"Cannot write the target of {f}")
        counts[split] = len(files)
    return counts

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_root", default="data/DatasetOrganoidsSplitted_70_15_15", help="segmentation dataset with <split>/images and <split>/masks")
    ap.add_argument("--output_root", default="data/ProcessedDatasetOrganoidsSplitted_70_15_15")
    ap.add_argument("--splits", nargs="+", default=None, help="splits to process (default: all those present in --input_root)")
    ap.add_argument("--pipeline_size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=list(PIPELINE_SIZE),
                    help="size the images are brought to before the pipeline (the filter sizes are in pixels)")
    ap.add_argument("--native", action="store_true", help="apply the pipeline at the original size of each image (no resize)")
    args = ap.parse_args()
    counts = build_processed_dataset(args.input_root, args.output_root, args.splits, None if args.native else tuple(args.pipeline_size))
    for split, n in counts.items():
        print(f"{split}: {n} images -> {os.path.join(args.output_root, split)}")

if __name__ == "__main__":
    main()
