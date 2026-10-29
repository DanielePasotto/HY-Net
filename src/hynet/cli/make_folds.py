import csv
import numpy as np
from hynet import data
from hynet.config import get_args
from hynet.evaluation import well_id
from hynet.folds import make_folds, save_folds, load_folds

def add_arguments(parser):
    group = parser.add_argument_group("folds")
    group.add_argument("--folds_k", type=int, default=5, help="Number of folds")
    group.add_argument("--folds_out", type=str, default="folds/k5_seed42.json", help="Where the folds are written")
    group.add_argument("--groups_csv", type=str, default=None,
                       help="Optional CSV with columns 'well,group' (e.g. donor or plate): the wells of a group are never split across folds")

def main():
    args = get_args(add_arguments)
    pool = data.pool_segmentation_samples(args.dataset_path)

    fractions = {}
    for key, (_, mask_path) in pool.items():
        fractions.setdefault(well_id(key), []).append(float(data._load_binary_mask(mask_path).mean()))
    well_fraction = {w: float(np.mean(v)) for w, v in fractions.items()}
    n_images = {w: len(v) for w, v in fractions.items()}

    groups = None
    if args.groups_csv:
        with open(args.groups_csv, newline="") as f:
            groups = {row["well"]: row["group"] for row in csv.DictReader(f)}
        missing = set(well_fraction) - set(groups)
        if missing:
            raise ValueError(f"Wells without a group in {args.groups_csv}: {sorted(missing)}")

    doc = make_folds(well_fraction, k=args.folds_k, seed=args.seed, groups=groups)
    save_folds(doc, args.folds_out)
    load_folds(args.folds_out)  # re-validates what was written

    print(f"{len(well_fraction)} wells, {sum(n_images.values())} images -> {doc['k']} folds grouped by {doc['grouped_by']} (seed {args.seed})")
    print(f"{'fold':>4} {'train':>6} {'val':>5} {'test':>5} {'test imgs':>10} {'test density':>13}")
    for i, f in enumerate(doc["folds"]):
        density = np.mean([well_fraction[w] for w in f["test"]])
        print(f"{i:>4} {len(f['train']):>6} {len(f['val']):>5} {len(f['test']):>5} {sum(n_images[w] for w in f['test']):>10} {100 * density:>12.1f}%")
    print(f"Folds written to {args.folds_out}")

if __name__ == "__main__":
    main()
