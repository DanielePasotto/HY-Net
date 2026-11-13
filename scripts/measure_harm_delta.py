import argparse, sys
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from hynet.models.factory import build_model
from hynet.splits import segmentation_dataset

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base = dict(folds_file="folds/lodo_organoid.json", fold=0, dataset_path="data/pool", augment=False, perturb=None, scale_aug=0.0,
            n_train=None, n_val=None, subset_seed=0, input_shape=1, hidden_units=64, output_shape=1, norm="batch")
args = argparse.Namespace(**base)
roles = {"test(organoid)": segmentation_dataset(args, "test"), "train(ours+orgaseg)": segmentation_dataset(args, "train")}

def r2_affine(x, d):
    A = np.stack([x, np.ones_like(x)], 1); coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    res = d - A @ coef
    return 1 - res.var() / max(d.var(), 1e-12), coef

for seed, root in (("42", "lodo_res"), ("43", "lodo_res_s43"), ("44", "lodo_res_s44")):
    model = build_model("hynet_res", args, dev)
    model.load_state_dict(torch.load(f"outputs/{root}/organoid/runs/hynet_res/fold0/model.pth", map_location=dev, weights_only=True))
    model.eval()
    print(f"=== seed {seed}  harmnet.scale={model.harmnet.scale}")
    rng = np.random.RandomState(0)
    for name, ds in roles.items():
        idx = np.arange(len(ds))
        if len(idx) > 60: idx = np.sort(rng.choice(idx, 60, replace=False))
        out = {k: [] for k in ("l1", "rms", "maxabs", "rel_rms", "r2_affine", "corr_hp", "d_fg", "d_bg", "frac_gt_.02", "mean_signed")}
        for i in idx:
            img, mask = ds[i]
            x = img[None].to(dev)
            with torch.no_grad(): h = model.harmnet(x)
            d = (h - x)[0, 0].cpu().numpy(); xx = x[0, 0].cpu().numpy(); m = mask[0].numpy() > 0.5
            hp = xx - F.avg_pool2d(x, 5, 1, 2)[0, 0].cpu().numpy()
            out["l1"].append(np.abs(d).mean()); out["rms"].append(np.sqrt((d ** 2).mean())); out["maxabs"].append(np.abs(d).max())
            out["rel_rms"].append(np.sqrt((d ** 2).mean()) / max(xx.std(), 1e-9))
            out["r2_affine"].append(r2_affine(xx.ravel(), d.ravel())[0])
            out["corr_hp"].append(np.corrcoef(d.ravel(), hp.ravel())[0, 1])
            out["d_fg"].append(np.abs(d[m]).mean() if m.any() else np.nan); out["d_bg"].append(np.abs(d[~m]).mean())
            out["frac_gt_.02"].append((np.abs(d) > 0.02).mean()); out["mean_signed"].append(d.mean())
        print(f"  {name:22s} n={len(idx)} " + "  ".join(f"{k}={np.nanmean(v):.4f}" for k, v in out.items()))
