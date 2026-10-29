"""What HarmNet does to images of our dataset and of the external ones.

For a few images of each dataset it saves a panel input | HarmNet pre-trained on our data (outputs/runs/harmnet/fold<k>/harmnet.pth) |
HarmNet after the end-to-end training inside HY-Net (outputs/runs/hynet/fold<k>/model.pth), and writes statistics of the change
(mean absolute difference from the input, mean and standard deviation of the grey levels) to a CSV.
"""
import argparse
import csv
import os
import numpy as np
import torch
from PIL import Image
from hynet import data
from hynet.models.harmnet import HarmNet

SETS = {
    "ours": "data/DatasetOrganoidsSplitted_70_15_15/test/images",
    "organosegment": "data/external/organosegment/test/images",
    "organoid": "data/external/organoid/test/images",
    "organoid_lb": "data/external/organoid_lb/test/images",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--runs_root", default="outputs/runs")
    ap.add_argument("--n", type=int, default=4, help="images per dataset in the panel")
    ap.add_argument("--n_stats", type=int, default=100, help="images per dataset for the statistics")
    ap.add_argument("--out_dir", default="outputs/eval_ext/harmonization")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pre = HarmNet(1, 64, 1).to(device).eval()
    pre.load_state_dict(torch.load(os.path.join(args.runs_root, "harmnet", f"fold{args.fold}", "harmnet.pth"), map_location=device, weights_only=True))
    tuned = HarmNet(1, 64, 1).to(device).eval()
    state = torch.load(os.path.join(args.runs_root, "hynet", f"fold{args.fold}", "model.pth"), map_location=device, weights_only=True)
    tuned.load_state_dict({k[len("harmnet."):]: v for k, v in state.items() if k.startswith("harmnet.")})

    os.makedirs(args.out_dir, exist_ok=True)
    to_u8 = lambda t: (t.squeeze().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    rows, panel = [], []
    for name, folder in SETS.items():
        if not os.path.isdir(folder):
            continue
        files = sorted(os.listdir(folder))
        step = max(1, len(files) // args.n)
        shown = set(files[::step][:args.n])
        for f in files[:args.n_stats] if len(files) > args.n_stats else files:
            x = data.IMG_TRANSFORM(data._load_normalized(os.path.join(folder, f))).unsqueeze(0).to(device)
            with torch.no_grad():
                outs = [pre(x), tuned(x)]
            for tag, y in zip(("pretrained", "finetuned"), outs):
                rows.append({"set": name, "image": f, "harmnet": tag, "mean_abs_diff": float((y - x).abs().mean()),
                             "in_mean": float(x.mean()), "in_std": float(x.std()), "out_mean": float(y.mean()), "out_std": float(y.std())})
            if f in shown:
                panel.append(np.concatenate([to_u8(x), to_u8(outs[0]), to_u8(outs[1])], axis=1))
        if panel:
            Image.fromarray(np.concatenate(panel, axis=0)).save(os.path.join(args.out_dir, f"panel_{name}.png"))
            panel = []
    with open(os.path.join(args.out_dir, "stats.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for name in SETS:
        for tag in ("pretrained", "finetuned"):
            sel = [r for r in rows if r["set"] == name and r["harmnet"] == tag]
            if sel:
                print(f"{name:<14}{tag:<12} n={len(sel):<4} mean|out-in| {np.mean([r['mean_abs_diff'] for r in sel]):.3f}   "
                      f"in mean/std {np.mean([r['in_mean'] for r in sel]):.2f}/{np.mean([r['in_std'] for r in sel]):.2f}   "
                      f"out mean/std {np.mean([r['out_mean'] for r in sel]):.2f}/{np.mean([r['out_std'] for r in sel]):.2f}")

if __name__ == "__main__":
    main()
