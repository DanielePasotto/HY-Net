import argparse, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "src")
from hynet.models.factory import build_model
from hynet.models.hynet import GatedFusionBlock
from hynet.splits import segmentation_dataset
from hynet.evaluation import well_id

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = argparse.Namespace(folds_file="folds/lodo_organoid.json", fold=0, dataset_path="data/pool", augment=False, perturb=None, scale_aug=0.0,
                          n_train=None, n_val=None, subset_seed=0, input_shape=1, hidden_units=64, output_shape=1, norm="batch")
test = segmentation_dataset(args, "test")
train = segmentation_dataset(args, "train")
os.makedirs("outputs/analysis", exist_ok=True)

def dice_by_well(counts):
    v = [2 * tp / max(2 * tp + fp + fn, 1) for tp, fp, fn in counts.values()]
    return 100 * float(np.mean(v))

for seed, root in (("42", "lodo_Wr"), ("43", "lodo_Wr_s43"), ("44", "lodo_Wr_s44")):
    model = build_model("hynet_res", args, dev)
    model.load_state_dict(torch.load(f"outputs/{root}/organoid/runs/hynet_res/fold0/model.pth", map_location=dev, weights_only=True))
    model.eval()
    # gate recorder: GatedFusionBlock weight alpha applies to the first input = harmonized stream
    alphas = []
    def hook(mod, inp, out):
        alphas.append(mod.gate_conv(torch.cat((inp[0], inp[1] if inp[0].shape == inp[1].shape else torch.nn.functional.interpolate(inp[1], size=inp[0].shape[2:])), 1)).mean().item())
    blocks = [(n, m) for n, m in model.ynet.named_modules() if isinstance(m, GatedFusionBlock)]
    for _, m in blocks: m.register_forward_hook(hook)
    modes = {"normal": lambda x, h: model.ynet(h, x), "harm->raw (stream1=x)": lambda x, h: model.ynet(x, x), "raw->harm (stream2=harm)": lambda x, h: model.ynet(h, h)}
    counts = {k: {} for k in modes}
    per_block = []
    for i in range(len(test)):
        x, mask = test[i]; x = x[None].to(dev); gt = mask[None].to(dev) > 0.5
        with torch.no_grad():
            h = model.harmnet(x)
            for k, f in modes.items():
                alphas.clear()
                pred = torch.sigmoid(f(x, h)) > 0.5
                if k == "normal": per_block.append(list(alphas))
                tp = (pred & gt).sum().item(); fp = (pred & ~gt).sum().item(); fn = (~pred & gt).sum().item()
                c = counts[k].setdefault(well_id(test.images[i]), [0, 0, 0])
                counts[k][well_id(test.images[i])] = [c[0] + tp, c[1] + fp, c[2] + fn]
    pb = np.mean(per_block, 0)
    print(f"=== seed {seed}")
    for k in modes: print(f"  Dice per well [{k:26s}] {dice_by_well(counts[k]):.2f}")
    print("  mean gate alpha (weight on harmonized stream), order of forward calls:", [f"{a:.3f}" for a in pb])
    if seed == "42":
        # gate alpha on training images too
        al_tr = []
        for i in range(0, len(train), 5):
            x, _ = train[i]; x = x[None].to(dev)
            with torch.no_grad(): alphas.clear(); model(x); al_tr.append(list(alphas))
        print("  train alpha:", [f"{a:.3f}" for a in np.mean(al_tr, 0)])
        # figure
        rows = []
        sel = [("test", test, j) for j in np.linspace(0, len(test) - 1, 4).astype(int)] + [("train", train, j) for j in (0, len(train) // 2)]
        for _, ds, j in sel:
            x, m = ds[j]; xb = x[None].to(dev)
            with torch.no_grad(): h = model.harmnet(xb)
            xi = x[0].numpy(); hi = h[0, 0].cpu().numpy(); d = hi - xi
            lo, hi_ = np.percentile(xi, 1), np.percentile(xi, 99)
            stretch = np.clip((xi - lo) / max(hi_ - lo, 1e-6), 0, 1)
            panels = [xi, stretch, np.clip(hi, 0, 1), np.clip(d + 0.5, 0, 1), m[0].numpy()]
            rows.append(np.concatenate([(p * 255).astype(np.uint8) for p in panels], 1))
        Image.fromarray(np.concatenate(rows, 0)).resize((5 * 320, len(rows) * 240)).save("outputs/analysis/res_vis_s42.png")
        print("  figure: columns = raw | raw contrast-stretched (p1-p99) | harmonized | delta+0.5 | mask; rows = 4 test, 2 train")
print("gate blocks:", [n for n, _ in blocks])
