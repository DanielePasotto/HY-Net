"""Measured analysis of a trained HY-Net / Y-Net: cost, behaviour of the gates and of the attention, and how much the prediction relies on each part.

Same data/model arguments as hynet-evaluate (--model, --folds_file/--fold/--dataset_path/--runs_root or --weights_path); results are written to
<out_dir>/<model>/<fold>/analysis.json and printed.

  cost          parameters, FLOPs, latency and peak memory per component
  harmonization how far the HarmNet output is from its input
  gates         statistics of the fusion weight alpha (weight of the harmonized stream, 1 - alpha = raw stream) at the bottleneck and the four skip levels:
                mean, spread inside an image, spread between images, saturation, mean on organoid and on background pixels
  attention     the same for the attention map applied to the raw skips
  interventions test-set Dice when one part is switched off at inference: alpha forced to 1 (harmonized only), 0 (raw only) or 0.5, at all levels or one level,
                HarmNet bypassed (the raw image in both streams), attention forced to 1
"""
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from hynet import data, evaluation
from hynet.config import get_args
from hynet.engine import seed_everything
from hynet.models.factory import build_model
from hynet.models.hynet import HYNet
from hynet.splits import segmentation_dataset, weights_paths, run_tag

def add_arguments(parser):
    group = parser.add_argument_group("analysis")
    group.add_argument("--split", type=str, default="test")
    group.add_argument("--weights_path", type=str, default=None)
    group.add_argument("--out_dir", type=str, default="outputs/analysis")
    group.add_argument("--max_images", type=int, default=None, help="Analyse only the first N test images (quick checks)")

def levels(ynet):
    """(name, gate module, attention module or None, resolution relative to the input) from the bottleneck to the full resolution."""
    out = [("bottleneck 1/16", ynet.bottleneck_fusion.gate_conv, None)]
    for i, (name, res) in enumerate(zip(("skip 1/8", "skip 1/4", "skip 1/2", "skip 1/1"), (8, 4, 2, 1))):
        out.append((name, ynet.fusion_blocks[i].gate_conv, ynet.attentions[i].psi if ynet.use_attention else None))
    return out

class Recorder:
    """Forward hooks that record the output of a module (and optionally replace it)."""
    def __init__(self, module):
        self.value = None
        self.override = None
        self.handle = module.register_forward_hook(self.hook)
    def hook(self, module, inputs, output):
        self.value = output.detach()
        if self.override is not None:
            return self.override(output)
    def close(self):
        self.handle.remove()

def dice_over_wells(model, dataset, masks, device, max_images, per_image=None):
    preds = []
    n = len(dataset) if max_images is None else min(max_images, len(dataset))
    with torch.no_grad():
        for i in range(n):
            image, _ = dataset[i]
            preds.append((torch.sigmoid(model(image.unsqueeze(0).to(device))) > 0.5).squeeze().cpu().numpy())
            if per_image is not None:
                per_image(i)
    scores = evaluation.well_dice(preds, masks[:n], dataset.images[:n])
    return float(np.mean([d for _, d in scores.values()]))

class Accumulator:
    def __init__(self):
        self.means, self.within, self.sat_lo, self.sat_hi, self.on, self.off = [], [], [], [], [], []
    def add(self, a, mask):
        """a: (1, 1, h, w) tensor in [0, 1]; mask: (H, W) bool organoid mask."""
        a = a[0, 0].float().cpu()
        m = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None], size=a.shape, mode="area")[0, 0] > 0.5
        self.means.append(a.mean().item())
        self.within.append(a.std().item())
        self.sat_lo.append((a < 0.1).float().mean().item())
        self.sat_hi.append((a > 0.9).float().mean().item())
        if m.any():
            self.on.append(a[m].mean().item())
        if (~m).any():
            self.off.append(a[~m].mean().item())
    def summary(self):
        mean = lambda v: float(np.mean(v)) if v else float("nan")
        return {"mean": mean(self.means), "std_within_image": mean(self.within), "std_between_images": float(np.std(self.means)),
                "fraction_below_0.1": mean(self.sat_lo), "fraction_above_0.9": mean(self.sat_hi),
                "mean_on_organoid": mean(self.on), "mean_on_background": mean(self.off)}

def cost(model, device, shape=(1, 1, 480, 640)):
    components = {}
    if model.harmonize:
        components["harmnet"] = model.harmnet
    for name in ("encoder1_layers", "encoder2_layers", "bottleneck_fusion", "bottleneck", "ups", "attentions", "fusion_blocks", "dec_convs", "final_conv"):
        module = getattr(model.ynet, name, None)
        if module is not None and sum(p.numel() for p in module.parameters()):
            components[name] = module
    params = {k: sum(p.numel() for p in m.parameters()) for k, m in components.items()}
    x = torch.rand(*shape, device=device)
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False, depth=4) as fc:
        model(x)
    counts = fc.get_flop_counts()
    flops = {k: sum(v.values()) for k, v in counts.items()}
    result = {"parameters": params, "parameters_total": sum(p.numel() for p in model.parameters()),
              "gflops_total": flops.get("Global", 0) / 1e9, "gflops_by_module": {k: v / 1e9 for k, v in flops.items() if k.count(".") <= 2}}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(5):
                model(x)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(30):
                model(x)
            torch.cuda.synchronize()
        result["latency_ms"] = (time.time() - t) / 30 * 1000
        result["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return result

def main():
    args = get_args(add_arguments)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    _, weights_path = weights_paths(args)
    weights_path = args.weights_path or weights_path
    model = build_model(args.model, args, device)
    if not isinstance(model, HYNet):
        raise SystemExit(f"{args.model}: the analysis needs a Y-Net based model (hynet, ynet_raw, ...)")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    dataset = segmentation_dataset(args, "test" if args.folds_file else args.split)
    masks = [data.load_mask_480x640(p) for p in dataset.mask_paths]
    tag = run_tag(args, args.split)
    out_dir = os.path.join(args.out_dir, args.model, tag)
    os.makedirs(out_dir, exist_ok=True)
    report = {"model": args.model, "weights": weights_path, "images": len(dataset) if args.max_images is None else args.max_images}

    report["cost"] = cost(model, device)

    ynet = model.ynet
    lv = levels(ynet)
    gates = [Recorder(g) for _, g, _ in lv]
    psis = [Recorder(p) if p is not None else None for _, _, p in lv]
    harm = Recorder(model.harmnet) if model.harmonize else None
    gate_acc = [Accumulator() for _ in lv]
    psi_acc = [Accumulator() for _ in lv]
    harm_stats = {"mean_abs_difference": [], "correlation": [], "std_ratio": []}
    state = {"x": None}

    def collect(i):
        for k in range(len(lv)):
            gate_acc[k].add(gates[k].value, masks[i])
            if psis[k] is not None:
                psi_acc[k].add(psis[k].value, masks[i])
        if harm is not None:
            y, x = harm.value.float().flatten(), state["x"].float().flatten()
            harm_stats["mean_abs_difference"].append((y - x).abs().mean().item())
            harm_stats["correlation"].append(torch.corrcoef(torch.stack([y, x]))[0, 1].item())
            harm_stats["std_ratio"].append((y.std() / x.std()).item())

    # the input of HarmNet is the image fed to the model: remember it for collect()
    model.register_forward_pre_hook(lambda m, inputs: state.__setitem__("x", inputs[0].detach()))
    baseline = dice_over_wells(model, dataset, masks, device, args.max_images, collect)
    report["gates"] = {name: acc.summary() for (name, _, _), acc in zip(lv, gate_acc)}
    report["attention"] = {name: acc.summary() for (name, _, p), acc in zip(lv, psi_acc) if p is not None}
    if harm is not None:
        report["harmonization"] = {k: float(np.mean(v)) for k, v in harm_stats.items()}

    interventions = {"baseline": baseline}
    def run(name, gate_levels=None, alpha=None, bypass_harmnet=False, attention_one=False):
        for k, rec in enumerate(gates):
            rec.override = (lambda o, a=alpha: torch.full_like(o, a)) if alpha is not None and (gate_levels is None or k in gate_levels) else None
        for rec in psis:
            if rec is not None:
                rec.override = (lambda o: torch.ones_like(o)) if attention_one else None
        if harm is not None:
            harm.override = (lambda o: state["x"]) if bypass_harmnet else None
        interventions[name] = dice_over_wells(model, dataset, masks, device, args.max_images)
    for alpha, label in ((1.0, "only the harmonized stream (alpha=1)"), (0.0, "only the raw stream (alpha=0)"), (0.5, "equal mix (alpha=0.5)")):
        run(f"all gates, {label}", alpha=alpha)
    run("bottleneck gate only, alpha=0.5", gate_levels={0}, alpha=0.5)
    run("skip gates only, alpha=0.5", gate_levels={1, 2, 3, 4}, alpha=0.5)
    for k, (name, _, _) in enumerate(lv):
        run(f"{name} gate only, alpha=0.5", gate_levels={k}, alpha=0.5)
    if ynet.use_attention:
        run("attention forced to 1 (raw skips unfiltered)", attention_one=True)
    if harm is not None:
        run("HarmNet bypassed (raw image in both streams)", bypass_harmnet=True)
    report["interventions_mean_well_dice"] = interventions

    with open(os.path.join(out_dir, "analysis.json"), "w") as f:
        json.dump(report, f, indent=2)
    c = report["cost"]
    print(f"{args.model} {tag}: {c['parameters_total'] / 1e6:.2f} M parameters, {c['gflops_total']:.1f} GFLOPs" + (f", {c['latency_ms']:.0f} ms, {c['peak_memory_mb']:.0f} MB" if "latency_ms" in c else ""))
    for k, v in report["gates"].items():
        print(f"gate {k}: " + ", ".join(f"{a}={b:.3f}" for a, b in v.items()))
    for k, v in interventions.items():
        print(f"Dice {k}: {v * 100:.2f}")

if __name__ == "__main__":
    main()
