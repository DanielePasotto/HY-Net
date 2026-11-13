"""Does HarmNet reduce the differences between datasets (microscopes)? Measured on images of the three datasets of the pool.

For the raw images, the output of the classical pipeline (the HarmNet training target) and the output of each HarmNet, per-image features
(32-bin intensity histogram, mean, std, mean gradient, Laplacian std) are computed after the same per-image min-max normalization, then:
  domain accuracy   cross-validated accuracy of a linear classifier that guesses the dataset from the features (chance = 1/3): lower = more harmonized
  intensity W1      Wasserstein-1 distance between the intensity histograms of two datasets, averaged over the three pairs: lower = more harmonized
  F(feature)        spread of a feature between datasets divided by its spread inside datasets: lower = more harmonized

  python scripts/analyze_harmonization.py --harmnets outputs/lodo/ours/runs/harmnet/fold0/harmnet.pth ... [--n 60]
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from hynet import data
from hynet.models.harmnet import HarmNet

DOMAINS = ("ours", "orgaseg", "organoid")

class DomainDiscriminator(nn.Module):
    """Small CNN with global mean/max pooling: it sees intensity statistics and texture, not the image layout."""
    def __init__(self, n_domains, width=32):
        super().__init__()
        c = [1, width, width * 2, width * 4, width * 4]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(c[i], c[i + 1], 5, stride=2, padding=2), nn.LeakyReLU(0.2, inplace=True)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(c[-1] * 2, n_domains)

    def forward(self, x):
        f = self.features(x)
        return self.head(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))

def domain_of(name):
    return name.split("_", 1)[1].split("-", 1)[0]

def normalize(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else x - lo

def features(x):
    hist = np.histogram(x, bins=32, range=(0, 1))[0] / x.size
    lap = 4 * x[1:-1, 1:-1] - x[:-2, 1:-1] - x[2:, 1:-1] - x[1:-1, :-2] - x[1:-1, 2:]
    return np.concatenate([hist, [x.mean(), x.std(), np.abs(np.diff(x, axis=1)).mean(), np.abs(np.diff(x, axis=0)).mean(), lap.std()]])

def domain_accuracy(feats, labels, folds=5, repeats=5, ridge=1.0):
    accs = []
    for rep in range(repeats):
        rng = np.random.default_rng(rep)
        idx = {d: rng.permutation(np.flatnonzero(labels == d)) for d in DOMAINS}
        correct = 0
        for k in range(folds):
            test = np.concatenate([idx[d][k::folds] for d in DOMAINS])
            train = np.setdiff1d(np.arange(len(labels)), test)
            mu, sd = feats[train].mean(0), feats[train].std(0) + 1e-8
            Xtr, Xte = (feats[train] - mu) / sd, (feats[test] - mu) / sd
            Y = np.stack([(labels[train] == d).astype(float) for d in DOMAINS], 1)
            W = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1]), Xtr.T @ Y)
            correct += (np.array(DOMAINS)[np.argmax(Xte @ W, 1)] == labels[test]).sum()
        accs.append(correct / len(labels))
    return float(np.mean(accs)), float(np.std(accs))

def intensity_w1(feats, labels):
    mean_hist = {d: feats[labels == d, :32].mean(0) for d in DOMAINS}
    cdf = {d: np.cumsum(h) for d, h in mean_hist.items()}
    pairs = [("ours", "orgaseg"), ("ours", "organoid"), ("orgaseg", "organoid")]
    return float(np.mean([np.abs(cdf[a] - cdf[b]).sum() / 32 for a, b in pairs]))

def f_ratio(values, labels):
    means = np.array([values[labels == d].mean() for d in DOMAINS])
    within = np.mean([values[labels == d].std() for d in DOMAINS])
    return float(means.std() / (within + 1e-12))

def probe_accuracy(images, labels, device, epochs=40, repeats=3):
    """Accuracy on held-out images (30%) of a CNN trained from scratch to guess the dataset; mean and std over `repeats` random splits."""
    import torch.nn.functional as F
    X = torch.tensor(np.stack(images))[:, None].float()
    y = torch.tensor([DOMAINS.index(l) for l in labels])
    accs = []
    for rep in range(repeats):
        g = torch.Generator().manual_seed(rep)
        train, test = [], []
        for k in range(len(DOMAINS)):
            idx = torch.nonzero(y == k).flatten()
            idx = idx[torch.randperm(len(idx), generator=g)]
            cut = int(0.7 * len(idx))
            train += idx[:cut].tolist()
            test += idx[cut:].tolist()
        train = torch.tensor(train)
        net = DomainDiscriminator(len(DOMAINS), 16).to(device)
        opt = torch.optim.Adam(net.parameters(), 3e-4)
        for _ in range(epochs):
            for batch in train[torch.randperm(len(train), generator=g)].split(16):
                loss = F.cross_entropy(net(X[batch].to(device)), y[batch].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(X[test[i:i + 32]].to(device)).argmax(1).cpu() for i in range(0, len(test), 32)])
        accs.append((pred == y[test]).float().mean().item())
    return float(np.mean(accs)), float(np.std(accs))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="data/pool/train/images")
    parser.add_argument("--processed", default="data/pool_processed/train/images", help="Classical-pipeline outputs (the HarmNet targets), same file names")
    parser.add_argument("--harmnets", nargs="*", default=[], help="HarmNet checkpoints to apply")
    parser.add_argument("--n", type=int, default=60, help="Images per dataset")
    parser.add_argument("--probe", action="store_true", help="Also train a fresh CNN (30%% of the images held out) to guess the dataset: the independent check of the harmonization")
    parser.add_argument("--out", default="outputs/analysis/harmonization.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(0)
    names = sorted(os.listdir(args.pool))
    chosen = []
    for d in DOMAINS:
        pool = [n for n in names if domain_of(n) == d]
        chosen += [pool[i] for i in sorted(rng.choice(len(pool), min(args.n, len(pool)), replace=False))]
    labels = np.array([domain_of(n) for n in chosen])

    def load(directory):
        return [data.IMG_TRANSFORM(data._load_normalized(os.path.join(directory, n))) for n in chosen]
    raw = load(args.pool)
    versions = {"raw": [x[0].numpy() for x in raw]}
    if os.path.isdir(args.processed):
        versions["classical pipeline (HarmNet target)"] = [x[0].numpy() for x in load(args.processed)]
    for path in args.harmnets:
        net = HarmNet(1, 64, 1).to(device)
        net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        net.eval()
        with torch.no_grad():
            versions["HarmNet " + ("/".join(path.split("/runs/")[0].split("/")[-2:]) if "/runs/" in path else os.path.basename(path))] = [net(x.unsqueeze(0).to(device))[0, 0].cpu().numpy() for x in raw]

    report = {}
    print(f"{'version':50s} {'domain accuracy':>16s} {'CNN probe':>10s} {'intensity W1':>13s}   F(mean) F(std) F(gradient) F(Laplacian)")
    for name, images in versions.items():
        feats = np.array([features(normalize(x)) for x in images])
        acc, sd = domain_accuracy(feats, labels)
        row = {"domain_accuracy": acc, "domain_accuracy_sd": sd, "intensity_w1": intensity_w1(feats, labels),
               "F_mean": f_ratio(feats[:, 32], labels), "F_std": f_ratio(feats[:, 33], labels),
               "F_gradient": f_ratio(feats[:, 34] + feats[:, 35], labels), "F_laplacian": f_ratio(feats[:, 36], labels)}
        if args.probe:
            row["cnn_probe_accuracy"], row["cnn_probe_sd"] = probe_accuracy(images, labels, device)
        report[name] = row
        probe = f"{row['cnn_probe_accuracy'] * 100:9.1f}%" if args.probe else f"{'-':>10s}"
        print(f"{name:50s} {acc * 100:10.1f} ±{sd * 100:3.1f}% {probe} {row['intensity_w1']:13.4f}   {row['F_mean']:7.2f} {row['F_std']:6.2f} {row['F_gradient']:11.2f} {row['F_laplacian']:12.2f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"images_per_domain": args.n, "versions": report}, f, indent=2)

if __name__ == "__main__":
    main()
