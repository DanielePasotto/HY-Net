"""Domain-adversarial training of HarmNet: next to the usual reconstruction loss, a discriminator tries to guess the dataset (microscope) of an image
from the HarmNet output and HarmNet is trained to make that guess impossible (cross-entropy to the uniform distribution), so the output no longer
carries the dataset of origin."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from hynet import data

class DomainLabelled(Dataset):
    """Wraps a pooled dataset and appends the index of the sample's domain (the IMG_<domain>-<id> prefix) to every item."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.image_paths = dataset.image_paths
        self.domains = sorted({data.domain_of(p) for p in self.image_paths})
        self.labels = [self.domains.index(data.domain_of(p)) for p in self.image_paths]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (*self.dataset[idx], self.labels[idx])

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

def training_adversarial(model, disc, loader, criterion, optimizer, optimizer_disc, device, lambda_adv, amp=False):
    """One epoch. Returns (reconstruction loss, adversarial loss, discriminator accuracy); the accuracy of a fooled discriminator is 1 / number of domains."""
    model.train()
    disc.train()
    totals = np.zeros(3)
    for inputs, masks, targets, domains in tqdm(loader, desc="Training"):
        inputs, masks, targets, domains = (t.to(device) for t in (inputs, masks, targets, domains))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            outputs = model(inputs)
            logits_detached = disc(outputs.detach())
        outputs = outputs.float()
        disc_loss = F.cross_entropy(logits_detached.float(), domains)
        optimizer_disc.zero_grad()
        disc_loss.backward()
        optimizer_disc.step()

        reconstruction, _ = criterion(outputs, targets, masks, inputs)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = disc(outputs)
        adversarial = -F.log_softmax(logits.float(), dim=1).mean()
        optimizer.zero_grad()
        (reconstruction + lambda_adv * adversarial).backward()
        optimizer.step()
        totals += [reconstruction.item(), adversarial.item(), (logits_detached.argmax(1) == domains).float().mean().item()]
    return tuple(totals / len(loader))

def image_statistics(x):
    """Differentiable per-image statistics of a (B, 1, H, W) batch: five intensity quantiles, mean, std, mean gradient, Laplacian std."""
    flat = x.flatten(1)
    q = torch.quantile(flat, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=x.device), dim=1).T
    grad = (x[..., :, 1:] - x[..., :, :-1]).abs().mean((1, 2, 3)) + (x[..., 1:, :] - x[..., :-1, :]).abs().mean((1, 2, 3))
    lap = (4 * x[..., 1:-1, 1:-1] - x[..., :-2, 1:-1] - x[..., 2:, 1:-1] - x[..., 1:-1, :-2] - x[..., 1:-1, 2:]).flatten(1).std(1)
    return torch.cat([q, flat.mean(1, keepdim=True), flat.std(1, keepdim=True), grad[:, None], lap[:, None]], dim=1)

def moment_penalty(stats, domains):
    """Mean squared relative difference of the per-domain average statistics, over the pairs of domains present in the batch (0 if only one)."""
    scale = stats.detach().abs().mean(0) + 1e-3
    means = [stats[domains == d].mean(0) / scale for d in domains.unique()]
    pairs = [((a - b) ** 2).mean() for i, a in enumerate(means) for b in means[i + 1:]]
    return torch.stack(pairs).mean() if pairs else stats.sum() * 0

def training_moments(model, loader, criterion, optimizer, device, weight, amp=False):
    """One epoch of HarmNet with, next to the reconstruction loss, a penalty on the differences between the datasets in the statistics of the output
    (no discriminator: stable, but it only removes the differences that are measured). Returns (reconstruction loss, penalty)."""
    model.train()
    totals = np.zeros(2)
    for inputs, masks, targets, domains in tqdm(loader, desc="Training"):
        inputs, masks, targets, domains = (t.to(device) for t in (inputs, masks, targets, domains))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            outputs = model(inputs)
        outputs = outputs.float()
        reconstruction, _ = criterion(outputs, targets, masks, inputs)
        penalty = moment_penalty(image_statistics(outputs), domains)
        optimizer.zero_grad()
        (reconstruction + weight * penalty).backward()
        optimizer.step()
        totals += [reconstruction.item(), penalty.item()]
    return tuple(totals / len(loader))
