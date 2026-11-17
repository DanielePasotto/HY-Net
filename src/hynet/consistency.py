"""Unsupervised consistency training: two photometric views of the same (unlabeled) image must give the same prediction and,
for the models with a HarmNet, the same harmonized image. The images can be the training images (their masks are not used here)
and, optionally, the unlabeled images of the test domain (transductive use: the masks of the test domain are never read).
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from hynet import data, perturb

class PairDataset(Dataset):
    """(view1, view2): two independent random photometric augmentations of the same image, same geometry."""
    def __init__(self, paths, ranges="default"):
        self.image_paths = list(paths)
        self.ranges = ranges

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = data._load_normalized(self.image_paths[idx])
        return tuple(data.IMG_TRANSFORM(perturb.random_augment(image, ranges=self.ranges)) for _ in range(2))

def _forever(loader):
    while True:
        yield from loader

class Consistency:
    """Draws pairs from `loader` (batch of one pair) and computes the weighted consistency loss of a model."""
    def __init__(self, loader, w_seg, w_harm, pairs=2, bn_eval=False):
        self.batches = _forever(loader)
        self.w_seg, self.w_harm, self.pairs, self.bn_eval = w_seg, w_harm, pairs, bn_eval

    def _forward(self, model, x):
        forward = getattr(model, "forward_proc", None)
        return forward(x) if forward else (model(x), None)

    def loss(self, model, device):
        v1, v2 = next(self.batches)
        x = torch.cat((v1, v2)).to(device)  # both views in one batch
        if self.w_seg == 0 and hasattr(model, "harmnet"):
            # only the harmonizer sees the views: the Y-Net is not run, so its BatchNorm statistics never include the perturbed images
            out, proc = None, model.harmnet(x)
        elif self.bn_eval:  # BatchNorm uses its running statistics and does not update them with the perturbed views
            bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training]
            for m in bns:
                m.eval()
            try:
                out, proc = self._forward(model, x)
            finally:
                for m in bns:
                    m.train()
        else:
            out, proc = self._forward(model, x)
        total = x.new_zeros(())
        if self.w_seg > 0:
            prob = torch.sigmoid(out.float())
            total = total + self.w_seg * F.mse_loss(prob[:len(v1)], prob[len(v1):])
        if proc is not None and self.w_harm > 0:
            proc = proc.float()
            total = total + self.w_harm * F.mse_loss(proc[:len(v1)], proc[len(v1):])
        return total

def make_consistency(args, source_paths, target_paths=()):
    """Consistency pairs from the training images and, if `target_paths` is not empty, half of them from the (unlabeled) target images."""
    paths = list(source_paths) + list(target_paths)
    share = 0.5 if target_paths else 0.0
    weights = [(1 - share) / len(source_paths)] * len(source_paths) + [share / max(len(target_paths), 1)] * len(target_paths)
    sampler = WeightedRandomSampler(weights, num_samples=100000, replacement=True)
    loader = DataLoader(PairDataset(paths, args.augment_range), batch_size=1, sampler=sampler)
    return Consistency(loader, args.consist, args.consist_harm, args.consist_pairs, args.consist_bn_eval)
