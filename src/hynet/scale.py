"""Random scale augmentation of an (image, mask) pair on the network grid.

The organoids of different datasets (and of different magnifications) span very different sizes: with a random zoom the model sees every organoid at several sizes.
Zoom in (factor > 1) crops a window and enlarges it; zoom out (factor < 1) shrinks the pair and fills the rest with the mirrored image and mask, so that the
borders show organoids (as real wells do) and not a flat frame.
"""
import math
import random
import torch.nn.functional as F

MAX_FACTOR = 2.0  # zooming out by more than 2 would need more mirrored border than the image has

def random_scale(image, mask, max_factor):
    """image, mask: (1, H, W) float tensors; the scale factor is log-uniform in [1/max_factor, max_factor]. Returns the transformed pair, same shapes."""
    if not 1 < max_factor <= MAX_FACTOR:
        raise ValueError(f"max_factor must be in (1, {MAX_FACTOR}]")
    factor = math.exp(random.uniform(-math.log(max_factor), math.log(max_factor)))
    if abs(factor - 1) < 0.05:
        return image, mask
    H, W = image.shape[-2:]
    if factor > 1:
        h, w = round(H / factor), round(W / factor)
        top, left = random.randint(0, H - h), random.randint(0, W - w)
        image, mask = image[:, top:top + h, left:left + w], mask[:, top:top + h, left:left + w]
        return (F.interpolate(image[None], size=(H, W), mode="bilinear", align_corners=False)[0],
                F.interpolate(mask[None], size=(H, W), mode="nearest")[0])
    h, w = round(H * factor), round(W * factor)
    image = F.interpolate(image[None], size=(h, w), mode="bilinear", align_corners=False, antialias=True)
    mask = F.interpolate(mask[None], size=(h, w), mode="nearest")
    top, left = _offset(H - h, h), _offset(W - w, w)
    pad = (left, W - w - left, top, H - h - top)
    return F.pad(image, pad, mode="reflect")[0], F.pad(mask, pad, mode="reflect")[0]

def _offset(total, size):
    """Random amount of padding before the content, such that neither side is larger than reflect padding allows (< size)."""
    return random.randint(max(0, total - (size - 1)), min(total, size - 1))
