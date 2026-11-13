"""Photometric perturbations of the input image: a fixed sweep to test the robustness of a trained model, and a random version used as training augmentation.

Every perturbation takes a float32 image in [0, 1] and returns one; the result is min-max renormalized, as every image is when it is loaded.
Brightness and contrast changes are not among them because that normalization undoes them.
"""
import io
import os
import zlib
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

def _renorm(img):
    img = np.asarray(img, dtype=np.float32)
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo) if hi > lo else img - lo

def gamma(img, level, rng):
    return img ** level

def noise(img, level, rng):
    """Additive Gaussian noise of standard deviation `level` (image range 1), clipped to the sensor range."""
    return np.clip(img + rng.normal(0.0, level, img.shape), 0.0, 1.0)

def blur(img, level, rng):
    """Gaussian blur of standard deviation `level` pixels (of the full-resolution image)."""
    size = 2 * int(np.ceil(3 * level)) + 1
    return TF.gaussian_blur(torch.from_numpy(img)[None, None], [size, size], [level, level])[0, 0].numpy()

def shading(img, level, rng):
    """Uneven illumination: a linear ramp in a random direction, from 1 - level to 1 + level across the image."""
    theta = rng.uniform(0.0, 2 * np.pi)
    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = (x - w / 2) * np.cos(theta) + (y - h / 2) * np.sin(theta)
    return img * (1.0 + level * ramp / np.abs(ramp).max())

def jpeg(img, level, rng):
    """JPEG compression at quality `level` (1-95)."""
    buffer = io.BytesIO()
    Image.fromarray(np.round(img * 255).astype(np.uint8)).save(buffer, "JPEG", quality=int(level))
    return np.asarray(Image.open(buffer), dtype=np.float32) / 255.0

PERTURBATIONS = {"gamma": gamma, "noise": noise, "blur": blur, "shading": shading, "jpeg": jpeg}

# levels of the robustness sweep ("name:level")
SWEEP = ("gamma:0.5", "gamma:0.7", "gamma:1.5", "gamma:2.0", "noise:0.02", "noise:0.05", "noise:0.1",
         "blur:1", "blur:2", "blur:4", "shading:0.2", "shading:0.4", "jpeg:50", "jpeg:20", "jpeg:10")

def apply(img, spec, rng):
    """`spec` is 'name:level', e.g. 'noise:0.05'."""
    name, level = spec.split(":")
    return _renorm(PERTURBATIONS[name](img, float(level), rng))

def fixed(spec):
    """Image function (image, path) -> image for the sweep: deterministic, the random part depends only on the file name (the same for every model)."""
    def image_fn(img, path):
        return apply(img, spec, np.random.default_rng(zlib.crc32(os.path.basename(path).encode())))
    return image_fn

# training augmentation: every perturbation is applied with probability 1/2 and a random level, in the order of the acquisition chain.
# Part of the sweep is inside these ranges, the strongest levels (gamma 0.5 and 2, noise 0.1, blur 4, JPEG 10) are outside.
AUGMENT = (("shading", lambda r: r.uniform(0.0, 0.3)),
           ("gamma", lambda r: float(np.exp(r.uniform(np.log(0.6), np.log(1.7))))),
           ("blur", lambda r: r.uniform(0.3, 2.5)),
           ("noise", lambda r: r.uniform(0.0, 0.06)),
           ("jpeg", lambda r: int(r.uniform(25, 96))))

# --augment_range wide: every level of the sweep is inside the ranges (the default ranges leave the strongest ones out), a control for the extrapolation
AUGMENT_WIDE = (("shading", lambda r: r.uniform(0.0, 0.5)),
                ("gamma", lambda r: float(np.exp(r.uniform(np.log(0.4), np.log(2.5))))),
                ("blur", lambda r: r.uniform(0.3, 5.0)),
                ("noise", lambda r: r.uniform(0.0, 0.12)),
                ("jpeg", lambda r: int(r.uniform(8, 96))))
AUGMENT_RANGES = {"default": AUGMENT, "wide": AUGMENT_WIDE}

def random_augment(img, path=None, ranges="default"):
    """Random photometric augmentation (draws from numpy's global RNG, which `seed_everything` seeds)."""
    for name, draw_level in AUGMENT_RANGES[ranges]:
        if np.random.random() < 0.5:
            img = _renorm(PERTURBATIONS[name](img, draw_level(np.random), np.random))
    return img
