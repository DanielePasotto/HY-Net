"""Deterministic harmonization of the acquisition signature of an image (no training).

The datasets differ in resolution and optics: once resized to the network input, the radial power spectrum of an image tells the datasets apart (a linear
classifier on it alone gets about 80%). `attenuate` lowers, at every spatial frequency, the power of an image to a common envelope (the lowest mean
power among the reference datasets) and never raises it: no noise is amplified and the structure is kept (correlation of the edges with the original 0.97).
`robust_normalize` then maps the 1st..99th percentile of the intensities to 0..1, which removes the brightness and contrast offsets between datasets.
"""
import numpy as np

SIZE = (480, 640)  # (height, width) of the network input
BINS = 32

class Grid:
    """Radial frequency bins of an image of a given shape."""
    def __init__(self, shape=SIZE, bins=BINS):
        fy, fx = np.fft.fftfreq(shape[0])[:, None], np.fft.fftfreq(shape[1])[None, :]
        self.r = np.sqrt(fy ** 2 + fx ** 2)
        edges = np.linspace(0, 0.5, bins + 1)
        self.index = np.clip(np.digitize(self.r, edges) - 1, 0, bins - 1)
        self.centers = (edges[:-1] + edges[1:]) / 2
        self.bins = bins
        self.window = np.outer(np.hanning(shape[0]), np.hanning(shape[1]))

def radial_profile(x, grid):
    """Mean power per radial frequency bin of the image rescaled to zero mean and unit variance (Hann window against the edge effects)."""
    z = (x - x.mean()) / (x.std() + 1e-9)
    power = np.abs(np.fft.fft2(z * grid.window)) ** 2
    return np.bincount(grid.index.ravel(), power.ravel(), grid.bins) / np.bincount(grid.index.ravel(), minlength=grid.bins)

def reference_profile(images_by_domain, grid):
    """Lowest, at every frequency, of the mean profiles of the datasets: the envelope that every reference dataset reaches."""
    return np.min([np.mean([radial_profile(x, grid) for x in images], 0) for images in images_by_domain.values()], 0)

def attenuate(x, reference, grid):
    """Lower the spectrum of `x` to `reference` where it is above it; frequencies already below it are left unchanged."""
    mean, std = x.mean(), x.std() + 1e-9
    gain = np.minimum(1.0, np.sqrt(reference / (radial_profile(x, grid) + 1e-12)))
    return np.fft.ifft2(np.fft.fft2((x - mean) / std) * np.interp(grid.r, grid.centers, gain)).real * std + mean

def robust_normalize(x, low=1, high=99):
    a, b = np.percentile(x, [low, high])
    return np.clip((x - a) / (b - a + 1e-9), 0, 1)

def harmonize(x, reference, grid):
    """Float image (network input size) -> float image in [0, 1]. reference None: only the intensity normalization (control)."""
    return robust_normalize(x if reference is None else attenuate(x, reference, grid))
