import numpy as np
from hynet import spectral

GRID = spectral.Grid((64, 96), 16)

def noise(seed=0):
    return np.random.default_rng(seed).normal(size=(64, 96))

def high_frequency_energy(x):
    return (np.abs(np.fft.fft2(x - x.mean())) ** 2)[GRID.r > 0.25].sum()

def test_attenuate_never_amplifies():
    x = noise()
    huge = np.full(16, 1e9)
    assert np.allclose(spectral.attenuate(x, huge, GRID), x)  # reference above the image: unchanged
    flat = spectral.radial_profile(x, GRID) * 0.01
    y = spectral.attenuate(x, flat, GRID)
    assert high_frequency_energy(y) < high_frequency_energy(x) * 0.5

def test_reference_is_lowest_mean_profile():
    sharp = [noise(i) for i in range(3)]
    smooth = [np.cumsum(np.cumsum(noise(i), 0), 1) for i in range(3)]  # integrated noise: power at high frequencies much lower
    ref = spectral.reference_profile({"a": sharp, "b": smooth}, GRID)
    means = [np.mean([spectral.radial_profile(x, GRID) for x in images], 0) for images in (sharp, smooth)]
    assert np.allclose(ref, np.minimum(*means))
    assert (ref <= means[0]).all() and (ref <= means[1]).all()

def test_harmonize_range_and_control():
    x = noise(1) * 7 + 100
    y = spectral.harmonize(x, None, GRID)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1 and y.max() > 0.9
    z = spectral.harmonize(x, spectral.radial_profile(x, GRID) * 0.1, GRID)
    assert z.min() >= 0 and z.max() <= 1
