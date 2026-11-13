import random
import pytest
import torch
from hynet.scale import random_scale

def pair(h=48, w=64):
    mask = torch.zeros(1, h, w)
    mask[:, 10:30, 20:44] = 1
    return mask.clone(), mask

def test_shapes_and_binary_mask_for_many_factors():
    for seed in range(60):
        random.seed(seed)
        image, mask = random_scale(*pair(), 2.0)
        assert image.shape == mask.shape == (1, 48, 64)
        assert set(mask.unique().tolist()) <= {0.0, 1.0}

def test_image_and_mask_stay_aligned():
    for seed in range(60):
        random.seed(seed)
        image, mask = random_scale(*pair(), 2.0)  # the image is the mask itself, so after any transformation they agree except at interpolated edges
        assert ((image > 0.5).float() != mask).float().mean() < 0.06

def test_zoom_changes_the_organoid_area():
    areas = set()
    for seed in range(40):
        random.seed(seed)
        areas.add(round(random_scale(*pair(), 2.0)[1].mean().item(), 3))
    assert len(areas) > 10 and min(areas) < 0.6 * pair()[1].mean().item() and max(areas) > 1.4 * pair()[1].mean().item()

def test_invalid_factor():
    with pytest.raises(ValueError):
        random_scale(*pair(), 3.0)
    with pytest.raises(ValueError):
        random_scale(*pair(), 1.0)
