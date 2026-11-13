import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from hynet import evaluation
from hynet.cli.postproc import probabilities

class _Dataset(torch.utils.data.Dataset):
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.images = [torch.rand(1, 8, 12, generator=g) for _ in range(3)]
    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        return self.images[i], torch.zeros(1, 8, 12)

def test_tta_does_not_change_a_pointwise_model():
    model = nn.Conv2d(1, 1, 1)
    plain, tta = (probabilities(model, _Dataset(), torch.device("cpu"), t) for t in (False, True))
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(plain, tta))

def test_tta_averages_a_model_that_sees_the_left_edge_only():
    class LeftEdge(nn.Module):
        def forward(self, x):
            out = torch.full_like(x, -10.0)
            out[..., 0] = 10.0
            return out
    p = probabilities(LeftEdge(), _Dataset(), torch.device("cpu"), True)[0]
    # the edge is seen by the image and its vertical flip (left column) and by the two horizontal flips (right column): half of the four views
    assert p.shape == (8, 12) and p[4, 0] == pytest.approx(0.5, abs=1e-3) and p[4, -1] == pytest.approx(0.5, abs=1e-3) and p[4, 6] < 1e-3

def test_well_dice_sums_the_images_of_a_well():
    mask = np.zeros((4, 4), bool)
    mask[:2] = True
    scores = evaluation.well_dice([mask, ~mask, mask], [mask, mask, mask], ["IMG_B1_Interval1.tif", "IMG_B1_Interval2.tif", "IMG_B2_Interval1.tif"])
    assert scores["B2"] == (1, 1.0) and scores["B1"][0] == 2 and scores["B1"][1] == pytest.approx(0.5)
