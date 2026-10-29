import argparse
import os

import pytest
import torch

from hynet.models.factory import MODELS, build_model, needs_harmnet
from hynet.models.hynet import HYNet

ARGS = argparse.Namespace(input_shape=1, hidden_units=64, output_shape=1)
CHECKPOINT = "checkpoints/hynet_weights.pth"
CPU = torch.device("cpu")

@pytest.mark.parametrize("name", list(MODELS))
def test_forward_shape_and_backward(name):
    torch.manual_seed(0)
    model = build_model(name, ARGS, CPU)
    x = torch.rand(2, 1, 32, 48)
    out = model(x)
    assert out.shape == (2, 1, 32, 48)
    out.mean().backward()
    assert any(p.grad is not None for p in model.parameters())

def test_unknown_model():
    with pytest.raises(ValueError):
        build_model("nope", ARGS, CPU)

def test_frozen_harmnet_gets_no_gradient():
    model = build_model("hynet_frozen", ARGS, CPU)
    model(torch.rand(2, 1, 32, 48)).mean().backward()
    assert all(p.grad is None for p in model.harmnet.parameters())
    assert any(p.grad is not None for p in model.ynet.parameters())

def test_end_to_end_harmnet_gets_gradient():
    model = build_model("hynet", ARGS, CPU)
    model(torch.rand(2, 1, 32, 48)).mean().backward()
    assert any(p.grad is not None for p in model.harmnet.parameters())

def test_raw_ynet_has_no_harmnet_and_same_ynet_size():
    raw, full = build_model("ynet_raw", ARGS, CPU), build_model("hynet", ARGS, CPU)
    assert not hasattr(raw, "harmnet")
    assert sum(p.numel() for p in raw.ynet.parameters()) == sum(p.numel() for p in full.ynet.parameters())

def test_needs_harmnet():
    assert needs_harmnet("hynet") and needs_harmnet("harm_unet")
    assert not needs_harmnet("unet") and not needs_harmnet("ynet_raw")

@pytest.mark.skipif(not os.path.exists(CHECKPOINT), reason="released checkpoint not available")
def test_released_checkpoint_still_loads_strictly():
    """The ablation options must not change the parameter names/shapes of the published model."""
    state = torch.load(CHECKPOINT, map_location=CPU, weights_only=True)
    model = HYNet(1, 64, 1, CPU, None)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(1, 1, 64, 64))
    assert out.shape == (1, 1, 64, 64) and torch.isfinite(out).all()
