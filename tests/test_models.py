import argparse
import os

import pytest
import torch

from hynet.models.factory import MODELS, build_model

ARGS = argparse.Namespace(input_shape=1, hidden_units=64, output_shape=1)
CHECKPOINT = "checkpoints/final/hynet_res_pool_all.pth"
CPU = torch.device("cpu")

@pytest.mark.parametrize("name", [m for m in MODELS if m not in ("campeod", "macpnet")])  # they need third-party code that is not in the repository
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

def test_raw_ynet_has_no_harmnet_and_same_ynet_size():
    raw, full = build_model("ynet_raw", ARGS, CPU), build_model("hynet_res", ARGS, CPU)
    assert not hasattr(raw, "harmnet")
    assert sum(p.numel() for p in raw.ynet.parameters()) == sum(p.numel() for p in full.ynet.parameters())

def test_residual_harmnet_is_identity_at_start_and_learns():
    model = build_model("hynet_res", ARGS, CPU)
    x = torch.rand(2, 1, 32, 48)
    assert torch.equal(model.harmnet(x), x)
    model(x).mean().backward()
    assert model.harmnet.out[0].weight.grad.abs().sum() > 0  # the zero-initialized layer does receive gradient

def test_fast_residual_harmnet_is_identity_at_start_and_works_on_a_reduced_image():
    model = build_model("hynet_fast", ARGS, CPU)
    x = torch.rand(2, 1, 32, 48)
    assert torch.equal(model.harmnet(x), x)
    model.harmnet.out[0].bias.data.fill_(0.5)  # a non-zero correction must come back at the input size
    assert model.harmnet(x).shape == x.shape
    model(x).mean().backward()
    assert model.harmnet.out[0].weight.grad.abs().sum() > 0

@pytest.mark.skipif(not os.path.exists(CHECKPOINT), reason="released checkpoint not available")
def test_released_checkpoint_still_loads_strictly():
    """The released final model must load into the current code without missing or unexpected parameters."""
    state = torch.load(CHECKPOINT, map_location=CPU, weights_only=True)
    model = build_model("hynet_res", ARGS, CPU)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(1, 1, 64, 64))
    assert out.shape == (1, 1, 64, 64) and torch.isfinite(out).all()
