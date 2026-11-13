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

def test_decoder_only_keeps_pretrained_encoders_fixed():
    model = build_model("hynet_decoder", ARGS, CPU)
    model.train()
    assert not model.harmnet.training
    assert not model.ynet.encoder1_layers.training
    assert not model.ynet.encoder2_layers.training
    assert all(not p.requires_grad for p in model.harmnet.parameters())
    assert all(not p.requires_grad for p in model.ynet.encoder1_layers.parameters())
    assert all(not p.requires_grad for p in model.ynet.encoder2_layers.parameters())
    assert any(p.requires_grad for p in model.ynet.dec_convs.parameters())
    assert any(p.requires_grad for p in model.ynet.fusion_blocks.parameters())

def test_sum_fusion_preserves_attention_without_gates():
    model = build_model("hynet_sum", ARGS, CPU)
    assert model.ynet.use_attention
    assert model.ynet.fusion == "sum"
    assert len(model.ynet.fusion_blocks) == 0
    assert not hasattr(model.ynet, "bottleneck_fusion")

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
    assert not needs_harmnet("unet") and not needs_harmnet("ynet_raw") and not needs_harmnet("hynet_res")

def test_residual_harmnet_is_identity_at_start_and_learns():
    model = build_model("hynet_res", ARGS, CPU)
    x = torch.rand(2, 1, 32, 48)
    assert torch.equal(model.harmnet(x), x)
    model(x).mean().backward()
    assert model.harmnet.out[0].weight.grad.abs().sum() > 0  # the zero-initialized layer does receive gradient

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

def test_group_norm_replaces_every_batchnorm_of_the_ynet():
    model = build_model("ynet_raw", argparse.Namespace(**vars(ARGS), norm="group"), CPU)
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    x = torch.rand(1, 1, 32, 48)
    out = model(x)
    assert out.shape == x.shape
    # per-image statistics: an image gives the same output alone and inside a batch
    model.eval()
    assert torch.allclose(model(x), model(torch.cat([x, torch.rand(1, 1, 32, 48)]))[:1], atol=1e-5)

def test_adaptive_batchnorm_changes_statistics_not_weights():
    from hynet.adabn import adapt_batchnorm
    torch.manual_seed(0)
    model = build_model("ynet_raw", ARGS, CPU)
    model.eval()
    weights = {k: v.clone() for k, v in model.state_dict().items() if "running" not in k and "num_batches" not in k}
    stats = [m.running_mean.clone() for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    images = [torch.rand(1, 1, 32, 48) * 0.2 + 0.7 for _ in range(3)]
    assert adapt_batchnorm(model, images, CPU) == len(stats)
    assert not model.training
    assert any(not torch.equal(a, m.running_mean) for a, m in zip(stats, [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in weights.items())
