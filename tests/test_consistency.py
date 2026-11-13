import argparse
import numpy as np
import pytest
import torch
from PIL import Image
from hynet.consistency import Consistency, PairDataset, make_consistency
from hynet.engine import training
from hynet.models.factory import build_model

ARGS = argparse.Namespace(input_shape=1, hidden_units=64, output_shape=1)
CPU = torch.device("cpu")

def loader(n=3, same=False):
    for _ in range(n):
        v1 = torch.rand(1, 1, 32, 48)
        yield v1, (v1.clone() if same else torch.rand(1, 1, 32, 48))

@pytest.mark.parametrize("name", ["hynet_res", "ynet_raw", "unet"])
def test_loss_is_finite_and_has_gradient(name):
    model = build_model(name, ARGS, CPU)
    loss = Consistency(list(loader()), 5.0, 5.0).loss(model, CPU)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

def test_identical_views_have_zero_loss():
    model = build_model("hynet_res", ARGS, CPU).eval()
    assert Consistency(list(loader(same=True)), 5.0, 5.0).loss(model, CPU).item() == 0.0

def test_harm_term_only_for_models_with_harmnet_and_positive_weight():
    torch.manual_seed(0)
    model = build_model("hynet_res", ARGS, CPU).eval()
    batch = list(loader(1))
    both = Consistency(batch, 0.0, 5.0).loss(model, CPU).item()
    seg_only = Consistency(batch, 0.0, 0.0).loss(model, CPU).item()
    assert seg_only == 0.0 and both > 0.0
    raw = build_model("ynet_raw", ARGS, CPU).eval()
    assert Consistency(batch, 0.0, 5.0).loss(raw, CPU).item() == 0.0  # no HarmNet: nothing to make consistent

def test_pair_dataset_gives_two_different_views(tmp_path):
    path = tmp_path / "a.png"
    Image.fromarray((np.random.RandomState(0).rand(60, 80) * 255).astype(np.uint8)).save(path)
    np.random.seed(1)
    v1, v2 = PairDataset([str(path)])[0]
    assert v1.shape == v2.shape == (1, 480, 640) and not torch.equal(v1, v2)

def test_target_images_get_half_of_the_draws(tmp_path):
    for name in ("s", "t"):
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / f"{name}.png")
    args = argparse.Namespace(augment_range="default", consist=1.0, consist_harm=0.0, consist_pairs=2)
    cons = make_consistency(args, [str(tmp_path / "s.png")], [str(tmp_path / "t.png")])
    sampler = cons.batches.gi_frame.f_locals["loader"].sampler
    idx = torch.tensor(list(sampler)[:4000])
    assert abs((idx == 1).float().mean().item() - 0.5) < 0.05

def test_training_step_with_consistency_updates_weights():
    torch.manual_seed(0)
    model = build_model("hynet_res", ARGS, CPU)
    before = [p.clone() for p in model.harmnet.parameters()]
    data = [(torch.rand(2, 1, 32, 48), (torch.rand(2, 1, 32, 48) > 0.5).float())]
    cons = Consistency(list(loader(4)), 5.0, 5.0, pairs=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = training(model, data, lambda out, m: torch.nn.functional.binary_cross_entropy_with_logits(out, m), opt, CPU, None, 1, cons)
    assert np.isfinite(loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.harmnet.parameters()))
