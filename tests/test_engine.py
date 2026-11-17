import pytest

torch = pytest.importorskip("torch")
from torch import nn
from hynet.engine import training

def _run(micro_batches):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 1, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    g = torch.Generator().manual_seed(1)
    batches = [(torch.rand(4, 1, 8, 8, generator=g), torch.rand(4, 1, 8, 8, generator=g)) for _ in range(3)]
    loss = training(model, batches, lambda out, mask: ((out - mask) ** 2).mean(), optimizer, torch.device("cpu"), micro_batches=micro_batches)
    return loss, torch.cat([p.flatten() for p in model.parameters()])

def test_micro_batches_match_the_full_batch():
    loss_full, weights_full = _run(1)
    loss_micro, weights_micro = _run(2)
    assert loss_micro == pytest.approx(loss_full, rel=1e-5)
    assert torch.allclose(weights_micro, weights_full, atol=1e-6)

def test_micro_batches_larger_than_the_batch():
    loss_full, _ = _run(1)
    loss_many, _ = _run(7)
    assert loss_many == pytest.approx(loss_full, rel=1e-5)

def test_small_object_weights_mark_only_components_under_the_area():
    import torch
    from hynet.losses import small_object_weights
    m = torch.zeros(1, 1, 60, 60)
    m[0, 0, 2:5, 2:5] = 1          # 9 px: small
    m[0, 0, 20:50, 20:50] = 1      # 900 px: large
    w = small_object_weights(m, 4.0, max_area=200)
    assert w[0, 0, 3, 3] == 4.0 and w[0, 0, 30, 30] == 1.0 and w[0, 0, 55, 55] == 1.0

def test_tversky_bce_weight_one_equals_no_weight_and_weights_change_the_loss():
    import torch
    from hynet.losses import TverskyBCELoss
    torch.manual_seed(0)
    crit = TverskyBCELoss()
    logits, masks = torch.randn(2, 1, 16, 16), (torch.rand(2, 1, 16, 16) > 0.5).float()
    assert torch.allclose(crit(logits, masks), crit(logits, masks, torch.ones_like(masks)), atol=1e-6)
    heavy = torch.where(masks > 0, torch.full_like(masks, 5.0), torch.ones_like(masks))
    assert not torch.allclose(crit(logits, masks), crit(logits, masks, heavy))
