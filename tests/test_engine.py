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
