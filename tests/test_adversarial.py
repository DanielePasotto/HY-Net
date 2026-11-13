import torch
from torch.utils.data import DataLoader, Dataset
from torch import optim

from hynet.adversarial import DomainDiscriminator, DomainLabelled, training_adversarial
from hynet.losses import RegionAwareCompositeLoss
from hynet.models.harmnet import HarmNet

class Fake(Dataset):
    """Two 'datasets': the second is brighter, so a discriminator can tell them apart from the image statistics."""
    def __init__(self):
        self.image_paths = [f"x/IMG_{d}-{i}_Interval0.png" for d in ("a", "b") for i in range(8)]
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, i):
        torch.manual_seed(i)
        base = 0.3 if i < 8 else 0.7
        x = (base + 0.1 * torch.rand(1, 32, 48)).clamp(0, 1)
        return x, (torch.rand(1, 32, 48) > 0.5).float(), x.clone()

def test_domain_labels_follow_the_file_name_prefix():
    ds = DomainLabelled(Fake())
    assert ds.domains == ["a", "b"]
    assert [ds[i][3] for i in (0, 7, 8, 15)] == [0, 0, 1, 1]
    assert len(ds[0]) == 4

def test_discriminator_learns_the_domain_and_harmnet_is_updated():
    torch.manual_seed(0)
    ds = DomainLabelled(Fake())
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = HarmNet(1, 8, 1)
    disc = DomainDiscriminator(2, width=8)
    before = [p.clone() for p in model.parameters()]
    criterion = RegionAwareCompositeLoss()
    opt, opt_d = optim.Adam(model.parameters(), 1e-3), optim.Adam(disc.parameters(), 1e-3)
    for _ in range(15):
        rec, adv, acc = training_adversarial(model, disc, loader, criterion, opt, opt_d, torch.device("cpu"), lambda_adv=0.1)
    assert acc > 0.8  # a weak adversarial weight leaves the brightness difference visible, so the discriminator wins
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))

def test_moment_penalty_is_zero_for_equal_domains_and_positive_otherwise():
    from hynet.adversarial import image_statistics, moment_penalty
    torch.manual_seed(0)
    same = torch.rand(1, 1, 32, 48).repeat(4, 1, 1, 1)
    domains = torch.tensor([0, 0, 1, 1])
    assert moment_penalty(image_statistics(same), domains).item() < 1e-8
    different = torch.cat([same[:2] * 0.3, same[2:] * 0.9])
    assert moment_penalty(image_statistics(different), domains).item() > 0.1
    assert moment_penalty(image_statistics(different), torch.zeros(4, dtype=torch.long)).item() == 0

def test_moment_training_step_runs():
    from hynet.adversarial import training_moments
    loader = DataLoader(DomainLabelled(Fake()), batch_size=4, shuffle=True)
    model = HarmNet(1, 8, 1)
    rec, pen = training_moments(model, loader, RegionAwareCompositeLoss(), optim.Adam(model.parameters(), 1e-3), torch.device("cpu"), 1.0)
    assert pen >= 0
