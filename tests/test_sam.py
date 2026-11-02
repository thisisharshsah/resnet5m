import torch
from torch import nn

from cifar_resnet.sam import SAM, frozen_bn_stats


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.Flatten(), nn.LazyLinear(2))


def test_sam_restores_weights_before_the_base_step():
    model = _model()
    x, y = torch.randn(8, 3, 8, 8), torch.randint(0, 2, (8,))
    model(x)
    base = torch.optim.SGD(model.parameters(), lr=0.0)
    sam = SAM(base, rho=0.5)
    before = [p.detach().clone() for p in model.parameters()]

    nn.functional.cross_entropy(model(x), y).backward()
    sam.first_step()
    assert any(not torch.equal(a, p) for a, p in zip(before, model.parameters()))
    base.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    sam.second_step()
    assert all(torch.allclose(a, p) for a, p in zip(before, model.parameters()))


def test_frozen_bn_stats_keeps_running_statistics():
    model = _model().train()
    bn = model[1]
    model(torch.randn(8, 3, 8, 8))
    mean = bn.running_mean.clone()
    with frozen_bn_stats(model):
        model(torch.randn(8, 3, 8, 8) + 5)
    assert torch.equal(mean, bn.running_mean)
    assert bn.momentum == 0.1
