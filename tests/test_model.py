from pathlib import Path

import pytest
import torch

from cifar_resnet.config import ModelConfig, load_config
from cifar_resnet.models import build_model, count_parameters

CONFIGS = sorted(p for p in Path("configs").glob("*.yaml") if p.stem != "teacher")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_submission_configs_fit_the_budget(path):
    cfg = load_config(path)
    model = build_model(cfg.model)
    assert count_parameters(model) <= 5_000_000


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_forward_shape(path):
    model = build_model(load_config(path).model).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_budget_is_enforced():
    with pytest.raises(ValueError, match="budget"):
        build_model(ModelConfig(channels=[256, 512, 1024], blocks=[2, 2, 2]))


def test_configurable_kernels_and_pool():
    cfg = ModelConfig(
        channels=[16, 32], blocks=[1, 1], kernel_size=5, skip_kernel_size=3, pool_size=4
    )
    model = build_model(cfg).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 32, 32)).shape == (1, 10)

    assert model.fc.in_features == 4 * 4 * 32
