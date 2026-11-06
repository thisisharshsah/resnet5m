import pytest
import torch

from cifar_resnet.engine import tta_views


@pytest.mark.parametrize(
    "mode, n_views", [("none", 1), (False, 1), ("flip", 2), (True, 2), ("shift", 10)]
)
def test_tta_view_counts_and_shapes(mode, n_views):
    x = torch.randn(4, 3, 32, 32)
    views = tta_views(x, mode)
    assert len(views) == n_views
    assert all(v.shape == x.shape for v in views)


def test_shift_views_are_shifted_copies():
    x = torch.randn(1, 3, 32, 32)
    views = tta_views(x, "shift")
    assert torch.equal(views[0], x)
    assert torch.equal(views[1], torch.flip(x, dims=[3]))
    assert torch.equal(views[2][:, :, :-2, :], x[:, :, 2:, :])
    assert torch.equal(views[8][:, :, :, 2:], x[:, :, :, :-2])


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        tta_views(torch.zeros(1, 3, 32, 32), "rotate")
