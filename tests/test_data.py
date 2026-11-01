import numpy as np

from cifar_resnet.config import AugmentConfig
from cifar_resnet.data import build_transforms, find_data_root, load_split, stratified_split


def test_find_data_root_searches_one_level_down(fake_data_root):
    assert find_data_root(fake_data_root) == fake_data_root
    assert find_data_root(fake_data_root.parent) == fake_data_root


def test_load_split_and_cache(fake_data_root, tmp_path):
    train = load_split(fake_data_root, "train", tmp_path)
    assert train.images.shape == (60, 32, 32, 3) and train.images.dtype == np.uint8
    assert train.labels is not None and set(train.labels) == set(range(10))
    cached = load_split(fake_data_root, "train", tmp_path)
    assert np.array_equal(train.images, cached.images)

    test = load_split(fake_data_root, "test", tmp_path)
    assert test.labels is None and len(test.ids) == 12


def test_stratified_split_is_balanced_and_disjoint():
    labels = np.repeat(np.arange(10), 100)
    train, val = stratified_split(labels, 200, seed=0)
    assert len(val) == 200 and len(set(train) & set(val)) == 0
    assert np.all(np.bincount(labels[val]) == 20)


def test_transforms_output_normalized_float():
    import torch

    img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    for train in (True, False):
        out = build_transforms(AugmentConfig(), train)(img)
        assert out.shape == (3, 32, 32) and out.dtype == torch.float32
