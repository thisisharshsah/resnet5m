"""Shared fixtures: a tiny fake dataset in the exact Kaggle layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

N_TRAIN_PER_CLASS = 6
N_TEST = 12


@pytest.fixture(scope="session")
def fake_data_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("kaggle_data")
    rng = np.random.default_rng(0)
    (root / "train_images").mkdir()
    (root / "test_images").mkdir()

    rows = []
    for i in range(10 * N_TRAIN_PER_CLASS):
        name = f"train_{i:05d}.png"
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
            root / "train_images" / name
        )
        rows.append({"id": i, "filename": name, "label": i % 10})
    pd.DataFrame(rows).to_csv(root / "train_labels.csv", index=False)

    test_rows = []
    for i in range(N_TEST):
        name = f"test_{i:04d}.png"
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
            root / "test_images" / name
        )
        test_rows.append({"id": i, "filename": name})
    pd.DataFrame(test_rows).to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": range(N_TEST), "label": 0}).to_csv(
        root / "sample_submission.csv", index=False
    )
    return root
