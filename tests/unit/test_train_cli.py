from __future__ import annotations

import pytest
import torch

from diffusion_sources.train_cli import calculate_pos_weight, load_train_config, set_seed
from torch_geometric.data import Data


def test_calculate_pos_weight_uses_only_candidate_mask():
    data = Data(
        source_labels=torch.tensor([1.0, 0.0, 0.0]),
        candidate_mask=torch.tensor([True, True, False]),
    )
    assert calculate_pos_weight([data]).item() == 1.0


def test_set_seed_reproduces_torch_values():
    set_seed(9)
    first = torch.rand(3)
    set_seed(9)
    second = torch.rand(3)
    assert torch.equal(first, second)


def test_load_train_config_validates_sections(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("data: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing training configuration"):
        load_train_config(config_path)
