from __future__ import annotations

import torch

from diffusion_sources.demo import run_demo
from diffusion_sources.models import JointSourceCountGCN


def test_run_demo_returns_complete_result(path_graph):
    torch.manual_seed(4)
    model = JointSourceCountGCN(input_dim=2, hidden_dim=8, dropout=0.0)
    result = run_demo(
        path_graph,
        model,
        [0, 1],
        {1, 3},
        probability=1.0,
        max_steps=3,
        observation_fraction=1.0,
        seed=5,
    )

    assert result.cascade.sources == frozenset({1, 3})
    assert 1 <= result.prediction.source_count <= 3
    assert 0.0 <= result.metrics["f1"] <= 1.0
