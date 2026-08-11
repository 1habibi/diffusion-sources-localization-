from __future__ import annotations

import numpy as np

from diffusion_sources.dataset import build_example, example_to_pyg
from diffusion_sources.diffusion import simulate_ic
from diffusion_sources.inference import predict_joint
from diffusion_sources.losses import joint_source_count_loss
from diffusion_sources.metrics import set_metrics
from diffusion_sources.models import JointSourceCountGCN
from diffusion_sources.observations import observe_cascade


def test_cascade_to_model_to_metrics_pipeline(path_graph):
    cascade = simulate_ic(path_graph, {1, 3}, 1.0, 3, np.random.default_rng(10))
    observation = observe_cascade(
        path_graph, cascade, 1.0, 0, np.random.default_rng(11)
    )
    example = build_example(
        "path", path_graph, cascade, observation, simulation_seed=10, observation_seed=11
    )
    data = example_to_pyg(path_graph, example)
    source_logits, count_logits = JointSourceCountGCN(hidden_dim=8, dropout=0.0)(data)
    loss = joint_source_count_loss(
        source_logits,
        count_logits,
        data.source_labels,
        data.source_count,
        data.candidate_mask,
    )
    prediction = predict_joint(source_logits, count_logits, data.candidate_mask)
    metrics = set_metrics(cascade.sources, prediction.sources)

    assert loss.total.isfinite()
    assert 1 <= prediction.source_count <= 3
    assert 0.0 <= metrics["f1"] <= 1.0
