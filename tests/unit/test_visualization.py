from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from diffusion_sources.demo import run_demo
from diffusion_sources.models import JointSourceCountGCN
from diffusion_sources.visualization import plot_demo_result


def test_plot_demo_result_builds_two_panel_figure(path_graph):
    result = run_demo(
        path_graph,
        JointSourceCountGCN(input_dim=2, hidden_dim=8, dropout=0.0),
        [0, 1],
        {2},
        probability=1.0,
        max_steps=3,
        observation_fraction=1.0,
        seed=8,
    )
    figure = plot_demo_result(result)

    assert len(figure.axes) == 3
    plt.close(figure)
