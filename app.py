from __future__ import annotations

from pathlib import Path

import streamlit as st

from diffusion_sources.demo import load_demo_graph, load_demo_model, run_demo
from diffusion_sources.visualization import plot_demo_result


DATA_DIR = Path("data/generated/pilot")
RUN_DIR = Path("reports/runs/pilot")


@st.cache_resource
def load_resources():
    return load_demo_graph(DATA_DIR), *load_demo_model(RUN_DIR)


def main() -> None:
    st.set_page_config(page_title="Diffusion source detective", layout="wide")
    st.title("Localization of multiple diffusion sources")
    st.caption(
        "Select 1-3 initial nodes, simulate an IC cascade, and compare the "
        "Joint Source-Count GCN prediction with the known sources."
    )
    if not (DATA_DIR / "graph.npz").exists() or not (RUN_DIR / "best_model.pt").exists():
        st.error("Pilot data or checkpoint is missing. Generate and train the pilot first.")
        return

    graph, model, feature_indices = load_resources()
    nodes = sorted(graph.nodes())
    with st.sidebar:
        st.header("Cascade controls")
        source_count = st.slider("Number of sources", 1, 3, 2)
        default_sources = nodes[:source_count]
        sources = st.multiselect(
            "Source nodes", nodes, default=default_sources, max_selections=source_count
        )
        probability = st.slider("Transmission probability", 0.05, 1.0, 0.4, 0.05)
        max_steps = st.slider("Diffusion steps", 1, 6, 3)
        observed_fraction = st.slider("Observed infected fraction", 0.25, 1.0, 0.75, 0.05)
        false_positives = st.slider("False-positive nodes", 0, 5, 0)
        seed = st.number_input("Random seed", min_value=0, value=2026, step=1)
        run_clicked = st.button("Run simulation and inference", type="primary")

    if len(sources) != source_count:
        st.info(f"Select exactly {source_count} source nodes.")
        return
    if not run_clicked:
        st.info("Configure the cascade in the sidebar and run the experiment.")
        return

    result = run_demo(
        graph,
        model,
        feature_indices,
        set(sources),
        probability=probability,
        max_steps=max_steps,
        observation_fraction=observed_fraction,
        false_positive_count=false_positives,
        seed=int(seed),
    )
    metric_columns = st.columns(5)
    values = (
        ("True k", len(result.cascade.sources)),
        ("Predicted k", result.prediction.source_count),
        ("F1", f"{result.metrics['f1']:.3f}"),
        ("Count error", f"{result.metrics['count_mae']:.0f}"),
        ("Graph distance", f"{result.metrics['symmetric_set_distance']:.3f}"),
    )
    for column, (label, value) in zip(metric_columns, values, strict=True):
        column.metric(label, value)
    st.pyplot(plot_demo_result(result), use_container_width=True)

    left, right = st.columns(2)
    left.subheader("Sources")
    left.write({"true": sorted(result.cascade.sources), "predicted": sorted(result.prediction.sources)})
    right.subheader("Cascade")
    right.write(
        {
            "infected": len(result.cascade.infected),
            "observed": len(result.observation.observed_infected),
            "candidates": len(result.observation.candidate_nodes),
        }
    )


if __name__ == "__main__":
    main()
