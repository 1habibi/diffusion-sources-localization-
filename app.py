"""Local, read-only Streamlit workbench for the frozen Temporal-v3 model."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from diffusion_sources.temporal_demo_assets import load_temporal_resources
from diffusion_sources.temporal_demo_inference import TemporalDemoResult, infer_demo
from diffusion_sources.temporal_demo_scenario import DemoSettings, generate_demo_scenario
from diffusion_sources.temporal_demo_view import build_demo_graph_view, plot_demo_graph


BACKUP_ROOT = Path(__file__).resolve().parent / "reports/backups/temporal_v3_20260929"


@st.cache_resource(show_spinner=False)
def load_resources():
    root = Path(os.environ.get("DIFFUSION_TEMPORAL_BACKUP_DIR", BACKUP_ROOT))
    return load_temporal_resources(root)


def _ids(nodes) -> str:
    return ", ".join(str(node) for node in sorted(nodes)) or "—"


def _candidate_rows(result: TemporalDemoResult, method: str) -> list[dict]:
    candidates = result.candidate_scores
    rows = zip(candidates.candidate_ids, candidates.scores, candidates.early_observed, strict=True)
    ranked = sorted(rows, key=lambda row: (
        -(row[1] + (.5 if method == "Temporal-v3" and row[2] else 0)), -row[1], row[0]
    ))
    chosen = result.temporal_sources if method == "Temporal-v3" else result.snapshot.sources
    return [
        {"Ранг": index, "Узел": node, "Сырой GCN score": round(score, 4),
         "Раннее наблюдение": "да" if early else "нет", "В прогнозе": "да" if node in chosen else "нет"}
        for index, (node, score, early) in enumerate(ranked, 1)
    ]


def _render_result(result: TemporalDemoResult) -> None:
    st.divider()
    head, note = st.columns([3, 1])
    head.subheader("Результат каскада")
    note.caption(f"Попытка {result.scenario.attempt} · seed симуляции {result.scenario.simulation_seed}")

    view_method = st.radio("Прогноз", ["Snapshot", "Temporal-v3"], horizontal=True, key="view_method")
    frame_label = st.radio("Слой графа", ["t = 1", "t = 3", "Скоринг"], horizontal=True, key="view_frame")
    show_truth = st.toggle("Показать истинные источники", value=True, key="show_truth")
    metrics = result.temporal_metrics if view_method == "Temporal-v3" else result.snapshot_metrics
    metric_columns = st.columns(5)
    metric_columns[0].metric("F1", f"{metrics['f1']:.3f}")
    metric_columns[1].metric("Предсказанное k", str(result.snapshot.source_count), help="Число источников, которое оценила модель.")
    metric_columns[2].metric("Истинное k", str(len(result.scenario.cascade.sources)), help="Число источников, заданное только симулятору.")
    metric_columns[3].metric("Точный набор", "да" if metrics["exact_set_accuracy"] else "нет", help="Совпал ли набор узлов-источников целиком.")
    metric_columns[4].metric("Симм. дистанция", f"{metrics['symmetric_set_distance']:.2f}", help="Среднее расстояние по рёбрам от истинных источников к ближайшему прогнозу и обратно; меньше — лучше.")
    st.caption("Метрики относятся только к этому симулированному каскаду, а не к среднему качеству модели.")

    selected = st.session_state.get("selected_node")
    view = build_demo_graph_view(result, selected_node=selected)
    st.session_state["demo_view"] = view
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Граф распространения")
        st.caption(f"Показано {view.displayed_count} из {view.total_count} узлов. Расчёт модели выполнен на полном графе.")
        frame = {"t = 1": "early", "t = 3": "final", "Скоринг": "scores"}[frame_label]
        method = "snapshot" if view_method == "Snapshot" else "temporal"
        figure = plot_demo_graph(view, result, method, frame, show_truth)
        event = st.plotly_chart(figure, width="stretch", on_select="rerun", selection_mode="points", key=f"graph_selection_{id(result)}")
        if event and event.selection.points:
            point = event.selection.points[0]
            original_id = point.get("customdata")
            if isinstance(original_id, (list, tuple)):
                original_id = original_id[0] if original_id else None
            if isinstance(original_id, int) and original_id in result.graph:
                st.session_state["selected_node"] = original_id
        st.caption("Синий — наблюдённое заражение; ромб — прогноз; зелёная обводка — истинный источник (если открыт). В слое «Скоринг» яркость показывает сырой GCN score, не вероятность.")
    with right:
        st.markdown("#### Сравнение")
        st.markdown(f"**Snapshot:** {_ids(result.snapshot.sources)} · F1 {result.snapshot_metrics['f1']:.3f}")
        st.markdown(f"**Temporal-v3:** {_ids(result.temporal_sources)} · F1 {result.temporal_metrics['f1']:.3f}")
        if show_truth:
            st.markdown(f"**Истинные источники:** {_ids(result.scenario.cascade.sources)}")
        st.caption("Обе версии используют одно предсказанное число источников; Temporal-v3 меняет только порядок кандидатов по раннему наблюдению.")
        st.markdown("#### Узел")
        choices = [None, *view.node_ids]
        if st.session_state.get("selected_node") not in choices:
            st.session_state["selected_node"] = None
        chosen = st.selectbox("Выбрать узел по ID", choices, format_func=lambda n: "—" if n is None else str(n), key="selected_node")
        if chosen is not None:
            status = "наблюдён к t = 3" if chosen in result.scenario.final.observed_infected else "не наблюдён к t = 3"
            early = "наблюдён к t = 1" if chosen in result.scenario.early_nodes else "не наблюдён к t = 1"
            st.write(f"Узел {chosen} · {early} · {status} · сырой GCN score {float(result.snapshot.scores[chosen]):.4f}")
            if show_truth and chosen in result.scenario.cascade.sources:
                st.write("Истинный источник")
        st.markdown("#### Кандидаты")
        st.dataframe(_candidate_rows(result, view_method), width="stretch", hide_index=True, height=300)
        st.caption("Ранг показан для выбранного метода; исходный GCN score одинаков в обоих методах.")


def main() -> None:
    st.set_page_config(page_title="Локализация источников", layout="wide", initial_sidebar_state="auto")
    st.markdown("<style>div.block-container{max-width:1500px;padding-top:2rem}div[data-testid='stMetric']{border:1px solid rgba(128,128,128,.25);border-radius:.65rem;padding:.8rem 1rem}</style>", unsafe_allow_html=True)
    st.title("Локализация источников распространения")
    st.caption("Temporal-v3 / S1b · фиксированный граф Facebook · локальная симуляция, не реальные наблюдения")
    with st.sidebar:
        st.header("Параметры каскада")
        true_k = st.selectbox("Истинное число источников", [1, 2, 3], index=1, key="true_k")
        probability = st.selectbox("Вероятность передачи", [.01, .02, .03], index=1,
                                   format_func=lambda value: f"{value:.2f}", key="probability")
        observation = st.selectbox("Доля наблюдаемых заражённых", [.5, .75, 1.0], index=1,
                                   format_func=lambda value: f"{value:.0%}", key="observation_fraction")
        seed = st.number_input("Seed", min_value=0, value=2026, step=1, key="seed")
        st.caption("Истинное число источников нужно симулятору; модель определяет его самостоятельно.")
        run_clicked = st.button("Рассчитать", type="primary", width="stretch", key="run_demo")
    if run_clicked:
        try:
            with st.spinner("Формируем каскад и вычисляем два прогноза…"):
                resources = load_resources()
                scenario = generate_demo_scenario(resources, DemoSettings(int(true_k), probability, observation, int(seed)))
                new_result = infer_demo(resources, scenario)
                new_view = build_demo_graph_view(new_result, selected_node=None)
            st.session_state["demo_result"] = new_result
            st.session_state["demo_view"] = new_view
            st.session_state["selected_node"] = None
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            st.error(f"Расчёт не завершён: {exc}. Проверьте локальный архив модели и параметры; предыдущий результат сохранён.")
    result = st.session_state.get("demo_result")
    if result is None:
        st.info("Задайте параметры слева и нажмите «Рассчитать». До этого приложение не запускает модель и не показывает метрики.")
        return
    _render_result(result)


if __name__ == "__main__":
    main()
