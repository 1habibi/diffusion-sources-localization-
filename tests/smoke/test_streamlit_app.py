from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_running_inference():
    app_path = Path(__file__).resolve().parents[2] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=10).run()
    assert not app.exception
    assert app.title[0].value == "Локализация источников распространения"
    assert not app.metric


def test_readme_streamlit_instructions_match_frozen_demo():
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Streamlit", 1)[1].split("## Google Colab", 1)[0]
    assert "reports/backups/temporal_v3_20260929" in section
    assert "python -m streamlit run app.py" in section
    assert "data/generated/pilot" not in section
