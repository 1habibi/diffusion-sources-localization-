from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_with_pilot_artifacts():
    app_path = Path(__file__).resolve().parents[2] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=10).run()
    assert not app.exception
    assert app.title[0].value == "Localization of multiple diffusion sources"
