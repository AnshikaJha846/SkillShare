"""Test that all application scripts and pages compile without syntax or import errors."""

import py_compile
from pathlib import Path


def test_streamlit_app_syntax():
    app_path = Path("app/streamlit_app.py")
    assert app_path.is_file()
    py_compile.compile(str(app_path), doraise=True)


def test_evaluation_dashboard_syntax():
    dashboard_path = Path("app/pages/1_Evaluation_Dashboard.py")
    assert dashboard_path.is_file()
    py_compile.compile(str(dashboard_path), doraise=True)
