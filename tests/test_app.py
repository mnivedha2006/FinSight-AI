"""Opens the real Streamlit app headlessly and visits every page and theme.

Needs streamlit and plotly installed (they are, in your venv).
Run:  python -m pytest tests/test_app.py -q --tb=short
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

# AppTest resolves a relative path against THIS file's folder (tests/), so give it the absolute path.
APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["Overview", "Check a message", "Investigate a transaction", "Counterfactual lab",
         "Spending insights", "Beneficiary network", "Model results"]


def _errors(at):
    return [str(e.value)[:300] for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(page):
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not _errors(at), _errors(at)
    at.sidebar.radio[0].set_value(page).run()
    assert not _errors(at), f"{page}: {_errors(at)}"


@pytest.mark.parametrize("theme", ["Daybreak", "Midnight", "Monsoon"])
def test_every_theme_renders(theme):
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.sidebar.selectbox[0].set_value(theme).run()
    assert not _errors(at), f"{theme}: {_errors(at)}"