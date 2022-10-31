from gptools.util.testing import discover_myst_notebooks, run_myst_notebook
from gptools import util
import pytest


@pytest.mark.parametrize("notebook", discover_myst_notebooks(util))
def test_notebook(notebook: str) -> None:
    run_myst_notebook(notebook)
