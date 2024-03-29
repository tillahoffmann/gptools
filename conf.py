from multiproject.utils import get_project
from sphinx.application import Sphinx


master_doc = "README"
extensions = [
    "matplotlib.sphinxext.plot_directive",
    "multiproject",
    "myst_nb",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
]

napoleon_custom_sections = [("Returns", "params_style")]
plot_formats = [
    ("png", 144),
]
html_theme = "sphinx_rtd_theme"
html_sidebars = {}
exclude_patterns = ["docs/_build", "docs/jupyter_execute", ".pytest_cache", "playground", "figures",
                    "**.ipynb_checkpoints"]

# Configure autodoc to avoid excessively long fully-qualified names.
add_module_names = False
autodoc_typehints_format = "short"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

source_suffix = {
    ".rst": "restructuredtext",
}

nb_execution_mode = "cache"
nb_execution_timeout = 300  # Some examples take a while to compile and sample.
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]
myst_dmath_double_inline = True

mathjax3_config = {
    "tex": {
        "macros": {
            "BigO": r"\mathcal{O}",
            "braces": [r"\left\{#1\right\}", 1],
            "cov": r"\operatorname{cov}",
            "dist": r"\sim",
            "E": [r"\mathbb{E}\parenth{#1}", 1],
            "imag": r"\mathrm{i}",
            "mat": [r"\mathbf{#1}", 1],
            "parenth": [r"\left(#1\right)", 1],
            "pred": r"\mathcal{P}",
            "proba": [r"p\parenth{#1}", 1],
        }
    }
}

multiproject_projects = {
    project: {
        "use_config_file": False,
        "config": {
            "project": f"gptools-{project}",
        },
    } for project in ["stan", "torch", "util"]
}
current_project = get_project(multiproject_projects)

if current_project == "stan":
    import cmdstanpy
    import logging
    extensions.append("sphinxcontrib.stan")
    intersphinx_mapping["cmdstanpy"] = \
        (f"https://cmdstanpy.readthedocs.io/en/v{cmdstanpy.__version__}", None)

    cmdstanpy_logger = cmdstanpy.utils.get_logger()
    for handler in cmdstanpy_logger.handlers:
        handler.setLevel(logging.WARNING)
elif current_project == "torch":
    intersphinx_mapping["torch"] = ("https://pytorch.org/docs/stable/", None)
elif current_project == "util":
    pass
else:
    raise ValueError(current_project)


def setup(app: Sphinx) -> None:
    # Ignore .ipynb and .html files (cf. https://github.com/executablebooks/MyST-NB/issues/363).
    app.registry.source_suffix.pop(".ipynb", None)
    app.registry.source_suffix.pop(".html", None)
