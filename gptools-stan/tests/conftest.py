import numpy as np
import os


if "CI" in os.environ:
    np.random.seed(0)

# Include fixtures from the base package.
pytest_plugins = [
   "gptools.util.testing",
]
