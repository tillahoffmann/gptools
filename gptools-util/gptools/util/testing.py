from myst_parser.config.main import MdParserConfig
from myst_nb.core.config import NbParserConfig
from myst_nb.core.read import create_nb_reader, NbReader
from nbconvert.preprocessors import ExecutePreprocessor
import numpy as np
import pathlib
import pytest
from typing import Any, Iterable, Type
from .kernels import ExpQuadKernel, HeatKernel, Kernel
from . import coordgrid, ArrayOrTensor


class KernelConfiguration:
    """
    Kernel configuration for testing.

    Args:
        dims: Sequence of dimension domains. `None` indicates an unbounded domain.
        kernel_cls: Class of the kernel to create.
        **kwargs: Keyword arguments passed to the kernel.
    """
    def __init__(self, dims: tuple, kernel_cls: Type[Kernel], **kwargs) -> None:
        self.dims = dims
        self.kernel_cls = kernel_cls
        self.kwargs = kwargs
        if all(self.dims):
            self.kwargs.setdefault("period", self.dims)

    def __call__(self) -> Kernel:
        return self.kernel_cls(**self.kwargs)

    def sample_locations(self, size: tuple = None) -> ArrayOrTensor:
        """
        Sample locations consistent with the domain on which to apply the kernel.
        """
        locations = []
        for dim in self.dims:
            if dim is None:
                locations.append(np.random.normal(0, 1, size))
            else:
                domain = dim
                locations.append(np.random.uniform(0, domain, size))
        locations = np.asarray(locations)
        return np.moveaxis(locations, 0, -1)

    def coordgrid(self, shape) -> ArrayOrTensor:
        """
        Create a coordinate grid on which to evaluate the kernel.
        """
        lins = []
        for dim, n in zip(self.dims, shape):
            if dim is None:  # pragma: no cover
                raise ValueError("cannot create coordinate grid for unbounded support")
            lins.append(np.linspace(0, dim, n, endpoint=False))
        return coordgrid(*lins)


_kernel_configurations = [
    KernelConfiguration([None], ExpQuadKernel, sigma=1.3, length_scale=0.2),
    KernelConfiguration([None, None, None], ExpQuadKernel, sigma=4, length_scale=0.2),
    KernelConfiguration([2, 3], ExpQuadKernel, sigma=1.7, length_scale=0.3),
    KernelConfiguration([1.5], ExpQuadKernel, sigma=1.5, length_scale=0.1),
    KernelConfiguration([2, 3, 4], ExpQuadKernel, sigma=2.1,
                        length_scale=np.asarray([0.1, 0.15, 0.2])),
    KernelConfiguration(np.asarray([1.7]), HeatKernel, sigma=2.5, length_scale=0.2),
    KernelConfiguration(np.asarray([1.7, 2.3]), HeatKernel, sigma=2.5,
                        length_scale=np.asarray([0.2, 0.3])),
]


@pytest.fixture(params=_kernel_configurations)
def kernel_configuration(request: pytest.FixtureRequest) -> KernelConfiguration:
    return request.param


def run_myst_notebook(path: str, timeout: float = 60) -> Any:
    """
    Run a myst example notebook.
    """
    md_config = MdParserConfig()
    nb_config = NbParserConfig()
    with open(path) as fp:
        content = fp.read()
    reader: NbReader = create_nb_reader(path, md_config, nb_config, content)
    notebook = reader.read(content)
    preprocessor = ExecutePreprocessor(timeout=timeout)
    return preprocessor.preprocess(notebook, {"metadata": {"path": pathlib.Path(path).parent}})


def discover_myst_notebooks(package) -> Iterable[str]:
    """
    Discover example notebooks within a package.
    """
    return [
        str(path.relative_to(pathlib.Path.cwd())) for path in
        pathlib.Path(package.__file__).parent.parent.parent.glob("**/*.md")
        if ".ipynb_checkpoints" not in path.parts
    ]
