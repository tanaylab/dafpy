"""
Concrete formats of ``Daf`` data sets.
"""

from typing import Optional
from typing import Union

from .data import DafWriter
from .julia_import import jl

__all__ = ["MemoryDaf", "FilesDaf", "H5df"]


class MemoryDaf(DafWriter):
    """
    Simple in-memory storage. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/memory_format.html>`_ for details.
    """

    def __init__(self, *, name: str = "memory") -> None:
        super().__init__(jl.Daf.MemoryDaf(name=name))


class FilesDaf(DafWriter):
    """
    A ``Daf`` storage format in disk files. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/files_format.html>`_ for details.
    """

    def __init__(self, path: str, mode: str = "r", *, name: Optional[str] = None) -> None:
        super().__init__(jl.Daf.FilesDaf(path, mode, name=name))


class H5df(DafWriter):
    """
    A ``Daf`` storage format in an HDF5 disk file. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/hdf5_format.html>`_ for details.

    Note that if you want to open the ``HDF5`` file yourself (e.g., to access a specific group in it as a ``Daf`` data
    set), you will need to use the Julia API to do so, in order to pass the result here. That is, the current Python
    ``Daf`` API does **not** support using the Python ``HDF5`` API. This is because the ``Daf`` Python API is just a
    thin wrapper for the Julia ``Daf`` implementation, which doesn't "speak Python".
    """

    def __init__(
        self, root: Union[str, jl.HDF5.File, jl.HDF5.Group], mode: str = "r", *, name: Optional[str] = None
    ) -> None:
        super().__init__(jl.Daf.H5df(root, mode, name=name))
