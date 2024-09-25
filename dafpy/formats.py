"""
Concrete formats of ``Daf`` data sets.
"""

from typing import Optional
from typing import Sequence
from typing import Union

from .data import DafReader
from .data import DafReadOnly
from .data import DafWriter
from .julia_import import jl

__all__ = ["MemoryDaf", "FilesDaf", "H5df", "chain_reader", "chain_writer"]


class MemoryDaf(DafWriter):
    """
    Simple in-memory storage. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/memory_format.html>`__ for details.
    """

    def __init__(self, jl_obj: Optional[jl.MemoryDaf] = None, *, name: str = "memory") -> None:
        if jl_obj is None:
            jl_obj = jl.DataAxesFormats.MemoryDaf(name=name)
        super().__init__(jl_obj)


class FilesDaf(DafWriter):
    """
    A ``Daf`` storage format in disk files. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/files_format.html>`__ for details.
    """

    def __init__(self, path: str, mode: str = "r", *, name: Optional[str] = None) -> None:
        super().__init__(jl.DataAxesFormats.FilesDaf(path, mode, name=name))


class H5df(DafWriter):
    """
    A ``Daf`` storage format in an HDF5 disk file. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/h5df_format.html>`__ for details.

    Note that if you want to open the ``HDF5`` file yourself (e.g., to access a specific group in it as a ``Daf`` data
    set), you will need to use the Julia API to do so, in order to pass the result here. That is, the current Python
    ``Daf`` API does **not** support using the Python ``HDF5`` API. This is because the ``Daf`` Python API is just a
    thin wrapper for the Julia ``Daf`` implementation, which doesn't "speak Python".
    """

    def __init__(
        self, root: Union[str, jl.HDF5.File, jl.HDF5.Group], mode: str = "r", *, name: Optional[str] = None
    ) -> None:
        super().__init__(jl.DataAxesFormats.H5df(root, mode, name=name))


def chain_reader(dsets: Sequence[DafReader], *, name: Optional[str] = None) -> DafReadOnly:
    """
    Create a read-only chain wrapper of ``DafReader``, presenting them as a single ``DafReader``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/chains.html#Daf.Chains.chain_reader>`__ for
    details.
    """
    return DafReadOnly(jl.chain_reader(jl._to_daf_readers([dset.jl_obj for dset in dsets]), name=name))


def chain_writer(dsets: Sequence[DafReader], *, name: Optional[str] = None) -> DafWriter:
    """
    Create a chain wrapper for a chain of ``DafReader`` data, presenting them as a single ``DafWriter``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/chains.html#Daf.Chains.chain_writer>`__ for
    details.
    """
    return DafWriter(jl.chain_writer(jl._to_daf_readers([dset.jl_obj for dset in dsets]), name=name))
