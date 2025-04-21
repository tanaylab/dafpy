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

__all__ = ["memory_daf", "files_daf", "h5df", "chain_reader", "chain_writer"]


def memory_daf(jl_obj: Optional[jl.MemoryDaf] = None, *, name: str = "memory") -> DafWriter:
    """
    Simple in-memory storage. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/memory_format.html>`__ for details.
    """
    if jl_obj is None:
        jl_obj = jl.DataAxesFormats.MemoryDaf(name=name)
    return DafWriter(jl_obj)


def files_daf(path: str, mode: str = "r", *, name: Optional[str] = None) -> DafReadOnly | DafWriter:
    """
    A ``Daf`` storage format in disk files. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/files_format.html>`__ for details.
    """
    jl_obj = jl.DataAxesFormats.FilesDaf(path, mode, name=name)
    if mode == "r":
        return DafReadOnly(jl_obj)
    return DafWriter(jl_obj)


def h5df(
    root: Union[str, jl.HDF5.File, jl.HDF5.Group], mode: str = "r", *, name: Optional[str] = None
) -> DafReadOnly | DafWriter:
    """
    A ``Daf`` storage format in an HDF5 disk file. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/h5df_format.html>`__ for details.

    Note that if you want to open the ``HDF5`` file yourself (e.g., to access a specific group in it as a ``Daf`` data
    set), you will need to use the Julia API to do so, in order to pass the result here. That is, the current Python
    ``Daf`` API does **not** support using the Python ``HDF5`` API. This is because the ``Daf`` Python API is just a
    thin wrapper for the Julia ``Daf`` implementation, which doesn't "speak Python".
    """
    jl_obj = jl.DataAxesFormats.H5df(root, mode, name=name)
    if mode == "r":
        return DafReadOnly(jl_obj)
    return DafWriter(jl_obj)


def chain_reader(dsets: Sequence[DafReader], *, name: Optional[str] = None) -> DafReadOnly:
    """
    Create a read-only chain wrapper of ``DafReader``, presenting them as a single ``DafReader``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/chains.html#DataAxesFormats.Chains.chain_reader>`__ for
    details.
    """
    return DafReadOnly(jl.chain_reader(jl._to_daf_readers([dset.jl_obj for dset in dsets]), name=name))


def chain_writer(dsets: Sequence[DafReader], *, name: Optional[str] = None) -> DafWriter:
    """
    Create a chain wrapper for a chain of ``DafReader`` data, presenting them as a single ``DafWriter``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/chains.html#DataAxesFormats.Chains.chain_writer>`__ for
    details.
    """
    return DafWriter(jl.chain_writer(jl._to_daf_readers([dset.jl_obj for dset in dsets]), name=name))
