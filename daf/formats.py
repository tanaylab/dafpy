"""
Concrete formats of ``Daf`` data sets.
"""

from typing import Optional
from typing import Sequence
from typing import Union

from .data import DafReader
from .data import DafWriter
from .julia_import import jl

__all__ = ["MemoryDaf", "FilesDaf", "H5df", "read_only", "chain_reader", "chain_writer"]


class MemoryDaf(DafWriter):
    """
    Simple in-memory storage. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/memory_format.html>`__ for details.
    """

    def __init__(self, *, name: str = "memory") -> None:
        super().__init__(jl.Daf.MemoryDaf(name=name))


class FilesDaf(DafWriter):
    """
    A ``Daf`` storage format in disk files. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/files_format.html>`__ for details.
    """

    def __init__(self, path: str, mode: str = "r", *, name: Optional[str] = None) -> None:
        super().__init__(jl.Daf.FilesDaf(path, mode, name=name))


class H5df(DafWriter):
    """
    A ``Daf`` storage format in an HDF5 disk file. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/h5df_format.html>`__ for details.

    Note that if you want to open the ``HDF5`` file yourself (e.g., to access a specific group in it as a ``Daf`` data
    set), you will need to use the Julia API to do so, in order to pass the result here. That is, the current Python
    ``Daf`` API does **not** support using the Python ``HDF5`` API. This is because the ``Daf`` Python API is just a
    thin wrapper for the Julia ``Daf`` implementation, which doesn't "speak Python".
    """

    def __init__(
        self, root: Union[str, jl.HDF5.File, jl.HDF5.Group], mode: str = "r", *, name: Optional[str] = None
    ) -> None:
        super().__init__(jl.Daf.H5df(root, mode, name=name))


class ReadOnlyView(DafReader):
    """
    A wrapper for any ``DafWriter`` data, protecting it against accidental modification. This isn't typically created
    manually; instead call ``read_only``. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.ReadOnly.ReadOnlyView>`__ for details.
    """


def read_only(dset: DafReader, *, name: Optional[str] = None) -> ReadOnlyView:
    """
    Wrap a ``Daf`` data set with a ``ReadOnlyView`` to protect it against accidental modification. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.ReadOnly.read_only>`__ for details.
    """
    if isinstance(dset, ReadOnlyView):
        return dset
    return ReadOnlyView(jl.Daf.read_only(dset.jl_obj, name=name))


def chain_reader(dsets: Sequence[DafReader], *, name: Optional[str] = None) -> DafReader:
    """
    Create a read-only chain wrapper of ``DafReader``, presenting them as a single ``DafReader``. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/chains.html#Daf.Chains.chain_reader>`__ for details.
    """
    return DafReader(jl.chain_reader(jl._to_daf_readers([dset.jl_obj for dset in dsets]), name=name))


def chain_writer(dsets: Sequence[DafReader], *, name: Optional[str] = None) -> DafWriter:
    """
    Create a chain wrapper for a chain of ``DafReader`` data, presenting them as a single ``DafWriter``. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/chains.html#Daf.Chains.chain_writer>`__ for details.
    """
    return DafWriter(jl.chain_writer(jl._to_daf_readers([dset.jl_obj for dset in dsets]), name=name))
