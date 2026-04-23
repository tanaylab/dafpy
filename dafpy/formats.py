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

__all__ = [
    "chain_reader",
    "chain_writer",
    "complete_daf",
    "files_daf",
    "files_to_zarr",
    "h5df",
    "http_daf",
    "memory_daf",
    "open_daf",
    "zarr_daf",
    "zarr_to_files",
]


# Wrap a Julia Daf object in the matching Python class: DafWriter for a Julia DafWriter subtype; DafReadOnly for
# anything else (a DafReadOnly subtype, or a bare DafReader such as HttpDaf).
def _wrap_daf(jl_obj) -> DafReadOnly | DafWriter:
    if jl.isa(jl_obj, jl.DataAxesFormats.DafWriter):
        return DafWriter(jl_obj)
    return DafReadOnly(jl_obj)


def complete_daf(path: str, mode: str = "r", *, name: Optional[str] = None) -> DafReadOnly | DafWriter:
    """
    Open a complete chain of ``Daf`` repositories by tracing back through the ``base_daf_repository``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/complete.html#DataAxesFormats.CompleteDaf.complete_daf>`__
    for details.
    """
    return _wrap_daf(jl.DataAxesFormats.complete_daf(path, mode, name=name))


def open_daf(path: str, mode: str = "r", *, name: Optional[str] = None) -> DafReadOnly | DafWriter:
    """
    Open a ``Daf`` data set, dispatching to the appropriate backend based on ``path``. Zarr suffixes (``.daf.zarr``,
    ``.daf.zarr.zip``, ``.dafs.zarr.zip#/...``) open a :py:func:`zarr_daf`; ``http://`` or ``https://`` URLs open a
    :py:func:`http_daf` (read-only); ``.h5df`` and ``.h5dfs#`` paths open an :py:func:`h5df`; anything else opens a
    :py:func:`files_daf`. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/complete.html#DataAxesFormats.CompleteDaf.open_daf>`__
    for details.
    """
    return _wrap_daf(jl.DataAxesFormats.open_daf(path, mode, name=name))


def memory_daf(jl_obj: Optional[jl.MemoryDaf] = None, *, name: str = "memory") -> DafWriter:
    """
    Simple in-memory storage. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/memory_format.html>`__ for details.
    """
    if jl_obj is None:
        jl_obj = jl.DataAxesFormats.MemoryDaf(name=name)
    return DafWriter(jl_obj)


def files_daf(path: str, mode: str = "r", *, name: Optional[str] = None) -> DafReadOnly | DafWriter:
    """
    A ``Daf`` storage format in disk files. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/files_format.html>`__ for details.
    """
    return _wrap_daf(jl.DataAxesFormats.FilesDaf(path, mode, name=name))


def h5df(
    root: Union[str, jl.HDF5.File, jl.HDF5.Group], mode: str = "r", *, name: Optional[str] = None
) -> DafReadOnly | DafWriter:
    """
    A ``Daf`` storage format in an HDF5 disk file. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/h5df_format.html>`__ for details.

    Note that if you want to open the ``HDF5`` file yourself (e.g., to access a specific group in it as a ``Daf`` data
    set), you will need to use the Julia API to do so, in order to pass the result here. That is, the current Python
    ``Daf`` API does **not** support using the Python ``HDF5`` API. This is because the ``Daf`` Python API is just a
    thin wrapper for the Julia ``Daf`` implementation, which doesn't "speak Python".
    """
    return _wrap_daf(jl.DataAxesFormats.H5df(root, mode, name=name))


def zarr_daf(path: str, mode: str = "r", *, name: Optional[str] = None) -> DafReadOnly | DafWriter:
    """
    A ``Daf`` storage format in a Zarr directory tree, Zarr ZIP archive, or remote HTTP(S) Zarr group. The ``path``
    follows one of these conventions: ``something.daf.zarr`` (directory), ``something.daf.zarr.zip`` (single-daf ZIP),
    ``something.dafs.zarr.zip#/group`` (sub-daf inside a multi-daf ZIP), or ``http(s)://...`` (remote zarr served over
    HTTP; read-only). See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/zarr_format.html>`__ for details.
    """
    return _wrap_daf(jl.DataAxesFormats.ZarrDaf(path, mode, name=name))


def http_daf(url: str, *, name: Optional[str] = None) -> DafReadOnly:
    """
    Read-only access to a :py:func:`files_daf` served over ``http://`` or ``https://``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/http_format.html>`__ for details.
    """
    return DafReadOnly(jl.DataAxesFormats.HttpDaf(url, name=name))


def files_to_zarr(*, files_path: str, zarr_path: str) -> None:
    """
    Hard-link convert a :py:func:`files_daf` directory at ``files_path`` into an equivalent :py:func:`zarr_daf`
    directory at ``zarr_path``. The ``zarr_path`` must not already exist, its name must end with ``.daf.zarr``, and the
    two paths must live on the same filesystem. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/zarr_convert.html#DataAxesFormats.ZarrConvert.files_to_zarr>`__
    for details.
    """
    jl.DataAxesFormats.files_to_zarr(files_path=files_path, zarr_path=zarr_path)


def zarr_to_files(*, zarr_path: str, files_path: str) -> None:
    """
    Hard-link convert a :py:func:`zarr_daf` directory at ``zarr_path`` into an equivalent :py:func:`files_daf`
    directory at ``files_path``. The ``zarr_path`` must be a ``.daf.zarr`` directory (ZIP and HTTP Zarr backends are
    rejected); the ``files_path`` must not already exist, and the two paths must live on the same filesystem. See the
    Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/zarr_convert.html#DataAxesFormats.ZarrConvert.zarr_to_files>`__
    for details.
    """
    jl.DataAxesFormats.zarr_to_files(zarr_path=zarr_path, files_path=files_path)


def chain_reader(dsets: Sequence[DafReader], *, name: Optional[str] = None) -> DafReadOnly:
    """
    Create a read-only chain wrapper of ``DafReader``, presenting them as a single ``DafReader``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/chains.html#DataAxesFormats.Chains.chain_reader>`__ for
    details.
    """
    return DafReadOnly(jl.chain_reader(jl._to_daf_readers([dset.jl_obj for dset in dsets]), name=name))


def chain_writer(dsets: Sequence[DafReader], *, name: Optional[str] = None) -> DafWriter:
    """
    Create a chain wrapper for a chain of ``DafReader`` data, presenting them as a single ``DafWriter``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/chains.html#DataAxesFormats.Chains.chain_writer>`__ for
    details.
    """
    return DafWriter(jl.chain_writer(jl._to_daf_readers([dset.jl_obj for dset in dsets]), name=name))
