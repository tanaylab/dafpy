"""
Copy data between ``Daf`` data sets. See the Julia
`documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/copies.html>`__ for details.
"""

from typing import Mapping
from typing import Optional

from .data import DafReader
from .data import DafWriter
from .data import DataKey
from .julia_import import Undef
from .julia_import import UndefInitializer
from .julia_import import _to_julia
from .julia_import import jl
from .storage_types import StorageScalar

__all__ = [
    "copy_all",
    "EmptyData",
    "copy_scalar",
    "copy_axis",
    "copy_vector",
    "copy_matrix",
]


def copy_scalar(
    *,
    destination: DafWriter,
    source: DafReader,
    name: str,
    rename: Optional[str] = None,
    default: StorageScalar | UndefInitializer | None = Undef,
    overwrite: bool = False,
) -> None:
    """
    Copy a scalar with some ``name`` from some ``source`` ``DafReader`` into some ``destination`` ``DafWriter``. See the
    Julia `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/copies.html#Daf.Copies.copy_scalar!>`__ for details.
    """
    jl.Daf.copy_scalar_b(
        destination=destination.jl_obj,
        source=source.jl_obj,
        name=name,
        rename=rename,
        default=_to_julia(default),
        overwrite=overwrite,
    )


def copy_axis(
    *,
    destination: DafWriter,
    source: DafReader,
    axis: str,
    rename: Optional[str] = None,
    default: StorageScalar | UndefInitializer | None = Undef,
) -> None:
    """
    Copy an axis from some ``source`` ``DafReader`` into some ``destination`` ``DafWriter``. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/copies.html#Daf.Copies.copy_axis!>`__ for details.
    """
    jl.Daf.copy_axis_b(
        destination=destination.jl_obj, source=source.jl_obj, axis=axis, rename=rename, default=_to_julia(default)
    )


def copy_vector(
    *,
    destination: DafWriter,
    source: DafReader,
    axis: str,
    name: str,
    reaxis: Optional[str] = None,
    rename: Optional[str] = None,
    default: StorageScalar | UndefInitializer | None = Undef,
    empty: Optional[StorageScalar] = None,
    overwrite: bool = False,
) -> None:
    """
    Copy a vector from some ``source`` ``DafReader`` into some ``destination`` ``DafWriter``. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/copies.html#Daf.Copies.copy_vector!>`__ for details.
    """
    jl.Daf.copy_vector_b(
        destination=destination.jl_obj,
        source=source.jl_obj,
        axis=axis,
        name=name,
        reaxis=reaxis,
        rename=rename,
        default=_to_julia(default),
        empty=empty,
        overwrite=overwrite,
    )


def copy_matrix(
    *,
    destination: DafWriter,
    source: DafReader,
    rows_axis: str,
    columns_axis: str,
    name: str,
    rows_reaxis: Optional[str] = None,
    columns_reaxis: Optional[str] = None,
    rename: Optional[str] = None,
    default: StorageScalar | UndefInitializer | None = Undef,
    empty: Optional[StorageScalar] = None,
    relayout: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Copy a matrix from some ``source`` ``DafReader`` into some ``destination`` ``DafWriter``. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/copies.html#Daf.Copies.copy_matrix!>`__ for details.
    """
    jl.Daf.copy_matrix_b(
        destination=destination.jl_obj,
        source=source.jl_obj,
        rows_axis=rows_axis,
        columns_axis=columns_axis,
        name=name,
        rows_reaxis=rows_reaxis,
        columns_reaxis=columns_reaxis,
        rename=rename,
        default=_to_julia(default),
        empty=empty,
        relayout=relayout,
        overwrite=overwrite,
    )


#: Specify the data to use for missing properties in a ``Daf`` data set. See the Julia
#: `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/copies.html#Daf.Copies.EmptyData>`__ for details.
EmptyData = Mapping[DataKey, StorageScalar]


def copy_all(
    *,
    destination: DafWriter,
    source: DafReader,
    empty: Optional[EmptyData] = None,
    overwrite: bool = False,
    relayout: bool = True,
) -> None:
    """
    Copy all the content of a ``source`` ``DafReader`` into a ``destination`` ``DafWriter``. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/copies.html#Daf.Copies.copy_all!>`__ for details.
    """
    jl.Daf.copy_all_b(
        destination=destination.jl_obj, source=source.jl_obj, empty=empty, overwrite=overwrite, relayout=relayout
    )
