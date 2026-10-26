"""
Copy data between ``Daf`` data sets. See the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/copies.html>`__ for details.
"""

from typing import Mapping
from typing import Type

from .data import DafReader
from .data import DafWriter
from .data import DataKey
from .julia_import import DEFAULT
from .julia_import import DefaultValue
from .julia_import import Undef
from .julia_import import UndefInitializer
from .julia_import import _given
from .julia_import import _to_julia_array
from .julia_import import _to_julia_type
from .julia_import import jl
from .storage_types import StorageScalar

__all__ = [
    "copy_all",
    "DataTypes",
    "EmptyData",
    "copy_scalar",
    "copy_axis",
    "copy_vector",
    "copy_matrix",
    "copy_tensor",
]


def copy_scalar(
    *,
    destination: DafWriter,
    source: DafReader,
    name: str,
    rename: str | None | DefaultValue = DEFAULT,
    type: Type | None | DefaultValue = DEFAULT,  # pylint: disable=redefined-builtin
    default: StorageScalar | UndefInitializer | None = Undef,
    overwrite: bool | DefaultValue = DEFAULT,
    insist: bool | DefaultValue = DEFAULT,
) -> None:
    """
    Copy a scalar with some ``name`` from some ``source`` ``DafReader`` into some ``destination`` ``DafWriter``. See the
    Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/copies.html#DataAxesFormats.Copies.copy_scalar!>`__
    for details.
    """
    jl.DataAxesFormats.copy_scalar_b(
        destination=destination,
        source=source,
        name=name,
        default=_to_julia_array(default),
        **_given(rename=rename, type=_to_julia_type(type), overwrite=overwrite, insist=insist),
    )


def copy_axis(
    *,
    destination: DafWriter,
    source: DafReader,
    axis: str,
    rename: str | None | DefaultValue = DEFAULT,
    default: StorageScalar | UndefInitializer | None = Undef,
    overwrite: bool | DefaultValue = DEFAULT,
    insist: bool | DefaultValue = DEFAULT,
) -> None:
    """
    Copy an axis from some ``source`` ``DafReader`` into some ``destination`` ``DafWriter``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/copies.html#DataAxesFormats.Copies.copy_axis!>`__
    for details.
    """
    jl.DataAxesFormats.copy_axis_b(
        destination=destination,
        source=source,
        axis=axis,
        default=_to_julia_array(default),
        **_given(rename=rename, overwrite=overwrite, insist=insist),
    )


def copy_vector(
    *,
    destination: DafWriter,
    source: DafReader,
    axis: str,
    name: str,
    reaxis: str | None | DefaultValue = DEFAULT,
    rename: str | None | DefaultValue = DEFAULT,
    eltype: Type | None | DefaultValue = DEFAULT,
    default: StorageScalar | UndefInitializer | None = Undef,
    empty: StorageScalar | None | DefaultValue = DEFAULT,
    bestify: bool | DefaultValue = DEFAULT,
    min_sparse_saving_fraction: float | DefaultValue = DEFAULT,
    overwrite: bool | DefaultValue = DEFAULT,
    insist: bool | DefaultValue = DEFAULT,
    packed: bool | None | DefaultValue = DEFAULT,
) -> None:
    """
    Copy a vector from some ``source`` ``DafReader`` into some ``destination`` ``DafWriter``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/copies.html#DataAxesFormats.Copies.copy_vector!>`__
    for details.
    """
    jl.DataAxesFormats.copy_vector_b(
        destination=destination,
        source=source,
        axis=axis,
        name=name,
        default=_to_julia_array(default),
        **_given(
            reaxis=reaxis,
            rename=rename,
            eltype=_to_julia_type(eltype),
            empty=empty,
            bestify=bestify,
            min_sparse_saving_fraction=min_sparse_saving_fraction,
            overwrite=overwrite,
            insist=insist,
            packed=packed,
        ),
    )


def copy_matrix(  # pylint: disable=too-many-locals
    *,
    destination: DafWriter,
    source: DafReader,
    rows_axis: str,
    columns_axis: str,
    name: str,
    rows_reaxis: str | None | DefaultValue = DEFAULT,
    columns_reaxis: str | None | DefaultValue = DEFAULT,
    rename: str | None | DefaultValue = DEFAULT,
    eltype: Type | None | DefaultValue = DEFAULT,
    default: StorageScalar | UndefInitializer | None = Undef,
    empty: StorageScalar | None | DefaultValue = DEFAULT,
    bestify: bool | DefaultValue = DEFAULT,
    min_sparse_saving_fraction: float | DefaultValue = DEFAULT,
    relayout: bool | DefaultValue = DEFAULT,
    overwrite: bool | DefaultValue = DEFAULT,
    insist: bool | DefaultValue = DEFAULT,
    packed: bool | None | DefaultValue = DEFAULT,
) -> None:
    """
    Copy a matrix from some ``source`` ``DafReader`` into some ``destination`` ``DafWriter``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/copies.html#DataAxesFormats.Copies.copy_matrix!>`__
    for details.
    """
    jl.DataAxesFormats.copy_matrix_b(
        destination=destination,
        source=source,
        rows_axis=rows_axis,
        columns_axis=columns_axis,
        name=name,
        default=_to_julia_array(default),
        **_given(
            rows_reaxis=rows_reaxis,
            columns_reaxis=columns_reaxis,
            rename=rename,
            eltype=_to_julia_type(eltype),
            empty=empty,
            bestify=bestify,
            min_sparse_saving_fraction=min_sparse_saving_fraction,
            relayout=relayout,
            overwrite=overwrite,
            insist=insist,
            packed=packed,
        ),
    )


def copy_tensor(  # pylint: disable=too-many-locals
    *,
    destination: DafWriter,
    source: DafReader,
    main_axis: str,
    rows_axis: str,
    columns_axis: str,
    name: str,
    rows_reaxis: str | None | DefaultValue = DEFAULT,
    columns_reaxis: str | None | DefaultValue = DEFAULT,
    rename: str | None | DefaultValue = DEFAULT,
    eltype: Type | None | DefaultValue = DEFAULT,
    empty: StorageScalar | None | DefaultValue = DEFAULT,
    bestify: bool | DefaultValue = DEFAULT,
    min_sparse_saving_fraction: float | DefaultValue = DEFAULT,
    relayout: bool | DefaultValue = DEFAULT,
    overwrite: bool | DefaultValue = DEFAULT,
    insist: bool | DefaultValue = DEFAULT,
    packed: bool | None | DefaultValue = DEFAULT,
) -> None:
    """
    Copy a tensor from some ``source`` ``DafReader`` into some ``destination`` ``DafWriter``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/copies.html#DataAxesFormats.Copies.copy_tensor!>`__
    for details.
    """
    jl.DataAxesFormats.copy_tensor_b(
        destination=destination,
        source=source,
        main_axis=main_axis,
        rows_axis=rows_axis,
        columns_axis=columns_axis,
        name=name,
        **_given(
            rows_reaxis=rows_reaxis,
            columns_reaxis=columns_reaxis,
            rename=rename,
            eltype=_to_julia_type(eltype),
            empty=empty,
            bestify=bestify,
            min_sparse_saving_fraction=min_sparse_saving_fraction,
            relayout=relayout,
            overwrite=overwrite,
            insist=insist,
            packed=packed,
        ),
    )


#: Specify the data to use for missing properties in a ``Daf`` data set. See the Julia
#: `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/copies.html#DataAxesFormats.Copies.EmptyData>`__
#: for details.
EmptyData = Mapping[DataKey, StorageScalar]

#: Specify the data type to use for overriding properties types in a ``Daf`` data set. See the Julia
#: `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/copies.html#DataAxesFormats.Copies.DataTypes>`__
#: for details.
DataTypes = Mapping[DataKey, Type]


def copy_all(
    *,
    destination: DafWriter,
    source: DafReader,
    empty: EmptyData | None | DefaultValue = DEFAULT,
    types: DataTypes | None | DefaultValue = DEFAULT,
    overwrite: bool | DefaultValue = DEFAULT,
    insist: bool | DefaultValue = DEFAULT,
    relayout: bool | DefaultValue = DEFAULT,
    packed: bool | None | DefaultValue = DEFAULT,
) -> None:
    """
    Copy all the content of a ``source`` ``DafReader`` into a ``destination`` ``DafWriter``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/copies.html#DataAxesFormats.Copies.copy_all!>`__
    for details.
    """
    if isinstance(types, Mapping):
        types = {key: _to_julia_type(value) for key, value in types.items()}
    jl.DataAxesFormats.copy_all_b(
        destination=destination,
        source=source,
        **_given(empty=empty, types=types, overwrite=overwrite, insist=insist, relayout=relayout, packed=packed),
    )
