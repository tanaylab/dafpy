"""
Reconstruct implicit axes. See the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/reconstruction.html>`__
for details.
"""

__all__ = ["connect_axes", "reconstruct_axis", "unify_empty_vector_values"]

from typing import AbstractSet
from typing import Collection
from typing import Mapping
from typing import Optional
from typing import Type
from typing import Union

from .data import DafWriter
from .julia_import import _given
from .julia_import import _to_julia_scalar_or_collection
from .julia_import import _to_julia_strings_set
from .julia_import import _to_julia_type
from .julia_import import jl
from .storage_types import StorageScalar


def reconstruct_axis(
    dset: DafWriter,
    *,
    existing_axis: str,
    implicit_axis: str,
    rename_axis: Optional[str] = None,
    implicit_properties: Optional[AbstractSet[str]] = None,
    skipped_properties: Optional[AbstractSet[str]] = None,
) -> Mapping[str, Optional[StorageScalar]]:
    """
    Given an ``existing_axis`` in a ``Daf`` data set, which has a property ``implicit_axis``, create a new axis with the
    same name (or, if specified, call it ``rename_axis``). An empty string means there is no value; data spelling that
    some other way should be passed through ``unify_empty_vector_values`` first. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/reconstruction.html#DataAxesFormats.Reconstruction.reconstruct_axis!>`__
    for details.
    """
    return jl.DataAxesFormats.reconstruct_axis_b(
        dset,
        existing_axis=existing_axis,
        implicit_axis=implicit_axis,
        rename_axis=rename_axis,
        implicit_properties=_to_julia_strings_set(implicit_properties),
        skipped_properties=_to_julia_strings_set(skipped_properties),
    )


def connect_axes(
    dset: DafWriter,
    *,
    base_axis: str,
    from_axis: str,
    from_property: Optional[str] = None,
    to_axis: str,
    to_property: Optional[str] = None,
    connect_property: Optional[str] = None,
    overwrite: Optional[bool] = None,
) -> None:
    """
    Given a ``base_axis`` with two vector properties, one holding a reference to ``from_axis`` and one to ``to_axis``,
    create a property of ``from_axis`` that references ``to_axis``. This is only possible if every entry of
    ``from_axis`` is always associated with a single entry of ``to_axis``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/reconstruction.html#DataAxesFormats.Reconstruction.connect_axes!>`__
    for details.
    """
    jl.DataAxesFormats.connect_axes_b(
        dset,
        base_axis=base_axis,
        from_axis=from_axis,
        from_property=from_property,
        to_axis=to_axis,
        to_property=to_property,
        connect_property=connect_property,
        **_given(overwrite=overwrite),
    )


def unify_empty_vector_values(
    dset: DafWriter,
    *,
    axis: str,
    property: str,  # pylint: disable=redefined-builtin
    empty_values: Union[StorageScalar, Collection[StorageScalar]],
    dtype: Optional[Type] = None,
    empty_value: Optional[StorageScalar] = None,
) -> None:
    """
    Replace every one of the ``empty_values`` of a ``property`` of an ``axis`` with a single ``empty_value``, so that
    "there is no value here" is spelled one way, converting the property to a ``dtype`` on the way if one is given. See
    the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/reconstruction.html#DataAxesFormats.Reconstruction.unify_empty_vector_values!>`__
    for details.
    """
    jl.DataAxesFormats.unify_empty_vector_values_b(
        dset,
        axis=axis,
        property=property,
        empty_values=_to_julia_scalar_or_collection(empty_values),
        **_given(dtype=_to_julia_type(dtype), empty_value=empty_value),
    )
