"""
Reconstruct implicit axes. See the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/reconstruction.html>`__
for details.
"""

__all__ = ["reconstruct_axis"]

from typing import AbstractSet
from typing import Collection
from typing import Mapping
from typing import Optional
from typing import Union

from .data import DafWriter
from .julia_import import _to_julia_scalar_or_collection
from .julia_import import _to_julia_strings_set
from .julia_import import jl
from .storage_types import StorageScalar


def reconstruct_axis(
    dset: DafWriter,
    *,
    existing_axis: str,
    implicit_axis: str,
    rename_axis: Optional[str] = None,
    empty_implicit: Optional[Union[StorageScalar, Collection[StorageScalar]]] = None,
    implicit_properties: Optional[AbstractSet[str]] = None,
    skipped_properties: Optional[AbstractSet[str]] = None,
) -> Mapping[str, Optional[StorageScalar]]:
    """
    Given an ``existing_axis`` in a ``Daf`` data set, which has a property ``implicit_axis``, create a new axis with the
    same name (or, if specified, call it ``rename_axis``). The ``empty_implicit`` value(s), meaning "there is no value",
    may be a single one or any collection of them, since data often spells "no value" in more than one way. See the
    Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/reconstruction.html#DataAxesFormats.Reconstruction.reconstruct_axis!>`__
    for details.
    """
    return jl.DataAxesFormats.reconstruct_axis_b(
        dset,
        existing_axis=existing_axis,
        implicit_axis=implicit_axis,
        rename_axis=rename_axis,
        empty_implicit=_to_julia_scalar_or_collection(empty_implicit),
        implicit_properties=_to_julia_strings_set(implicit_properties),
        skipped_properties=_to_julia_strings_set(skipped_properties),
    )
