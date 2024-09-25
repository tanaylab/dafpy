"""
Reconstruct implicit axes. See the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/reconstruction.html>`__
for details.
"""

__all__ = ["reconstruct_axis"]

from typing import AbstractSet
from typing import Mapping
from typing import Optional

from .data import DafWriter
from .julia_import import jl
from .storage_types import StorageScalar


def reconstruct_axis(
    dset: DafWriter,
    *,
    existing_axis: str,
    implicit_axis: str,
    rename_axis: Optional[str] = None,
    empty_implicit: Optional[StorageScalar] = None,
    implicit_properties: Optional[AbstractSet[str]] = None,
) -> Mapping[str, Optional[StorageScalar]]:
    """
    Given an ``existing_axis`` in a ``Daf`` data set, which has a property ``implicit_axis``, create a new axis with the
    same name (or, if specified, call it ``rename_axis``). See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/reconstruction.html#Daf.Reconstruction.reconstruct_axis!>`__
    for details.
    """
    return jl.reconstruct_axis_b(
        dset,
        existing_axis=existing_axis,
        implicit_axis=implicit_axis,
        rename_axis=rename_axis,
        empty_implicit=empty_implicit,
        implicit_properties=implicit_properties,
    )
