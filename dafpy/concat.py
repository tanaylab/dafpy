"""
Concatenate multiple ``Daf`` data sets along some axis. See the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/concat.html>`__ for details.
"""

from typing import AbstractSet
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Sequence

import numpy as np

from .copies import EmptyData
from .data import DafReader
from .data import DafWriter
from .data import PropertyKey
from .julia_import import _to_julia_array
from .julia_import import jl

__all__ = [
    "concatenate",
]

#: The action for merging the values of a property from the concatenated data sets into the result data set. See the
#: Julia
#: `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/concat.html#DataAxesFormats.Concats.MergeData>`__
#: for details.
MergeAction = Literal["SkipProperty"] | Literal["LastValue"] | Literal["CollectAxis"]

JL_MERGE_ACTION = {
    "SkipProperty": jl.DataAxesFormats.SkipProperty,
    "LastValue": jl.DataAxesFormats.LastValue,
    "CollectAxis": jl.DataAxesFormats.CollectAxis,
}

#: A mapping where the key is a ``PropertyKey`` and the value is ``MergeAction``. See the Julia
#: `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/concat.html#DataAxesFormats.Concats.MergeData>`__
#: for details.
#:
#: Note that in Python this is a dictionary and not a vector. This allows using the ``key: value`` notation,
#: and preserves the order of the entries since in Python dictionaries are ordered by default.
MergeData = Mapping[PropertyKey, MergeAction]


def concatenate(
    destination: DafWriter,
    axis: str | Sequence[str],
    sources: Sequence[DafReader],
    names: Optional[Sequence[str]] = None,
    dataset_axis: Optional[str] = "dataset",
    dataset_property: bool = True,
    prefix: bool | Sequence[bool] = False,
    prefixed: Optional[AbstractSet[str] | Sequence[AbstractSet[str]]] = None,
    empty: Optional[EmptyData] = None,
    sparse_if_saves_storage_fraction: float = 0.1,
    merge: Optional[MergeData] = None,
    overwrite: bool = False,
) -> None:
    """
    Concatenate data from a ``sources`` sequence of ``Daf`` data sets into a single ``destination`` data set along one
    or more concatenation ``axis``. See the Julia
    `documentation <DafAxesFormats://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/concatenate.html>`__ for details.
    """
    if merge is None:
        merge_data = None
    else:
        merge_data = jl._pairify_merge([(key, JL_MERGE_ACTION[value]) for key, value in merge.items()])

    jl.DataAxesFormats.concatenate_b(
        destination,
        _to_julia_array(axis),
        jl.pyconvert(jl._DafReadersVector, np.array(sources)),
        names=_to_julia_array(names),
        dataset_axis=dataset_axis,
        dataset_property=dataset_property,
        prefix=_to_julia_array(prefix),
        prefixed=_to_julia_array(prefixed),
        empty=empty,
        sparse_if_saves_storage_fraction=sparse_if_saves_storage_fraction,
        merge=merge_data,
        overwrite=overwrite,
    )
