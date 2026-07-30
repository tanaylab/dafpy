"""
Concatenate multiple ``Daf`` data sets along some axis. See the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/concat.html>`__ for details.
"""

# The enum values are named exactly as they are in Julia, so they are not UPPER_CASE.
# pylint: disable=invalid-name


from typing import AbstractSet
from typing import Mapping
from typing import Optional
from typing import Sequence

import numpy as np

from .copies import EmptyData
from .data import DafReader
from .data import DafWriter
from .data import PropertyKey
from .julia_import import JlEnum
from .julia_import import _given
from .julia_import import _to_julia_array
from .julia_import import jl

__all__ = [
    "concatenate",
    "MergeAction",
]


class MergeAction(JlEnum):
    """
    The action for merging the values of a property from the concatenated data sets into the result data set. See the
    Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/concat.html#DataAxesFormats.Concat.MergeAction>`__
    for details.
    """

    #: Do not include the property in the result.
    SkipProperty = "SkipProperty"
    #: Use the value from the last concatenated data set.
    LastValue = "LastValue"
    #: Collect the values along the concatenation axis.
    CollectAxis = "CollectAxis"


JL_MERGE_ACTION = {
    MergeAction.SkipProperty: jl.DataAxesFormats.SkipProperty,
    MergeAction.LastValue: jl.DataAxesFormats.LastValue,
    MergeAction.CollectAxis: jl.DataAxesFormats.CollectAxis,
}

#: A mapping where the key is a ``PropertyKey`` and the value is ``MergeAction``. See the Julia
#: `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/concat.html#DataAxesFormats.Concat.MergeData>`__
#: for details.
#:
#: Note that in Python this is a dictionary and not a vector. This allows using the ``key: value`` notation,
#: and preserves the order of the entries since in Python dictionaries are ordered by default.
MergeData = Mapping[PropertyKey, MergeAction]


def concatenate(  # pylint: disable=too-many-positional-arguments
    destination: DafWriter,
    axis: str | Sequence[str],
    sources: Sequence[DafReader],
    names: Optional[Sequence[str]] = None,
    # This is the one parameter whose Julia default isn't ``nothing`` even though it accepts ``nothing`` (to not add
    # a dataset axis at all), so the default has to be restated here and passed unconditionally.
    dataset_axis: Optional[str] = "dataset",
    dataset_property: Optional[bool] = None,
    prefix: Optional[bool | Sequence[bool]] = None,
    prefixed: Optional[AbstractSet[str] | Sequence[AbstractSet[str]]] = None,
    empty: Optional[EmptyData] = None,
    sparse_if_saves_storage_fraction: Optional[float] = None,
    merge: Optional[MergeData] = None,
    overwrite: Optional[bool] = None,
) -> None:
    """
    Concatenate data from a ``sources`` sequence of ``Daf`` data sets into a single ``destination`` data set along one
    or more concatenation ``axis``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/concat.html#DataAxesFormats.Concat.concatenate!>`__
    for details.
    """
    if merge is None:
        merge_data = None
    else:
        merge_data = jl.DafPy._pairify_merge([(key, JL_MERGE_ACTION[value]) for key, value in merge.items()])

    jl.DataAxesFormats.concatenate_b(
        destination,
        _to_julia_array(axis),
        jl.pyconvert(jl.DafPy._DafReadersVector, np.array(sources)),
        dataset_axis=dataset_axis,
        **_given(
            names=_to_julia_array(names),
            dataset_property=dataset_property,
            prefix=_to_julia_array(prefix),
            prefixed=_to_julia_array(prefixed),
            empty=empty,
            sparse_if_saves_storage_fraction=sparse_if_saves_storage_fraction,
            merge=merge_data,
            overwrite=overwrite,
        ),
    )
