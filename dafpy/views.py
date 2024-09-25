"""
Create a different view of a ``Daf`` data set using queries. See the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/views.html>`__
for details.
"""

from typing import Mapping
from typing import Optional

from .data import DafReader
from .data import DafReadOnly
from .data import DataKey
from .julia_import import _jl_pairs
from .julia_import import jl
from .queries import Query

__all__ = [
    "viewer",
    "ViewAxes",
    "ViewData",
    "ALL_SCALARS",
    "ALL_AXES",
    "ALL_VECTORS",
    "ALL_MATRICES",
]

#: A key to use in the ``data`` parameter of ``viewer`` to specify all the base data scalars. See the Julia
#: `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/views.html#Daf.Views.ALL_SCALARS>`__
#: for details.
ALL_SCALARS = "*"

#: A pair to use in the ``axes`` parameter of ``viewer`` to specify all the base data axes. See the Julia
#: `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/views.html#Daf.Views.ALL_AXES>`__
#: for details.
ALL_AXES = "*"

#: A key to use in the ``data`` parameter of ``viewer`` to specify all the vectors of the exposed axes. See the Julia
#: `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/views.html#Daf.Views.ALL_VECTORS>`__
#: for details.
ALL_VECTORS = ("*", "*")

#: A key to use in the ``data`` parameter of ``viewer`` to specify all the matrices of the exposed axes. See the Julia
#: `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/views.html#Daf.Views.ALL_MATRICES>`__
#: for details.
ALL_MATRICES = ("*", "*", "*")

#: Specify axes to expose from a view. See the Julia
#: `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/views.html#Daf.Views.ViewAxes>`__
#: for details.
#:
#: Note that in Python this is a dictionary and not a vector. This allows using the ``key: value`` notation,
#: and preserves the order of the entries since in Python dictionaries are ordered by default.
ViewAxes = Mapping[str, str | Query | None]

#: Specify data to expose from view. See the Julia
#: `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/views.html#Daf.Views.ViewAxes>`__
#: for details.
#:
#: Note that in Python this is a dictionary and not a vector. This allows using the ``key: value`` notation,
#: and preserves the order of the entries since in Python dictionaries are ordered by default.
ViewData = Mapping[DataKey, str | Query | None]


def viewer(
    dset: DafReader, *, name: Optional[str] = None, axes: Optional[ViewAxes] = None, data: Optional[ViewData] = None
) -> DafReadOnly:
    """
    Wrap ``Daf`` data set with a read-only ``DafView``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/views.html#Daf.Views.viewer>`__
    for details.

    The order of the axes and data matters. Luckily, the default dictionary type is ordered in modern Python, write
    ``axes = {ALL_AXES: None, "cell": "obs"}`` you can trust that the ``cell`` axis will be exposed as ``obs`` (and
    similarly for ``data``).
    """
    return DafReadOnly(
        jl.DataAxesFormats.viewer(
            dset, name=name, axes=jl._pairify_axes(_jl_pairs(axes)), data=jl._pairify_data(_jl_pairs(data))
        )
    )
