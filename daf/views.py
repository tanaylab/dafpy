"""
Create a different view of a ``Daf`` data set using queries. See the Julia
`documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/views.html>`__ for details.
"""

from typing import Mapping
from typing import Optional

from .data import DafReader
from .formats import ReadOnlyView
from .julia_import import _jl_pairs
from .julia_import import _to_julia
from .julia_import import jl

__all__ = [
    "ALL_AXES",
    "ALL_MATRICES",
    "ALL_SCALARS",
    "ALL_VECTORS",
    "DafView",
    "viewer",
]

"""
A key to use in the ``data`` parameter of ``viewer`` to specify all the base data scalars. See the Julia
`documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/views.html#Daf.Views.ALL_SCALARS>`__ for details.
"""
ALL_SCALARS = "*"

"""
A pair to use in the ``axes`` parameter of ``viewer`` to specify all the base data axes. See the Julia
`documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/views.html#Daf.Views.ALL_AXES>`__ for details.
"""
ALL_AXES = "*"

"""
A key to use in the ``data`` parameter of ``viewer`` to specify all the vectors of the exposed axes. See the Julia
`documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/views.html#Daf.Views.ALL_VECTORS>`__ for details.
"""
ALL_VECTORS = ("*", "*")


"""
A key to use in the ``data`` parameter of ``viewer`` to specify all the matrices of the exposed axes. See the Julia
`documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/views.html#Daf.Views.ALL_MATRICES>`__ for details.
"""
ALL_MATRICES = ("*", "*", "*")


class DafView(DafReader):
    """
    A read-only wrapper for any ``DafReader`` data, which exposes an arbitrary view of it as another ``DafReader``. This
    isn't typically created manually; instead call ``viewer``. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/views.html#Daf.Views.DafView>`__ for details.
    """


def viewer(
    dset: DafReader, *, name: Optional[str] = None, axes: Optional[Mapping] = None, data: Optional[Mapping] = None
) -> DafView | ReadOnlyView:
    """
    Wrap ``Daf`` data set with a read-only ``DafView``. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/views.html#Daf.Views.viewer>`__ for details.

    The order of the axes and data matters. Luckily, the default dictionary type is ordered in modern Python, write
    ``axes = {ALL_AXES: None, "cell": "obs"}`` you can trust that the ``cell`` axis will be exposed as ``obs`` (and
    similarly for ``data``).
    """
    return DafView(
        jl.Daf.viewer(
            _to_julia(dset), name=name, axes=jl._pairify_axes(_jl_pairs(axes)), data=jl._pairify_data(_jl_pairs(data))
        )
    )
