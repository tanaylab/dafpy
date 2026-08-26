"""
Reorder the entries of ``Daf`` axes. See the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/reorder.html>`__ for details.
"""

from typing import Any
from typing import Mapping
from typing import Sequence
from typing import Union

import numpy as np

from .data import DafWriter
from .julia_import import _jl_pairs
from .julia_import import jl

__all__ = ["reorder_axes", "reset_reorder_axes"]


def _to_julia_writers(daf: Union[DafWriter, Sequence[DafWriter]]) -> Any:
    if isinstance(daf, DafWriter):
        daf = [daf]
    return jl.DafPy._to_daf_writers([writer.jl_obj for writer in daf])


def reorder_axes(daf: Union[DafWriter, Sequence[DafWriter]], axes_permutations: Mapping[str, Sequence[int]]) -> None:
    """
    Reorder the entries of one or more axes, in one or more leaf ``Daf`` repositories. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/reorder.html#DataAxesFormats.Reorder.reorder_axes!>`__
    for details.

    The permutations passed here are 0-based to fit the Python conventions.
    """
    jl.DataAxesFormats.reorder_axes_b(
        _to_julia_writers(daf),
        jl.DafPy._pairify_permutations(
            _jl_pairs(
                {name: np.asarray(permutation, dtype=np.int64) + 1 for name, permutation in axes_permutations.items()}
            )
        ),
    )


def reset_reorder_axes(daf: Union[DafWriter, Sequence[DafWriter]]) -> bool:
    """
    Roll back an interrupted :py:obj:`reorder_axes` of one or more leaf ``Daf`` repositories, and return whether any of
    them had one to roll back. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/reorder.html#DataAxesFormats.Reorder.reset_reorder_axes!>`__
    for details.
    """
    return bool(jl.DataAxesFormats.reset_reorder_axes_b(_to_julia_writers(daf)))
