"""
Adapt ``Daf`` data to a ``computation``. See the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/adapters.html>`__ for details.

The Julia package has support for creating self-documenting computations (see the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/computations.html>`__ for details) which explicitly
declare a contract describing the inputs and outputs of the computation (see the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/contracts.html>`__ for details). The Python package
doesn't provide these utilities, since we'd like to promote implementing such computations in Julia, so they would be
efficient (in particular, effectively use parallelism), and be available to be called from Julia, Python (using this
package) and R (using the equivalent R package). That said, nothing prevents the creation of ``Daf`` computational
pipelines in Python or any other language, if one insists on doing so.
"""

from contextlib import contextmanager
from typing import Callable
from typing import Iterator
from typing import Optional

from .copies import EmptyData
from .copies import copy_all
from .data import DafWriter
from .formats import MemoryDaf
from .formats import chain_writer
from .views import ViewAxes
from .views import ViewData
from .views import viewer

__all__ = [
    "adapter",
]


@contextmanager
def adapter(
    daf: DafWriter,
    *,
    input_axes: Optional[ViewAxes] = None,
    input_data: Optional[ViewData] = None,
    capture: Callable[..., DafWriter] = MemoryDaf,
    output_axes: Optional[ViewAxes] = None,
    output_data: Optional[ViewData] = None,
    empty: Optional[EmptyData] = None,
    relayout: bool = True,
    overwrite: bool = False,
) -> Iterator[DafWriter]:
    """
    Invoke a computation on a view of some ``daf`` data set and return the result; copy a view of the results into the
    base ``daf`` data set. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/adapters.html#DataAxesFormats.Adapters.adapter>`__
    for details.
    """
    base_name = daf.name
    input_daf = viewer(daf, axes=input_axes, data=input_data, name=f"{base_name}.input")
    captured_daf = capture(name=f"{base_name}.capture")
    adapted_daf = chain_writer([input_daf, captured_daf], name=f"{base_name}.adapted")
    result = yield adapted_daf
    output_daf = viewer(adapted_daf, axes=output_axes, data=output_data, name=f"{base_name}.output")
    copy_all(source=output_daf, destination=daf, empty=empty, relayout=relayout, overwrite=overwrite)
    return result
