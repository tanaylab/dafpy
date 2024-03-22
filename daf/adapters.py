"""
Adapt ``Daf`` data to a ``computation``. See the Julia
`documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/adapters.html>`__ for details.

The Julia package has support for creating self-documenting computations (see the Julia
`documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/computations.html>`__ for details) which explicitly declare a
contract describing the inputs and outputs of the computation (see the Julia
`documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/contracts.html>`__ for details). The Python package doesn't
provide these utilities, since we'd like to promote implementing such computations in Julia, so they would be
efficient (in particular, effectively use parallelism), and be available to be called from Julia, Python (using this
package) and R (using the equivalent R package). That said, nothing prevents the creation of ``Daf`` computational
pipelines in Python or any other language, if one insists on doing so.
"""

from contextlib import contextmanager
from typing import Callable
from typing import Iterator
from typing import Optional

from .copies import EmptyData
from .data import DafReadOnly
from .data import DafWriter
from .formats import MemoryDaf
from .julia_import import _jl_pairs
from .julia_import import jl
from .views import ViewAxes
from .views import ViewData

__all__ = [
    "daf_adapter",
]


@contextmanager
def daf_adapter(
    view: DafWriter | DafReadOnly,
    name: Optional[str] = None,
    capture: Callable[..., DafWriter] = MemoryDaf,
    axes: Optional[ViewAxes] = None,
    data: Optional[ViewData] = None,
    empty: Optional[EmptyData] = None,
    relayout: bool = True,
    overwrite: bool = False,
) -> Iterator[DafWriter]:
    """
    Invoke a computation on a ``view`` data set; copy a ``daf_view`` of the updated data set into the base ``Daf`` data
    set of the view. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/adapters.html#Daf.Adapters.daf_adapter>`__ for details.
    """
    writer = capture(name=jl.Daf.Adapters.get_adapter_capture_name(view, name=name))
    adapted = jl.Daf.Adapters.get_adapter_input(view, name=name, writer=writer)
    yield DafWriter(adapted)
    jl.Daf.Adapters.copy_adapter_output(
        view,
        adapted,
        name=name,
        axes=axes,
        data=jl._pairify_data(_jl_pairs(data)),
        empty=empty,
        relayout=relayout,
        overwrite=overwrite,
    )
