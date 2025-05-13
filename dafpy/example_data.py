"""
Example data. See the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/example_data.html>`__ for details.
"""

from .data import DafWriter
from .julia_import import jl

__all__ = [
    "example_cells_daf",
    "example_metacells_daf",
    "example_chain_daf",
]


def example_cells_daf(
    *,
    name: str = "cells!",
) -> DafWriter:
    """
    Load the cells example data into a ``MemoryDaf``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/example_data.html#DataAxesFormats.ExampleData.example_cells_daf>`__
    for details.
    """
    jl_obj = jl.DataAxesFormats.example_cells_daf(name=name)
    return DafWriter(jl_obj)


def example_metacells_daf(
    *,
    name: str = "metacells!",
) -> DafWriter:
    """
    Load the metacells example data into a ``MemoryDaf``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/example_data.html#DataAxesFormats.ExampleData.example_metacells_daf>`__
    for details.
    """
    jl_obj = jl.DataAxesFormats.example_metacells_daf(name=name)
    return DafWriter(jl_obj)


def example_chain_daf(
    *,
    name: str = "chain!",
) -> DafWriter:
    """
    Load a chain of both the cells and metacells example data. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/example_data.html#DataAxesFormats.ExampleData.example_chain_daf>`__
    for details.
    """
    jl_obj = jl.DataAxesFormats.example_chain_daf(name=name)
    return DafWriter(jl_obj)
