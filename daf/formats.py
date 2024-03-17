"""
Concrete formats of ``Daf`` data sets.
"""

from .data import DafWriter
from .julia_import import jl

__all__ = ["MemoryDaf"]


class MemoryDaf(DafWriter):
    """
    Simple in-memory storage. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/memory_format.html>`_ for details.
    """

    def __init__(self, *, name: str = "memory") -> None:
        super().__init__(jl.Daf.MemoryDaf(name=name))
