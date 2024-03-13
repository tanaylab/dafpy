"""
Concrete formats of ``Daf`` data sets.
"""

from .data import DafWriter
from .julia_import import jl


class MemoryDaf(DafWriter):
    """
    Simple in-memory storage.
    """

    def __init__(self, *, name: str = "memory") -> None:
        super().__init__(jl.Daf.MemoryDaf(name=name))
