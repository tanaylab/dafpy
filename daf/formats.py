"""
Concrete formats of ``Daf`` data sets.
"""

from typing import Optional

from .data import DafWriter
from .julia_import import jl

__all__ = ["MemoryDaf", "FilesDaf"]


class MemoryDaf(DafWriter):
    """
    Simple in-memory storage. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/memory_format.html>`_ for details.
    """

    def __init__(self, *, name: str = "memory") -> None:
        super().__init__(jl.Daf.MemoryDaf(name=name))


class FilesDaf(DafWriter):
    """
    A ``Daf`` storage format in disk files. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/files_format.html>`_ for details.
    """

    def __init__(self, path: str, mode: str = "r", *, name: Optional[str] = None) -> None:
        super().__init__(jl.Daf.FilesDaf(path, mode, name=name))
