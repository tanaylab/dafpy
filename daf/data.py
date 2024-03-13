"""
Interface of ``DafReader`` and ``DafWriter``.

In Julia, the API is defined as a set of functions, which take the ``Daf`` object as the 1st parameter. In Python, this
is implemented as member functions of the ``DafReader`` and ``DafWriter`` classes that wrap the matching Julia objects.
Python also doesn't support the ``!`` trailing character in function names (to indicate modifying the object), so it is
removed from the Python method names.
"""

from typing import AbstractSet

from .julia_import import jl
from .storage_types import StorageScalar


class DafReader:
    """
    Read-only access to ``Daf`` data.
    """

    def __init__(self, daf_jl) -> None:
        self.daf_jl = daf_jl

    @property
    def name(self) -> str:
        """
        Return the (hopefully unique) name of the ``Daf`` data set.
        """
        return self.daf_jl.name

    def description(self) -> str:
        """
        Return a (multi-line) description of the contents of ``Daf`` data.
        """
        return jl.Daf.description(self.daf_jl)

    def has_scalar(self, name: str) -> bool:
        """
        Check whether a scalar property with some ``name`` exists in the ``Daf`` data set.
        """
        return jl.Daf.has_scalar(self.daf_jl, name)

    def get_scalar(self, name: str) -> StorageScalar:
        """
        Get the value of a scalar property with some ``name`` in the ``Daf`` data set.
        """
        return jl.Daf.get_scalar(self.daf_jl, name)

    def scalar_names(self) -> AbstractSet[str]:
        """
        The names of the scalar properties in the ``Daf`` data set.
        """
        return jl.Daf.scalar_names(self.daf_jl)


class DafWriter(DafReader):
    """
    Read-write access to ``Daf`` data.
    """

    def set_scalar(self, name: str, value: StorageScalar, *, overwrite: bool = False) -> None:
        """
        Set the ``value`` of a scalar property with some ``name`` in a ``Daf`` data set.
        """
        jl.Daf.set_scalar_b(self.daf_jl, name, value, overwrite=overwrite)

    def delete_scalar(self, name: str, *, must_exist: bool = True) -> None:
        """
        Delete a scalar property with some ``name`` from the ``Daf`` data set.
        """
        jl.Daf.delete_scalar_b(self.daf_jl, name, must_exist=must_exist)
