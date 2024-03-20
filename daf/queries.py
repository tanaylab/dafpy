"""
Extract data from a ``DafReader``. See the Julia
`documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html>`__ for details.
"""

from typing import Optional
from typing import Type
from typing import Union

from .julia_import import _to_julia
from .julia_import import jl
from .operations import QueryOperation
from .operations import QuerySequence
from .storage_types import StorageScalar

__all__ = [
    "And",
    "AndNot",
    "AsAxis",
    "Axis",
    "CountBy",
    "Fetch",
    "GroupBy",
    "IfMissing",
    "IfNot",
    "IsEqual",
    "IsGreater",
    "IsGreaterEqual",
    "IsLess",
    "IsLessEqual",
    "IsMatch",
    "IsNotEqual",
    "IsNotMatch",
    "Lookup",
    "Names",
    "Or",
    "OrNot",
    "Query",
    "parse_query",
    "Xor",
    "XorNot",
    "query_result_dimensions",
]


class Names(QueryOperation):
    """
    A query operation for looking up a set of names. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.Names>`__ for details.
    """

    def __init__(self, kind: Optional[str] = None) -> None:
        super().__init__(jl.Daf.Names(kind))


class Lookup(QueryOperation):
    """
    A query operation for looking up the value of a property with some name. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.Lookup>`__ for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.Daf.Lookup(property))


class Fetch(QueryOperation):
    """
    A query operation for fetching the value of a property from another axis, based on a vector property whose values
    are entry names of the axis. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.Fetch>`__ for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.Daf.Fetch(property))


class IfMissing(QueryOperation):
    """
    A query operation providing a value to use if the data is missing some property. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.IfMissing>`__ for details.
    """

    def __init__(self, missing_value: StorageScalar, *, dtype: Optional[Type] = None) -> None:
        super().__init__(jl.Daf.IfMissing(missing_value, dtype=_to_julia(dtype)))


class IfNot(QueryOperation):
    """
    A query operation providing a value to use for "false-ish" values in a vector (empty strings, zero numeric values,
    or false Boolean values). See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.IfNot>`__ for details.
    """

    def __init__(self, value: Optional[StorageScalar] = None) -> None:
        super().__init__(jl.Daf.IfNot(value))


class AsAxis(QueryOperation):
    """
    There are three cases where we may want to take a vector property and consider each value to be the name of an entry
    of some axis: ``Fetch``, ``CountBy`` and ``GroupBy``. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.IfNot>`__ for details.
    """

    def __init__(self, axis: Optional[str] = None) -> None:
        super().__init__(jl.Daf.AsAxis(axis))


class Axis(QueryOperation):
    """
    A query operation for specifying a result axis. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.IfNot>`__ for details.
    """

    def __init__(self, axis: Optional[str] = None) -> None:
        super().__init__(jl.Daf.Axis(axis))


class And(QueryOperation):
    """
    A query operation for restricting the set of entries of an ``Axis``. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.And>`__ for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.Daf.And(property))


class AndNot(QueryOperation):
    """
    Same as ``And`` but use the inverse of the mask. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.AndNot>`__ for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.Daf.AndNot(property))


class Or(QueryOperation):
    """
    A query operation for expanding the set of entries of an ``Axis``. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.Or>`__ for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.Daf.Or(property))


class OrNot(QueryOperation):
    """
    Same as ``Or`` but use the inverse of the mask. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.OrNot>`__ for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.Daf.OrNot(property))


class Xor(QueryOperation):
    """
    A query operation for flipping the set of entries of an ``Axis``. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.Xor>`__ for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.Daf.Xor(property))


class XorNot(QueryOperation):
    """
    Same as ``Xor`` but use the inverse of the mask. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.XorNot>`__ for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.Daf.XorNot(property))


class IsLess(QueryOperation):
    """
    A query operation for converting a vector value to a Boolean mask by comparing it some value. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.IsLess>`__ for details.
    """

    def __init__(self, value: StorageScalar) -> None:
        super().__init__(jl.Daf.IsLess(value))


class IsLessEqual(QueryOperation):
    """
    Similar to ``IsLess`` except that uses ``<=`` instead of ``<`` for the comparison. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.IsLessEqual>`__ for details.
    """

    def __init__(self, value: StorageScalar) -> None:
        super().__init__(jl.Daf.IsLessEqual(value))


class IsEqual(QueryOperation):
    """
    Equality is used for two purposes: As a comparison operator, similar to ``IsLess`` except that uses ``=`` instead of
    ``<`` for the comparison; and To select a single entry from a vector. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.IsEqual>`__ for details.
    """

    def __init__(self, value: StorageScalar) -> None:
        super().__init__(jl.Daf.IsEqual(value))


class IsNotEqual(QueryOperation):
    """
    Similar to ``IsLess`` except that uses ``!=`` instead of ``<`` for the comparison. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.IsNotEqual>`__ for details.
    """

    def __init__(self, value: StorageScalar) -> None:
        super().__init__(jl.Daf.IsNotEqual(value))


class IsGreater(QueryOperation):
    """
    Similar to ``IsLess`` except that uses ``>`` instead of ``<`` for the comparison. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.IsGreater>`__ for details.
    """

    def __init__(self, value: StorageScalar) -> None:
        super().__init__(jl.Daf.IsGreater(value))


class IsGreaterEqual(QueryOperation):
    """
    Similar to ``IsLess`` except that uses ``>=`` instead of ``<`` for the comparison. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.IsGreaterEqual>`__ for details.
    """

    def __init__(self, value: StorageScalar) -> None:
        super().__init__(jl.Daf.IsGreaterEqual(value))


class IsMatch(QueryOperation):
    """
    Similar to ``IsLess`` except that the compared values must be strings, and the mask
    is of the values that match the given regular expression. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.IsMatch>`__ for details.
    """

    def __init__(self, value: StorageScalar) -> None:
        super().__init__(jl.Daf.IsMatch(value))


class IsNotMatch(QueryOperation):
    """
    Similar to ``IsMatch`` except that looks for entries that do not match the pattern. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.IsNotMatch>`__ for details.
    """

    def __init__(self, value: StorageScalar) -> None:
        super().__init__(jl.Daf.IsNotMatch(value))


class CountBy(QueryOperation):
    """
    A query operation that generates a matrix of counts of combinations of pairs of values for the same entries of an
    axis. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.CountBy>`__ for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.Daf.CountBy(property))


class GroupBy(QueryOperation):
    """
    A query operation that uses a (following) ``ReductionOperation`` to aggregate the values of each group of
    values. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.GroupBY>`__ for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.Daf.GroupBy(property))


#: A Python class to use instead of Julia's ``Daf.Query``. See the Julia
#: `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.Query>`__ for details.
Query = Union[Axis, Lookup, Names, QuerySequence]


def parse_query(query_string: str) -> QuerySequence:
    """
    Parse a ``query_string`` into a ``QuerySequence``.
    """
    return QuerySequence(jl.Daf.Queries.Query(query_string))


def query_result_dimensions(query: str | Query) -> int:
    """
    Return the number of dimensions (-1 - names, 0 - scalar, 1 - vector, 2 - matrix) of the results of a query. See the
    Julia `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/queries.html#Daf.Queries.query_result_dimensions>`_
    for details.
    """
    return jl.Daf.Queries.query_result_dimensions(_to_julia(query))
