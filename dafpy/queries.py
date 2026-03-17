"""
Extract data from a ``DafReader``. See the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html>`__ for details.
"""

from typing import Optional
from typing import Union

from .julia_import import jl
from .operations import QueryOperation
from .operations import QuerySequence
from .storage_types import StorageScalar

__all__ = [
    "AndMask",
    "AndNegatedMask",
    "AsAxis",
    "Axis",
    "BeginMask",
    "BeginNegatedMask",
    "CountBy",
    "EndMask",
    "GroupBy",
    "GroupColumnsBy",
    "GroupRowsBy",
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
    "LookupMatrix",
    "LookupScalar",
    "LookupVector",
    "Names",
    "OrMask",
    "OrNegatedMask",
    "Query",
    "QuerySequence",
    "ReduceToColumn",
    "ReduceToRow",
    "SquareColumnIs",
    "SquareRowIs",
    "XorMask",
    "XorNegatedMask",
    "parse_query",
    "is_axis_query",
    "query_axis_name",
    "query_result_dimensions",
]


class Names(QueryOperation):
    """
    A query operation for looking up a set of names. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.Names>`__
    for details.
    """

    def __init__(self) -> None:
        super().__init__(jl.DataAxesFormats.Names())


class LookupScalar(QueryOperation):
    """
    Lookup the value of a scalar property. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.LookupScalar>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.LookupScalar(property))


class LookupVector(QueryOperation):
    """
    Lookup the value of a vector property. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.LookupVector>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.LookupVector(property))


class LookupMatrix(QueryOperation):
    """
    Lookup the value of a matrix property. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.LookupMatrix>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.LookupMatrix(property))


class IfMissing(QueryOperation):
    """
    A query operator for specifying a value to use for a property that is missing from the data. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.IfMissing>`__
    for details.
    """

    def __init__(
        self,
        default_value: StorageScalar,
    ) -> None:
        super().__init__(jl.DataAxesFormats.IfMissing(default_value))


class IfNot(QueryOperation):
    """
    Specify a final value to use when, having looked up some base property values, we use them as axis entry names to
    lookup another property of that axis. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.IfNot>`__
    for details.
    """

    def __init__(self, final_value: Optional[StorageScalar] = None) -> None:
        super().__init__(jl.DataAxesFormats.IfNot(final_value))


class AsAxis(QueryOperation):
    """
    A query operator for specifying that the values of a property we looked up are the names of entries in some axis.
    See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.AsAxis>`__
    for details.
    """

    def __init__(self, axis: Optional[str] = None) -> None:
        super().__init__(jl.DataAxesFormats.AsAxis(axis))


class Axis(QueryOperation):
    """
    A query operator for specifying an axis. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.Axis>`__
    for details.
    """

    def __init__(self, axis: Optional[str] = None) -> None:
        super().__init__(jl.DataAxesFormats.Axis(axis))


class BeginMask(QueryOperation):
    """
    Start specifying a mask to apply to an axis of the result. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.BeginMask>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.BeginMask(property))


class BeginNegatedMask(QueryOperation):
    """
    Start specifying a mask to apply to an axis of the result, negating the first mask. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.BeginNegatedMask>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.BeginNegatedMask(property))


class EndMask(QueryOperation):
    """
    Finish specifying a mask to apply to an axis of the result, following ``BeginMask`` or ``BeginNegatedMask``. See the
    Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.EndMask>`__
    for details.
    """

    def __init__(self) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.EndMask())


class AndMask(QueryOperation):
    """
    Combine a mask with another, using the bitwise AND operator. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.AndMask>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.And(property))


class AndNegatedMask(QueryOperation):
    """
    Combine a mask with another, using the bitwise AND-NOT operator. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.AndNegatedMask>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.AndNot(property))


class OrMask(QueryOperation):
    """
    Combine a mask with another, using the bitwise OR operator. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.OrMask>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.OrMask(property))


class OrNegatedMask(QueryOperation):
    """
    Combine a mask with another, using the bitwise OR-NOT operator. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.OrNegatedMask>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.OrNegatedMask(property))


class XorMask(QueryOperation):
    """
    Combine a mask with another, using the bitwise XOR operator. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.XorMask>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.XorMask(property))


class XorNegatedMask(QueryOperation):
    """
    Combine a mask with another, using the bitwise XOR operator. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.XorNegatedMask>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.XorNegatedMask(property))


class IsLess(QueryOperation):
    """
    Convert a vector of values to a vector of Booleans, is true for entries that are less than the ``comparison_value``.
    See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.IsLess>`__
    for details.
    """

    def __init__(self, comparison_value: StorageScalar) -> None:
        super().__init__(jl.DataAxesFormats.IsLess(comparison_value))


class IsLessEqual(QueryOperation):
    """
    Convert a vector of values to a vector of Booleans, is true for entries that are less than or equal to the
    ``comparison_value``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.IsLessEqual>`__
    for details.
    """

    def __init__(self, comparison_value: StorageScalar) -> None:
        super().__init__(jl.DataAxesFormats.IsLessEqual(comparison_value))


class IsEqual(QueryOperation):
    """
    Convert a vector of values to a vector of Booleans, is true for entries that are equal to the ``comparison_value``.
    See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.IsEqual>`__
    for details.
    """

    def __init__(self, comparison_value: StorageScalar) -> None:
        super().__init__(jl.DataAxesFormats.IsEqual(comparison_value))


class IsNotEqual(QueryOperation):
    """
    Convert a vector of values to a vector of Booleans, is true for entries that are not equal to the
    ``comparison_value``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.IsNotEqual>`__
    for details.
    """

    def __init__(self, comparison_value: StorageScalar) -> None:
        super().__init__(jl.DataAxesFormats.IsNotEqual(comparison_value))


class IsGreater(QueryOperation):
    """
    Convert a vector of values to a vector of Booleans, is true for entries that are greater than the
    ``comparison_value``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.IsGreater>`__
    for details.
    """

    def __init__(self, comparison_value: StorageScalar) -> None:
        super().__init__(jl.DataAxesFormats.IsGreater(comparison_value))


class IsGreaterEqual(QueryOperation):
    """
    Convert a vector of values to a vector of Booleans, is true for entries that are greater than or equal to the
    ``comparison_value``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.IsGreaterEqual>`__
    for details.
    """

    def __init__(self, comparison_value: StorageScalar) -> None:
        super().__init__(jl.DataAxesFormats.IsGreaterEqual(comparison_value))


class IsMatch(QueryOperation):
    """
    Convert a vector of values to a vector of Booleans, is true for (string!) entries that are a (complete!) match to
    the ``comparison_value`` regular expression. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.IsMatch>`__
    for details.
    """

    def __init__(self, comparison_value: StorageScalar) -> None:
        super().__init__(jl.DataAxesFormats.IsMatch(comparison_value))


class IsNotMatch(QueryOperation):
    """
    Convert a vector of values to a vector of Booleans, is true for (string!) entries that are not a (complete!) match
    to the ``comparison_value`` regular expression. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.IsNotMatch>`__
    for details.
    """

    def __init__(self, comparison_value: StorageScalar) -> None:
        super().__init__(jl.DataAxesFormats.IsNotMatch(comparison_value))


class CountBy(QueryOperation):
    """
    Specify a second property for each vector entry, to compute a matrix of counts of the entries with each combination
    of values. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.CountBy>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.CountBy(property))


class GroupBy(QueryOperation):
    """
    Specify value per vector entry to group vector values by, must be followed by a ``ReductionOperation`` to reduce
    each group of values to a single value. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.GroupBy>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.GroupBy(property))


class GroupColumnsBy(QueryOperation):
    """
    Specify value per matrix column to group the columns by, must be followed by a ``ReduceToColumn`` to reduce each
    group of columns to a single column. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.GroupColumnsBy>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.GroupBy(property))


class GroupRowsBy(QueryOperation):
    """
    Specify value per matrix row to group the rows by, must be followed by a ``ReduceToRow`` to reduce each group of
    rows to a single row. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.GroupRowsBy>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.GroupBy(property))


class ReduceToColumn(QueryOperation):
    """
    Specify a ``ReductionOperation`` to convert each row of a matrix to a single value, reducing the matrix to a single
    column. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.ReduceToColumn>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.ReduceToColumn(property))


class ReduceToRow(QueryOperation):
    """
    Specify a ``ReductionOperation`` to convert each column of a matrix to a single value, reducing the matrix to a
    single row. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.ReduceToRow>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.ReduceToRow(property))


class SquareColumnIs(QueryOperation):
    """
    Whenever extracting a vector from a square matrix, specify the axis entry that identifies the column to extract. See
    the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.SquareColumnIs>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.SquareColumnIs(property))


class SquareRowIs(QueryOperation):
    """
    Whenever extracting a vector from a square matrix, specify the axis entry that identifies the row to extract. See
    the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.SquareRowIs>`__
    for details.
    """

    def __init__(self, property: str) -> None:  # pylint: disable=redefined-builtin
        super().__init__(jl.DataAxesFormats.SquareRowIs(property))


#: A Python class to use instead of Julia's ``Daf.Query``. See the Julia
#: `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.Query>`__
#: for details.
Query = Union[Axis, LookupScalar, Names, QuerySequence]


def parse_query(query_string: str) -> QueryOperation:
    """
    Parse a query (or a fragment of a query). See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.parse_query>`__
    for details.
    """
    query = jl.DataAxesFormats.Queries.parse_query(query_string)
    for julia_type, wrapper_type in (
        (jl.DataAxesFormats.Axis, Axis),
        (jl.DataAxesFormats.LookupScalar, LookupScalar),
        (jl.DataAxesFormats.Names, Names),
        (jl.DataAxesFormats.QuerySequence, QuerySequence),
        (jl.DataAxesFormats.QueryOperation, QueryOperation),
    ):
        if jl.isa(query, julia_type):
            return wrapper_type(query)
    assert False


def is_axis_query(query: str | Query) -> bool:
    """
    Returns whether the ``query`` specifies a (possibly masked) axis. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.is_axis_query>`__
    for details.
    """
    return jl.DataAxesFormats.Queries.is_axis_query(query)


def query_axis_name(query: str | Query) -> bool:
    """
    Return the axis name of a query. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.query_axis_name>`__
    for details.
    """
    return jl.DataAxesFormats.Queries.query_axis_name(query)


def query_result_dimensions(query: str | Query) -> int:
    """
    Return the number of dimensions (-1 - names, 0 - scalar, 1 - vector, 2 - matrix) of the results of a query. See the
    Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.2.0/queries.html#DataAxesFormats.Queries.query_result_dimensions>`__
    for details.
    """
    return jl.DataAxesFormats.Queries.query_result_dimensions(query)
