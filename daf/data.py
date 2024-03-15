"""
Interface of ``DafReader`` and ``DafWriter``.

In Julia, the API is defined as a set of functions, which take the ``Daf`` object as the 1st parameter. In Python, this
is implemented as member functions of the ``DafReader`` and ``DafWriter`` classes that wrap the matching Julia objects.
Python also doesn't support the ``!`` trailing character in function names (to indicate modifying the object), so it is
removed from the Python method names.
"""

from contextlib import contextmanager
from typing import AbstractSet
from typing import Any
from typing import Iterator
from typing import Never
from typing import Optional
from typing import Sequence
from typing import Type
from typing import overload
from weakref import WeakValueDictionary

import numpy as np
import pandas as pd  # type: ignore

from .julia_import import jl
from .storage_types import StorageScalar

__all__ = ["Undef", "undef", "DafReader", "DafWriter"]


class Undef:  # pylint: disable=too-few-public-methods
    """
    Python equivalent for Julia's ``UndefInitializer``.
    """

    def __str__(self) -> str:
        return "undef"


#: Python equivalent for Julia's ``undef``.
undef = Undef()


class DafReader:
    """
    Read-only access to ``Daf`` data.
    """

    def __init__(self, daf_jl) -> None:
        self.daf_jl = daf_jl
        self.weakrefs: WeakValueDictionary[Any, Any] = WeakValueDictionary()

    @property
    def name(self) -> str:
        """
        Return the (hopefully unique) name of the ``Daf`` data set.
        """
        return self.daf_jl.name

    def description(self, *, deep: bool = False) -> str:
        """
        Return a (multi-line) description of the contents of ``Daf`` data.
        """
        return jl.Daf.description(self.daf_jl, deep=deep)

    def has_scalar(self, name: str) -> bool:
        """
        Check whether a scalar property with some ``name`` exists in the ``Daf`` data set.
        """
        return jl.Daf.has_scalar(self.daf_jl, name)

    def get_scalar(self, name: str) -> StorageScalar:
        """
        Get the value of a scalar property with some ``name`` in the ``Daf`` data set.

        Numeric scalars are always returned as ``int`` or ``float``, regardless of the specific data type they are
        stored in the ``Daf`` data set (e.g., a ``UInt8`` will be returned as an ``int`` instead of a ``np.uint8``).
        """
        return jl.Daf.get_scalar(self.daf_jl, name)

    def scalar_names(self) -> AbstractSet[str]:
        """
        The names of the scalar properties in the ``Daf`` data set.
        """
        return jl.Daf.scalar_names(self.daf_jl)

    def has_axis(self, axis: str) -> bool:
        """
        Check whether some ``axis`` exists in the ``Daf`` data set.
        """
        return jl.Daf.has_axis(self.daf_jl, axis)

    def axis_names(self) -> AbstractSet[str]:
        """
        The names of the axes of the ``Daf`` data set.
        """
        return jl.Daf.axis_names(self.daf_jl)

    def axis_length(self, axis: str) -> int:
        """
        The number of entries along the ``axis`` i the ``Daf`` data set.n
        """
        return jl.Daf.axis_length(self.daf_jl, axis)

    def get_axis(self, axis: str) -> np.ndarray:
        """
        The unique names of the entries of some ``axis`` of the ``Daf`` data set.

        This creates an in-memory copy of the data, which is cached for repeated calls.
        """
        axis_version_counter = jl.Daf.axis_version_counter(self.daf_jl, axis)
        axis_key = (axis_version_counter, axis)
        axis_entries = self.weakrefs.get(axis_key)
        if axis_entries is None:
            axis_entries = _from_julia(jl.Daf.get_axis(self.daf_jl, axis))
            self.weakrefs[axis_key] = axis_entries
        return axis_entries

    def has_vector(self, axis: str, name: str) -> bool:
        """
        Check whether a vector property with some ``name`` exists for the ``axis`` in the ``Daf`` data set.
        """
        return jl.Daf.has_vector(self.daf_jl, axis, name)

    def vector_names(self, axis: str) -> AbstractSet[str]:
        """
        The names of the vector properties for the ``axis`` in ``Daf`` data set, **not** including the special ``name``
        property.
        """
        return jl.Daf.vector_names(self.daf_jl, axis)

    @overload
    def get_np_vector(
        self,
        axis: str,
        name: str,
        *,
        default: None,
    ) -> Optional[np.ndarray]: ...

    @overload
    def get_np_vector(
        self,
        axis: str,
        name: str,
        *,
        default: StorageScalar | Sequence[StorageScalar] | np.ndarray | Undef = undef,
    ) -> np.ndarray: ...

    def get_np_vector(
        self,
        axis,
        name,
        *,
        default: None | StorageScalar | Sequence[StorageScalar] | np.ndarray | Undef = undef,
    ) -> Optional[np.ndarray]:
        """
        Get the vector property with some ``name`` for some ``axis`` in the ``Daf`` data set.

        This always returns a ``numpy`` vector (unless ``default`` is ``None`` and the vector does not exist). If the
        stored data is numeric and dense, this is a zero-copy view of the data stored in the ``Daf`` data set.
        Otherwise, a Python copy of the data as a dense ``numpy`` array is returned (and cached for repeated calls).
        Since Python has no concept of sparse vectors (because "reasons"), you can't zero-copy view a sparse ``Daf``
        vector using the Python API.
        """
        if not jl.Daf.has_vector(self.daf_jl, axis, name):
            if default is None:
                return None
            return _from_julia(jl.Daf.get_vector(self.daf_jl, axis, name, default=_to_julia(default)).array)

        vector_version_counter = jl.Daf.vector_version_counter(self.daf_jl, axis, name)
        vector_key = (vector_version_counter, axis, name)
        vector_value = self.weakrefs.get(vector_key)
        if vector_value is None:
            vector_value = _from_julia(jl.Daf.get_vector(self.daf_jl, axis, name).array)
            self.weakrefs[vector_key] = vector_value
        return vector_value

    @overload
    def get_pd_vector(
        self,
        axis: str,
        name: str,
        *,
        default: None,
    ) -> Optional[pd.Series]: ...

    @overload
    def get_pd_vector(
        self,
        axis: str,
        name: str,
        *,
        default: StorageScalar | Sequence[StorageScalar] | np.ndarray | Undef = undef,
    ) -> pd.Series: ...

    def get_pd_vector(
        self,
        axis: str,
        name: str,
        *,
        default: None | StorageScalar | Sequence[StorageScalar] | np.ndarray | Undef = undef,
    ) -> Optional[pd.Series]:
        """
        Get the vector property with some ``name`` for some ``axis`` in the ``Daf`` data set.

        This is a wrapper around ``get_np_vector`` which returns a ``pandas`` series using the entry names of the axis
        as the index.
        """
        vector_value = self.get_np_vector(axis, name, default=_to_julia(default))
        if vector_value is None:
            return None
        return pd.Series(vector_value, index=self.get_axis(axis))

    def has_matrix(self, rows_axis: str, columns_axis: str, name: str, *, relayout: bool = True) -> bool:
        """
        Check whether a matrix property with some ``name`` exists for the ``rows_axis`` and the ``columns_axis`` in the
        ``Daf`` data set.
        """
        return jl.Daf.has_matrix(self.daf_jl, rows_axis, columns_axis, name, relayout=relayout)

    def matrix_names(self, rows_axis: str, columns_axis: str, *, relayout: bool = True) -> AbstractSet[str]:
        """
        The names of the matrix properties for the ``rows_axis`` and ``columns_axis`` in the ``Daf`` data set.
        """
        return jl.Daf.matrix_names(self.daf_jl, rows_axis, columns_axis, relayout=relayout)

    @overload
    def get_np_matrix(
        self, rows_axis: str, columns_axis: str, name: str, *, default: None, relayout: bool = True
    ) -> Optional[np.ndarray]: ...

    @overload
    def get_np_matrix(
        self,
        rows_axis: str,
        columns_axis: str,
        name: str,
        *,
        default: StorageScalar | Sequence[StorageScalar] | np.ndarray | Undef = undef,
        relayout: bool = True,
    ) -> np.ndarray: ...

    def get_np_matrix(
        self,
        rows_axis: str,
        columns_axis: str,
        name: str,
        *,
        default: None | StorageScalar | Sequence[StorageScalar] | np.ndarray | Undef = undef,
        relayout: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Get the column-major matrix property with some ``name`` for some ``rows_axis`` and ``columns_axis`` in the
        ``Daf`` data set.

        This always returns a ``numpy`` matrix (unless ``default`` is ``None`` and the matrix does not exist). Note that
        by default ``numpy`` matrices are in row-major (C) layout and not in column-major (Fortran) layout. To get a
        row-major matrix, simply flip the order of the axes, and call transpose on the result (which is an efficient
        zero-copy operation).

        Also note that although we call this ``get_np_matrix``, the result is a simple ``np.ndarray`` with two
        dimensions (which is what you want) and **not** the deprecated ``np.matrix`` (which is to be avoided at all
        costs).
        """
        if not jl.Daf.has_matrix(self.daf_jl, rows_axis, columns_axis, name, relayout=relayout):
            if default is None:
                return None
            return _from_julia(
                jl.Daf.get_matrix(
                    self.daf_jl, rows_axis, columns_axis, name, default=_to_julia(default), relayout=relayout
                ).array
            )

        matrix_version_counter = jl.Daf.matrix_version_counter(self.daf_jl, rows_axis, columns_axis, name)
        matrix_key = (matrix_version_counter, rows_axis, columns_axis, name)
        matrix_value = self.weakrefs.get(matrix_key)
        if matrix_value is None:
            matrix_value = _from_julia(
                jl.Daf.get_matrix(self.daf_jl, rows_axis, columns_axis, name, relayout=relayout).array
            )
            self.weakrefs[matrix_key] = matrix_value
        return matrix_value

    @overload
    def get_pd_matrix(
        self,
        rows_axis: str,
        columns_axis: str,
        name: str,
        *,
        default: None,
        relayout: bool = True,
    ) -> Optional[pd.DataFrame]: ...

    @overload
    def get_pd_matrix(
        self,
        rows_axis: str,
        columns_axis: str,
        name: str,
        *,
        default: StorageScalar | Sequence[StorageScalar] | np.ndarray | Undef = undef,
        relayout: bool = True,
    ) -> pd.DataFrame: ...

    def get_pd_matrix(
        self,
        rows_axis: str,
        columns_axis: str,
        name: str,
        *,
        default: None | StorageScalar | Sequence[StorageScalar] | np.ndarray | Undef = undef,
        relayout: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Get the column-major matrix property with some ``name`` for some ``rows_axis`` and ``columns_axis`` in the
        ``Daf`` data set.

        This is a wrapper around ``get_np_matrix`` which returns a ``pandas`` data frame using the entry names of the
        axes as the indices.
        """
        matrix_value = self.get_np_matrix(rows_axis, columns_axis, name, default=_to_julia(default), relayout=relayout)
        if matrix_value is None:
            return None
        return pd.DataFrame(matrix_value, index=self.get_axis(rows_axis), columns=self.get_axis(columns_axis))


class DafWriter(DafReader):
    """
    Read-write access to ``Daf`` data.
    """

    def set_scalar(self, name: str, value: StorageScalar, *, overwrite: bool = False) -> None:
        """
        Set the ``value`` of a scalar property with some ``name`` in a ``Daf`` data set.

        You can force the data type numeric scalars are stored in by using the appropriate ``numpy`` type (e.g., a
        ``np.uint8`` will be stored as a ``UInt8``).
        """
        jl.Daf.set_scalar_b(self.daf_jl, name, value, overwrite=overwrite)

    def delete_scalar(self, name: str, *, must_exist: bool = True) -> None:
        """
        Delete a scalar property with some ``name`` from the ``Daf`` data set.
        """
        jl.Daf.delete_scalar_b(self.daf_jl, name, must_exist=must_exist)

    def add_axis(self, axis: str, entries: Sequence[str] | np.ndarray) -> None:
        """
        Add a new ``axis`` to the ``Daf`` data set.
        """
        jl.Daf.add_axis_b(self.daf_jl, axis, _to_julia(entries))

    def delete_axis(self, axis: str, *, must_exist: bool = True) -> None:
        """
        Delete an ``axis`` from the ``Daf`` data set.
        """
        jl.Daf.delete_axis_b(self.daf_jl, axis, must_exist=must_exist)

    def set_vector(
        self, axis: str, name: str, value: Sequence[StorageScalar] | np.ndarray, *, overwrite: bool = False
    ) -> None:
        """
        Set a vector property with some ``name`` for some ``axis`` in the ``Daf`` data set.

        If the provided ``value`` is numeric and dense, this passes a zero-copy view of the data to the ``Daf`` data
        set. Otherwise, a Python copy of the data as a dense ``numpy`` array is made, and passed to ``Daf``. Since
        Python has no concept of sparse vectors (because "reasons"), you can't create a sparse ``Daf`` vector using the
        Python API.

        As a convenience, you can pass a 1xN or Nx1 matrix here and it will be mercifully interpreted as a vector.
        """
        jl.Daf.set_vector_b(self.daf_jl, axis, name, _as_vector(_to_julia(value)), overwrite=overwrite)

    @contextmanager
    def empty_dense_vector(self, axis: str, name: str, dtype: Type, *, overwrite: bool = False) -> Iterator[pd.Series]:
        """
        Create an empty dense vector property with some ``name`` for some ``axis`` in the ``Daf`` data set, and pass it
        to the block (as a ``pandas`` series) to be filled.
        """

        def fill(named: Any) -> Iterator[Never]:
            vector_value = _from_julia(named.array)
            yield pd.Series(vector_value, index=self.get_axis(axis))

        yield from jl.Daf.empty_dense_vector_b(
            jl.py_function_to_fulia_function(fill), self.daf_jl, axis, name, _to_julia(dtype), overwrite=overwrite
        )

    def delete_vector(self, axis: str, name: str, *, must_exist: bool = True) -> None:
        """
        Delete a vector property with some ``name`` for some ``axis`` from the ``Daf`` data set.
        """
        jl.Daf.delete_vector_b(self.daf_jl, axis, name, must_exist=must_exist)

    def set_matrix(
        self,
        rows_axis: str,
        columns_axis: str,
        name: str,
        value: np.ndarray,
        *,
        overwrite: bool = False,
        relayout: bool = True,
    ) -> None:
        """
        Set the matrix property with some ``name`` for some ``rows_axis`` and ``columns_axis`` in the ``Daf`` data set.
        Since ``Daf`` is implemented Julia, this should be a column-major ``matrix``, so if you have a standard
        ``numpy`` matrix, flip the order of the axes and pass the ``transpose`` (which is an efficient zero-copy
        operation).
        """
        jl.Daf.set_matrix_b(self.daf_jl, rows_axis, columns_axis, name, value, overwrite=overwrite, relayout=relayout)

    @contextmanager
    def empty_dense_matrix(
        self, rows_axis: str, columns_axis: str, name: str, dtype: Type, *, overwrite: bool = False
    ) -> Iterator[pd.DataFrame]:
        """
        Create an empty dense matrix property with some ``name`` for some ``rows_axis`` and ``columns_axis`` in the
        ``Daf`` data set, and pass it to the block (as a ``pandas`` data frame) to be filled.
        """

        def fill(named: Any) -> Iterator[Never]:
            matrix_value = _from_julia(named.array)
            yield pd.DataFrame(matrix_value, index=self.get_axis(rows_axis), columns=self.get_axis(columns_axis))

        yield from jl.Daf.empty_dense_matrix_b(
            jl.py_function_to_fulia_function(fill),
            self.daf_jl,
            rows_axis,
            columns_axis,
            name,
            _to_julia(dtype),
            overwrite=overwrite,
        )

    def relayout_matrix(self, rows_axis: str, columns_axis: str, name: str, *, overwrite: bool = False) -> None:
        """
        Given a matrix property with some ``name`` exists (in column-major layout) in the ``Daf`` data set for the
        ``rows_axis`` and the ``columns_axis``, then relayout it and store the row-major result as well (that is, with
        flipped axes).
        """
        jl.Daf.relayout_matrix_b(self.daf_jl, rows_axis, columns_axis, name, overwrite=overwrite)

    def delete_matrix(self, rows_axis: str, columns_axis: str, name: str, *, must_exist: bool = True) -> None:
        """
        Delete a matrix property with some ``name`` for some ``rows_axis`` and ``columns_axis`` from the ``Daf`` data
        set.
        """
        jl.Daf.delete_matrix_b(self.daf_jl, rows_axis, columns_axis, name, must_exist=must_exist)


JULIA_TYPE_OF_PY_TYPE = {
    bool: jl.Bool,
    int: jl.Int64,
    float: jl.Float64,
    np.int8: jl.Int8,
    np.int16: jl.Int16,
    np.int32: jl.Int32,
    np.int64: jl.Int64,
    np.uint8: jl.UInt8,
    np.uint16: jl.UInt16,
    np.uint32: jl.UInt32,
    np.uint64: jl.UInt64,
    np.float32: jl.Float32,
    np.float64: jl.Float64,
}


def _to_julia(value: Any) -> Any:
    if isinstance(value, type):
        value = JULIA_TYPE_OF_PY_TYPE[value]
    elif isinstance(value, Undef):
        value = jl.undef
    elif isinstance(value, Sequence) and not isinstance(value, np.ndarray):
        value = np.array(value)
    return value


def _from_julia(julia_vector: Any) -> np.ndarray:
    python_array = np.asarray(julia_vector)
    if python_array.dtype == "object":
        python_array = np.array([str(obj) for obj in python_array], dtype=str)
    return python_array


def _as_vector(vector_ish: Any) -> Any:
    if isinstance(vector_ish, np.ndarray):
        shape = vector_ish.shape
        if len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
            vector_ish = vector_ish.reshape(-1)
    return vector_ish
