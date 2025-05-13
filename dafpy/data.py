"""
Interface of ``DafReader`` and ``DafWriter``. See the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/formats.html>`__,
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html>`__ and
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/writers.html>`__
for details.
"""

from contextlib import contextmanager
from typing import AbstractSet
from typing import Any
from typing import Iterator
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Self
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import overload
from weakref import WeakValueDictionary

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from .julia_import import JlObject
from .julia_import import Undef
from .julia_import import UndefInitializer
from .julia_import import _as_vector
from .julia_import import _from_julia_array
from .julia_import import _from_julia_frame
from .julia_import import _jl_pairs
from .julia_import import _to_julia_array
from .julia_import import _to_julia_type
from .julia_import import jl
from .operations import PendingNumpyQuery
from .operations import PendingPandasQuery
from .queries import Query
from .storage_types import StorageScalar

__all__ = ["DafReader", "DafReadOnly", "DafWriter", "CacheGroup"]


#: Types of cached data inside ``Daf``. See the Julia
#: `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/formats.html#DataAxesFormats.Formats.CacheGroup>`__
#: for details.
CacheGroup = Literal["MappedData"] | Literal["MemoryData"] | Literal["QueryData"]

JL_CACHE_TYPE = {
    "MappedData": jl.DataAxesFormats.MappedData,
    "MemoryData": jl.DataAxesFormats.MemoryData,
    "QueryData": jl.DataAxesFormats.QueryData,
}

#: A key specifying an atomic data property in ``Daf``. See the Julia
#: `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/keys.html#DataAxesFormats.Keys.PropertyKey>`__
#: for details.
PropertyKey = str | Tuple[str, str] | Tuple[str, str, str]

#: A key specifying some data property in ``Daf`` (including tensors). See the Julia
#: `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/keys.html#DataAxesFormats.Keys.DataKey>`__
#: for details.
DataKey = PropertyKey | Tuple[str, str, str, str]


def _to_jl_cache_group(cache_group: Optional[CacheGroup]) -> jl.DataAxesFormats.Formats.CacheGroup:
    if cache_group is not None:
        return JL_CACHE_TYPE[cache_group]
    return None


class WeakRefAbleDict(dict):
    """
    Inheritance is required to allow weak reference to the built-in ``dict`` type (in CPython, anyway).
    """


class DafReader(JlObject):
    """
    Read-only access to ``Daf`` data. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/formats.html#DataAxesFormats.Formats.DafReader>`__
    for details.
    """

    def __init__(self, jl_obj) -> None:
        super().__init__(jl_obj)
        self.weakrefs: WeakValueDictionary[Any, Any] = WeakValueDictionary()

    @property
    def name(self) -> str:
        """
        Return the (hopefully unique) name of the ``Daf`` data set.
        """
        return self.jl_obj.name

    def description(self, *, cache: bool = False, deep: bool = False, tensors: bool = True) -> str:
        """
        Return a (multi-line) description of the contents of ``Daf`` data. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.description>`__
        for details.
        """
        return jl.DataAxesFormats.description(self.jl_obj, cache=cache, deep=deep, tensors=tensors)

    def has_scalar(self, name: str) -> bool:
        """
        Check whether a scalar property with some ``name`` exists in the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.has_scalar>`__
        for details.
        """
        return jl.DataAxesFormats.has_scalar(self.jl_obj, name)

    def get_scalar(self, name: str) -> StorageScalar:
        """
        Get the value of a scalar property with some ``name`` in the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.get_scalar>`__
        for details.

        Numeric scalars are always returned as ``int`` or ``float``, regardless of the specific data type they are
        stored in the ``Daf`` data set (e.g., a ``UInt8`` will be returned as an ``int`` instead of a ``np.uint8``).
        """
        return jl.DataAxesFormats.get_scalar(self.jl_obj, name)

    def scalars_set(self) -> AbstractSet[str]:
        """
        The names of the scalar properties in the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.scalars_set>`__
        for details.
        """
        return jl.DataAxesFormats.scalars_set(self.jl_obj)

    def has_axis(self, axis: str) -> bool:
        """
        Check whether some ``axis`` exists in the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.has_axis>`__
        for details.
        """
        return jl.DataAxesFormats.has_axis(self.jl_obj, axis)

    def axes_set(self) -> AbstractSet[str]:
        """
        The set of names of the axes of the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.axes_set>`__
        for details.
        """
        return jl.DataAxesFormats.axes_set(self.jl_obj)

    def axis_length(self, axis: str) -> int:
        """
        The number of entries along the ``axis`` in the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.axis_length>`__
        for details.
        """
        return jl.DataAxesFormats.axis_length(self.jl_obj, axis)

    def axis_np_vector(self, axis: str) -> np.ndarray:
        """
        A ``numpy`` vector of unique names of the entries of some ``axis`` of the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.axis_vector>`__
        for details.

        This creates an in-memory copy of the data, which is cached for repeated calls.
        """
        axis_version_counter = jl.DataAxesFormats.axis_version_counter(self.jl_obj, axis)
        axis_key = (axis_version_counter, axis, True)
        axis_entries = self.weakrefs.get(axis_key)
        if axis_entries is None:
            axis_entries = _from_julia_array(jl.DataAxesFormats.axis_vector(self.jl_obj, axis))
            self.weakrefs[axis_key] = axis_entries
        return axis_entries

    def axis_np_entries(
        self, axis: str, indices: Optional[Sequence[int]] = None, *, allow_empty: bool = False
    ) -> np.ndarray:
        """
        Return a ``numpy`` vector of the names of entries of the ``indices`` in the ``axis``. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.axis_entries>`__
        for details.

        The ``indices`` passed here are 0-based to fit the Python conventions. This means that if ``allow_empty``,
        *negative* ``indices`` are converted to the empty string.
        """
        entries = self.axis_np_vector(axis)

        if indices is None:
            return entries

        if allow_empty:
            return np.array(["" if index < 0 else entries[index] for index in indices], dtype="str")

        return entries[indices]

    def axis_dict(self, axis: str) -> Mapping[str, int]:
        """
        Return a dictionary converting ``axis`` entry names to their (0-based) integer index.
        """
        axis_version_counter = jl.DataAxesFormats.axis_version_counter(self.jl_obj, axis)
        axis_key = (axis_version_counter, axis, False)
        axis_dictionary = self.weakrefs.get(axis_key)
        if axis_dictionary is None:
            axis_dictionary = WeakRefAbleDict()
            for name, index in jl.DataAxesFormats.axis_dict(self.jl_obj, axis).items():
                axis_dictionary[str(name)] = index - 1
            self.weakrefs[axis_key] = axis_dictionary
        return axis_dictionary

    def axis_np_indices(self, axis: str, entries: Sequence[str], *, allow_empty: bool = False) -> np.ndarray:
        """
        Return a ``numpy`` vector of the indices of the ``entries`` in the ``axis``. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.axis_indices>`__
        for details.

        The indices returned here are 0-based to fit the Python conventions. This means that if ``allow_empty``,
        the empty string is converted to the index -1.
        """
        axis_dictionary = self.axis_dict(axis)
        result = np.empty(len(entries), "int32")
        for index, entry in enumerate(entries):
            if allow_empty and entry == "":
                result[index] = -1
            else:
                result[index] = axis_dictionary[entry]
        return result

    def axis_pd_indices(self, axis: str, entries: Sequence[str], *, allow_empty: bool = False) -> pd.Series:
        """
        Return a ``pandas`` series of the indices of the ``entries`` in the ``axis``. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.axis_indices>`__
        for details.
        """
        return pd.Series(self.axis_np_indices(axis, entries, allow_empty=allow_empty), index=np.array(entries))

    def has_vector(self, axis: str, name: str) -> bool:
        """
        Check whether a vector property with some ``name`` exists for the ``axis`` in the ``Daf`` data set. See the
        Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.has_vector>`__
        for details.
        """
        return jl.DataAxesFormats.has_vector(self.jl_obj, axis, name)

    def vectors_set(self, axis: str) -> AbstractSet[str]:
        """
        The set of names of the vector properties for the ``axis`` in ``Daf`` data set, **not** including the special
        ``name`` property. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.vectors_set>`__
        for details.
        """
        return jl.DataAxesFormats.vectors_set(self.jl_obj, axis)

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
        default: StorageScalar | Sequence[StorageScalar] | np.ndarray | UndefInitializer = Undef,
    ) -> np.ndarray: ...

    def get_np_vector(
        self,
        axis,
        name,
        *,
        default: None | StorageScalar | Sequence[StorageScalar] | np.ndarray | UndefInitializer = Undef,
    ) -> Optional[np.ndarray]:
        """
        Get the vector property with some ``name`` for some ``axis`` in the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.get_vector>`__
        for details.

        This always returns a ``numpy`` vector (unless ``default`` is ``None`` and the vector does not exist). If the
        stored data is numeric and dense, this is a zero-copy view of the data stored in the ``Daf`` data set.
        Otherwise, a Python copy of the data as a dense ``numpy`` array is returned (and cached for repeated calls).
        Since Python has no concept of sparse vectors (because "reasons"), you can't zero-copy view a sparse ``Daf``
        vector using the Python API.
        """
        if not jl.DataAxesFormats.has_vector(self.jl_obj, axis, name):
            if default is None:
                return None
            return _from_julia_array(
                jl.DataAxesFormats.get_vector(self.jl_obj, axis, name, default=_to_julia_array(default))
            )

        vector_version_counter = jl.DataAxesFormats.vector_version_counter(self.jl_obj, axis, name)
        vector_key = (vector_version_counter, axis, name)
        vector_value = self.weakrefs.get(vector_key)
        if vector_value is None:
            vector_value = _from_julia_array(jl.DataAxesFormats.get_vector(self.jl_obj, axis, name))
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
        default: StorageScalar | Sequence[StorageScalar] | np.ndarray | UndefInitializer = Undef,
    ) -> pd.Series: ...

    def get_pd_vector(
        self,
        axis: str,
        name: str,
        *,
        default: None | StorageScalar | Sequence[StorageScalar] | np.ndarray | UndefInitializer = Undef,
    ) -> Optional[pd.Series]:
        """
        Get the vector property with some ``name`` for some ``axis`` in the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.get_vector>`__
        for details.

        This is a wrapper around ``get_np_vector`` which returns a ``pandas`` series using the entry names of the axis
        as the index.
        """
        vector_value = self.get_np_vector(axis, name, default=_to_julia_array(default))
        if vector_value is None:
            return None
        return pd.Series(vector_value, index=self.axis_np_vector(axis))

    def has_matrix(self, rows_axis: str, columns_axis: str, name: str, *, relayout: bool = True) -> bool:
        """
        Check whether a matrix property with some ``name`` exists for the ``rows_axis`` and the ``columns_axis`` in the
        ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.has_matrix>`__
        for details.
        """
        return jl.DataAxesFormats.has_matrix(self.jl_obj, rows_axis, columns_axis, name, relayout=relayout)

    def matrices_set(self, rows_axis: str, columns_axis: str, *, relayout: bool = True) -> AbstractSet[str]:
        """
        The names of the matrix properties for the ``rows_axis`` and ``columns_axis`` in the ``Daf`` data set. See the
        Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.matrices_set>`__
        for details.
        """
        return jl.DataAxesFormats.matrices_set(self.jl_obj, rows_axis, columns_axis, relayout=relayout)

    @overload
    def get_np_matrix(
        self, rows_axis: str, columns_axis: str, name: str, *, default: None, relayout: bool = True
    ) -> Optional[np.ndarray | sp.csc_matrix]: ...

    @overload
    def get_np_matrix(
        self,
        rows_axis: str,
        columns_axis: str,
        name: str,
        *,
        default: StorageScalar | Sequence[StorageScalar] | np.ndarray | UndefInitializer = Undef,
        relayout: bool = True,
    ) -> np.ndarray | sp.csc_matrix: ...

    def get_np_matrix(
        self,
        rows_axis: str,
        columns_axis: str,
        name: str,
        *,
        default: None | StorageScalar | Sequence[StorageScalar] | np.ndarray | UndefInitializer = Undef,
        relayout: bool = True,
    ) -> Optional[np.ndarray | sp.csc_matrix]:
        """
        Get the column-major matrix property with some ``name`` for some ``rows_axis`` and ``columns_axis`` in the
        ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.get_matrix>`__
        for details.

        This always returns a column-major ``numpy`` matrix or a ``scipy`` sparse ``csc_matrix``, (unless ``default`` is
        ``None`` and the matrix does not exist). If the stored data is numeric and dense, this is a zero-copy view of
        the data stored in the ``Daf`` data set.

        Note that by default ``numpy`` matrices are in row-major (C) layout and not in column-major (Fortran) layout. To
        get a row-major matrix, simply flip the order of the axes, and call transpose on the result (which is an
        efficient zero-copy operation). This will also (zero-copy) convert the ``csc_matrix`` into a ``csr_matrix``.

        Also note that although we call this ``get_np_matrix``, the result is **not** the deprecated ``np.matrix``
        (which is to be avoided at all costs).
        """
        if not jl.DataAxesFormats.has_matrix(self.jl_obj, rows_axis, columns_axis, name, relayout=relayout):
            if default is None:
                return None
            return _from_julia_array(
                jl.DataAxesFormats.get_matrix(
                    self.jl_obj, rows_axis, columns_axis, name, default=_to_julia_array(default), relayout=relayout
                )
            )

        matrix_version_counter = jl.DataAxesFormats.matrix_version_counter(self.jl_obj, rows_axis, columns_axis, name)
        matrix_key = (matrix_version_counter, rows_axis, columns_axis, name)
        matrix_value = self.weakrefs.get(matrix_key)
        if matrix_value is None:
            matrix_value = _from_julia_array(
                jl.DataAxesFormats.get_matrix(self.jl_obj, rows_axis, columns_axis, name, relayout=relayout)
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
        default: StorageScalar | Sequence[StorageScalar] | np.ndarray | UndefInitializer = Undef,
        relayout: bool = True,
    ) -> pd.DataFrame: ...

    def get_pd_matrix(
        self,
        rows_axis: str,
        columns_axis: str,
        name: str,
        *,
        default: None | StorageScalar | Sequence[StorageScalar] | np.ndarray | UndefInitializer = Undef,
        relayout: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Get the column-major matrix property with some ``name`` for some ``rows_axis`` and ``columns_axis`` in the
        ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/readers.html#DataAxesFormats.Readers.get_matrix>`__
        for details.

        This is a wrapper around ``get_np_matrix`` which returns a ``pandas`` data frame using the entry names of the
        axes as the indices.

        Note that since ``pandas`` data frames can't contain a sparse matrix, the data will always be in a dense
        ``numpy`` matrix, so take care not to invoke this for a too-large sparse data matrix.

        This is not to be confused with ``get_frame`` which returns a "real" ``pandas`` data frame, with arbitrary
        (query) columns, possibly using a different data type for each.
        """
        matrix_value = self.get_np_matrix(
            rows_axis, columns_axis, name, default=_to_julia_array(default), relayout=relayout
        )
        if matrix_value is None:
            return None
        if sp.issparse(matrix_value):
            matrix_value = matrix_value.toarray()  # type: ignore
        return pd.DataFrame(
            matrix_value, index=self.axis_np_vector(rows_axis), columns=self.axis_np_vector(columns_axis)
        )

    def empty_cache(self, *, clear: Optional[CacheGroup] = None, keep: Optional[CacheGroup] = None) -> None:
        """
        Clear some cached data. By default, completely empties the caches. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/formats.html#DataAxesFormats.Formats.empty_cache!>`__
        for details.
        """
        jl.DataAxesFormats.empty_cache_b(self.jl_obj, clear=_to_jl_cache_group(clear), keep=_to_jl_cache_group(keep))

    def has_query(self, query: str | Query) -> bool:
        """
        Return whether the ``query`` can be applied to the ``Daf`` data. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/queries.html#DataAxesFormats.Queries.has_query>`__
        for details.
        """
        return jl.DataAxesFormats.Queries.has_query(self.jl_obj, query)

    @overload
    def get_np_query(
        self, query: str | Query, *, cache: bool = True
    ) -> StorageScalar | np.ndarray | AbstractSet[str]: ...

    @overload
    def get_np_query(self, query: None = None, *, cache: bool = True) -> PendingNumpyQuery: ...

    def get_np_query(
        self, query: str | Query | None = None, *, cache: bool = True
    ) -> StorageScalar | np.ndarray | AbstractSet[str] | PendingNumpyQuery:
        """
        Apply the full ``query`` to the ``Daf`` data set and return the result. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/queries.html#DataAxesFormats.Queries.get_query>`__
        for details.

        If the result isn't a scalar, and isn't an array of names, then we return a ``numpy`` array or a ``scipy``
        ``csc_matrix``.

        If the ``query`` is not specified, this is intended to be used as ``query | daf.get_np_query()``. This is
        useful when constructing the query in parts (e.g. ``Axis("cell") |> Lookup("metacell") |> daf.get_np_query()``).
        """
        if query is None:
            return PendingNumpyQuery(lambda query: self.get_np_query(query, cache=cache))

        result = jl.DataAxesFormats.Queries.get_query(self.jl_obj, query, cache=cache)
        if not isinstance(result, (str, int, float, AbstractSet)):
            result = _from_julia_array(result)
        return result

    @overload
    def get_pd_query(
        self, query: str | Query, *, cache: bool = True
    ) -> StorageScalar | pd.Series | pd.DataFrame | AbstractSet[str]: ...

    @overload
    def get_pd_query(self, query: None = None, *, cache: bool = True) -> PendingPandasQuery: ...

    def get_pd_query(
        self, query: str | Query | None = None, *, cache: bool = True
    ) -> StorageScalar | pd.Series | pd.DataFrame | AbstractSet[str] | PendingPandasQuery:
        """
        Similar to ``get_np_query``, but return a ``pandas`` series or data frame for vector and matrix data.

        Note that since ``pandas`` data frames can't contain a sparse matrix, the data will always be in a dense
        ``numpy`` matrix, so take care not to invoke this for a too-large sparse data matrix.

        If the ``query`` is not specified, this is intended to be used as ``query | daf.get_np_query()``. This is
        useful when constructing the query in parts (e.g. ``Axis("cell") |> Lookup("metacell") |> daf.get_np_query()``).
        """
        if query is None:
            return PendingPandasQuery(lambda query: self.get_pd_query(query, cache=cache))

        result = jl.DataAxesFormats.Queries.get_query(self.jl_obj, query, cache=cache)
        if not isinstance(result, (str, int, float, AbstractSet)):
            values = _from_julia_array(result)
            if sp.issparse(values):
                values = values.toarray()  # type: ignore
            assert 1 <= values.ndim <= 2
            if values.ndim == 1:
                result = pd.Series(values, index=_from_julia_array(jl._optional_julia_vector_names(result)))
            else:
                result = pd.DataFrame(
                    values,
                    index=_from_julia_array(jl.names(result, 1)),
                    columns=_from_julia_array(jl.names(result, 2)),
                )
        return result

    def __getitem__(self, query: str | Query) -> StorageScalar | pd.Series | pd.DataFrame | AbstractSet[str]:
        """
        The shorthand ``data[query]`` is equivalent to ``data.get_pd_query(query, cache = False)``.
        """
        return self.get_pd_query(query, cache=False)

    def get_pd_frame(
        self,
        axis: str | Query,
        columns: Optional[Sequence[str | Tuple[str, str]] | Mapping[str, str | Query]] = None,
        *,
        cache: bool = False,
    ) -> pd.DataFrame:
        """
        Return a ``DataFrame`` containing multiple vectors of the same ``axis``. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/queries.html#DataAxesFormats.Queries.get_frame>`__
        for details.

        Note this is different from ``get_pd_matrix`` which returns some 2D data as a ``pandas`` data frame. Here, each
        column can be the result of an arbitrary query and may have a different data type.

        The order of the columns matters. Luckily, the default dictionary type is ordered in modern Python, so if you
        write ``columns = {"color": ": type => color", "age": ": batch => age"}`` you can trust that the ``color``
        column will be first and the ``age`` column will be second.
        """
        if isinstance(columns, Mapping):
            columns = jl._pairify_columns(_jl_pairs(columns))
        else:
            columns = _to_julia_array(columns)
        jl_frame = jl.DataAxesFormats.Queries.get_frame(self.jl_obj, axis, columns, cache=cache)
        return _from_julia_frame(jl_frame)

    def read_only(self, *, name: Optional[str] = None) -> "DafReadOnly":
        """
        Wrap the ``Daf`` data sett with a ``DafReadOnlyWrapper`` to protect it against accidental modification. See the
        Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/read_only.html#DataAxesFormats.ReadOnly.read_only>`__
        for details.
        """
        return DafReadOnly(jl.DataAxesFormats.read_only(self.jl_obj, name=name))


class DafReadOnly(DafReader):
    """
    A read-only ``DafReader``, which doesn't allow any modification of the data. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/read_only.html#DataAxesFormats.ReadOnly.DafReadOnly>`__
    for details.
    """

    def read_only(self, *, name: Optional[str] = None) -> "DafReadOnly":
        if name is None:
            return self
        return super().read_only(name=name)


class DafWriter(DafReader):
    """
    Read-write access to ``Daf`` data. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/formats.html#DataAxesFormats.Formats.DafWriter>`__
    for details.
    """

    def set_scalar(self, name: str, value: StorageScalar, *, overwrite: bool = False) -> Self:
        """
        Set the ``value`` of a scalar property with some ``name`` in a ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/writers.html#DataAxesFormats.Writers.set_scalar!>`__
        for details.

        Returns ``self`` for chaining.

        You can force the data type numeric scalars are stored in by using the appropriate ``numpy`` type (e.g., a
        ``np.uint8`` will be stored as a ``UInt8``).
        """
        jl.DataAxesFormats.set_scalar_b(self.jl_obj, name, value, overwrite=overwrite)
        return self

    def delete_scalar(self, name: str, *, must_exist: bool = True) -> Self:
        """
        Delete a scalar property with some ``name`` from the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/writers.html#DataAxesFormats.Writers.delete_scalar!>`__
        for details.

        Returns ``self`` for chaining.
        """
        jl.DataAxesFormats.delete_scalar_b(self.jl_obj, name, must_exist=must_exist)
        return self

    def add_axis(self, axis: str, entries: Sequence[str] | np.ndarray, *, overwrite: bool = False) -> Self:
        """
        Add a new ``axis`` to the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/writers.html#DataAxesFormats.Writers.add_axis!>`__
        for details.

        Returns ``self`` for chaining.
        """
        jl.DataAxesFormats.add_axis_b(self.jl_obj, axis, _to_julia_array(entries), overwrite=overwrite)
        return self

    def delete_axis(self, axis: str, *, must_exist: bool = True) -> Self:
        """
        Delete an ``axis`` from the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/writers.html#DataAxesFormats.Writers.delete_axis!>`__
        for details.

        Returns ``self`` for chaining.
        """
        jl.DataAxesFormats.delete_axis_b(self.jl_obj, axis, must_exist=must_exist)
        return self

    def set_vector(
        self,
        axis: str,
        name: str,
        value: Sequence[StorageScalar] | np.ndarray | sp.csc_matrix | sp.csr_matrix,
        *,
        overwrite: bool = False,
    ) -> Self:
        """
        Set a vector property with some ``name`` for some ``axis`` in the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/writers.html#DataAxesFormats.Writers.set_vector!>`__
        for details.

        If the provided ``value`` is numeric and dense, this passes a zero-copy view of the data to the ``Daf`` data
        set. Otherwise, a Python copy of the data is made (as a dense ``numpy`` array), and passed to ``Daf``.

        As a convenience, you can pass a 1xN or Nx1 matrix here and it will be mercifully interpreted as a vector. This
        allows creating sparse vectors in ``Daf`` by passing a 1xN slice of a sparse (column-major) Python matrix.

        Returns ``self`` for chaining.
        """
        if (isinstance(value, sp.csc_matrix) and value.shape[1] == 1) or (
            isinstance(value, sp.csr_matrix) and value.shape[0] == 1
        ):
            with self.empty_sparse_vector(
                axis,
                name,
                value.data.dtype,
                value.nnz,
                value.indptr.dtype,
                overwrite=overwrite,
            ) as (nzind, nzval):
                nzind[:] = value.indices[:]
                nzind += 1
                nzval[:] = value.data[:]
            return self

        if (isinstance(value, sp.csc_matrix) and value.shape[0] == 1) or (
            isinstance(value, sp.csr_matrix) and value.shape[1] == 1
        ):
            with self.empty_sparse_vector(
                axis,
                name,
                value.data.dtype,
                value.nnz,
                value.indptr.dtype,
                overwrite=overwrite,
            ) as (nzind, nzval):
                nzind[:] = np.where(np.ediff1d(value.indptr) == 1)[0]
                nzind += 1
                nzval[:] = value.data[:]
            return self

        jl.DataAxesFormats.set_vector_b(
            self.jl_obj, axis, name, _as_vector(_to_julia_array(value)), overwrite=overwrite
        )
        return self

    @contextmanager
    def empty_dense_vector(
        self, axis: str, name: str, eltype: Type, *, overwrite: bool = False
    ) -> Iterator[np.ndarray]:
        """
        Create an empty dense vector property with some ``name`` for some ``axis`` in the ``Daf`` data set, and pass it
        to the block to be filled. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/writers.html#DataAxesFormats.Writers.empty_dense_vector!>`__
        for details.

        Note this is a Python ``contextmanager``, that is, is meant to be used with the ``with`` statement:
        ``with empty_dense_vector(dset, ...) as empty_vector: ...``.
        """
        vector = jl.DataAxesFormats.get_empty_dense_vector_b(
            self.jl_obj, axis, name, _to_julia_type(eltype), overwrite=overwrite
        )
        try:
            yield _from_julia_array(vector, writeable=True)
        finally:
            jl.DataAxesFormats.end_data_write_lock(self.jl_obj)

    @contextmanager
    def empty_sparse_vector(  # pylint: disable=too-many-positional-arguments
        self, axis: str, name: str, eltype: Type, nnz: int, indtype: Type, *, overwrite: bool = False
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Create an empty sparse vector property with some ``name`` for some ``axis`` in the ``Daf`` data set, pass its
        parts (``nzind`` and ``nzval``) to the block to be filled. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/writers.html#DataAxesFormats.Writers.empty_sparse_vector!>`__
        for details.

        Note this is a Python ``contextmanager``, that is, is meant to be used with the ``with`` statement:
        ``with empty_sparse_vector(dset, ...) as (empty_nzind, empty_nzval): ...``. The arrays are to be filled with
        *Julia's* ``SparseVector`` data, that is, ``empty_nzind`` needs to be filled with **1**-based indices (as
        opposed to 0-based indices typically used by ``scipy.sparse``). Due to this difference in the indexing, we can't
        zero-copy share sparse data between Python and Julia. Sigh.
        """
        nzind, nzval = jl.DataAxesFormats.get_empty_sparse_vector_b(
            self.jl_obj, axis, name, _to_julia_type(eltype), nnz, _to_julia_type(indtype), overwrite=overwrite
        )
        try:
            yield (_from_julia_array(nzind, writeable=True), _from_julia_array(nzval, writeable=True))
            jl.DataAxesFormats.filled_empty_sparse_vector_b(self.jl_obj, axis, name, nzind, nzval)
        finally:
            jl.DataAxesFormats.end_data_write_lock(self.jl_obj)

    def delete_vector(self, axis: str, name: str, *, must_exist: bool = True) -> Self:
        """
        Delete a vector property with some ``name`` for some ``axis`` from the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/writers.html#DataAxesFormats.Writers.delete_vector!>`__
        for details.

        Returns ``self`` for chaining.
        """
        jl.DataAxesFormats.delete_vector_b(self.jl_obj, axis, name, must_exist=must_exist)
        return self

    def set_matrix(
        self,
        rows_axis: str,
        columns_axis: str,
        name: str,
        value: np.ndarray | sp.csc_matrix,
        *,
        overwrite: bool = False,
        relayout: bool = True,
    ) -> Self:
        """
        Set the matrix property with some ``name`` for some ``rows_axis`` and ``columns_axis`` in the ``Daf`` data set.
        See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/writers.html#DataAxesFormats.Writers.set_matrix!>`__
        for details.

        Since ``Daf`` is implemented Julia, this should be a column-major ``matrix``, so if you have a standard
        ``numpy`` or ``scipy`` row-major matrix, flip the order of the axes and pass the ``transpose`` (which is an
        efficient zero-copy operation).

        Returns ``self`` for chaining.
        """
        jl.DataAxesFormats.set_matrix_b(
            self.jl_obj, rows_axis, columns_axis, name, _to_julia_array(value), overwrite=overwrite, relayout=relayout
        )
        return self

    @contextmanager
    def empty_dense_matrix(
        self, rows_axis: str, columns_axis: str, name: str, eltype: Type, *, overwrite: bool = False
    ) -> Iterator[np.ndarray]:
        """
        Create an empty (column-major) dense matrix property with some ``name`` for some ``rows_axis`` and
        ``columns_axis`` in the ``Daf`` data set, and pass it to the block to be filled. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/writers.html#DataAxesFormats.Writers.empty_dense_matrix!>`__
        for details.

        Note this is a Python ``contextmanager``, that is, is meant to be used with the ``with`` statement:
        ``with empty_dense_matrix(dset, ...) as empty_matrix: ...``.
        """
        matrix = jl.DataAxesFormats.get_empty_dense_matrix_b(
            self.jl_obj, rows_axis, columns_axis, name, _to_julia_type(eltype), overwrite=overwrite
        )
        try:
            yield _from_julia_array(matrix, writeable=True)
        finally:
            jl.DataAxesFormats.end_data_write_lock(self.jl_obj)

    @contextmanager
    def empty_sparse_matrix(  # pylint: disable=too-many-positional-arguments
        self,
        rows_axis: str,
        columns_axis: str,
        name: str,
        eltype: Type,
        nnz: int,
        indtype: Type,
        *,
        overwrite: bool = False,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create an empty (column-major) sparse matrix property with some ``name`` for some ``rows_axis`` and
        ``columns_axis`` in the ``Daf`` data set, and pass its parts (``colptr``, ``rowval`` and ``nzval``) to the block
        to be filles. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/writers.html#DataAxesFormats.Writers.empty_sparse_matrix!>`__
        for details.

        Note this is a Python ``contextmanager``, that is, is meant to be used with the ``with`` statement:
        ``with empty_sparse_vector(dset, ...) as (empty_colptr, empty_rowval, empty_nzval): ...``. The arrays are to be
        filled with *Julia's* ``SparseVector`` data, that is, ``empty_colptr`` and ``empty_rowval`` need to be filled
        with **1**-based indices (as opposed to 0-based indices used by ``scipy.sparse.cs[cr]_matrix``). Due to this
        difference in the indexing, we can't zero-copy share sparse data between Python and Julia. Sigh.
        """
        colptr, rowval, nzval = jl.DataAxesFormats.get_empty_sparse_matrix_b(
            self.jl_obj,
            rows_axis,
            columns_axis,
            name,
            _to_julia_type(eltype),
            nnz,
            _to_julia_type(indtype),
            overwrite=overwrite,
        )
        try:
            yield (
                _from_julia_array(colptr, writeable=True),
                _from_julia_array(rowval, writeable=True),
                _from_julia_array(nzval, writeable=True),
            )
            jl.DataAxesFormats.filled_empty_sparse_matrix_b(
                self.jl_obj, rows_axis, columns_axis, name, colptr, rowval, nzval
            )
        finally:
            jl.DataAxesFormats.end_data_write_lock(self.jl_obj)

    def relayout_matrix(self, rows_axis: str, columns_axis: str, name: str, *, overwrite: bool = False) -> Self:
        """
        Given a matrix property with some ``name`` exists (in column-major layout) in the ``Daf`` data set for the
        ``rows_axis`` and the ``columns_axis``, then relayout it and store the row-major result as well (that is, with
        flipped axes). See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/writers.html#DataAxesFormats.Writers.relayout_matrix!>`__ for
        details.

        Returns ``self`` for chaining.
        """
        jl.DataAxesFormats.relayout_matrix_b(self.jl_obj, rows_axis, columns_axis, name, overwrite=overwrite)
        return self

    def delete_matrix(self, rows_axis: str, columns_axis: str, name: str, *, must_exist: bool = True) -> Self:
        """
        Delete a matrix property with some ``name`` for some ``rows_axis`` and ``columns_axis`` from the ``Daf`` data
        set. See the Julia
        `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.2/writers.html#DataAxesFormats.Writers.delete_matrix!>`__ for
        details.

        Returns ``self`` for chaining.
        """
        jl.DataAxesFormats.delete_matrix_b(self.jl_obj, rows_axis, columns_axis, name, must_exist=must_exist)
        return self
