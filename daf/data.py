"""
Interface of ``DafReader`` and ``DafWriter``. See the Julia
`documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html>`_ for details.
"""

from contextlib import contextmanager
from typing import AbstractSet
from typing import Any
from typing import Iterator
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import overload
from weakref import WeakValueDictionary

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

from .julia_import import Undef
from .julia_import import UndefInitializer
from .julia_import import jl
from .storage_types import StorageScalar

__all__ = ["DafReader", "DafWriter"]


class DafReader:
    """
    Read-only access to ``Daf`` data. See the Julia
    `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html>`_ for details.
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
        Return a (multi-line) description of the contents of ``Daf`` data. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.description>`_ for details.
        """
        return jl.Daf.description(self.daf_jl, deep=deep)

    def has_scalar(self, name: str) -> bool:
        """
        Check whether a scalar property with some ``name`` exists in the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.has_scalar>`_ for details.
        """
        return jl.Daf.has_scalar(self.daf_jl, name)

    def get_scalar(self, name: str) -> StorageScalar:
        """
        Get the value of a scalar property with some ``name`` in the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.get_scalar>`_ for details.

        Numeric scalars are always returned as ``int`` or ``float``, regardless of the specific data type they are
        stored in the ``Daf`` data set (e.g., a ``UInt8`` will be returned as an ``int`` instead of a ``np.uint8``).
        """
        return jl.Daf.get_scalar(self.daf_jl, name)

    def scalar_names(self) -> AbstractSet[str]:
        """
        The names of the scalar properties in the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.scalar_names>`_ for details.
        """
        return jl.Daf.scalar_names(self.daf_jl)

    def has_axis(self, axis: str) -> bool:
        """
        Check whether some ``axis`` exists in the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.has_axis>`_ for details.
        """
        return jl.Daf.has_axis(self.daf_jl, axis)

    def axis_names(self) -> AbstractSet[str]:
        """
        The names of the axes of the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.axis_names>`_ for details.
        """
        return jl.Daf.axis_names(self.daf_jl)

    def axis_length(self, axis: str) -> int:
        """
        The number of entries along the ``axis`` i the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.axis_length>`_ for details.
        """
        return jl.Daf.axis_length(self.daf_jl, axis)

    def get_axis(self, axis: str) -> np.ndarray:
        """
        The unique names of the entries of some ``axis`` of the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.get_axis>`_ for details.

        This creates an in-memory copy of the data, which is cached for repeated calls.
        """
        axis_version_counter = jl.Daf.axis_version_counter(self.daf_jl, axis)
        axis_key = (axis_version_counter, axis)
        axis_entries = self.weakrefs.get(axis_key)
        if axis_entries is None:
            axis_entries = _from_julia_array(jl.Daf.get_axis(self.daf_jl, axis))
            self.weakrefs[axis_key] = axis_entries
        return axis_entries

    def has_vector(self, axis: str, name: str) -> bool:
        """
        Check whether a vector property with some ``name`` exists for the ``axis`` in the ``Daf`` data set. See the
        Julia `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.has_vector>`_ for details.
        """
        return jl.Daf.has_vector(self.daf_jl, axis, name)

    def vector_names(self, axis: str) -> AbstractSet[str]:
        """
        The names of the vector properties for the ``axis`` in ``Daf`` data set, **not** including the special ``name``
        property. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.vector_names>`_ for details.
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
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.get_vector>`_ for details.

        This always returns a ``numpy`` vector (unless ``default`` is ``None`` and the vector does not exist). If the
        stored data is numeric and dense, this is a zero-copy view of the data stored in the ``Daf`` data set.
        Otherwise, a Python copy of the data as a dense ``numpy`` array is returned (and cached for repeated calls).
        Since Python has no concept of sparse vectors (because "reasons"), you can't zero-copy view a sparse ``Daf``
        vector using the Python API.
        """
        if not jl.Daf.has_vector(self.daf_jl, axis, name):
            if default is None:
                return None
            return _from_julia_array(jl.Daf.get_vector(self.daf_jl, axis, name, default=_to_julia(default)).array)

        vector_version_counter = jl.Daf.vector_version_counter(self.daf_jl, axis, name)
        vector_key = (vector_version_counter, axis, name)
        vector_value = self.weakrefs.get(vector_key)
        if vector_value is None:
            vector_value = _from_julia_array(jl.Daf.get_vector(self.daf_jl, axis, name).array)
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
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.get_vector>`_ for details.

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
        ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.has_matrix>`_ for details.
        """
        return jl.Daf.has_matrix(self.daf_jl, rows_axis, columns_axis, name, relayout=relayout)

    def matrix_names(self, rows_axis: str, columns_axis: str, *, relayout: bool = True) -> AbstractSet[str]:
        """
        The names of the matrix properties for the ``rows_axis`` and ``columns_axis`` in the ``Daf`` data set. See the
        Julia `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.matrix_names>`_ for details.
        """
        return jl.Daf.matrix_names(self.daf_jl, rows_axis, columns_axis, relayout=relayout)

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
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.get_matrix>`_ for details.

        This always returns a column-major ``numpy`` matrix or a ``scipy`` sparse ``csc_matrix``, (unless ``default`` is
        ``None`` and the matrix does not exist). If the stored data is numeric and dense, this is a zero-copy view of
        the data stored in the ``Daf`` data set.

        Note that by default ``numpy`` matrices are in row-major (C) layout and not in column-major (Fortran) layout. To
        get a row-major matrix, simply flip the order of the axes, and call transpose on the result (which is an
        efficient zero-copy operation). This will also (zero-copy) convert the ``csc_matrix`` into a ``csr_matrix``.

        Also note that although we call this ``get_np_matrix``, the result is **not** the deprecated ``np.matrix``
        (which is to be avoided at all costs).
        """
        if not jl.Daf.has_matrix(self.daf_jl, rows_axis, columns_axis, name, relayout=relayout):
            if default is None:
                return None
            return _from_julia_array(
                jl.Daf.get_matrix(
                    self.daf_jl, rows_axis, columns_axis, name, default=_to_julia(default), relayout=relayout
                ).array
            )

        matrix_version_counter = jl.Daf.matrix_version_counter(self.daf_jl, rows_axis, columns_axis, name)
        matrix_key = (matrix_version_counter, rows_axis, columns_axis, name)
        matrix_value = self.weakrefs.get(matrix_key)
        if matrix_value is None:
            matrix_value = _from_julia_array(
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
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.get_matrix>`_ for details.

        This is a wrapper around ``get_np_matrix`` which returns a ``pandas`` data frame using the entry names of the
        axes as the indices. Since ``pandas`` data frames can't contain a sparse matrix, the data will always be
        in a dense ``numpy`` matrix, so take care not to invoke this for a too-large sparse data matrix.

        This is not to be confused with ``get_frame`` which returns a "real" ``pandas`` data frame, with arbitrary
        (query) columns, possibly using a different data type for each.
        """
        matrix_value = self.get_np_matrix(rows_axis, columns_axis, name, default=_to_julia(default), relayout=relayout)
        if matrix_value is None:
            return None
        if sp.issparse(matrix_value):
            matrix_value = matrix_value.toarray()
        return pd.DataFrame(matrix_value, index=self.get_axis(rows_axis), columns=self.get_axis(columns_axis))


class DafWriter(DafReader):
    """
    Read-write access to ``Daf`` data.
    """

    def set_scalar(self, name: str, value: StorageScalar, *, overwrite: bool = False) -> None:
        """
        Set the ``value`` of a scalar property with some ``name`` in a ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.set_scalar!>`_ for details.

        You can force the data type numeric scalars are stored in by using the appropriate ``numpy`` type (e.g., a
        ``np.uint8`` will be stored as a ``UInt8``).
        """
        jl.Daf.set_scalar_b(self.daf_jl, name, value, overwrite=overwrite)

    def delete_scalar(self, name: str, *, must_exist: bool = True) -> None:
        """
        Delete a scalar property with some ``name`` from the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.delete_scalar!>`_ for details.
        """
        jl.Daf.delete_scalar_b(self.daf_jl, name, must_exist=must_exist)

    def add_axis(self, axis: str, entries: Sequence[str] | np.ndarray) -> None:
        """
        Add a new ``axis`` to the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.add_axis!>`_ for details.
        """
        jl.Daf.add_axis_b(self.daf_jl, axis, _to_julia(entries))

    def delete_axis(self, axis: str, *, must_exist: bool = True) -> None:
        """
        Delete an ``axis`` from the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.delete_axis!>`_ for details.
        """
        jl.Daf.delete_axis_b(self.daf_jl, axis, must_exist=must_exist)

    def set_vector(
        self,
        axis: str,
        name: str,
        value: Sequence[StorageScalar] | np.ndarray | sp.csc_matrix | sp.csr_matrix,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Set a vector property with some ``name`` for some ``axis`` in the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.set_vector!>`_ for details.

        If the provided ``value`` is numeric and dense, this passes a zero-copy view of the data to the ``Daf`` data
        set. Otherwise, a Python copy of the data is made (as a dense ``numpy`` array), and passed to ``Daf``.

        As a convenience, you can pass a 1xN or Nx1 matrix here and it will be mercifully interpreted as a vector. This
        allows creating sparse vectors in ``Daf`` by passing a 1xN slice of a sparse (column-major) Python matrix.
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
            return

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
            return

        jl.Daf.set_vector_b(self.daf_jl, axis, name, _as_vector(_to_julia(value)), overwrite=overwrite)

    @contextmanager
    def empty_dense_vector(
        self, axis: str, name: str, eltype: Type, *, overwrite: bool = False
    ) -> Iterator[np.ndarray]:
        """
        Create an empty dense vector property with some ``name`` for some ``axis`` in the ``Daf`` data set, and pass it
        to the block to be filled. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.empty_dense_vector!>`_ for details.
        """
        vector = jl.Daf.get_empty_dense_vector_b(self.daf_jl, axis, name, _to_julia(eltype), overwrite=overwrite)
        try:
            yield _from_julia_array(vector)
            jl.Daf.filled_empty_dense_vector_b(self.daf_jl, axis, name, vector, overwrite=overwrite)
        finally:
            jl.Daf.end_write_lock(self.daf_jl)

    @contextmanager
    def empty_sparse_vector(
        self, axis: str, name: str, eltype: Type, nnz: int, indtype: Type, *, overwrite: bool = False
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Create an empty sparse vector property with some ``name`` for some ``axis`` in the ``Daf`` data set, pass its
        parts (``nzind`` and ``nzval``) to the block to be filled. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.empty_sparse_vector!>`_ for details.

        Note that the code block will get a tuple of ``(nzind, nzval)`` arrays for *Julia's* ``SparseVector``, **not** a
        tuple of ``(data, indices, indptr)`` for Python's ``csc_matrix``. First, ``numpy`` (that is, ``scipy``) has no
        concept of sparse vectors. In addition ``nzind`` is 1-based (Julia) and not 0-based (Python).
        """
        nzind, nzval, extra = jl.Daf.get_empty_sparse_vector_b(
            self.daf_jl, axis, name, _to_julia(eltype), nnz, _to_julia(indtype), overwrite=overwrite
        )
        try:
            yield (_from_julia_array(nzind), _from_julia_array(nzval))
            jl.Daf.filled_empty_sparse_vector_b(self.daf_jl, axis, name, nzind, nzval, extra, overwrite=overwrite)
        finally:
            jl.Daf.end_write_lock(self.daf_jl)

    def delete_vector(self, axis: str, name: str, *, must_exist: bool = True) -> None:
        """
        Delete a vector property with some ``name`` for some ``axis`` from the ``Daf`` data set. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.delete_vector!>`_ for details.
        """
        jl.Daf.delete_vector_b(self.daf_jl, axis, name, must_exist=must_exist)

    def set_matrix(
        self,
        rows_axis: str,
        columns_axis: str,
        name: str,
        value: np.ndarray | sp.csc_matrix,
        *,
        overwrite: bool = False,
        relayout: bool = True,
    ) -> None:
        """
        Set the matrix property with some ``name`` for some ``rows_axis`` and ``columns_axis`` in the ``Daf`` data set.
        See the Julia `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.set_matrix!>`_ for
        details.

        Since ``Daf`` is implemented Julia, this should be a column-major ``matrix``, so if you have a standard
        ``numpy`` or ``scipy`` row-major matrix, flip the order of the axes and pass the ``transpose`` (which is an
        efficient zero-copy operation).
        """
        jl.Daf.set_matrix_b(
            self.daf_jl, rows_axis, columns_axis, name, _to_julia(value), overwrite=overwrite, relayout=relayout
        )

    @contextmanager
    def empty_dense_matrix(
        self, rows_axis: str, columns_axis: str, name: str, eltype: Type, *, overwrite: bool = False
    ) -> Iterator[np.ndarray]:
        """
        Create an empty (column-major) dense matrix property with some ``name`` for some ``rows_axis`` and
        ``columns_axis`` in the ``Daf`` data set, and pass it to the block to be filled. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.empty_dense_matrix!>`_ for details.
        """
        matrix = jl.Daf.get_empty_dense_matrix_b(
            self.daf_jl, rows_axis, columns_axis, name, _to_julia(eltype), overwrite=overwrite
        )
        try:
            yield _from_julia_array(matrix)
            jl.Daf.filled_empty_dense_matrix_b(self.daf_jl, rows_axis, columns_axis, name, matrix, overwrite=overwrite)
        finally:
            jl.Daf.end_write_lock(self.daf_jl)

    @contextmanager
    def empty_sparse_matrix(
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
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.empty_sparse_matrix!>`_ for details.

        Note that the code block will get a tuple of ``(colptr, rowval, nzval)`` arrays for Julia's ``SparseMatrixCSC``,
        **not** a tuple of ``(data, indices, indptr)`` for Python's ``csc_matrix``. Yes, ``data`` is the same as
        ``nzval``, but ``colptr = indptr + 1`` and ``rowval = indices + 1``, because Julia uses 1-based indexing, and
        Python uses 0-based indexing. For this reason, sparse data can't ever be zero-copy between Julia and Python.
        Sigh.
        """
        colptr, rowval, nzval, extra = jl.Daf.get_empty_sparse_matrix_b(
            self.daf_jl, rows_axis, columns_axis, name, _to_julia(eltype), nnz, _to_julia(indtype), overwrite=overwrite
        )
        try:
            yield (_from_julia_array(colptr), _from_julia_array(rowval), _from_julia_array(nzval))
            jl.Daf.filled_empty_sparse_matrix_b(
                self.daf_jl, rows_axis, columns_axis, name, colptr, rowval, nzval, extra, overwrite=overwrite
            )
        finally:
            jl.Daf.end_write_lock(self.daf_jl)

    def relayout_matrix(self, rows_axis: str, columns_axis: str, name: str, *, overwrite: bool = False) -> None:
        """
        Given a matrix property with some ``name`` exists (in column-major layout) in the ``Daf`` data set for the
        ``rows_axis`` and the ``columns_axis``, then relayout it and store the row-major result as well (that is, with
        flipped axes). See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.relayout_matrix!>`_ for details.
        """
        jl.Daf.relayout_matrix_b(self.daf_jl, rows_axis, columns_axis, name, overwrite=overwrite)

    def delete_matrix(self, rows_axis: str, columns_axis: str, name: str, *, must_exist: bool = True) -> None:
        """
        Delete a matrix property with some ``name`` for some ``rows_axis`` and ``columns_axis`` from the ``Daf`` data
        set. See the Julia
        `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/data.html#Daf.Data.delete_matrix!>`_ for details.
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
    if isinstance(value, np.dtype):
        return JULIA_TYPE_OF_PY_TYPE[value.type]

    if isinstance(value, type):
        return JULIA_TYPE_OF_PY_TYPE[value]

    if isinstance(value, UndefInitializer):
        return jl.undef

    if isinstance(value, (sp.csc_matrix, sp.csr_matrix)):
        colptr = jl.Vector(value.indptr)
        rowval = jl.Vector(value.indices)
        nzval = jl.Vector(value.data)

        colptr_as_array = np.asarray(colptr)
        rowval_as_array = np.asarray(rowval)

        colptr_as_array += 1
        rowval_as_array += 1

        nrows, ncols = value.shape
        if isinstance(value, sp.csr_matrix):
            nrows, ncols = ncols, nrows

        julia_matrix = jl.SparseArrays.SparseMatrixCSC(nrows, ncols, colptr, rowval, nzval)

        if isinstance(value, sp.csr_matrix):
            julia_matrix = jl.LinearAlgebra.transpose(julia_matrix)

        return julia_matrix

    if isinstance(value, Sequence) and not isinstance(value, np.ndarray):
        value = np.array(value)

    if isinstance(value, np.ndarray) and value.dtype.type == np.str_:
        value = jl.Vector(value)

    return value


def _from_julia_array(julia_array: Any) -> np.ndarray | sp.csc_matrix:
    try:
        indptr = np.array(julia_array.colptr)
        indptr -= 1
        indices = np.array(julia_array.rowval)
        indices -= 1
        data = np.asarray(julia_array.nzval)
        return sp.csc_matrix((data, indices, indptr), julia_array.shape)
    except:
        pass

    python_array = np.asarray(julia_array)
    if python_array.dtype == "object":
        python_array = np.array([str(obj) for obj in python_array], dtype=str)
    return python_array


def _as_vector(vector_ish: Any) -> Any:
    if isinstance(vector_ish, np.ndarray):
        shape = vector_ish.shape
        if len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
            vector_ish = vector_ish.reshape(-1)
    return vector_ish
