"""
Test ``Daf`` storage formats.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from contextlib import contextmanager
from textwrap import dedent
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Never
from typing import Sequence
from typing import Tuple

import numpy as np
import pytest
import scipy.sparse as sp  # type: ignore

from daf import *

FORMATS = [("MemoryDaf", lambda: MemoryDaf(name="test!"))]


@contextmanager
def assert_raises(expected: str) -> Iterator[Never]:
    try:
        yield  # type: ignore
        raise AssertionError("no exception was thrown")
    except Exception as exception:  # pylint: disable=broad-exception-caught
        actual = str(exception)
        if expected not in actual:
            raise exception


@pytest.mark.parametrize("format_data", FORMATS)
@pytest.mark.parametrize("scalar_data", [("1.0.1", "String"), (np.int8(1), "Int8"), (0.5, "Float64")])
def test_scalars(format_data: Tuple[str, Callable[[], DafWriter]], scalar_data: Tuple[StorageScalar, str]) -> None:
    format_name, create_empty = format_data
    scalar_value, julia_type = scalar_data

    data = create_empty()
    assert data.name == "test!"

    assert len(data.scalar_names()) == 0
    assert not data.has_scalar("foo")
    data.set_scalar("foo", scalar_value)
    assert data.has_scalar("foo")
    assert data.get_scalar("foo") == scalar_value
    assert set(data.scalar_names()) == set(["foo"])

    if julia_type == "String":
        scalar_value = '"' + str(scalar_value) + '"'
    else:
        scalar_value = f"{scalar_value} ({julia_type})"

    assert (
        data.description()
        == dedent(
            f"""
                name: test!
                type: {format_name}
                scalars:
                  foo: {scalar_value}
            """
        )[1:]
    )

    data.delete_scalar("foo")
    assert len(data.scalar_names()) == 0
    assert not data.has_scalar("foo")


@pytest.mark.parametrize("format_data", FORMATS)
@pytest.mark.parametrize("axis_data", [("cell", ["A", "B"]), ("gene", np.array(["X", "Y", "Z"]))])
def test_axes(format_data: Tuple[str, Callable[[], DafWriter]], axis_data: Tuple[str, Sequence[str]]) -> None:
    format_name, create_empty = format_data
    axis_name, axis_entries = axis_data

    data = create_empty()

    assert len(data.axis_names()) == 0
    assert not data.has_axis(axis_name)

    data.add_axis(axis_name, axis_entries)

    assert data.has_axis(axis_name)
    assert data.axis_length(axis_name) == len(axis_entries)
    assert list(data.get_axis(axis_name)) == list(axis_entries)
    assert set(data.axis_names()) == set([axis_name])
    assert (
        data.description()
        == dedent(
            f"""
                name: test!
                type: {format_name}
                axes:
                  {axis_name}: {len(axis_entries)} entries
            """
        )[1:]
    )

    data.delete_axis(axis_name)

    assert len(data.axis_names()) == 0
    assert not data.has_axis(axis_name)


@pytest.mark.parametrize("format_data", FORMATS)
def test_vectors_defaults(format_data: Tuple[str, Callable[[], DafWriter]]) -> None:
    _format_name, create_empty = format_data

    data = create_empty()
    data.add_axis("cell", ["A", "B"])

    with assert_raises(
        dedent(
            """
                missing vector: foo
                for the axis: cell
                in the daf data: test!
            """[
                1:
            ]
        )
    ):
        data.get_np_vector("cell", "foo")

    assert data.get_np_vector("cell", "foo", default=None) is None
    assert data.get_pd_vector("cell", "foo", default=None) is None
    assert list(data.get_np_vector("cell", "foo", default=1)) == [1, 1]
    assert list(data.get_np_vector("cell", "foo", default=[1, 2])) == [1, 2]
    assert list(data.get_np_vector("cell", "foo", default=np.array([1, 2]))) == [1, 2]


@pytest.mark.parametrize("format_data", FORMATS)
@pytest.mark.parametrize(
    "vector_data",
    [(["X", "Y"], "String"), ([1, 2], "Int64"), ([1.0, 2.0], "Float64"), (np.array([[1, 2]], dtype="i1"), "Int8")],
)
def test_dense_vectors(format_data: Tuple[str, Callable[[], DafWriter]], vector_data: Tuple[Any, str]) -> None:
    format_name, create_empty = format_data
    vector_entries, julia_type = vector_data

    data = create_empty()
    data.add_axis("cell", ["A", "B"])
    assert len(data.vector_names("cell")) == 0
    assert not data.has_vector("cell", "foo")

    vector_entries = np.array(vector_entries)
    data.set_vector("cell", "foo", vector_entries)

    assert data.has_vector("cell", "foo")
    assert set(data.vector_names("cell")) == set(["foo"])
    stored_vector = data.get_np_vector("cell", "foo")
    stored_series = data.get_pd_vector("cell", "foo")
    repeated_vector = data.get_np_vector("cell", "foo")
    assert id(repeated_vector) == id(stored_vector)

    assert list(stored_vector) == list(vector_entries.reshape(-1))
    assert list(stored_series.values) == list(vector_entries.reshape(-1))
    assert list(stored_series.index) == ["A", "B"]
    if isinstance(stored_vector[0], str):
        julia_type = f"PythonCall.Utils.Static{julia_type}{{UInt32, 1}}"
    else:
        vector_entries.reshape(-1)[0] = vector_entries.reshape(-1)[1]
        assert list(stored_vector) == list(vector_entries.reshape(-1))
        assert list(stored_series.values) == list(vector_entries.reshape(-1))
    assert (
        data.description()
        == dedent(
            f"""
                name: test!
                type: {format_name}
                axes:
                  cell: 2 entries
                vectors:
                  cell:
                    foo: 2 x {julia_type} (PyArray{{{julia_type}, 1, true, true, {julia_type}}} - Dense)
            """
        )[1:]
    )

    data.delete_vector("cell", "foo")

    assert len(data.vector_names("cell")) == 0
    assert not data.has_vector("cell", "foo")

    data.set_vector("cell", "foo", list(vector_entries.reshape(-1)))
    new_vector = data.get_np_vector("cell", "foo")
    assert id(new_vector) != id(stored_vector)

    if isinstance(stored_vector[0], str):
        return

    with data.empty_dense_vector("cell", "foo", np.float32, overwrite=True) as empty_vector:
        empty_vector[:] = [-1.5, 2.5]
    assert np.all(data.get_np_vector("cell", "foo") == np.array([-1.5, 2.5]))

    with data.empty_sparse_vector("cell", "foo", np.float32, 1, np.int8, overwrite=True) as (
        nzind,
        nzval,
    ):
        nzind[0] = 2
        nzval[0] = 2.5
    assert np.all(data.get_pd_vector("cell", "foo").values == np.array([0.0, 2.5]))

    sparse_vector = sp.csc_matrix(np.array([[2.5, 0.0]]))
    data.set_vector("cell", "foo", sparse_vector, overwrite=True)
    assert np.all(data.get_np_vector("cell", "foo") == np.array([2.5, 0.0]))

    sparse_vector = sp.csr_matrix(np.array([[0.0, 2.5]]))
    data.set_vector("cell", "foo", sparse_vector, overwrite=True)
    assert np.all(data.get_np_vector("cell", "foo") == np.array([0.0, 2.5]))


@pytest.mark.parametrize("format_data", FORMATS)
def test_matrices_defaults(format_data: Tuple[str, Callable[[], DafWriter]]) -> None:
    _format_name, create_empty = format_data

    data = create_empty()
    data.add_axis("cell", ["A", "B"])
    data.add_axis("gene", ["X", "Y", "Z"])

    with assert_raises(
        dedent(
            """
                missing matrix: UMIs
                for the rows axis: cell
                and the columns axis: gene
                in the daf data: test!
            """[
                1:
            ]
        )
    ):
        data.get_np_matrix("cell", "gene", "UMIs", relayout=False)

    assert data.get_np_matrix("cell", "gene", "UMIs", default=None) is None
    assert data.get_pd_matrix("cell", "gene", "UMIs", default=None) is None
    assert np.all(data.get_np_matrix("cell", "gene", "UMIs", default=1) == np.array([[1, 1, 1], [1, 1, 1]]))
    default_matrix = np.array([[1, 2], [3, 4], [5, 6]]).transpose()
    assert id(data.get_np_matrix("cell", "gene", "UMIs", default=default_matrix)) == id(default_matrix)

    fill_matrix = np.array([[0.0, 2.5], [3.5, 0.0], [0.0, 6.5]]).transpose()
    with data.empty_dense_matrix("cell", "gene", "UMIs", np.float32, overwrite=True) as empty_matrix:
        empty_matrix[:, :] = fill_matrix
    assert np.all(data.get_np_matrix("cell", "gene", "UMIs") == fill_matrix)

    sparse_matrix = sp.csc_matrix(fill_matrix)
    with data.empty_sparse_matrix("cell", "gene", "UMIs", np.float32, sparse_matrix.nnz, np.int8, overwrite=True) as (
        colptr,
        rowval,
        nzval,
    ):
        colptr[:] = sparse_matrix.indptr[:]
        colptr += 1
        rowval[:] = sparse_matrix.indices[:]
        rowval += 1
        nzval[:] = sparse_matrix.data[:]
    assert isinstance(data.get_np_matrix("cell", "gene", "UMIs"), sp.csc_matrix)
    assert np.all(data.get_pd_matrix("cell", "gene", "UMIs").values == fill_matrix)

    fill_matrix = np.array([[0.0, 1.5], [2.5, 0.0], [0.0, 3.5]]).transpose()
    data.set_matrix("cell", "gene", "UMIs", sp.csc_matrix(fill_matrix), overwrite=True)
    assert isinstance(data.get_np_matrix("cell", "gene", "UMIs"), sp.csc_matrix)
    assert np.all(data.get_pd_matrix("cell", "gene", "UMIs").values == fill_matrix)

    with assert_raises("type not in column-major layout: 2 x 3 x Float64 in Rows (transposed Sparse 50%)"):
        data.set_matrix("cell", "gene", "UMIs", sp.csr_matrix(fill_matrix), overwrite=True)


@pytest.mark.parametrize("format_data", FORMATS)
def test_dense_matrices(format_data: Tuple[str, Callable[[], DafWriter]]) -> None:
    format_name, create_empty = format_data

    data = create_empty()
    data.add_axis("cell", ["A", "B"])
    data.add_axis("gene", ["X", "Y", "Z"])

    assert len(data.matrix_names("cell", "gene", relayout=False)) == 0
    assert not data.has_matrix("cell", "gene", "UMIs", relayout=False)

    row_major_umis = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int8)
    column_major_umis = row_major_umis.transpose()

    data.set_matrix("cell", "gene", "UMIs", column_major_umis, relayout=False)

    assert data.has_matrix("cell", "gene", "UMIs", relayout=False)
    assert set(data.matrix_names("cell", "gene", relayout=False)) == set(["UMIs"])
    stored_matrix = data.get_np_matrix("cell", "gene", "UMIs", relayout=False)
    stored_frame = data.get_pd_matrix("cell", "gene", "UMIs", relayout=False)
    assert list(stored_frame.index) == ["A", "B"]
    assert list(stored_frame.columns) == ["X", "Y", "Z"]
    repeated_matrix = data.get_np_matrix("cell", "gene", "UMIs", relayout=False)
    assert id(repeated_matrix) == id(stored_matrix)

    assert np.all(stored_matrix == column_major_umis)
    column_major_umis[1, 2] += 1
    assert np.all(stored_matrix == column_major_umis)

    assert (
        data.description()
        == dedent(
            f"""
                name: test!
                type: {format_name}
                axes:
                  cell: 2 entries
                  gene: 3 entries
                matrices:
                  cell,gene:
                    UMIs: 2 x 3 x Int8 in Columns (PyArray{{Int8, 2, true, true, Int8}} - Dense)
            """
        )[1:]
    )

    assert not data.has_matrix("gene", "cell", "UMIs", relayout=False)
    data.relayout_matrix("cell", "gene", "UMIs")
    assert data.has_matrix("gene", "cell", "UMIs", relayout=False)
    assert np.all(
        data.get_np_matrix("cell", "gene", "UMIs", relayout=False)
        == data.get_np_matrix("gene", "cell", "UMIs", relayout=False).transpose()
    )

    assert (
        data.description()
        == dedent(
            f"""
                name: test!
                type: {format_name}
                axes:
                  cell: 2 entries
                  gene: 3 entries
                matrices:
                  cell,gene:
                    UMIs: 2 x 3 x Int8 in Columns (PyArray{{Int8, 2, true, true, Int8}} - Dense)
                  gene,cell:
                    UMIs: 3 x 2 x Int8 in Columns (Dense)
            """
        )[1:]
    )

    data.delete_matrix("cell", "gene", "UMIs")

    assert len(data.matrix_names("cell", "gene")) == 0
    assert not data.has_matrix("cell", "gene", "UMIs", relayout=False)
