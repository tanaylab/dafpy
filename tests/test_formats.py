"""
Test ``Daf`` storage formats.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from contextlib import contextmanager
from tempfile import TemporaryDirectory
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


def make_files() -> FilesDaf:
    tmpdir = TemporaryDirectory()  # pylint: disable=consider-using-with
    files = FilesDaf(tmpdir.name, "w", name="test!")
    setattr(files, "__gc_anchor__", tmpdir)
    return files


def make_h5df() -> H5df:
    tmpdir = TemporaryDirectory()  # pylint: disable=consider-using-with
    h5df = H5df(tmpdir.name + "/test.h5df", "w", name="test!")
    setattr(h5df, "__gc_anchor__", tmpdir)
    return h5df


FORMATS = [("MemoryDaf", lambda: MemoryDaf(name="test!")), ("FilesDaf", make_files), ("H5df", make_h5df)]


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

    dset = create_empty()
    assert dset.name == "test!"

    assert len(dset.scalar_names()) == 0
    assert not dset.has_scalar("foo")
    dset.set_scalar("foo", scalar_value)
    assert dset.has_scalar("foo")
    assert dset.get_scalar("foo") == scalar_value
    assert set(dset.scalar_names()) == set(["foo"])

    if julia_type == "String":
        scalar_value = '"' + str(scalar_value) + '"'
    else:
        scalar_value = f"{scalar_value} ({julia_type})"

    assert (
        dset.description()
        == dedent(
            f"""
                name: test!
                type: {format_name}
                scalars:
                  foo: {scalar_value}
            """
        )[1:]
    )

    dset.delete_scalar("foo")
    assert len(dset.scalar_names()) == 0
    assert not dset.has_scalar("foo")


@pytest.mark.parametrize("format_data", FORMATS)
@pytest.mark.parametrize("axis_data", [("cell", ["A", "B"]), ("gene", np.array(["X", "Y", "Z"]))])
def test_axes(format_data: Tuple[str, Callable[[], DafWriter]], axis_data: Tuple[str, Sequence[str]]) -> None:
    format_name, create_empty = format_data
    axis_name, axis_entries = axis_data

    dset = create_empty()

    assert len(dset.axis_names()) == 0
    assert not dset.has_axis(axis_name)

    dset.add_axis(axis_name, axis_entries)

    assert dset.has_axis(axis_name)
    assert dset.axis_length(axis_name) == len(axis_entries)
    assert list(dset.get_axis(axis_name)) == list(axis_entries)
    assert set(dset.axis_names()) == set([axis_name])
    assert (
        dset.description()
        == dedent(
            f"""
                name: test!
                type: {format_name}
                axes:
                  {axis_name}: {len(axis_entries)} entries
            """
        )[1:]
    )

    dset.delete_axis(axis_name)

    assert len(dset.axis_names()) == 0
    assert not dset.has_axis(axis_name)


@pytest.mark.parametrize("format_data", FORMATS)
def test_vectors_defaults(format_data: Tuple[str, Callable[[], DafWriter]]) -> None:
    _format_name, create_empty = format_data

    dset = create_empty()
    dset.add_axis("cell", ["A", "B"])

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
        dset.get_np_vector("cell", "foo")

    assert dset.get_np_vector("cell", "foo", default=None) is None
    assert dset.get_pd_vector("cell", "foo", default=None) is None
    assert list(dset.get_np_vector("cell", "foo", default=1)) == [1, 1]
    assert list(dset.get_np_vector("cell", "foo", default=[1, 2])) == [1, 2]
    assert list(dset.get_np_vector("cell", "foo", default=np.array([1, 2]))) == [1, 2]


@pytest.mark.parametrize("format_data", FORMATS)
@pytest.mark.parametrize(
    "vector_data",
    [(["X", "Y"], "String"), ([1, 2], "Int64"), ([1.0, 2.0], "Float64"), (np.array([[1, 2]], dtype="i1"), "Int8")],
)
def test_dense_vectors(  # pylint: disable=too-many-locals
    format_data: Tuple[str, Callable[[], DafWriter]], vector_data: Tuple[Any, str]
) -> None:
    format_name, create_empty = format_data
    vector_entries, julia_type = vector_data

    dset = create_empty()
    dset.add_axis("cell", ["A", "B"])
    assert len(dset.vector_names("cell")) == 0
    assert not dset.has_vector("cell", "foo")

    vector_entries = np.array(vector_entries)
    dset.set_vector("cell", "foo", vector_entries)

    assert dset.has_vector("cell", "foo")
    assert set(dset.vector_names("cell")) == set(["foo"])
    stored_vector = dset.get_np_vector("cell", "foo")
    stored_series = dset.get_pd_vector("cell", "foo")
    repeated_vector = dset.get_np_vector("cell", "foo")
    assert id(repeated_vector) == id(stored_vector)

    assert list(stored_vector) == list(vector_entries.reshape(-1))
    assert list(stored_series.values) == list(vector_entries.reshape(-1))
    assert list(stored_series.index) == ["A", "B"]
    if isinstance(stored_vector[0], str):
        if isinstance(dset, MemoryDaf):
            julia_type = f"PythonCall.Utils.Static{julia_type}{{UInt32, 1}}"
        elif isinstance(dset, FilesDaf):
            julia_type = f"Sub{julia_type}{{StringViews.StringView{{Vector{{UInt8}}}}}}"
    elif isinstance(dset, MemoryDaf):
        vector_entries.reshape(-1)[0] = vector_entries.reshape(-1)[1]
        assert list(stored_vector) == list(vector_entries.reshape(-1))
        assert list(stored_series.values) == list(vector_entries.reshape(-1))
    description = dset.description()
    assert description.startswith(
        dedent(
            f"""
                name: test!
                type: {format_name}
                axes:
                  cell: 2 entries
                vectors:
                  cell:
                    foo: 2 x {julia_type}
            """
        )[1:-1]
    )

    dset.delete_vector("cell", "foo")

    assert len(dset.vector_names("cell")) == 0
    assert not dset.has_vector("cell", "foo")

    dset.set_vector("cell", "foo", list(vector_entries.reshape(-1)))
    new_vector = dset.get_np_vector("cell", "foo")
    assert id(new_vector) != id(stored_vector)

    if isinstance(stored_vector[0], str):
        return

    with dset.empty_dense_vector("cell", "foo", np.float32, overwrite=True) as empty_vector:
        empty_vector[:] = [-1.5, 2.5]
    assert np.all(dset.get_np_vector("cell", "foo") == np.array([-1.5, 2.5]))

    with dset.empty_sparse_vector("cell", "foo", np.float32, 1, np.int8, overwrite=True) as (
        nzind,
        nzval,
    ):
        nzind[0] = 2
        nzval[0] = 2.5
    assert np.all(dset.get_pd_vector("cell", "foo").values == np.array([0.0, 2.5]))

    sparse_vector = sp.csc_matrix(np.array([[2.5, 0.0]]))
    dset.set_vector("cell", "foo", sparse_vector, overwrite=True)
    assert np.all(dset.get_np_vector("cell", "foo") == np.array([2.5, 0.0]))

    sparse_vector = sp.csr_matrix(np.array([[0.0, 2.5]]))
    dset.set_vector("cell", "foo", sparse_vector, overwrite=True)
    assert np.all(dset.get_np_vector("cell", "foo") == np.array([0.0, 2.5]))


@pytest.mark.parametrize("format_data", FORMATS)
def test_matrices_defaults(format_data: Tuple[str, Callable[[], DafWriter]]) -> None:
    _format_name, create_empty = format_data

    dset = create_empty()
    dset.add_axis("cell", ["A", "B"])
    dset.add_axis("gene", ["X", "Y", "Z"])

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
        dset.get_np_matrix("cell", "gene", "UMIs", relayout=False)

    assert dset.get_np_matrix("cell", "gene", "UMIs", default=None) is None
    assert dset.get_pd_matrix("cell", "gene", "UMIs", default=None) is None
    assert np.all(dset.get_np_matrix("cell", "gene", "UMIs", default=1) == np.array([[1, 1, 1], [1, 1, 1]]))
    default_matrix = np.array([[1, 2], [3, 4], [5, 6]]).transpose()
    assert id(dset.get_np_matrix("cell", "gene", "UMIs", default=default_matrix)) == id(default_matrix)

    fill_matrix = np.array([[0.0, 2.5], [3.5, 0.0], [0.0, 6.5]]).transpose()
    with dset.empty_dense_matrix("cell", "gene", "UMIs", np.float32) as empty_matrix:
        empty_matrix[:, :] = fill_matrix
    assert np.all(dset.get_np_matrix("cell", "gene", "UMIs") == fill_matrix)

    sparse_matrix = sp.csc_matrix(fill_matrix)
    with dset.empty_sparse_matrix("cell", "gene", "UMIs", np.float32, sparse_matrix.nnz, np.int8, overwrite=True) as (
        colptr,
        rowval,
        nzval,
    ):
        colptr[:] = sparse_matrix.indptr[:]
        colptr += 1
        rowval[:] = sparse_matrix.indices[:]
        rowval += 1
        nzval[:] = sparse_matrix.data[:]
    assert isinstance(dset.get_np_matrix("cell", "gene", "UMIs"), sp.csc_matrix)
    assert np.all(dset.get_pd_matrix("cell", "gene", "UMIs").values == fill_matrix)

    fill_matrix = np.array([[0.0, 1.5], [2.5, 0.0], [0.0, 3.5]]).transpose()
    dset.set_matrix("cell", "gene", "UMIs", sp.csc_matrix(fill_matrix), overwrite=True)
    assert isinstance(dset.get_np_matrix("cell", "gene", "UMIs"), sp.csc_matrix)
    assert np.all(dset.get_pd_matrix("cell", "gene", "UMIs").values == fill_matrix)

    with assert_raises("type not in column-major layout: 2 x 3 x Float64 in Rows (transposed Sparse 50%)"):
        dset.set_matrix("cell", "gene", "UMIs", sp.csr_matrix(fill_matrix), overwrite=True)


@pytest.mark.parametrize("format_data", FORMATS)
def test_dense_matrices(format_data: Tuple[str, Callable[[], DafWriter]]) -> None:
    format_name, create_empty = format_data

    dset = create_empty()
    dset.add_axis("cell", ["A", "B"])
    dset.add_axis("gene", ["X", "Y", "Z"])

    assert len(dset.matrix_names("cell", "gene", relayout=False)) == 0
    assert not dset.has_matrix("cell", "gene", "UMIs", relayout=False)

    row_major_umis = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int8)
    column_major_umis = row_major_umis.transpose()

    dset.set_matrix("cell", "gene", "UMIs", column_major_umis, relayout=False)

    assert dset.has_matrix("cell", "gene", "UMIs", relayout=False)
    assert set(dset.matrix_names("cell", "gene", relayout=False)) == set(["UMIs"])
    stored_matrix = dset.get_np_matrix("cell", "gene", "UMIs", relayout=False)
    stored_frame = dset.get_pd_matrix("cell", "gene", "UMIs", relayout=False)
    assert list(stored_frame.index) == ["A", "B"]
    assert list(stored_frame.columns) == ["X", "Y", "Z"]
    repeated_matrix = dset.get_np_matrix("cell", "gene", "UMIs", relayout=False)
    assert id(repeated_matrix) == id(stored_matrix)

    assert np.all(stored_matrix == column_major_umis)
    if isinstance(dset, MemoryDaf):
        column_major_umis[1, 2] += 1
        assert np.all(stored_matrix == column_major_umis)

    description = dset.description()
    assert description.startswith(
        dedent(
            f"""
                name: test!
                type: {format_name}
                axes:
                  cell: 2 entries
                  gene: 3 entries
                matrices:
                  cell,gene:
                    UMIs: 2 x 3 x Int8 in Columns
            """
        )[1:-1]
    )

    assert not dset.has_matrix("gene", "cell", "UMIs", relayout=False)
    dset.relayout_matrix("cell", "gene", "UMIs")
    assert dset.has_matrix("gene", "cell", "UMIs", relayout=False)
    assert np.all(
        dset.get_np_matrix("cell", "gene", "UMIs", relayout=False)
        == dset.get_np_matrix("gene", "cell", "UMIs", relayout=False).transpose()
    )

    description = dset.description()
    assert description.startswith(
        dedent(
            f"""
                name: test!
                type: {format_name}
                axes:
                  cell: 2 entries
                  gene: 3 entries
                matrices:
                  cell,gene:
                    UMIs: 2 x 3 x Int8 in Columns
            """
        )[1:-1]
    )

    dset.delete_matrix("cell", "gene", "UMIs")

    assert len(dset.matrix_names("cell", "gene")) == 0
    assert not dset.has_matrix("cell", "gene", "UMIs", relayout=False)


def test_chains() -> None:
    first_writer = MemoryDaf(name="first!")
    first_writer.set_scalar("version", 1.0)

    first = read_only(first_writer)
    assert id(first) != id(first_writer)
    assert id(first) == id(read_only(first))

    second = MemoryDaf(name="second!")
    second.set_scalar("version", 2.0)

    chain = chain_reader([first, second], name="chain!")
    assert chain.get_scalar("version") == 2.0

    chain = chain_writer([first, second], name="chain!")
    chain.set_scalar("version", 3.0, overwrite=True)
    assert second.get_scalar("version") == 3.0
