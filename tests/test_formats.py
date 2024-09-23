"""
Test ``Daf`` storage formats.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from tempfile import TemporaryDirectory
from textwrap import dedent
from typing import Any
from typing import Callable
from typing import Sequence
from typing import Tuple

import numpy as np
import pytest
import scipy.sparse as sp  # type: ignore

import dafpy as dp

from .utilities import assert_raises


def make_files() -> Tuple[dp.FilesDaf, str]:
    tmpdir = TemporaryDirectory()  # pylint: disable=consider-using-with
    files = dp.FilesDaf(tmpdir.name, "w", name="test!")
    setattr(files, "__gc_anchor__", tmpdir)
    return (
        files,
        dedent(
            f"""
            path: {tmpdir.name}
            mode: w
            """
        )[1:],
    )


def make_h5df() -> Tuple[dp.H5df, str]:
    tmpdir = TemporaryDirectory()  # pylint: disable=consider-using-with
    h5df = dp.H5df(tmpdir.name + "/test.h5df", "w", name="test!")
    setattr(h5df, "__gc_anchor__", tmpdir)
    return (
        h5df,
        dedent(
            f"""
            root: HDF5.File: (read-write) {tmpdir.name}/test.h5df
            mode: w
            """
        )[1:],
    )


FORMATS = [("MemoryDaf", lambda: (dp.MemoryDaf(name="test!"), "")), ("FilesDaf", make_files), ("H5df", make_h5df)]


@pytest.mark.parametrize("format_data", FORMATS)
@pytest.mark.parametrize("scalar_data", [("1.0.1", "String"), (np.int8(1), "Int8"), (0.5, "Float64")])
def test_scalars(
    format_data: Tuple[str, Callable[[], Tuple[dp.DafWriter, str]]], scalar_data: Tuple[dp.StorageScalar, str]
) -> None:
    format_name, create_empty = format_data
    scalar_value, julia_type = scalar_data

    daf, extra = create_empty()
    assert daf.name == "test!"

    assert len(daf.scalars_set()) == 0
    assert not daf.has_scalar("foo")
    daf.set_scalar("foo", scalar_value)
    assert daf.has_scalar("foo")
    assert daf.get_scalar("foo") == scalar_value
    assert set(daf.scalars_set()) == set(["foo"])

    if julia_type == "String":
        scalar_value = '"' + str(scalar_value) + '"'
    else:
        scalar_value = f"{scalar_value} ({julia_type})"

    assert (
        daf.description()
        == dedent(
            f"""
                name: test!
                type: {format_name}
            """
        )[1:]
        + extra
        + dedent(
            f"""
                scalars:
                  foo: {scalar_value}
            """
        )[1:]
    )

    daf.delete_scalar("foo")
    assert len(daf.scalars_set()) == 0
    assert not daf.has_scalar("foo")


@pytest.mark.parametrize("format_data", FORMATS)
@pytest.mark.parametrize("axis_data", [("cell", ["A", "B"]), ("gene", np.array(["X", "Y", "Z"]))])
def test_axes(
    format_data: Tuple[str, Callable[[], Tuple[dp.DafWriter, str]]], axis_data: Tuple[str, Sequence[str]]
) -> None:
    format_name, create_empty = format_data
    axis_name, axis_entries = axis_data

    daf, extra = create_empty()

    assert len(daf.axes_set()) == 0
    assert not daf.has_axis(axis_name)

    daf.add_axis(axis_name, axis_entries)

    assert daf.has_axis(axis_name)
    assert daf.axis_length(axis_name) == len(axis_entries)
    assert list(daf.axis_array(axis_name)) == list(axis_entries)
    assert list(sorted([(str(name), index) for (name, index) in daf.axis_dict(axis_name).items()])) == sorted(
        [(name, index) for (index, name) in enumerate(axis_entries)]
    )
    assert np.all(daf.axis_np_indices(axis_name, axis_entries) == np.arange(len(axis_entries)))
    assert np.all(daf.axis_pd_indices(axis_name, axis_entries).values == np.arange(len(axis_entries)))
    assert list(daf.axis_pd_indices(axis_name, axis_entries).index) == list(axis_entries)
    assert set(daf.axes_set()) == set([axis_name])
    assert (
        daf.description()
        == dedent(
            f"""
                name: test!
                type: {format_name}
            """
        )[1:]
        + extra
        + dedent(
            f"""
                axes:
                  {axis_name}: {len(axis_entries)} entries
            """
        )[1:]
    )

    daf.delete_axis(axis_name)

    assert len(daf.axes_set()) == 0
    assert not daf.has_axis(axis_name)


@pytest.mark.parametrize("format_data", FORMATS)
def test_vectors_defaults(format_data: Tuple[str, Callable[[], Tuple[dp.DafWriter, str]]]) -> None:
    _format_name, create_empty = format_data

    daf, _extra = create_empty()
    daf.add_axis("cell", ["A", "B"])

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
        daf.get_np_vector("cell", "foo")

    assert daf.get_np_vector("cell", "foo", default=None) is None
    assert daf.get_pd_vector("cell", "foo", default=None) is None
    assert list(daf.get_np_vector("cell", "foo", default=1)) == [1, 1]
    assert list(daf.get_np_vector("cell", "foo", default=[1, 2])) == [1, 2]
    assert list(daf.get_np_vector("cell", "foo", default=np.array([1, 2]))) == [1, 2]


@pytest.mark.parametrize("format_data", FORMATS)
@pytest.mark.parametrize(
    "vector_data",
    [(["X", "Y"], "String"), ([1, 2], "Int64"), ([1.0, 2.0], "Float64"), (np.array([[1, 2]], dtype="i1"), "Int8")],
)
def test_dense_vectors(  # pylint: disable=too-many-locals,too-many-statements
    format_data: Tuple[str, Callable[[], Tuple[dp.DafWriter, str]]], vector_data: Tuple[Any, str]
) -> None:
    format_name, create_empty = format_data
    vector_entries, julia_type = vector_data

    daf, extra = create_empty()
    daf.add_axis("cell", ["A", "B"])
    assert len(daf.vectors_set("cell")) == 0
    assert not daf.has_vector("cell", "foo")

    vector_entries = np.array(vector_entries)
    daf.set_vector("cell", "foo", vector_entries)

    assert daf.has_vector("cell", "foo")
    assert set(daf.vectors_set("cell")) == set(["foo"])
    stored_vector = daf.get_np_vector("cell", "foo")
    stored_series = daf.get_pd_vector("cell", "foo")
    repeated_vector = daf.get_np_vector("cell", "foo")
    assert id(repeated_vector) == id(stored_vector)

    assert list(stored_vector) == list(vector_entries.reshape(-1))
    assert list(stored_series.values) == list(vector_entries.reshape(-1))
    assert list(stored_series.index) == ["A", "B"]
    if isinstance(stored_vector[0], str):
        if isinstance(daf, dp.MemoryDaf):
            julia_type = f"PythonCall.Utils.Static{julia_type}{{UInt32, 1}}"
        elif isinstance(daf, dp.FilesDaf):
            julia_type = f"Sub{julia_type}{{StringViews.StringView{{Vector{{UInt8}}}}}}"
    elif isinstance(daf, dp.MemoryDaf):
        if not vector_entries.flags.writeable:  # Due to get_np_vector above.
            vector_entries.flags.writeable = True
        vector_entries.reshape(-1)[0] = vector_entries.reshape(-1)[1]
        assert list(stored_vector) == list(vector_entries.reshape(-1))
        assert list(stored_series.values) == list(vector_entries.reshape(-1))
    description = daf.description()
    assert description.startswith(
        dedent(
            f"""
                name: test!
                type: {format_name}
            """
        )[1:]
        + extra
        + dedent(
            f"""
                axes:
                  cell: 2 entries
                vectors:
                  cell:
                    foo: 2 x {julia_type}
            """
        )[1:-1]
    )

    daf.delete_vector("cell", "foo")

    assert len(daf.vectors_set("cell")) == 0
    assert not daf.has_vector("cell", "foo")

    daf.set_vector("cell", "foo", list(vector_entries.reshape(-1)))
    new_vector = daf.get_np_vector("cell", "foo")
    assert id(new_vector) != id(stored_vector)

    if isinstance(stored_vector[0], str):
        return

    with daf.empty_dense_vector("cell", "foo", np.float32, overwrite=True) as empty_vector:
        empty_vector[:] = [-1.5, 2.5]
    assert np.all(daf.get_np_vector("cell", "foo") == np.array([-1.5, 2.5]))

    with daf.empty_sparse_vector("cell", "foo", np.float32, 1, np.int8, overwrite=True) as (
        nzind,
        nzval,
    ):
        nzind[0] = 2
        nzval[0] = 2.5
    assert np.all(daf.get_pd_vector("cell", "foo").values == np.array([0.0, 2.5]))

    sparse_vector = sp.csc_matrix(np.array([[2.5, 0.0]]))
    daf.set_vector("cell", "foo", sparse_vector, overwrite=True)
    assert np.all(daf.get_np_vector("cell", "foo") == np.array([2.5, 0.0]))

    sparse_vector = sp.csr_matrix(np.array([[0.0, 2.5]]))
    daf.set_vector("cell", "foo", sparse_vector, overwrite=True)
    assert np.all(daf.get_np_vector("cell", "foo") == np.array([0.0, 2.5]))


@pytest.mark.parametrize("format_data", FORMATS)
def test_matrices_defaults(format_data: Tuple[str, Callable[[], Tuple[dp.DafWriter, str]]]) -> None:
    _format_name, create_empty = format_data

    daf, _extra = create_empty()
    daf.add_axis("cell", ["A", "B"])
    daf.add_axis("gene", ["X", "Y", "Z"])

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
        daf.get_np_matrix("cell", "gene", "UMIs", relayout=False)

    assert daf.get_np_matrix("cell", "gene", "UMIs", default=None) is None
    assert daf.get_pd_matrix("cell", "gene", "UMIs", default=None) is None
    assert np.all(daf.get_np_matrix("cell", "gene", "UMIs", default=1) == np.array([[1, 1, 1], [1, 1, 1]]))
    default_matrix = np.array([[1, 2], [3, 4], [5, 6]]).transpose()
    assert id(daf.get_np_matrix("cell", "gene", "UMIs", default=default_matrix)) == id(default_matrix)

    fill_matrix = np.array([[0.0, 2.5], [3.5, 0.0], [0.0, 6.5]]).transpose()
    with daf.empty_dense_matrix("cell", "gene", "UMIs", np.float32) as empty_matrix:
        empty_matrix[:, :] = fill_matrix
    assert np.all(daf.get_np_matrix("cell", "gene", "UMIs") == fill_matrix)

    sparse_matrix = sp.csc_matrix(fill_matrix)
    with daf.empty_sparse_matrix("cell", "gene", "UMIs", np.float32, sparse_matrix.nnz, np.int8, overwrite=True) as (
        colptr,
        rowval,
        nzval,
    ):
        colptr[:] = sparse_matrix.indptr[:]
        colptr += 1
        rowval[:] = sparse_matrix.indices[:]
        rowval += 1
        nzval[:] = sparse_matrix.data[:]
    assert isinstance(daf.get_np_matrix("cell", "gene", "UMIs"), sp.csc_matrix)
    assert np.all(daf.get_pd_matrix("cell", "gene", "UMIs").values == fill_matrix)

    fill_matrix = np.array([[0.0, 1.5], [2.5, 0.0], [0.0, 3.5]]).transpose()
    daf.set_matrix("cell", "gene", "UMIs", sp.csc_matrix(fill_matrix), overwrite=True)
    assert isinstance(daf.get_np_matrix("cell", "gene", "UMIs"), sp.csc_matrix)
    assert np.all(daf.get_pd_matrix("cell", "gene", "UMIs").values == fill_matrix)

    with assert_raises("type not in column-major layout: 2 x 3 x Float64 in Rows (Transpose Sparse Int32 50%)"):
        daf.set_matrix("cell", "gene", "UMIs", sp.csr_matrix(fill_matrix), overwrite=True)


@pytest.mark.parametrize("format_data", FORMATS)
def test_dense_matrices(format_data: Tuple[str, Callable[[], Tuple[dp.DafWriter, str]]]) -> None:
    format_name, create_empty = format_data

    daf, extra = create_empty()
    daf.add_axis("cell", ["A", "B"])
    daf.add_axis("gene", ["X", "Y", "Z"])

    assert len(daf.matrices_set("cell", "gene", relayout=False)) == 0
    assert not daf.has_matrix("cell", "gene", "UMIs", relayout=False)

    row_major_umis = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int8)
    column_major_umis = row_major_umis.transpose()

    daf.set_matrix("cell", "gene", "UMIs", column_major_umis, relayout=False)

    assert daf.has_matrix("cell", "gene", "UMIs", relayout=False)
    assert set(daf.matrices_set("cell", "gene", relayout=False)) == set(["UMIs"])
    stored_matrix = daf.get_np_matrix("cell", "gene", "UMIs", relayout=False)
    stored_frame = daf.get_pd_matrix("cell", "gene", "UMIs", relayout=False)
    assert list(stored_frame.index) == ["A", "B"]
    assert list(stored_frame.columns) == ["X", "Y", "Z"]
    repeated_matrix = daf.get_np_matrix("cell", "gene", "UMIs", relayout=False)
    assert id(repeated_matrix) == id(stored_matrix)

    assert np.all(stored_matrix == column_major_umis)
    if isinstance(daf, dp.MemoryDaf):
        if not column_major_umis.flags.writeable:  # Due to get_np_matrix above.
            column_major_umis.flags.writeable = True
        column_major_umis[1, 2] += 1
        assert np.all(stored_matrix == column_major_umis)

    description = daf.description()
    assert description.startswith(
        dedent(
            f"""
                name: test!
                type: {format_name}
            """
        )[1:]
        + extra
        + dedent(
            """
                axes:
                  cell: 2 entries
                  gene: 3 entries
                matrices:
                  cell,gene:
                    UMIs: 2 x 3 x Int8 in Columns
            """
        )[1:-1]
    )

    assert not daf.has_matrix("gene", "cell", "UMIs", relayout=False)
    daf.relayout_matrix("cell", "gene", "UMIs")
    assert daf.has_matrix("gene", "cell", "UMIs", relayout=False)
    assert np.all(
        daf.get_np_matrix("cell", "gene", "UMIs", relayout=False)
        == daf.get_np_matrix("gene", "cell", "UMIs", relayout=False).transpose()
    )

    description = daf.description()
    assert description.startswith(
        dedent(
            f"""
                name: test!
                type: {format_name}
            """
        )[1:]
        + extra
        + dedent(
            """
                axes:
                  cell: 2 entries
                  gene: 3 entries
                matrices:
                  cell,gene:
                    UMIs: 2 x 3 x Int8 in Columns
            """
        )[1:-1]
    )

    daf.delete_matrix("cell", "gene", "UMIs")

    assert len(daf.matrices_set("cell", "gene")) == 0
    assert not daf.has_matrix("cell", "gene", "UMIs", relayout=False)


def test_chains() -> None:
    first_writer = dp.MemoryDaf(name="first!")
    first_writer.set_scalar("version", 1.0)

    first = first_writer.read_only()
    assert id(first) != id(first_writer)
    assert id(first) == id(first.read_only())
    assert id(first) != id(first.read_only(name="renamed"))

    second = dp.MemoryDaf(name="second!")
    second.set_scalar("version", 2.0)

    read_chain = dp.chain_reader([first, second], name="chain!")
    assert read_chain.get_scalar("version") == 2.0

    write_chain = dp.chain_writer([first, second], name="chain!")
    write_chain.set_scalar("version", 3.0, overwrite=True)
    assert second.get_scalar("version") == 3.0
