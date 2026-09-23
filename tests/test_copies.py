"""
Test ``Daf`` copy operations.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

import numpy as np
import scipy.sparse as sp  # type: ignore

import dafpy as dp


def test_copies() -> None:  # pylint: disable=too-many-statements
    source = dp.memory_daf(name="source!")
    destination = dp.memory_daf(name="destination!")

    source.set_scalar("version", "1.0")
    dp.copy_scalar(source=source, destination=destination, name="version")
    assert destination.get_scalar("version") == "1.0"

    source.add_axis("cell", ["A", "B"])
    dp.copy_axis(source=source, destination=destination, axis="cell")
    assert list(destination.axis_np_vector("cell")) == ["A", "B"]

    source.add_axis("gene", ["X", "Y", "Z"])
    dp.copy_axis(source=source, destination=destination, axis="gene")
    assert list(destination.axis_np_vector("gene")) == ["X", "Y", "Z"]

    source.set_vector("cell", "age", [0.0, 1.0])
    dp.copy_vector(source=source, destination=destination, axis="cell", name="age")
    assert list(destination.get_np_vector("cell", "age")) == [0.0, 1.0]

    source.set_matrix("gene", "cell", "UMIs", np.array([[0, 1, 2], [3, 4, 5]]).transpose())
    dp.copy_matrix(
        source=source, destination=destination, rows_axis="gene", columns_axis="cell", name="UMIs", relayout=False
    )
    assert np.all(destination.get_np_matrix("gene", "cell", "UMIs") == np.array([[0, 1, 2], [3, 4, 5]]).transpose())

    destination.add_axis("batch", ["U", "V"])
    source.set_matrix("gene", "cell", "U_is_high", np.array([[True, False, True], [False, True, False]]).transpose())
    dp.copy_tensor(
        source=source,
        destination=destination,
        main_axis="batch",
        rows_axis="gene",
        columns_axis="cell",
        name="is_high",
        empty=False,
    )
    assert np.all(
        destination.get_np_matrix("gene", "cell", "U_is_high")
        == np.array([[True, False, True], [False, True, False]]).transpose()
    )
    assert np.all(
        destination.get_np_matrix("gene", "cell", "V_is_high")
        == np.array([[False, False, False], [False, False, False]]).transpose()
    )

    destination = dp.memory_daf(name="destination!")
    dp.copy_all(source=source, destination=destination)
    assert destination.get_scalar("version") == "1.0"
    dp.copy_all(source=source, destination=destination, insist=False)
    assert destination.get_scalar("version") == "1.0"
    assert list(destination.axis_np_vector("cell")) == ["A", "B"]
    assert list(destination.axis_np_vector("gene")) == ["X", "Y", "Z"]
    assert list(destination.get_np_vector("cell", "age")) == [0.0, 1.0]
    assert np.all(destination.get_np_matrix("gene", "cell", "UMIs") == np.array([[0, 1, 2], [3, 4, 5]]).transpose())


def test_copies_keywords() -> None:
    source = dp.memory_daf(name="source!")
    source.set_scalar("count", 3)
    source.add_axis("cell", ["A", "B"])
    source.add_axis("gene", ["X", "Y", "Z"])
    source.set_vector("cell", "age", [0, 1])
    source.set_matrix("gene", "cell", "UMIs", np.array([[0, 0, 7], [0, 0, 0]]).transpose())

    destination = dp.memory_daf(name="destination!")
    dp.copy_scalar(source=source, destination=destination, name="count", type=float)
    assert destination.get_scalar("count") == 3.0
    assert isinstance(destination.get_scalar("count"), float)
    dp.copy_scalar(source=source, destination=destination, name="count", insist=False)
    assert isinstance(destination.get_scalar("count"), float)

    destination.add_axis("cell", ["A", "B"])
    destination.add_axis("gene", ["X", "Y", "Z"])
    dp.copy_vector(source=source, destination=destination, axis="cell", name="age", eltype=np.float32)
    assert destination.get_np_vector("cell", "age").dtype == np.float32
    dp.copy_vector(source=source, destination=destination, axis="cell", name="age", insist=False)
    assert destination.get_np_vector("cell", "age").dtype == np.float32

    dp.copy_matrix(
        source=source,
        destination=destination,
        rows_axis="gene",
        columns_axis="cell",
        name="UMIs",
        eltype=np.float32,
        bestify=True,
        min_sparse_saving_fraction=0.1,
    )
    copied_umis = destination.get_np_matrix("gene", "cell", "UMIs")
    assert isinstance(copied_umis, sp.csc_matrix)
    assert copied_umis.dtype == np.float32
    assert copied_umis.nnz == 1

    destination = dp.memory_daf(name="destination!")
    dp.copy_all(source=source, destination=destination, types={("cell", "age"): np.float32, "count": float})
    assert destination.get_np_vector("cell", "age").dtype == np.float32
    assert isinstance(destination.get_scalar("count"), float)
