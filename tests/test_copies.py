"""
Test ``Daf`` copy operations.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

import numpy as np

import dafpy as dp


def test_copies() -> None:  # pylint: disable=too-many-statements
    source = dp.MemoryDaf(name="source!")
    destination = dp.MemoryDaf(name="destination!")

    source.set_scalar("version", "1.0")
    dp.copy_scalar(source=source, destination=destination, name="version")
    assert destination.get_scalar("version") == "1.0"

    source.add_axis("cell", ["A", "B"])
    dp.copy_axis(source=source, destination=destination, axis="cell")
    assert list(destination.axis_array("cell")) == ["A", "B"]

    source.add_axis("gene", ["X", "Y", "Z"])
    dp.copy_axis(source=source, destination=destination, axis="gene")
    assert list(destination.axis_array("gene")) == ["X", "Y", "Z"]

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

    destination = dp.MemoryDaf(name="destination!")
    dp.copy_all(source=source, destination=destination)
    assert destination.get_scalar("version") == "1.0"
    assert list(destination.axis_array("cell")) == ["A", "B"]
    assert list(destination.axis_array("gene")) == ["X", "Y", "Z"]
    assert list(destination.get_np_vector("cell", "age")) == [0.0, 1.0]
    assert np.all(destination.get_np_matrix("gene", "cell", "UMIs") == np.array([[0, 1, 2], [3, 4, 5]]).transpose())
