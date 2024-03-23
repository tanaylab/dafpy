"""
Test ``Daf`` view operations.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from textwrap import dedent

import numpy as np

from daf import *


def test_views() -> None:  # pylint: disable=too-many-statements
    dset = MemoryDaf(name="test!")
    dset.set_scalar("version", "1.0")
    dset.add_axis("cell", ["A", "B"])
    dset.add_axis("gene", ["X", "Y", "Z"])
    dset.set_vector("cell", "batch", ["U", "V"])
    dset.set_vector("cell", "age", [-1.0, 2.0])
    dset.set_matrix("gene", "cell", "UMIs", np.array([[1, 2, 3], [4, 5, 6]]).transpose(), relayout=False)
    dset.add_axis("batch", ["U", "V", "W"])
    dset.set_vector("batch", "sex", ["Male", "Female", "Male"])

    assert (
        dset.description()
        == dedent(
            """
            name: test!
            type: MemoryDaf
            scalars:
              version: "1.0"
            axes:
              batch: 3 entries
              cell: 2 entries
              gene: 3 entries
            vectors:
              batch:
                sex: 3 x PythonCall.Utils.StaticString{UInt32, 6} (Dense)
              cell:
                age: 2 x Float64 (PyArray{Float64, 1, true, true, Float64} - Dense)
                batch: 2 x PythonCall.Utils.StaticString{UInt32, 1} (Dense)
            matrices:
              gene,cell:
                UMIs: 3 x 2 x Int64 in Columns (PyArray{Int64, 2, true, true, Int64} - Dense)
            """
        )[1:]
    )

    view = viewer(
        dset,
        name="view!",
        axes={"obs": "/ cell", "var": "/ gene"},
        data={ALL_SCALARS: None, ALL_VECTORS: "=", ("obs", "var", "X"): ": UMIs"},
    )

    assert (
        view.description()
        == dedent(
            """
            name: view!
            type: View MemoryDaf
            axes:
              obs: 2 entries
              var: 3 entries
            vectors:
              obs:
                age: 2 x Float64 (PyArray{Float64, 1, true, true, Float64} - Dense)
                batch: 2 x PythonCall.Utils.StaticString{UInt32, 1} (Dense)
            matrices:
              obs,var:
                X: 2 x 3 x Int64 in Columns (Dense)
            """
        )[1:]
    )
