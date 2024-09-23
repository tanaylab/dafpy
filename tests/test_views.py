"""
Test ``Daf`` view operations.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from textwrap import dedent

import numpy as np

import dafpy as dp


def test_views() -> None:  # pylint: disable=too-many-statements
    daf = dp.MemoryDaf(name="test!")
    daf.set_scalar("version", "1.0")
    daf.add_axis("cell", ["A", "B"])
    daf.add_axis("gene", ["X", "Y", "Z"])
    daf.set_vector("cell", "batch", ["U", "V"])
    daf.set_vector("cell", "age", [-1.0, 2.0])
    daf.set_matrix("gene", "cell", "UMIs", np.array([[1, 2, 3], [4, 5, 6]]).transpose(), relayout=False)
    daf.add_axis("batch", ["U", "V", "W"])
    daf.set_vector("batch", "sex", ["Male", "Female", "Male"])

    assert (
        daf.description()
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
                age: 2 x Float64 (PyArray - Dense)
                batch: 2 x PythonCall.Utils.StaticString{UInt32, 1} (Dense)
            matrices:
              gene,cell:
                UMIs: 3 x 2 x Int64 in Columns (PyArray - Dense)
            """
        )[1:]
    )

    view = dp.viewer(
        daf,
        name="view!",
        axes={"obs": "/ cell", "var": "/ gene"},
        data={dp.ALL_SCALARS: None, dp.ALL_VECTORS: "=", ("obs", "var", "X"): ": UMIs"},
    )

    assert (
        view.description()
        == dedent(
            """
            name: view!
            type: View
            base: MemoryDaf test!
            axes:
              obs: 2 entries
              var: 3 entries
            vectors:
              obs:
                age: 2 x Float64 (PyArray - Dense)
                batch: 2 x PythonCall.Utils.StaticString{UInt32, 1} (Dense)
            matrices:
              var,obs:
                X: 3 x 2 x Int64 in Columns (PyArray - Dense)
            """
        )[1:]
    )
