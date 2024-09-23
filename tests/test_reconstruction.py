"""
Test ``Daf`` axis reconstruction.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from textwrap import dedent

import dafpy as dp


def test_reconstruction() -> None:
    memory = dp.MemoryDaf(name="memory!")

    memory.add_axis("cell", ["A", "B", "C", "D"])
    memory.set_vector("cell", "age", [1, 1, 2, 3])
    memory.set_vector("cell", "score", [0.0, 0.5, 1.0, 2.0])
    memory.set_vector("cell", "batch", ["X", "X", "Y", ""])
    results = dp.reconstruct_axis(memory, existing_axis="cell", implicit_axis="batch")
    assert list(results.keys()) == ["age"]
    assert list(results.values()) == [3]
    assert (
        memory.description()
        == dedent(
            """
        name: memory!
        type: MemoryDaf
        axes:
          batch: 2 entries
          cell: 4 entries
        vectors:
          batch:
            age: 2 x Int64 (Dense)
          cell:
            batch: 4 x PythonCall.Utils.StaticString{UInt32, 1} (Dense)
            score: 4 x Float64 (PyArray - Dense)
    """
        )[1:]
    )
