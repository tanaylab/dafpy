"""
Test ``Daf`` concatenation.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from textwrap import dedent

import dafpy as dp


def test_concatenate() -> None:  # pylint: disable=too-many-statements
    sources = [dp.MemoryDaf(name="source.1!"), dp.MemoryDaf(name="source.2!")]
    sources[0].add_axis("cell", ["A", "B"])
    sources[1].add_axis("cell", ["C", "D", "E"])
    sources[0].set_scalar("version", 1)
    sources[1].set_scalar("version", 2)

    destination = dp.MemoryDaf(name="destination!")
    dp.concatenate(destination, "cell", sources, merge={"version": "CollectAxis"})
    assert (
        destination.description()
        == dedent(
            """
            name: destination!
            type: MemoryDaf
            axes:
              cell: 5 entries
              dataset: 2 entries
            vectors:
              cell:
                dataset: 5 x AbstractString (Dense)
              dataset:
                version: 2 x Int64 (Dense)
            """
        )[1:]
    )

    destination = dp.MemoryDaf(name="destination!")
    dp.concatenate(destination, "cell", sources)
    assert (
        destination.description()
        == dedent(
            """
            name: destination!
            type: MemoryDaf
            axes:
              cell: 5 entries
              dataset: 2 entries
            vectors:
              cell:
                dataset: 5 x AbstractString (Dense)
            """
        )[1:]
    )
