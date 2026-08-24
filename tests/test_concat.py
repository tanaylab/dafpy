"""
Test ``Daf`` concatenation.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from textwrap import dedent

import dafpy as dp


def test_prefixed() -> None:
    # The names to prefix are given either as one set, applying to every concatenation axis, or as one set per axis.
    # A Python set is not an ``AbstractSet`` and a list of them is not an ``AbstractVector`` of one, so both shapes
    # have to be converted; passing either used to fail with a type error naming the keyword.
    for prefixed in ({"kind"}, [{"kind"}]):
        sources = [dp.memory_daf(name="source.1!"), dp.memory_daf(name="source.2!")]
        for index, source in enumerate(sources):
            source.add_axis("cell", ["A", "B"] if index == 0 else ["C", "D"])
            source.set_vector("cell", "kind", ["x", "y"])

        destination = dp.memory_daf(name="destination!")
        dp.concatenate(destination, "cell", sources, prefixed=prefixed)

        # Prefixing makes each value unique to the data set it came from, which is what asks for the conversion.
        assert list(destination.get_np_vector("cell", "kind")) == [
            "source.1!.x",
            "source.1!.y",
            "source.2!.x",
            "source.2!.y",
        ]


def test_concatenate() -> None:  # pylint: disable=too-many-statements
    sources = [dp.memory_daf(name="source.1!"), dp.memory_daf(name="source.2!")]
    sources[0].add_axis("cell", ["A", "B"])
    sources[1].add_axis("cell", ["C", "D", "E"])
    sources[0].set_scalar("version", 1)
    sources[1].set_scalar("version", 2)

    destination = dp.memory_daf(name="destination!")
    dp.concatenate(destination, "cell", sources, merge={"version": dp.MergeAction.CollectAxis})
    assert destination.description() == dedent("""
            name: destination!
            type: MemoryDaf
            axes:
              cell: 5 entries
              dataset: 2 entries
            vectors:
              cell:
                dataset: 5 x Str (Dense)
              dataset:
                version: 2 x Int64 (Dense)
            """)[1:]

    destination = dp.memory_daf(name="destination!")
    dp.concatenate(destination, "cell", sources)
    assert destination.description() == dedent("""
            name: destination!
            type: MemoryDaf
            axes:
              cell: 5 entries
              dataset: 2 entries
            vectors:
              cell:
                dataset: 5 x Str (Dense)
            """)[1:]
