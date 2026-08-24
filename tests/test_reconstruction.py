"""
Test ``Daf`` axis reconstruction.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from textwrap import dedent

import dafpy as dp


def test_empty_implicit() -> None:
    # The value(s) meaning "there is no batch" may be given as one of them, or as any collection of them. A string is
    # one of them, even though Python considers a string to be a collection of characters.
    for empties, n_batches in (
        ("Outliers", 2),
        (("Outliers", "Doublet"), 1),
        (["Outliers", "Doublet"], 1),
        ({"Outliers", "Doublet"}, 1),
    ):
        memory = dp.memory_daf(name="memory!")
        memory.add_axis("cell", ["A", "B", "C", "D"])
        memory.set_vector("cell", "age", [1, 1, 3, 3])
        memory.set_vector("cell", "batch", ["X", "X", "Outliers", "Doublet"])

        dp.reconstruct_axis(memory, existing_axis="cell", implicit_axis="batch", empty_implicit=empties)

        # Giving only one of them leaves the other as a batch of its own, which is what tells the cases apart.
        assert memory.description() == dedent(f"""
            name: memory!
            type: MemoryDaf
            axes:
              batch: {n_batches} entries
              cell: 4 entries
            vectors:
              batch:
                age: {n_batches} x Int64 (Dense)
              cell:
                batch: 4 x Str (Dense)
        """)[1:]


def test_properties_sets() -> None:
    # A Python set has to be converted to a Julia one, which is what a parameter declared as a set expects.
    memory = dp.memory_daf(name="memory!")
    memory.add_axis("cell", ["A", "B", "C", "D"])
    memory.set_vector("cell", "age", [1, 1, 2, 3])
    memory.set_vector("cell", "score", [0.0, 0.5, 1.0, 2.0])
    memory.set_vector("cell", "batch", ["X", "X", "Y", ""])

    results = dp.reconstruct_axis(
        memory,
        existing_axis="cell",
        implicit_axis="batch",
        implicit_properties={"age"},
        skipped_properties=frozenset({"score"}),
    )

    assert list(results.keys()) == ["age"]


def test_no_properties() -> None:
    # An empty set of properties to convert is how one says "create the axis and nothing else", which is what to do
    # when the axis was created in advance and holds entries the data does not use: converting a property would then
    # need a default for each of them. An empty Python set says nothing about what it would have held, so what is
    # built for Julia has to be a set of names because it was asked for, not because of what is in it.
    memory = dp.memory_daf(name="memory!")
    memory.add_axis("cell", ["A", "B", "C"])
    memory.set_vector("cell", "age", [1, 1, 2])
    memory.set_vector("cell", "batch", ["X", "X", "Y"])
    memory.add_axis("batch", ["X", "Y", "Z"])

    results = dp.reconstruct_axis(
        memory,
        existing_axis="cell",
        implicit_axis="batch",
        implicit_properties=set(),
    )

    assert not results
    assert list(memory.axis_np_vector("batch")) == ["X", "Y", "Z"]
    assert memory.has_vector("cell", "age")
    assert not memory.has_vector("batch", "age")


def test_reconstruction() -> None:
    memory = dp.memory_daf(name="memory!")

    memory.add_axis("cell", ["A", "B", "C", "D"])
    memory.set_vector("cell", "age", [1, 1, 2, 3])
    memory.set_vector("cell", "score", [0.0, 0.5, 1.0, 2.0])
    memory.set_vector("cell", "batch", ["X", "X", "Y", ""])
    results = dp.reconstruct_axis(memory, existing_axis="cell", implicit_axis="batch")
    assert list(results.keys()) == ["age"]
    assert list(results.values()) == [3]
    assert memory.description() == dedent("""
        name: memory!
        type: MemoryDaf
        axes:
          batch: 2 entries
          cell: 4 entries
        vectors:
          batch:
            age: 2 x Int64 (Dense)
          cell:
            batch: 4 x Str (Dense)
            score: 4 x Float64 (PyArray; Dense)
    """)[1:]
