"""
Test ``Daf`` axis reconstruction.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from textwrap import dedent

import numpy as np

import dafpy as dp


def test_empty_values() -> None:
    # The value(s) meaning "there is no batch" may be given as one of them, or as any collection of them. A string is
    # one of them, even though Python considers a string to be a collection of characters. Saying which values mean
    # nothing is ``unify_empty_vector_values``, so that ``reconstruct_axis`` need only know about the empty string.
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

        dp.unify_empty_vector_values(memory, axis="cell", property="batch", empty_values=empties)
        dp.reconstruct_axis(memory, existing_axis="cell", implicit_axis="batch")

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


def test_unify_to_a_type() -> None:
    # A column of measurements is a column of strings because a few of its entries say ``NA``. The type has to reach
    # Julia as a Julia type, which is the one thing this wrapper does beyond passing its arguments along.
    memory = dp.memory_daf(name="memory!")
    memory.add_axis("cell", ["A", "B", "C"])
    memory.set_vector("cell", "qc", ["23.5", "NA", "24.5"])

    dp.unify_empty_vector_values(memory, axis="cell", property="qc", empty_values="NA", dtype=np.float32)

    values = memory.get_np_vector("cell", "qc")
    assert values.dtype == np.float32
    assert list(values[[0, 2]]) == [np.float32(23.5), np.float32(24.5)]
    assert np.isnan(values[1])


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


def test_connect_axes() -> None:
    # Plates and sequencing runs are both properties of a batch, and each plate belongs to one run, but nothing says so
    # where a plate can be asked about it. The last batch has no plate, which is not a problem: nothing is moved, so it
    # keeps its own run.
    memory = dp.memory_daf(name="memory!")
    memory.add_axis("batch", ["B1", "B2", "B3", "B4"])
    memory.add_axis("plate", ["P1", "P2", "P3"])
    memory.add_axis("run", ["R1", "R2"])
    memory.set_vector("batch", "plate", ["P1", "P1", "P2", ""])
    memory.set_vector("batch", "run", ["R1", "R1", "R2", "R2"])

    dp.connect_axes(memory, base_axis="batch", from_axis="plate", to_axis="run")

    # P3 is named by no batch, so it is connected to nothing.
    assert list(memory.get_np_vector("plate", "run")) == ["R1", "R2", ""]

    # Nothing was moved, so the batch with no plate still has its run.
    assert list(memory.get_np_vector("batch", "run")) == ["R1", "R1", "R2", "R2"]


def test_connect_axes_names() -> None:
    # The properties holding the references need not be named after the axes they refer to.
    memory = dp.memory_daf(name="memory!")
    memory.add_axis("batch", ["B1", "B2"])
    memory.add_axis("plate", ["P1"])
    memory.add_axis("run", ["R1"])
    memory.set_vector("batch", "on_plate", ["P1", "P1"])
    memory.set_vector("batch", "sequenced_by", ["R1", "R1"])

    dp.connect_axes(
        memory,
        base_axis="batch",
        from_axis="plate",
        from_property="on_plate",
        to_axis="run",
        to_property="sequenced_by",
        connect_property="sequenced_by",
    )

    assert list(memory.get_np_vector("plate", "sequenced_by")) == ["R1"]
    assert not memory.has_vector("plate", "run")


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
