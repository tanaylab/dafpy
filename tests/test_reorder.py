"""
Test reordering ``Daf`` axes.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

import dafpy as dp


def test_reorder_axes(tmp_path) -> None:
    # The permutation says where each new entry comes from, and is 0-based here, so reversing three entries is
    # ``[2, 1, 0]`` rather than Julia's ``[3, 2, 1]``.
    files = dp.files_daf(f"{tmp_path}/reordered", "w", name="reordered!")
    assert isinstance(files, dp.DafWriter)
    files.add_axis("cell", ["A", "B", "C"])
    files.set_vector("cell", "age", [1, 2, 3])

    dp.reorder_axes(files, {"cell": [2, 1, 0]})

    assert list(files.axis_np_vector("cell")) == ["C", "B", "A"]
    assert list(files.get_np_vector("cell", "age")) == [3, 2, 1]


def test_reorder_several(tmp_path) -> None:
    # Repositories sharing an axis are reordered together, so that they keep agreeing about it.
    first = dp.files_daf(f"{tmp_path}/first", "w", name="first!")
    assert isinstance(first, dp.DafWriter)
    first.add_axis("cell", ["A", "B"])
    first.set_vector("cell", "age", [1, 2])

    second = dp.files_daf(f"{tmp_path}/second", "w", name="second!")
    assert isinstance(second, dp.DafWriter)
    second.add_axis("cell", ["A", "B"])
    second.set_vector("cell", "score", [0.5, 1.5])

    dp.reorder_axes([first, second], {"cell": [1, 0]})

    assert list(first.axis_np_vector("cell")) == ["B", "A"]
    assert list(first.get_np_vector("cell", "age")) == [2, 1]
    assert list(second.axis_np_vector("cell")) == ["B", "A"]
    assert list(second.get_np_vector("cell", "score")) == [1.5, 0.5]


def test_reset_reorder_axes(tmp_path) -> None:
    # Nothing was interrupted, so there is nothing to roll back.
    files = dp.files_daf(f"{tmp_path}/intact", "w", name="intact!")
    assert isinstance(files, dp.DafWriter)
    files.add_axis("cell", ["A", "B"])

    assert not dp.reset_reorder_axes(files)


def test_is_leaf(tmp_path) -> None:
    # A repository which owns its data is a leaf; a wrapper around one is not, which includes the read-only wrapper
    # that opening for reading puts in the way.
    memory = dp.memory_daf(name="memory!")
    assert memory.is_leaf()

    files = dp.files_daf(f"{tmp_path}/leaf", "w", name="leaf!")
    assert isinstance(files, dp.DafWriter)
    files.add_axis("cell", ["A", "B"])
    assert files.is_leaf()

    assert not dp.files_daf(f"{tmp_path}/leaf", "r", name="read!").is_leaf()
    assert not dp.chain_writer([memory, files], name="chain!").is_leaf()
