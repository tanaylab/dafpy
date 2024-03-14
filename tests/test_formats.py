"""
Test ``Daf`` storage formats.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from textwrap import dedent
from typing import Callable
from typing import Tuple

import numpy as np
import pytest

from daf import *

FORMATS = [("MemoryDaf", lambda: MemoryDaf(name="test!"))]


@pytest.mark.parametrize("format_data", FORMATS)
@pytest.mark.parametrize("scalar_data", [("1.0.1", "String"), (np.int8(1), "Int8"), (0.5, "Float64")])
def test_scalars(format_data: Tuple[str, Callable[[], DafWriter]], scalar_data: Tuple[StorageScalar, str]) -> None:
    format_name, create_empty = format_data
    scalar_value, julia_type = scalar_data

    data = create_empty()
    assert data.name == "test!"

    assert len(data.scalar_names()) == 0
    assert not data.has_scalar("foo")
    data.set_scalar("foo", scalar_value)
    assert data.has_scalar("foo")
    assert data.get_scalar("foo") == scalar_value
    assert set(data.scalar_names()) == set(["foo"])

    if julia_type == "String":
        scalar_value = '"' + str(scalar_value) + '"'
    else:
        scalar_value = f"{scalar_value} ({julia_type})"

    assert (
        data.description()
        == dedent(
            f"""
                name: test!
                type: {format_name}
                scalars:
                  foo: {scalar_value}
            """
        )[1:]
    )

    data.delete_scalar("foo")
    assert len(data.scalar_names()) == 0
    assert not data.has_scalar("foo")


@pytest.mark.parametrize("format_data", FORMATS)
@pytest.mark.parametrize("axis_data", [("cell", ["A", "B"]), ("gene", np.array(["X", "Y", "Z"]))])
def test_axes(format_data: Tuple[str, Callable[[], DafWriter]], axis_data: Tuple[str, Sequence[str]]) -> None:
    format_name, create_empty = format_data
    axis_name, axis_entries = axis_data

    data = create_empty()

    assert len(data.axis_names()) == 0
    assert not data.has_axis(axis_name)

    data.add_axis(axis_name, axis_entries)

    assert data.axis_length(axis_name) == len(axis_entries)
    assert list(data.get_axis(axis_name)) == list(axis_entries)
    assert set(data.axis_names()) == set([axis_name])
    assert (
        data.description()
        == dedent(
            f"""
                name: test!
                type: {format_name}
                axes:
                  {axis_name}: {len(axis_entries)} entries
            """
        )[1:]
    )

    data.delete_axis(axis_name)

    assert len(data.axis_names()) == 0
    assert not data.has_axis(axis_name)


@pytest.mark.parametrize("format_data", FORMATS)
@pytest.mark.parametrize(
    "vector_data",
    [(["X", "Y"], "String"), ([1, 2], "Int64"), ([1.0, 2.0], "Float64"), (np.array([1, 2], dtype="i1"), "Int8")],
)
def test_np_vectors(format_data: Tuple[str, Callable[[], DafWriter]], vector_data: Tuple[Any, str]) -> None:
    format_name, create_empty = format_data
    vector_entries, julia_type = vector_data

    data = create_empty()
    data.add_axis("cell", ["A", "B"])
    assert len(data.vector_names("cell")) == 0
    assert not data.has_vector("cell", "foo")

    vector_entries = np.array(vector_entries)
    data.set_vector("cell", "foo", vector_entries)

    assert set(data.vector_names("cell")) == set(["foo"])
    stored_vector = data.get_np_vector("cell", "foo")

    assert list(stored_vector) == list(vector_entries)
    if isinstance(stored_vector[0], str):
        julia_type = f"PythonCall.Utils.Static{julia_type}{{UInt32, 1}}"
    else:
        vector_entries[0] = vector_entries[1]
        assert list(stored_vector) == list(vector_entries)
    assert (
        data.description()
        == dedent(
            f"""
                name: test!
                type: {format_name}
                axes:
                  cell: 2 entries
                vectors:
                  cell:
                    foo: 2 x {julia_type} (PyArray{{{julia_type}, 1, true, true, {julia_type}}} - Dense)
            """
        )[1:]
    )

    data.delete_vector("cell", "foo")

    assert len(data.vector_names("cell")) == 0
    assert not data.has_vector("cell", "foo")
