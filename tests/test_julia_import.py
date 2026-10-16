"""
Test the Julia environment set up by ``Daf``.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from typing import Any

import numpy as np
import pandas as pd
import pytest

from dafpy.julia_import import _from_julia_frame
from dafpy.julia_import import _to_julia_array
from dafpy.julia_import import _to_julia_frame
from dafpy.julia_import import jl

#: Helpers that ``Daf`` defines for its own use. They live in the ``DafPy`` module so that other Python packages
#: wrapping Julia packages can define helpers of their own without clashing with these.
HELPER_NAMES = (
    "_DafReadersVector",
    "_inefficient_action_handler",
    "_optional_julia_vector_names",
    "_pairify_axes",
    "_pairify_columns",
    "_pairify_data",
    "_pairify_merge",
    "_strip_wrappers",
    "_to_daf_readers",
    "_to_frame",
    "pyconvert_rule_jl_object",
    "pyconvert_rule_undef",
)

#: Names exported by the Julia packages we wrap. Importing (rather than ``using``) them keeps these out of ``Main``.
EXPORTED_NAMES = (
    "AbnormalHandler",
    "DafReader",
    "MemoryDaf",
    "ReadOnlyArray",
    "chain_reader",
    "daf_as_anndata",
    "reconstruct_axis_b",
)


def _is_defined_in_main(name: str) -> bool:
    return bool(jl.seval(f"isdefined(Main, :{name})"))


def test_helpers_are_in_their_own_module() -> None:
    assert _is_defined_in_main("DafPy")
    for name in HELPER_NAMES:
        assert not _is_defined_in_main(name), f"the helper {name} leaked into Julia's Main"
        assert jl.seval(f"isdefined(Main.DafPy, :{name})"), f"the helper {name} is missing from Main.DafPy"


def test_wrapped_packages_do_not_leak() -> None:
    for name in EXPORTED_NAMES:
        assert not _is_defined_in_main(name), f"the exported {name} leaked into Julia's Main"


def test_the_enums_namespace_holds_the_same_types() -> None:
    import dafpy as dp  # pylint: disable=import-outside-toplevel

    assert sorted(dp.enums.__all__) == ["AbnormalHandler", "CacheGroup", "LogLevel", "MergeAction"]
    for name in dp.enums.__all__:
        assert getattr(dp.enums, name) is getattr(dp, name)


def test_frames_round_trip() -> None:
    frame = pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "count": np.array([1, 2, 3], dtype="int32"),
            "score": [0.5, 1.5, 2.5],
            "is_good": [True, False, True],
        }
    )

    jl_frame = _to_julia_frame(frame)
    assert int(jl.DataFrames.nrow(jl_frame)) == 3
    assert [str(name) for name in jl.names(jl_frame)] == ["name", "count", "score", "is_good"]
    assert str(jl.eltype(jl.getindex(jl_frame, jl.Colon(), "count"))) == "Int32"

    back = _from_julia_frame(jl_frame)
    assert list(back.columns) == ["name", "count", "score", "is_good"]
    assert list(back["name"]) == ["A", "B", "C"]
    assert list(back["count"]) == [1, 2, 3]
    assert list(back["score"]) == [0.5, 1.5, 2.5]
    assert list(back["is_good"]) == [True, False, True]


def _is_julia_strings_vector(value: Any) -> bool:
    return bool(jl.seval("value -> value isa AbstractVector{<:AbstractString}")(value))  # NOT F-STRING


def test_to_julia_array_of_strings() -> None:
    for value in (
        np.array(["a", "b"], dtype=object),
        np.array([b"a", b"b"]),
        pd.Series(["a", "b"]),
        pd.Index(["a", "b"]),
        pd.Categorical(["a", "b"]),
    ):
        converted = _to_julia_array(value)
        assert _is_julia_strings_vector(converted), f"not a vector of strings: {type(value)}"
        assert [str(entry) for entry in converted] == ["a", "b"]


def test_to_julia_array_of_numbers() -> None:
    converted = _to_julia_array(np.array([1, 2], dtype=object))
    assert converted.dtype == np.int64
    assert list(converted) == [1, 2]

    converted = _to_julia_array(pd.Series([0.5, 1.5]))
    assert converted.dtype == np.float64
    assert list(converted) == [0.5, 1.5]


def test_to_julia_array_rejects_missing_strings() -> None:
    with pytest.raises(TypeError, match="entry 0 is a str and whose entry 1 is a NoneType"):
        _to_julia_array(np.array(["a", None], dtype=object))

    with pytest.raises(TypeError, match="entry 0 is a str and whose entry 1 is a float"):
        _to_julia_array(pd.Categorical(["a", None]))

    with pytest.raises(TypeError, match="an array of NoneType"):
        _to_julia_array(np.array([None, None], dtype=object))
