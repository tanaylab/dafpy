"""
Test ``Daf`` query operations.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from textwrap import dedent
from typing import Tuple

import numpy as np
import pytest
import scipy.sparse as sp  # type: ignore

from daf import *


@pytest.mark.parametrize(
    "query_data",
    [
        (Axis("cell"), "/ cell"),
        (Names("axes"), "? axes"),
        (Lookup("version"), ": version"),
        (Axis("cell") | Lookup("batch") | Fetch("age"), "/ cell : batch => age"),
        (Axis("cell") | Lookup("manual") | AsAxis("type") | Fetch("color"), "/ cell : manual ! type => color"),
        (
            Axis("cell") | Lookup("batch") | AsAxis() | CountBy("manual") | AsAxis("type"),
            "/ cell : batch ! * manual ! type",
        ),
        (Axis("cell") | Lookup("age") | GroupBy("batch") | AsAxis() | Mean(), "/ cell : age @ batch ! %> Mean"),
        (Axis("cell") | Lookup("batch") | IfNot() | Fetch("age"), "/ cell : batch ?? => age"),
        (Axis("cell") | Lookup("type") | IfNot("Outlier"), "/ cell : type ?? Outlier"),
        (Axis("cell") | Lookup("batch") | IfNot(1) | Fetch("age"), "/ cell : batch ?? 1 => age"),
        (Axis("gene") | Lookup("is_marker") | IfMissing(False), "/ gene : is_marker || false Bool"),
        (Axis("cell") | Lookup("type") | IfMissing("red") | Fetch("color"), "/ cell : type || red => color"),
        (
            Axis("cell") | Lookup("type") | IsEqual("LMPP") | Fetch("age") | Max() | IfMissing(0),
            "/ cell : type = LMPP => age %> Max || 0 Int64",
        ),
        (
            Axis("cell") | Axis("gene") | Lookup("UMIs") | Log(base=2, eps=1),
            "/ cell / gene : UMIs % Log base 2.0 eps 1.0",
        ),
        (Axis("cell") | Axis("gene") | Lookup("UMIs") | Sum(), "/ cell / gene : UMIs %> Sum"),
        (Axis("cell") | Lookup("age") | CountBy("type"), "/ cell : age * type"),
        (Axis("cell") | Lookup("age") | GroupBy("type") | Mean(), "/ cell : age @ type %> Mean"),
        (Axis("cell") | Axis("gene") | Lookup("UMIs") | GroupBy("type") | Max(), "/ cell / gene : UMIs @ type %> Max"),
        (Axis("gene") | And("is_marker"), "/ gene & is_marker"),
        (Axis("gene") | AndNot("is_marker"), "/ gene &! is_marker"),
        (Axis("gene") | And("is_marker") | Or("is_noisy"), "/ gene & is_marker | is_noisy"),
        (Axis("gene") | And("is_marker") | OrNot("is_noisy"), "/ gene & is_marker |! is_noisy"),
        (Axis("gene") | And("is_marker") | Xor("is_noisy"), "/ gene & is_marker ^ is_noisy"),
        (Axis("gene") | And("is_marker") | XorNot("is_noisy"), "/ gene & is_marker ^! is_noisy"),
        (Axis("cell") | Axis("gene") | IsEqual("FOX1") | Lookup("UMIs"), "/ cell / gene = FOX1 : UMIs"),
        (Axis("cell") | And("age") | IsEqual(1), "/ cell & age = 1"),
        (Axis("cell") | And("age") | IsNotEqual(1), "/ cell & age != 1"),
        (Axis("cell") | And("age") | IsLess(1), "/ cell & age < 1"),
        (Axis("cell") | And("age") | IsLessEqual(1), "/ cell & age <= 1"),
        (Axis("cell") | And("age") | IsGreater(1), "/ cell & age > 1"),
        (Axis("cell") | And("age") | IsGreaterEqual(1), "/ cell & age >= 1"),
        (Axis("gene") | And("name") | IsMatch("RP[SL]"), r"/ gene & name ~ RP\[SL\]"),
        (Axis("gene") | And("name") | IsNotMatch(r"RP[SL]"), r"/ gene & name !~ RP\[SL\]"),
    ],
)
def test_query_formatting(query_data: Tuple[Query, str]) -> None:
    query_object, query_string = query_data
    assert str(query_object) == query_string


def test_query_result() -> None:  # pylint: disable=too-many-statements
    dset = MemoryDaf(name="test!")
    dset.set_scalar("version", "1.0")
    dset.add_axis("cell", ["A", "B"])
    dset.add_axis("gene", ["X", "Y", "Z"])
    dset.add_axis("batch", ["U", "V", "W"])
    dset.set_vector("cell", "batch", ["U", "V"])
    dset.set_vector("cell", "age", [-1.0, 2.0])
    dset.set_vector("batch", "sex", ["Male", "Female", "Male"])
    dset.set_matrix("gene", "cell", "UMIs", np.array([[1, 2, 3], [4, 5, 6]]).transpose())

    assert query_result_dimensions(parse_query(": version")) == 0
    assert query_result_dimensions("/ cell : age") == 1
    assert query_result_dimensions("/ cell / gene : UMIs") == 2

    assert dset[": version"] == "1.0"
    assert set(dset["? scalars"]) == set(["version"])  # type: ignore
    assert set(dset["/ cell ?"]) == set(["batch", "age"])  # type: ignore
    assert set(dset["/ gene / cell ?"]) == set(["UMIs"])  # type: ignore

    series = dset.get_pd_query("/ cell : age")
    assert (
        str(series)
        == dedent(
            """
        A   -1.0
        B    2.0
        dtype: float64
    """
        )[1:-1]
    )

    frame = dset.get_pd_query(parse_query("/ cell / gene : UMIs"))
    assert (
        str(frame)
        == dedent(
            """
           X  Y  Z
        A  1  2  3
        B  4  5  6
    """
        )[1:-1]
    )

    assert np.all(dset["/ cell : age"] == np.array([-1, 2]))
    assert np.all(dset["/ cell : batch"] == np.array(["U", "V"]))
    assert np.all(dset["/ cell : batch"] == ["U", "V"])
    assert np.all(dset["/ cell /gene : UMIs"] == np.array([[1, 2, 3], [4, 5, 6]]))

    assert np.all(dset[parse_query("/ cell : age") | Abs()] == np.array([1.0, 2.0]))
    assert np.all(dset[parse_query("/ cell : age") | Clamp(min=0.5)] == np.array([0.5, 2.0]))
    assert np.all(dset[parse_query("/ cell : age") | Convert(dtype=np.int8)] == np.array([-1, 2]))
    assert np.all(dset[parse_query("/ cell : age % Abs") | Fraction()] == np.array([1 / 3, 2 / 3]))
    assert np.all(dset[parse_query("/ cell : age % Abs") | Log(base=2)] == np.array([0.0, 1.0]))
    assert np.all(dset[parse_query("/ cell : age") | Significant(high=2)] == [0, 2])

    assert dset[parse_query("/ cell : age") | Max()] == 2
    assert dset[parse_query("/ cell : age") | Median()] == 0.5
    assert dset[parse_query("/ cell : age") | Mean()] == 0.5
    assert dset[parse_query("/ cell : age %> Mean") | Round()] == 0
    assert dset[parse_query("/ cell : age") | Quantile(p=0.5)] == 0.5
    assert dset[parse_query("/ cell : age") | Min()] == -1
    assert dset[parse_query("/ cell : age") | Std()] == 1.5
    assert dset[parse_query("/ cell : age") | StdN()] == 3.0
    assert dset[parse_query("/ cell : age") | Var()] == 2.25
    assert dset[parse_query("/ cell : age") | VarN()] == 4.5

    dset.set_matrix("cell", "gene", "UMIs", sp.csc_matrix([[0, 1, 2], [3, 4, 0]]), overwrite=True)
    frame = dset.get_pd_query(parse_query("/ cell / gene : UMIs"))
    assert np.all(frame.values == np.array([[0, 1, 2], [3, 4, 0]]))  # type: ignore

    frame = dset.get_pd_frame("cell")
    assert (
        str(frame)
        == dedent(
            """
          name  age batch
        0    A -1.0     U
        1    B  2.0     V
    """
        )[1:-1]
    )

    frame = dset.get_pd_frame("cell", ["age"])
    assert (
        str(frame)
        == dedent(
            """
           age
        0 -1.0
        1  2.0
    """
        )[1:-1]
    )

    frame = dset.get_pd_frame("/ cell", {"age": ": age", "sex": ": batch => sex"})
    assert (
        str(frame)
        == dedent(
            """
           age     sex
        0 -1.0    Male
        1  2.0  Female
    """
        )[1:-1]
    )

    dset.empty_cache(clear="MappedData")
