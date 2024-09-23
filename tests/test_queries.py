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

import dafpy as dp
from dafpy import parse_query as q


@pytest.mark.parametrize(
    "query_data",
    [
        (dp.Axis("cell"), "/ cell"),
        (dp.Names("axes"), "? axes"),
        (dp.Lookup("version"), ": version"),
        (dp.Axis("cell") | dp.Lookup("batch") | dp.Fetch("age"), "/ cell : batch => age"),
        (
            dp.Axis("cell") | dp.Lookup("manual") | dp.AsAxis("type") | dp.Fetch("color"),
            "/ cell : manual ! type => color",
        ),
        (
            dp.Axis("cell") | dp.Lookup("batch") | dp.AsAxis() | dp.CountBy("manual") | dp.AsAxis("type"),
            "/ cell : batch ! * manual ! type",
        ),
        (
            dp.Axis("cell") | dp.Lookup("age") | dp.GroupBy("batch") | dp.AsAxis() | dp.Mean(),
            "/ cell : age @ batch ! %> Mean",
        ),
        (dp.Axis("cell") | dp.Lookup("batch") | dp.IfNot() | dp.Fetch("age"), "/ cell : batch ?? => age"),
        (dp.Axis("cell") | dp.Lookup("type") | dp.IfNot("Outlier"), "/ cell : type ?? Outlier"),
        (dp.Axis("cell") | dp.Lookup("batch") | dp.IfNot(1) | dp.Fetch("age"), "/ cell : batch ?? 1 => age"),
        (dp.Axis("gene") | dp.Lookup("is_marker") | dp.IfMissing(False), "/ gene : is_marker || false Bool"),
        (
            dp.Axis("cell") | dp.Lookup("type") | dp.IfMissing("red") | dp.Fetch("color"),
            "/ cell : type || red => color",
        ),
        (
            dp.Axis("cell") | dp.Lookup("type") | dp.IsEqual("LMPP") | dp.Fetch("age") | dp.Max() | dp.IfMissing(0),
            "/ cell : type = LMPP => age %> Max || 0 Int64",
        ),
        (
            dp.Axis("cell") | dp.Axis("gene") | dp.Lookup("UMIs") | dp.Log(base=2, eps=1),
            "/ cell / gene : UMIs % Log base 2.0 eps 1.0",
        ),
        (dp.Axis("cell") | dp.Axis("gene") | dp.Lookup("UMIs") | dp.Sum(), "/ cell / gene : UMIs %> Sum"),
        (dp.Axis("cell") | dp.Lookup("age") | dp.CountBy("type"), "/ cell : age * type"),
        (dp.Axis("cell") | dp.Lookup("age") | dp.GroupBy("type") | dp.Mean(), "/ cell : age @ type %> Mean"),
        (
            dp.Axis("cell") | dp.Axis("gene") | dp.Lookup("UMIs") | dp.GroupBy("type") | dp.Max(),
            "/ cell / gene : UMIs @ type %> Max",
        ),
        (dp.Axis("gene") | dp.And("is_marker"), "/ gene & is_marker"),
        (dp.Axis("gene") | dp.AndNot("is_marker"), "/ gene &! is_marker"),
        (dp.Axis("gene") | dp.And("is_marker") | dp.Or("is_noisy"), "/ gene & is_marker | is_noisy"),
        (dp.Axis("gene") | dp.And("is_marker") | dp.OrNot("is_noisy"), "/ gene & is_marker |! is_noisy"),
        (dp.Axis("gene") | dp.And("is_marker") | dp.Xor("is_noisy"), "/ gene & is_marker ^ is_noisy"),
        (dp.Axis("gene") | dp.And("is_marker") | dp.XorNot("is_noisy"), "/ gene & is_marker ^! is_noisy"),
        (dp.Axis("cell") | dp.Axis("gene") | dp.IsEqual("FOX1") | dp.Lookup("UMIs"), "/ cell / gene = FOX1 : UMIs"),
        (dp.Axis("cell") | dp.And("age") | dp.IsEqual(1), "/ cell & age = 1"),
        (dp.Axis("cell") | dp.And("age") | dp.IsNotEqual(1), "/ cell & age != 1"),
        (dp.Axis("cell") | dp.And("age") | dp.IsLess(1), "/ cell & age < 1"),
        (dp.Axis("cell") | dp.And("age") | dp.IsLessEqual(1), "/ cell & age <= 1"),
        (dp.Axis("cell") | dp.And("age") | dp.IsGreater(1), "/ cell & age > 1"),
        (dp.Axis("cell") | dp.And("age") | dp.IsGreaterEqual(1), "/ cell & age >= 1"),
        (dp.Axis("gene") | dp.And("name") | dp.IsMatch("RP[SL]"), r"/ gene & name ~ RP\[SL\]"),
        (dp.Axis("gene") | dp.And("name") | dp.IsNotMatch(r"RP[SL]"), r"/ gene & name !~ RP\[SL\]"),
    ],
)
def test_query_formatting(query_data: Tuple[dp.Query, str]) -> None:
    query_object, query_string = query_data
    assert str(query_object) == query_string


def test_query_result() -> None:  # pylint: disable=too-many-statements
    daf = dp.MemoryDaf(name="test!")
    daf.set_scalar("version", "1.0")
    daf.add_axis("cell", ["A", "B"])
    daf.add_axis("gene", ["X", "Y", "Z"])
    daf.add_axis("batch", ["U", "V", "W"])
    daf.set_vector("cell", "batch", ["U", "V"])
    daf.set_vector("cell", "age", [-1.0, 2.0])
    daf.set_vector("batch", "sex", ["Male", "Female", "Male"])
    daf.set_matrix("gene", "cell", "UMIs", np.array([[1, 2, 3], [4, 5, 6]]).transpose())

    assert dp.query_result_dimensions(q(": version")) == 0
    assert dp.query_result_dimensions("/ cell : age") == 1
    assert dp.query_result_dimensions("/ cell / gene : UMIs") == 2

    assert daf[": version"] == "1.0"
    assert set(daf["? scalars"]) == set(["version"])  # type: ignore
    assert set(daf["/ cell ?"]) == set(["batch", "age"])  # type: ignore
    assert set(daf["/ gene / cell ?"]) == set(["UMIs"])  # type: ignore

    series = daf.get_pd_query("/ cell : age")
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

    frame = daf.get_pd_query(q("/ cell / gene : UMIs"))
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

    assert np.all(daf["/ cell : age"] == np.array([-1, 2]))
    assert np.all(daf["/ cell : batch"] == np.array(["U", "V"]))
    assert np.all(daf["/ cell : batch"] == ["U", "V"])
    assert np.all(daf["/ cell /gene : UMIs"] == np.array([[1, 2, 3], [4, 5, 6]]))

    assert np.all(daf[q("/ cell : age") | dp.Abs()] == np.array([1.0, 2.0]))
    assert np.all(daf[q("/ cell : age") | dp.Clamp(min=0.5)] == np.array([0.5, 2.0]))
    assert np.all(daf[q("/ cell : age") | dp.Convert(dtype=np.int8)] == np.array([-1, 2]))
    assert np.all(daf[q("/ cell : age % Abs") | dp.Fraction()] == np.array([1 / 3, 2 / 3]))
    assert np.all(daf[q("/ cell : age % Abs") | dp.Log(base=2)] == np.array([0.0, 1.0]))
    assert np.all(daf["/ cell : age % Significant high 2"] == [0, 2])

    assert daf[q("/ cell : age") | dp.Max()] == 2
    assert daf[q("/ cell : age") | dp.Min()] == -1
    assert daf[q("/ cell : age") | dp.Median()] == 0.5
    assert daf[q("/ cell : age") | dp.Quantile(p=0.5)] == 0.5
    assert daf[q("/ cell : age") | dp.Mean()] == 0.5
    assert daf[q("/ cell : age %> Mean") | dp.Round()] == 0
    assert daf[q("/ cell : age") | dp.Std()] == 1.5
    assert daf[q("/ cell : age") | dp.StdN()] == 3.0
    assert daf[q("/ cell : age") | dp.Var()] == 2.25
    assert daf[q("/ cell : age") | dp.VarN()] == 4.5

    daf.set_matrix("cell", "gene", "UMIs", sp.csc_matrix([[0, 1, 2], [3, 4, 0]]), overwrite=True)
    frame = daf.get_pd_query(q("/ cell / gene : UMIs"))
    assert np.all(frame.values == np.array([[0, 1, 2], [3, 4, 0]]))  # type: ignore

    frame = daf.get_pd_frame("cell")
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

    frame = daf.get_pd_frame("cell", ["age"])
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

    frame = daf.get_pd_frame("/ cell", {"age": ": age", "sex": ": batch => sex"})
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

    daf.empty_cache(clear="MappedData")
