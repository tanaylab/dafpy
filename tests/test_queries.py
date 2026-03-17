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
        (dp.Axis("cell"), "@ cell"),
        (dp.Axis() | dp.Names(), "@ ?"),
        (dp.LookupScalar("version"), ". version"),
        (dp.Axis("cell") | dp.LookupVector("batch") | dp.LookupVector("age"), "@ cell : batch : age"),
        (
            dp.Axis("cell") | dp.LookupVector("manual") | dp.AsAxis("type") | dp.LookupVector("color"),
            "@ cell : manual =@ type : color",
        ),
        (
            dp.Axis("cell") | dp.LookupVector("batch") | dp.AsAxis() | dp.CountBy("manual") | dp.AsAxis("type"),
            "@ cell : batch =@ * manual =@ type",
        ),
        (
            dp.Axis("cell") | dp.LookupVector("age") | dp.GroupBy("batch") | dp.AsAxis() | dp.Mean(),
            "@ cell : age / batch =@ >> Mean",
        ),
        (dp.Axis("cell") | dp.LookupVector("batch") | dp.IfNot() | dp.LookupVector("age"), "@ cell : batch ?? : age"),
        (dp.Axis("cell") | dp.LookupVector("type") | dp.IfNot("Outlier"), "@ cell : type ?? Outlier"),
        (
            dp.Axis("cell") | dp.LookupVector("batch") | dp.IfNot(1) | dp.LookupVector("age"),
            "@ cell : batch ?? 1 : age",
        ),
        (dp.Axis("gene") | dp.LookupVector("is_marker") | dp.IfMissing(False), "@ gene : is_marker || false"),
        (
            dp.Axis("cell") | dp.LookupVector("type") | dp.IfMissing("red") | dp.LookupVector("color"),
            "@ cell : type || red : color",
        ),
        (
            dp.Axis("cell")
            | dp.LookupVector("type")
            | dp.IsEqual("LMPP")
            | dp.LookupVector("age")
            | dp.Max()
            | dp.IfMissing(0),
            "@ cell : type = LMPP : age >> Max || 0",
        ),
        (
            dp.Axis("cell") | dp.Axis("gene") | dp.LookupMatrix("UMIs") | dp.Log(base=2, eps=1),
            "@ cell @ gene :: UMIs % Log base 2.0 eps 1.0",
        ),
        (dp.Axis("cell") | dp.Axis("gene") | dp.LookupMatrix("UMIs") | dp.Sum(), "@ cell @ gene :: UMIs >> Sum"),
        (dp.Axis("cell") | dp.LookupVector("age") | dp.CountBy("type"), "@ cell : age * type"),
        (dp.Axis("cell") | dp.LookupVector("age") | dp.GroupBy("type") | dp.Mean(), "@ cell : age / type >> Mean"),
        (
            dp.Axis("cell") | dp.Axis("gene") | dp.LookupMatrix("UMIs") | dp.GroupBy("type") | dp.Max(),
            "@ cell @ gene :: UMIs / type >> Max",
        ),
        (dp.Axis("gene") | dp.BeginMask("is_marker") | dp.EndMask(), "@ gene [ is_marker ]"),
        (dp.Axis("gene") | dp.BeginNegatedMask("is_marker") | dp.EndMask(), "@ gene [ ! is_marker ]"),
        (
            dp.Axis("gene") | dp.BeginMask("is_marker") | dp.OrMask("is_noisy") | dp.EndMask(),
            "@ gene [ is_marker | is_noisy ]",
        ),
        (
            dp.Axis("gene") | dp.BeginMask("is_marker") | dp.OrNegatedMask("is_noisy") | dp.EndMask(),
            "@ gene [ is_marker | ! is_noisy ]",
        ),
        (
            dp.Axis("gene") | dp.BeginMask("is_marker") | dp.XorMask("is_noisy") | dp.EndMask(),
            "@ gene [ is_marker ^ is_noisy ]",
        ),
        (
            dp.Axis("gene") | dp.BeginMask("is_marker") | dp.XorNegatedMask("is_noisy") | dp.EndMask(),
            "@ gene [ is_marker ^ ! is_noisy ]",
        ),
        (
            dp.Axis("cell") | dp.Axis("gene") | dp.IsEqual("FOX1") | dp.LookupMatrix("UMIs"),
            "@ cell @ gene = FOX1 :: UMIs",
        ),
        (dp.Axis("cell") | dp.BeginMask("age") | dp.IsEqual(1) | dp.EndMask(), "@ cell [ age = 1 ]"),
        (dp.Axis("cell") | dp.BeginMask("age") | dp.IsNotEqual(1) | dp.EndMask(), "@ cell [ age != 1 ]"),
        (dp.Axis("cell") | dp.BeginMask("age") | dp.IsLess(1) | dp.EndMask(), "@ cell [ age < 1 ]"),
        (dp.Axis("cell") | dp.BeginMask("age") | dp.IsLessEqual(1) | dp.EndMask(), "@ cell [ age <= 1 ]"),
        (dp.Axis("cell") | dp.BeginMask("age") | dp.IsGreater(1) | dp.EndMask(), "@ cell [ age > 1 ]"),
        (dp.Axis("cell") | dp.BeginMask("age") | dp.IsGreaterEqual(1) | dp.EndMask(), "@ cell [ age >= 1 ]"),
        (dp.Axis("gene") | dp.BeginMask("name") | dp.IsMatch("RP[SL]") | dp.EndMask(), r"@ gene [ name ~ RP\[SL\] ]"),
        (
            dp.Axis("gene") | dp.BeginMask("name") | dp.IsNotMatch(r"RP[SL]") | dp.EndMask(),
            r"@ gene [ name !~ RP\[SL\] ]",
        ),
        (
            dp.Axis("gene") | dp.BeginMask("is_marker") | dp.AndMask("is_lateral") | dp.EndMask(),
            "@ gene [ is_marker & is_lateral ]",
        ),
        (
            dp.Axis("gene") | dp.BeginMask("is_marker") | dp.AndNegatedMask("is_lateral") | dp.EndMask(),
            "@ gene [ is_marker & ! is_lateral ]",
        ),
        (
            dp.Axis("metacell") | dp.LookupMatrix("edge_weight") | dp.SquareColumnIs("mc1"),
            "@ metacell :: edge_weight @| mc1",
        ),
        (
            dp.Axis("metacell") | dp.LookupMatrix("edge_weight") | dp.SquareRowIs("mc1"),
            "@ metacell :: edge_weight @| mc1",
        ),
        (
            dp.Axis("cell")
            | dp.Axis("gene")
            | dp.LookupMatrix("UMIs")
            | dp.GroupColumnsBy("metacell")
            | dp.ReduceToColumn(dp.Sum()),
            "@ cell @ gene :: UMIs |/ metacell >| Sum",
        ),
        (
            dp.Axis("cell")
            | dp.Axis("gene")
            | dp.LookupMatrix("UMIs")
            | dp.GroupRowsBy("type")
            | dp.ReduceToRow(dp.Mean()),
            "@ cell @ gene :: UMIs -/ type >- Mean",
        ),
    ],
)
def test_query_formatting(query_data: Tuple[dp.Query, str]) -> None:
    query_object, query_string = query_data
    assert str(query_object) == query_string


def test_query_result() -> None:  # pylint: disable=too-many-statements
    daf = dp.memory_daf(name="test!")
    daf.set_scalar("version", "1.0")
    daf.add_axis("cell", ["A", "B"])
    daf.add_axis("gene", ["X", "Y", "Z"])
    daf.add_axis("batch", ["U", "V", "W"])
    daf.set_vector("cell", "batch", ["U", "V"])
    daf.set_vector("cell", "age", [-1.0, 2.0])
    daf.set_vector("batch", "sex", ["Male", "Female", "Male"])
    daf.set_matrix("gene", "cell", "UMIs", np.array([[1, 2, 3], [4, 5, 6]]).transpose())

    assert dp.query_result_dimensions(q(". version")) == 0  # type: ignore
    assert dp.query_result_dimensions("@ cell : age") == 1
    assert dp.query_result_dimensions("@ cell @ gene :: UMIs") == 2

    assert daf[". version"] == "1.0"
    assert set(daf[". ?"]) == set(["version"])  # type: ignore
    assert set(daf["@ cell : ?"]) == set(["batch", "age"])  # type: ignore
    assert set(daf["@ gene @ cell :: ?"]) == set(["UMIs"])  # type: ignore

    assert daf.has_query("@ cell : age")
    assert not daf.has_query("@ cell : youth")

    series = daf.get_pd_query("@ cell : age")
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

    frame = q("@ cell @ gene :: UMIs") | daf.get_pd_query()
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

    assert np.all(daf["@ cell : age"] == np.array([-1, 2]))
    assert np.all(daf["@ cell : batch"] == np.array(["U", "V"]))
    assert np.all(daf["@ cell : batch"] == ["U", "V"])
    assert np.all(daf["@ cell @ gene :: UMIs"] == np.array([[1, 2, 3], [4, 5, 6]]))

    assert np.all(daf[q("@ cell : age") | dp.Abs()] == np.array([1.0, 2.0]))
    assert np.all(daf[q("@ cell : age") | dp.Clamp(min=0.5)] == np.array([0.5, 2.0]))
    assert np.all(daf[q("@ cell : age") | dp.Convert(type=np.int8)] == np.array([-1, 2]))
    assert np.all(daf[q("@ cell : age % Abs") | dp.Fraction()] == np.array([1 / 3, 2 / 3]))
    assert np.all(daf[q("@ cell : age % Abs") | dp.Log(base=2)] == np.array([0.0, 1.0]))
    assert np.all(daf["@ cell : age % Significant high 2"] == [0, 2])

    assert daf[q("@ cell : age") | dp.Max()] == 2
    assert daf[q("@ cell : age") | dp.Min()] == -1
    assert daf[q("@ cell : age") | dp.Median()] == 0.5
    assert daf[q("@ cell : age") | dp.Quantile(p=0.5)] == 0.5
    assert daf[q("@ cell : age") | dp.Mean()] == 0.5
    assert daf[q("@ cell : age >> Mean") | dp.Round()] == 0
    assert daf[q("@ cell : age") | dp.Std()] == 1.5
    assert daf[q("@ cell : age") | dp.StdN()] == 3.0
    assert daf[q("@ cell : age") | dp.Var()] == 2.25
    assert daf[q("@ cell : age") | dp.VarN()] == 4.5

    assert daf[q("@ cell @ gene :: UMIs") | dp.Sum()] == 21
    assert np.all(daf[q("@ cell @ gene :: UMIs") | dp.ReduceToColumn(dp.Sum())] == np.array([6, 15]))
    assert np.all(daf[q("@ cell @ gene :: UMIs") | dp.ReduceToRow(dp.Sum())] == np.array([5, 7, 9]))

    daf.set_matrix("cell", "gene", "UMIs", sp.csc_matrix([[0, 1, 2], [3, 4, 0]]), overwrite=True)
    frame = daf.get_pd_query(q("@ cell @ gene :: UMIs"))  # type: ignore
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

    frame = daf.get_pd_frame("@ cell", {"age": ": age", "sex": ": batch : sex"})
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

    frame = daf.get_pd_frame("@ cell", ["age", ("sex", ": batch : sex")])
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

    assert np.all(daf[q("@ cell : batch") | dp.CountBy("age")] == np.array([[1, 0], [0, 1]]))

    daf.add_axis("type", ["T", "B"])
    daf.set_vector("cell", "type", ["T", "B"])
    daf.set_matrix("type", "type", "overlap", np.array([[1, 0], [0, 1]]).T.astype(np.float32))
    assert np.all(daf[q("@ type :: overlap") | dp.SquareColumnIs("T")] == np.array([1.0, 0.0]))
    assert np.all(daf[q("@ type :: overlap") | dp.SquareRowIs("T")] == np.array([1.0, 0.0]))

    assert dp.is_axis_query("@ cell") is True
    assert dp.is_axis_query("@ cell : age") is False
    assert dp.is_axis_query(". version") is False
    assert dp.is_axis_query(dp.Axis("cell")) is True

    assert dp.query_axis_name("@ cell") == "cell"
    assert dp.query_axis_name(dp.Axis("gene")) == "gene"

    axis_query = dp.parse_query("@ cell")
    assert isinstance(axis_query, dp.Axis)

    scalar_query = dp.parse_query(". version")
    assert isinstance(scalar_query, dp.LookupScalar)

    sequence_query = dp.parse_query("@ cell : age")
    assert isinstance(sequence_query, dp.QuerySequence)

    np_result = q("@ cell : age") | daf.get_np_query()
    assert np.all(np_result == np.array([-1, 2]))
