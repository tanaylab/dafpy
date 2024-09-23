"""
Test ``Daf`` conversion to and from ``AnnData``.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from tempfile import TemporaryDirectory
from textwrap import dedent

import numpy as np

import dafpy as dp


def test_h5ad() -> None:  # pylint: disable=too-many-statements
    origin = dp.MemoryDaf(name="memory!")
    origin.set_scalar("version", 1)
    origin.add_axis("cell", ["A", "B"])
    origin.add_axis("gene", ["X", "Y", "Z"])
    origin.set_matrix("gene", "cell", "UMIs", np.array([[0, 1, 2], [3, 4, 5]]).transpose(), relayout=False)

    assert (
        origin.description()
        == dedent(
            """
        name: memory!
        type: MemoryDaf
        scalars:
          version: 1 (Int64)
        axes:
          cell: 2 entries
          gene: 3 entries
        matrices:
          gene,cell:
            UMIs: 3 x 2 x Int64 in Columns (PyArray - Dense)
        """
        )[1:]
    )

    with TemporaryDirectory() as tmpdir:
        dp.daf_as_h5ad(origin, obs_is="cell", var_is="gene", X_is="UMIs", h5ad=f"{tmpdir}/test.h5ad")
        back = dp.h5ad_as_daf(f"{tmpdir}/test.h5ad", obs_is="cell", var_is="gene", X_is="UMIs")
        assert (
            back.description()
            == dedent(
                """
            name: anndata
            type: MemoryDaf
            scalars:
              X_is: "UMIs"
              obs_is: "cell"
              var_is: "gene"
              version: 1 (Int64)
            axes:
              cell: 2 entries
              gene: 3 entries
            matrices:
              gene,cell:
                UMIs: 3 x 2 x Int64 in Columns (Dense)
            """
            )[1:]
        )
