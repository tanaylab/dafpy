"""
Import/export ``Daf`` data from/to ``AnnData``. See the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/anndata_format.html>`__ for details.
"""

__all__ = ["h5ad_as_daf", "daf_as_h5ad"]

from typing import Optional

from .data import DafReader
from .formats import MemoryDaf
from .generic_functions import JL_ABNORMAL_HANDLER
from .generic_functions import AbnormalHandler
from .julia_import import jl


def h5ad_as_daf(
    h5ad: str,
    *,
    name: Optional[str] = None,
    obs_is: Optional[str] = None,
    var_is: Optional[str] = None,
    X_is: Optional[str] = None,
    unsupported_handler: AbnormalHandler = "WarnHandler",
) -> MemoryDaf:
    """
    View ``AnnData`` as a ``Daf`` data set, specifically using a ``MemoryDaf``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/anndata_format.html#anndata_as_daf>`__ for
    details.

    Note that you only pass an ``h5ad`` path, since the Julia ``AnnData`` object comes from the ``Muon.jl`` package and
    is not compatible with the Python ``anndata`` object."""
    return MemoryDaf(
        jl.DataAxesFormats.anndata_as_daf(
            h5ad,
            name=name,
            obs_is=obs_is,
            var_is=var_is,
            X_is=X_is,
            unsupported_handler=JL_ABNORMAL_HANDLER[unsupported_handler],
        )
    )


def daf_as_h5ad(
    daf: DafReader, *, obs_is: Optional[str] = None, var_is: Optional[str] = None, X_is: Optional[str] = None, h5ad: str
) -> None:
    """
    View the ``Daf`` data set as ``AnnData``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/anndata_format.html#DataAxesFormats.AnnDataFormat.daf_as_anndata>`__
    for details.

    Note this just creates the ``h5ad`` file. We do not return the ``AnnData`` object, because it is a Julia
    (``Muon.jl``) ``AnnData`` object, which is **not** a Python ``anndata`` ``AnnData`` object.
    """
    jl.daf_as_anndata(daf, obs_is=obs_is, var_is=var_is, X_is=X_is, h5ad=h5ad)
