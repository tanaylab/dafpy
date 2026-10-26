"""
Import/export ``Daf`` data from/to ``AnnData``. See the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/anndata_format.html>`__ for details.
"""

__all__ = ["h5ad_as_daf", "daf_as_h5ad"]

from typing import Any

from .data import DafReader
from .data import DafWriter
from .formats import memory_daf
from .generic_functions import JL_ABNORMAL_HANDLER
from .generic_functions import AbnormalHandler
from .julia_import import DEFAULT
from .julia_import import DefaultValue
from .julia_import import _given
from .julia_import import jl


def _jl_handler(handler: AbnormalHandler | DefaultValue) -> Any:
    # The Julia handler of the ``handler``, passing ``DEFAULT`` through as it is.
    return handler if isinstance(handler, DefaultValue) else JL_ABNORMAL_HANDLER[handler]


def h5ad_as_daf(
    h5ad: str,
    *,
    name: str | None | DefaultValue = DEFAULT,
    obs_is: str | None | DefaultValue = DEFAULT,
    var_is: str | None | DefaultValue = DEFAULT,
    X_is: str | None | DefaultValue = DEFAULT,
    unsupported_handler: AbnormalHandler | DefaultValue = DEFAULT,
) -> DafWriter:
    """
    View ``AnnData`` as a ``Daf`` data set, specifically using a ``MemoryDaf``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/anndata_format.html#DataAxesFormats.AnnDataFormat.anndata_as_daf>`__ for
    details.

    Note that you only pass an ``h5ad`` path, since the Julia ``AnnData`` object comes from the ``Muon.jl`` package and
    is not compatible with the Python ``anndata`` object."""
    return memory_daf(
        jl.DataAxesFormats.anndata_as_daf(
            h5ad,
            **_given(
                name=name,
                obs_is=obs_is,
                var_is=var_is,
                X_is=X_is,
                unsupported_handler=_jl_handler(unsupported_handler),
            ),
        )
    )


def daf_as_h5ad(
    daf: DafReader,
    *,
    obs_is: str | None | DefaultValue = DEFAULT,
    var_is: str | None | DefaultValue = DEFAULT,
    X_is: str | None | DefaultValue = DEFAULT,
    h5ad: str,
) -> None:
    """
    View the ``Daf`` data set as ``AnnData``. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.3.0/anndata_format.html#DataAxesFormats.AnnDataFormat.daf_as_anndata>`__
    for details.

    Note this just creates the ``h5ad`` file. We do not return the ``AnnData`` object, because it is a Julia
    (``Muon.jl``) ``AnnData`` object, which is **not** a Python ``anndata`` ``AnnData`` object.
    """
    jl.DataAxesFormats.daf_as_anndata(daf, h5ad=h5ad, **_given(obs_is=obs_is, var_is=var_is, X_is=X_is))
