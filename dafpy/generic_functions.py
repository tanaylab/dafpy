"""
Types that arguably should belong in a more general-purpose package. See the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/generic.html>`__ for details.
"""

from typing import Literal

from .julia_import import jl

__all__ = ["AbnormalHandler", "inefficient_action_handler"]


#: The action to take when encountering an "abnormal" (but recoverable) operation. See the Julia
#: `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/generic.html#Daf.Generic.AbnormalHandler>`__ for
#: details.
AbnormalHandler = Literal["IgnoreHandler"] | Literal["WarnHandler"] | Literal["ErrorHandler"]

JL_ABNORMAL_HANDLER = {
    "IgnoreHandler": jl.DataAxesFormats.GenericFunctions.IgnoreHandler,
    "WarnHandler": jl.DataAxesFormats.GenericFunctions.WarnHandler,
    "ErrorHandler": jl.DataAxesFormats.GenericFunctions.ErrorHandler,
}

PY_ABNORMAL_HANDLER = {
    jl.DataAxesFormats.GenericFunctions.IgnoreHandler: "IgnoreHandler",
    jl.DataAxesFormats.GenericFunctions.WarnHandler: "WarnHandler",
    jl.DataAxesFormats.GenericFunctions.ErrorHandler: "ErrorHandler",
}


def inefficient_action_handler(handler: AbnormalHandler) -> AbnormalHandler:
    """
    Specify the ``AbnormalHandler`` to use when accessing a matrix in an inefficient way ("against the grain"). Returns
    the previous handler. See the Julia
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/generic_functions.html#Daf.MatrixLayouts.inefficient_action_handler>`__
    for details.
    """
    return PY_ABNORMAL_HANDLER[jl.DataAxesFormats.MatrixLayouts.inefficient_action_handler(JL_ABNORMAL_HANDLER[handler])]  # type: ignore
