"""
Types that arguably should belong in a more general-purpose package. See the Julia
`documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/generic.html>`__ for details.
"""

from typing import Literal

from .julia_import import jl

__all__ = ["AbnormalHandler"]


#: The action to take when encountering an "abnormal" (but recoverable) operation. See the Julia
#: `documentation <https://tanaylab.github.io/Daf.jl/v0.1.0/generic.html#Daf.Generic.AbnormalHandler>`__ for details.
AbnormalHandler = Literal["IgnoreHandler"] | Literal["WarnHandler"] | Literal["ErrorHandler"]

JL_ABNORMAL_HANDLER = {
    "IgnoreHandler": jl.Daf.Generic.IgnoreHandler,
    "WarnHandler": jl.Daf.Generic.WarnHandler,
    "ErrorHandler": jl.Daf.Generic.ErrorHandler,
}
