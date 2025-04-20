"""
Functions from `TanayLabUtilities <https://tanaylab.github.io/TanayLabUtilities.jl>`__ which it is useful to make
available. In principle we should put these in a separate ``TanayLabUtilities.py`` wrapper package, but that's too much of
a hassle.
"""

from sys import stderr
from sys import stdout
from typing import Literal
from typing import TextIO

from .julia_import import jl

__all__ = ["AbnormalHandler", "inefficient_action_handler", "LogLevel", "setup_logger"]

#: The action to take when encountering an "abnormal" (but recoverable) operation. See the Julia
#: `documentation <https://tanaylab.github.io/TanayLabUtilities.jl/v0.1.1/handlers.html#TanayLabUtilities.Handlers.AbnormalHandler>`__
#: for details.
AbnormalHandler = Literal["IgnoreHandler"] | Literal["WarnHandler"] | Literal["ErrorHandler"]

JL_ABNORMAL_HANDLER = {
    "IgnoreHandler": jl.TanayLabUtilities.IgnoreHandler,
    "WarnHandler": jl.TanayLabUtilities.WarnHandler,
    "ErrorHandler": jl.TanayLabUtilities.ErrorHandler,
}

PY_ABNORMAL_HANDLER = {
    jl.TanayLabUtilities.IgnoreHandler: "IgnoreHandler",
    jl.TanayLabUtilities.WarnHandler: "WarnHandler",
    jl.TanayLabUtilities.ErrorHandler: "ErrorHandler",
}


def inefficient_action_handler(handler: AbnormalHandler) -> AbnormalHandler:
    """
    Specify the ``AbnormalHandler`` to use when accessing a matrix in an inefficient way ("against the grain"). Returns
    the previous handler. See the Julia
    `documentation <https://tanaylab.github.io/TanayLabUtilities.jl/v0.1.1/matrix_layouts.html#TanayLabUtilities.MatrixLayouts.GLOBAL_INEFFICIENT_ACTION_HANDLER>`__
    for details.
    """
    return PY_ABNORMAL_HANDLER[jl._inefficient_action_handler(JL_ABNORMAL_HANDLER[handler])]  # type: ignore


#: The (Julia) log levels.
LogLevel = Literal["Debug"] | Literal["Info"] | Literal["Warn"] | Literal["Error"] | int

JL_LOG_LEVEL = {
    "Debug": jl.Logging.Debug,
    "Info": jl.Logging.Info,
    "Warn": jl.Logging.Warn,
    "Error": jl.Logging.Error,
}


def setup_logger(
    io: TextIO = stderr,
    *,
    level: LogLevel = "Warn",
    show_time: bool = True,
    show_module: bool = True,
    show_location: bool = False,
) -> None:
    """
    Setup a global logger that will print into ``io`` (which currently must be either ``sys.stdout`` or ``sys.stderr``),
    printing messages with a timestamp prefix. See the Julia
    `documentation <https://tanaylab.github.io/TanayLabUtilities.jl/v0.1.1/logger.html#TanayLabUtilities.Logger.setup_logger>`__
    for details.
    """
    if id(io) == id(stdout):
        jl_io = jl.stdout
    elif id(io) == id(stderr):
        jl_io = jl.stderr
    else:
        raise ValueError("not implemented: logging into anything other than stdout and stderr")

    if isinstance(level, int):
        jl_level = jl.Logging.LogLevel(level)
    else:
        jl_level = JL_LOG_LEVEL[level]

    jl.TanayLabUtilities.Logger.setup_logger(
        jl_io, level=jl_level, show_time=show_time, show_module=show_module, show_location=show_location
    )
