"""
Functions from `TanayLabUtilities <https://tanaylab.github.io/TanayLabUtilities.jl>`__ which it is useful to make
available. In principle we should put these in a separate ``TanayLabUtilities.py`` wrapper package, but that's too much of
a hassle.
"""

# The enum values are named exactly as they are in Julia, so they are not UPPER_CASE.
# pylint: disable=invalid-name


from sys import stderr
from sys import stdout
from typing import Optional
from typing import TextIO

from .julia_import import JlEnum
from .julia_import import _given
from .julia_import import jl

__all__ = ["AbnormalHandler", "inefficient_action_handler", "LogLevel", "setup_logger"]


class AbnormalHandler(JlEnum):
    """
    The action to take when encountering an "abnormal" (but recoverable) operation. See the Julia
    `documentation <https://tanaylab.github.io/TanayLabUtilities.jl/v0.1.0/handlers.html#TanayLabUtilities.Handlers.AbnormalHandler>`__
    for details.
    """

    #: Ignore the abnormal operation.
    IgnoreHandler = "IgnoreHandler"
    #: Print a warning for the abnormal operation.
    WarnHandler = "WarnHandler"
    #: Raise an error for the abnormal operation.
    ErrorHandler = "ErrorHandler"


JL_ABNORMAL_HANDLER = {
    AbnormalHandler.IgnoreHandler: jl.TanayLabUtilities.IgnoreHandler,
    AbnormalHandler.WarnHandler: jl.TanayLabUtilities.WarnHandler,
    AbnormalHandler.ErrorHandler: jl.TanayLabUtilities.ErrorHandler,
}

PY_ABNORMAL_HANDLER = {
    jl.TanayLabUtilities.IgnoreHandler: AbnormalHandler.IgnoreHandler,
    jl.TanayLabUtilities.WarnHandler: AbnormalHandler.WarnHandler,
    jl.TanayLabUtilities.ErrorHandler: AbnormalHandler.ErrorHandler,
}


def inefficient_action_handler(handler: AbnormalHandler) -> AbnormalHandler:
    """
    Specify the ``AbnormalHandler`` to use when accessing a matrix in an inefficient way ("against the grain"). Returns
    the previous handler. See the Julia
    `documentation <https://tanaylab.github.io/TanayLabUtilities.jl/v0.1.0/matrix_layouts.html#TanayLabUtilities.MatrixLayouts.GLOBAL_INEFFICIENT_ACTION_HANDLER>`__
    for details.
    """
    return PY_ABNORMAL_HANDLER[jl.DafPy._inefficient_action_handler(JL_ABNORMAL_HANDLER[handler])]  # type: ignore


class LogLevel(JlEnum):
    """
    The (Julia) log levels. A specific ``int`` level can be used instead of any of these.
    """

    #: Show debug messages and above.
    Debug = "Debug"
    #: Show informational messages and above.
    Info = "Info"
    #: Show warning messages and above.
    Warn = "Warn"
    #: Show only error messages.
    Error = "Error"


JL_LOG_LEVEL = {
    LogLevel.Debug: jl.Logging.Debug,
    LogLevel.Info: jl.Logging.Info,
    LogLevel.Warn: jl.Logging.Warn,
    LogLevel.Error: jl.Logging.Error,
}


def setup_logger(
    io: TextIO = stderr,
    *,
    level: Optional[LogLevel | int] = None,
    show_time: Optional[bool] = None,
    show_module: Optional[bool] = None,
    show_location: Optional[bool] = None,
) -> None:
    """
    Setup a global logger that will print into ``io`` (which currently must be either ``sys.stdout`` or ``sys.stderr``),
    printing messages with a timestamp prefix. See the Julia
    `documentation <https://tanaylab.github.io/TanayLabUtilities.jl/v0.1.0/logger.html#TanayLabUtilities.Logger.setup_logger>`__
    for details.
    """
    if id(io) == id(stdout):
        jl_io = jl.stdout
    elif id(io) == id(stderr):
        jl_io = jl.stderr
    else:
        raise ValueError("not implemented: logging into anything other than stdout and stderr")

    if level is None:
        jl_level = None
    elif isinstance(level, int):
        jl_level = jl.Logging.LogLevel(level)
    else:
        jl_level = JL_LOG_LEVEL[level]

    jl.TanayLabUtilities.Logger.setup_logger(
        jl_io,
        **_given(level=jl_level, show_time=show_time, show_module=show_module, show_location=show_location),
    )
