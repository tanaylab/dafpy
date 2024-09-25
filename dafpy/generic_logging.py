"""
Generic macros and functions for logging, that arguably should belong in a more general-purpose package. See the Julia
`documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/generic_logging.html>`__ for details.
"""

from sys import stderr
from sys import stdout
from typing import Literal
from typing import TextIO

from .julia_import import jl

__all__ = ["LogLevel", "setup_logger"]

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
    `documentation <https://tanaylab.github.io/DataAxesFormats.jl/v0.1.1/generic_logging.html#Daf.GenericLogging.setup_logger>`__
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

    jl.DataAxesFormats.GenericLogging.setup_logger(
        jl_io, level=jl_level, show_time=show_time, show_module=show_module, show_location=show_location
    )
