"""
Import the Julia environment.

This provides additional functionality on top of ``juliapkg`` (which automatically downloads Julia packages for us). It
will import the ``juliacall`` module to create a Julia environment (as ``jl``), and uses it to import the ``Daf.jl``
Julia package.

You can control the behavior using the following environment variables:

* ``PYTHON_JULIACALL_HANDLE_SIGNALS`` (by default, ``yes``). This is needed to avoid segfaults when calling
  Julia from Python.

* ``PYTHON_JULIACALL_THREADS`` (by default, ``auto``). By default Julia will use all the threads available
  in the machine. On machines with hyper-threading you may want to specify only half the number of threads (that is,
  just have one thread per physical core) as that should provide better performance for compute-intensive (as opposed to
  IO-intensive) code.

* ``PYTHON_JULIACALL_OPTLEVEL`` (by default, ``3``).

This code is based on the code from the ``pysr`` Python package, adapted to our needs.
"""

import os
import sys
import warnings

__all__ = ["jl", "jl_version", "UndefInitializer", "Undef"]

IGNORE_REIMPORT = False

# Check if JuliaCall is already loaded, and if so, warn the user
# about the relevant environment variables. If not loaded,
# set up sensible defaults.
if "juliacall" in sys.modules:
    warnings.warn(
        "juliacall module already imported. "
        "Make sure that you have set the environment variable "
        "PYTHON_JULIACALL_HANDLE_SIGNALS=yes to avoid segfaults. "
        "Also note that Daf will not be able to configure "
        "PYTHON_JULIACALL_THREADS or PYTHON_JULIACALL_OPTLEVEL for you."
    )
else:
    # Required to avoid segfaults (https://juliapy.github.io/PythonCall.jl/dev/faq/)
    if os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes") != "yes":
        warnings.warn(
            "PYTHON_JULIACALL_HANDLE_SIGNALS environment variable is set to something other than 'yes' or ''. "
            + "You will experience segfaults if running with multithreading."
        )

    if os.environ.get("PYTHON_JULIACALL_THREADS", "auto") != "auto":
        warnings.warn(
            "PYTHON_JULIACALL_THREADS environment variable is set to something other than 'auto', "
            "so Daf was not able to set it. You may wish to set it to 'auto' for full use "
            "of your CPU."
        )

    # TODO: Remove these when juliapkg lets you specify this
    for k, default in (
        ("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes"),
        ("PYTHON_JULIACALL_THREADS", "auto"),
        ("PYTHON_JULIACALL_OPTLEVEL", "3"),
    ):
        os.environ[k] = os.environ.get(k, default)

from juliacall import Main  # type: ignore

#: The interface to the Julia run-time.
jl = Main

#: The version of Julia being used.
jl_version = (jl.VERSION.major, jl.VERSION.minor, jl.VERSION.patch)

jl.seval("using Daf")

jl.seval("import LinearAlgebra")

jl.seval("import SparseArrays")

#: This would not be needed if/when this `issue <https://github.com/JuliaPy/PythonCall.jl/issues/477>`_ is resolved.
jl.seval(
    """
function py_function_to_fulia_function(py_object::Py)::Function
    return (args...; kwargs...) -> pycall(py_object, args...; kwargs...)
end
"""
)


class UndefInitializer:
    """
    A Python class to use instead of Julia's ``UndefInitializer``. We need this to allow ``@overload`` to work in the
    presence of ``Undef``.
    """


#: A Python value to use instead of Julia's ``undef``. We need this to allow ``@overload`` to work in the presence of
#: ``undef``.
Undef = UndefInitializer()
