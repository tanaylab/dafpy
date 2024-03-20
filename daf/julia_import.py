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
from typing import Any
from typing import Dict
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp  # type: ignore

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

jl.seval("import DataFrames")
jl.seval("import HDF5")
jl.seval("import LinearAlgebra")
jl.seval("import NamedArrays")
jl.seval("import SparseArrays")

jl.seval(
    """
    function _pairify_columns(items::AbstractVector)::Daf.QueryColumns
        return [name => query for (name, query) in items]
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


JULIA_TYPE_OF_PY_TYPE = {
    bool: jl.Bool,
    int: jl.Int64,
    float: jl.Float64,
    np.int8: jl.Int8,
    np.int16: jl.Int16,
    np.int32: jl.Int32,
    np.int64: jl.Int64,
    np.uint8: jl.UInt8,
    np.uint16: jl.UInt16,
    np.uint32: jl.UInt32,
    np.uint64: jl.UInt64,
    np.float32: jl.Float32,
    np.float64: jl.Float64,
}


def _to_julia(value: Any) -> Any:  # pylint: disable=too-many-return-statements
    if isinstance(value, np.dtype):
        return JULIA_TYPE_OF_PY_TYPE[value.type]

    if isinstance(value, type):
        return JULIA_TYPE_OF_PY_TYPE[value]

    if isinstance(value, UndefInitializer):
        return jl.undef

    if isinstance(value, str):
        return value

    if hasattr(value, "jl_obj"):
        return value.jl_obj

    if isinstance(value, (sp.csc_matrix, sp.csr_matrix)):
        colptr = jl.Vector(value.indptr)
        rowval = jl.Vector(value.indices)
        nzval = jl.Vector(value.data)

        colptr_as_array = np.asarray(colptr)
        rowval_as_array = np.asarray(rowval)

        colptr_as_array += 1
        rowval_as_array += 1

        nrows, ncols = value.shape
        if isinstance(value, sp.csr_matrix):
            nrows, ncols = ncols, nrows

        julia_matrix = jl.SparseArrays.SparseMatrixCSC(nrows, ncols, colptr, rowval, nzval)

        if isinstance(value, sp.csr_matrix):
            julia_matrix = jl.LinearAlgebra.transpose(julia_matrix)

        return julia_matrix

    if isinstance(value, Sequence) and not isinstance(value, np.ndarray):
        value = np.array(value)

    if isinstance(value, np.ndarray) and value.dtype.type == np.str_:
        value = jl.Vector(value)

    return value


def _from_julia_array(julia_array: Any) -> np.ndarray | sp.csc_matrix:
    try:
        indptr = np.array(julia_array.colptr)
        indptr -= 1
        indices = np.array(julia_array.rowval)
        indices -= 1
        data = np.asarray(julia_array.nzval)
        return sp.csc_matrix((data, indices, indptr), julia_array.shape)
    except:
        pass

    python_array = np.asarray(julia_array)
    if python_array.dtype == "object":
        python_array = np.array([str(obj) for obj in python_array], dtype=str)
    return python_array


def _as_vector(vector_ish: Any) -> Any:
    if isinstance(vector_ish, np.ndarray):
        shape = vector_ish.shape
        if len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
            vector_ish = vector_ish.reshape(-1)
    return vector_ish


def _from_julia_frame(
    jl_frame: jl.DataFrames.DataFrame,  # type: ignore
) -> pd.DataFrame:
    data: Dict[str, Any] = {}
    for name in jl.names(jl_frame):
        value = jl.getindex(jl_frame, jl.Colon(), name)
        data[str(name)] = _from_julia_array(value)
    return pd.DataFrame(data)
