"""
Import the Julia environment.

This provides additional functionality on top of ``juliapkg`` (which automatically downloads Julia packages for us). It
will import the ``juliacall`` module to create a Julia environment (as ``jl``), and uses it to import the
``DataAxesFormats.jl`` Julia package.

You can control the behavior using the following environment variables:

* ``PYTHON_JULIACALL_HANDLE_SIGNALS`` (by default, ``yes``). This is needed to avoid segfaults when calling
  Julia from Python.

* ``PYTHON_JULIACALL_THREADS`` (by default, ``auto``). By default Julia will use all the threads available
  in the machine. On machines with hyper-threading you may want to specify only half the number of threads (that is,
  just have one thread per physical core) as that should provide better performance for compute-intensive (as opposed to
  IO-intensive) code.

* ``PYTHON_JULIACALL_OPTLEVEL`` (by default, ``3``).

This code is based on the code from the ``pysr`` Python package, adapted to our needs. TODO: Much of this is replicated
in all our Python packages that invoke Julia.
"""

import os
import sys
import warnings
from typing import Any
from typing import Mapping
from typing import MutableMapping
from typing import Sequence
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp  # type: ignore

__all__ = ["jl", "jl_version", "UndefInitializer", "Undef", "use_default_julia_environment"]

IGNORE_REIMPORT = False

# Check if JuliaCall is already loaded. If not, setup sensible defaults. If it was loaded manually, warn the user about
# the relevant environment variables.
if "juliacall" not in sys.modules:
    # TODO: Remove these when juliapkg lets you specify this.
    for k, default in (
        ("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes"),
        ("PYTHON_JULIACALL_THREADS", "auto"),
        ("PYTHON_JULIACALL_OPTLEVEL", "3"),
    ):
        os.environ[k] = os.environ.get(k, default)

if os.environ.get("PYTHON_JULIACALL_PROPER_IMPORT", "") == "":
    # Required to avoid segfaults (https://juliapy.github.io/PythonCall.jl/dev/faq/)
    if os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes") != "yes":
        warnings.warn(
            "PYTHON_JULIACALL_HANDLE_SIGNALS environment variable is set to something other than 'yes'. "
            "You will experience segfaults if running with multithreading."
        )

    if os.environ.get("PYTHON_JULIACALL_THREADS", "auto") != "auto":
        warnings.warn(
            "PYTHON_JULIACALL_THREADS environment variable is set to something other than 'auto', "
            "You may wish to set it to 'auto' for full use of your CPU."
        )

os.environ["PYTHON_JULIACALL_PROPER_IMPORT"] = "dafpy"

from juliacall import Main  # type: ignore

#: The interface to the Julia run-time.
jl = Main


def use_default_julia_environment() -> None:
    """
    Force JuliaCall to use your default environment (whatever that is). By default JuliaCall creates its own separate
    independent environment, which means that it re-downloads and re-installs all the dependency packages there.
    This is nice if you don't otherwise use Julia and/or don't care about the wasted disk space and initial installation
    time and manual control over package versions. It sucks if you do otherwise use Julia, and/or you want manual
    control over the version of some of the packages (e.g., if you are actively developing them). You would think
    there would a built-in method for doing this in JuliaCall, right?

    To make this easier (and trigger it before actually importing packages), you can set the
    ``PYTHON_JULIACALL_USE_DEFAULT_ENVIRONMENT`` environment variable to ``yes``, and then ``import dafpy`` will
    call ``use_default_julia_environment`` for you. Sigh.
    """
    default_env = jl.seval('joinpath(DEPOT_PATH[1], "environments", "v$(VERSION.major).$(VERSION.minor)")')
    jl.seval('using Pkg; Pkg.activate("' + default_env + '")')


if os.environ.get("PYTHON_JULIACALL_USE_DEFAULT_ENVIRONMENT", "yes") == "yes":
    use_default_julia_environment()

#: The version of Julia being used.
jl_version = (jl.VERSION.major, jl.VERSION.minor, jl.VERSION.patch)

jl.seval("using Pkg")

for verb, package in (
    ("using", "TanayLabUtilities"),
    ("using", "DataAxesFormats"),
    ("import", "DataFrames"),
    ("import", "HDF5"),
    ("import", "LinearAlgebra"),
    ("import", "Logging"),
    ("import", "Muon"),
    ("import", "NamedArrays"),
    ("import", "PythonCall"),
    ("import", "SparseArrays"),
):
    if jl.seval('Base.find_package("' + package + '")') is None:
        jl.seval('Pkg.add("' + package + '")')
    jl.seval(verb + " " + package)


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


class JlObject:
    """
    A Python base class for wrapping a Julia object.
    """

    def __init__(self, jl_obj) -> None:
        self.jl_obj = jl_obj

    def __str__(self) -> str:
        return jl.string(self)


def _to_julia_type(value: Any) -> Any:  # pylint: disable=too-many-return-statements
    if isinstance(value, np.dtype):
        return JULIA_TYPE_OF_PY_TYPE[value.type]

    if isinstance(value, type):
        return JULIA_TYPE_OF_PY_TYPE[value]

    return value


def _to_julia_array(value: Any) -> Any:  # pylint: disable=too-many-return-statements
    if isinstance(value, str):
        return value

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
        try:
            value = np.array(value)
        except ValueError:
            return jl.Vector(value)

    if isinstance(value, np.ndarray) and value.dtype.type == np.str_:
        value = jl.Vector(value)

    return value


def _from_julia_array(julia_array: Any, *, writeable: bool = False) -> np.ndarray | sp.csc_matrix:
    if julia_array is None:
        return None

    julia_array = jl._strip_wrappers(julia_array)

    try:
        indptr = np.array(julia_array.colptr)
        indptr -= 1

        indices = np.array(julia_array.rowval)
        indices -= 1

        data = np.asarray(julia_array.nzval)

        indptr.flags.writeable = writeable
        indices.flags.writeable = writeable
        data.flags.writeable = writeable

        return sp.csc_matrix((data, indices, indptr), julia_array.shape)
    except:
        pass

    python_array = np.asarray(julia_array)
    if python_array.dtype == "object":
        python_array = np.array([str(obj) for obj in python_array], dtype=str)
    if python_array.flags.writeable != writeable:
        python_array.flags.writeable = writeable
    return python_array


def _as_vector(vector_ish: Any) -> Any:
    if isinstance(vector_ish, np.ndarray):
        shape = vector_ish.shape
        if len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
            vector_ish = vector_ish.reshape(-1)
    return vector_ish


def _from_julia_frame(
    jl_frame: jl.DataFrames.DataFrame,  # type: ignore
    *,
    writeable: bool = False,
) -> pd.DataFrame:
    data: MutableMapping[str, Any] = {}
    for name in jl.names(jl_frame):
        value = jl.getindex(jl_frame, jl.Colon(), name)
        data[str(name)] = _from_julia_array(value, writeable=writeable)
    return pd.DataFrame(data)


jl.seval(
    """
    function _inefficient_action_handler(new_handler::AbnormalHandler)::AbnormalHandler
        old_handler = TanayLabUtilities.MatrixLayouts.GLOBAL_INEFFICIENT_ACTION_HANDLER
        TanayLabUtilities.MatrixLayouts.GLOBAL_INEFFICIENT_ACTION_HANDLER = new_handler
        return old_handler
    end
    """
)

jl.seval(
    """
    function _to_daf_readers(readers::AbstractVector)::Vector{DafReader}
        return Vector{DafReader}(readers)
    end
    """
)


def _jl_pairs(mapping: Mapping | None) -> Sequence[Tuple[str, Any]] | None:
    if mapping is None:
        return None
    return list(mapping.items())


jl.seval("_DafReadersVector = Vector{DafReader}")  # NOT F-STRING

jl.seval(
    """
    function _optional_julia_vector_names(vector::NamedArrays.NamedVector)::AbstractVector
        return names(vector, 1)
    end
    function _optional_julia_vector_names(array::AbstractVector)::Nothing
        return nothing
    end
    """
)

jl.seval(
    """
    function _strip_wrappers(array::Union{ReadOnlyArray, NamedArrays.NamedArray})::AbstractArray
        array = parent(array)
        return _strip_wrappers(array)
    end
    function _strip_wrappers(array::AbstractArray)::AbstractArray
        return array
    end
    """
)

jl.seval(
    """
    function _pairify_columns(items::Maybe{AbstractVector})::Maybe{DataAxesFormats.FrameColumns}
        if items == nothing
            return nothing
        else
            return [name => query for (name, query) in items]
        end
    end
    """
)

jl.seval(
    """
    function _pairify_axes(items::Maybe{AbstractVector})::Maybe{DataAxesFormats.ViewAxes}
        if items == nothing
            return nothing
        else
            return [key => query for (key, query) in items]
        end
    end
    """
)

jl.seval(
    """
    function _pairify_data(items::Maybe{AbstractVector})::Maybe{DataAxesFormats.ViewData}
        if items == nothing
            return nothing
        else
            return [key => query for (key, query) in items]
        end
    end
    """
)

jl.seval(
    """
    function _pairify_merge(items::Maybe{AbstractVector})::Maybe{DataAxesFormats.MergeData}
        if items == nothing
            return nothing
        else
            return [key => query for (key, query) in items]
        end
    end
    """
)

jl.seval(
    """
    function pyconvert_rule_jl_object(::Type{T}, x::Py) where {T}
        return PythonCall.pyconvert_return(pyconvert(T, x.jl_obj))
    end
    """
)

jl.seval(
    """
    PythonCall.pyconvert_add_rule("dafpy.julia_import:JlObject", Any, pyconvert_rule_jl_object)
    """
)

jl.seval(
    """
    function pyconvert_rule_undef(::Type{T}, x::Py) where {T}
        return PythonCall.pyconvert_return(undef)
    end
    """
)

jl.seval(
    """
    PythonCall.pyconvert_add_rule("dafpy.julia_import:UndefInitializer", UndefInitializer, pyconvert_rule_undef)
    """
)
