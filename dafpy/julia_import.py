"""
Import the Julia environment.

This imports the ``juliacall`` module to obtain a Julia run-time (as ``jl``), and uses it to import the
``DataAxesFormats.jl`` Julia package. Our own Julia code is the Julia package ``DafPy``, which ships inside this
package, in ``julia/DafPy``. It also caches the code our calls compile, so a session does not compile it again.

How Julia is run, and which Julia is run, is left to ``juliacall``, and is configured by its own environment variables,
which must be set before importing anything that reaches Julia. This adds one thing to them: ``@default``.

By default ``juliacall`` has ``juliapkg`` install a Julia of its own, and an environment of its own, and populates that
environment with what each installed Python package declares in its ``juliapkg.json``. That is a reasonable default, and
it is not always what you want: if you use Julia yourself, it means a second copy of everything, which you cannot see
from a Julia prompt, and whose versions you do not choose.

``juliacall`` can be pointed at a Julia instead, through ``PYTHON_JULIACALL_EXE`` and ``PYTHON_JULIACALL_PROJECT``, but
it has no way to say "the Julia I already have": the first must be an executable and the second a directory which
exists. Setting either of them to ``@default`` here means exactly that - the ``julia`` in the path, and the environment
that Julia would use by itself, which it is asked for rather than being worked out from the depot and the version.
They are expanded before ``juliacall`` sees them, and are independent, so one may be ``@default`` while the other is
given explicitly.

Setting them is a deliberate act, so nothing is assumed if you do not. In particular ``PYTHON_JULIACALL_THREADS`` and
``PYTHON_JULIACALL_HANDLE_SIGNALS`` are left exactly as you set them: Julia runs on one thread unless you ask for more,
and asking for more without also setting the signal handling to ``yes`` is what makes it crash. ``juliacall`` warns
about that combination itself; this warns, once, about the single thread, which nothing else would tell you about.

Three packages provide this expansion: ``dafpy``, ``somegraphspy``, and ``metacellspy`` (transitively, through
``dafpy``). Importing any of them expands ``@default``, so the order does not matter. If ``juliacall`` is imported
before any of them, it sees ``@default`` itself, and rejects it as a path which does not exist, naming the variable it
could not use. That is why this is a value of a variable ``juliacall`` reads, rather than a variable of our own, which
it would silently ignore.

This code is based on the code from the ``pysr`` Python package, adapted to our needs. TODO: Much of this is replicated
in all our Python packages that invoke Julia.
"""

import builtins
import os
import shutil
import sys
import tomllib
import warnings
from enum import Enum
from typing import Any
from typing import Collection
from typing import Mapping
from typing import MutableMapping
from typing import Sequence
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp  # type: ignore

__all__ = ["jl", "jl_version", "DefaultValue", "DEFAULT", "JlEnum", "UndefInitializer", "Undef"]

# The value of ``PYTHON_JULIACALL_EXE`` or ``PYTHON_JULIACALL_PROJECT`` asking for the Julia you already have, rather
# than the one ``juliapkg`` would install for itself. Not exported: it has to be in the environment before anything
# which reaches Julia is imported, so by the time it could be read from here it would be too late to use.
_DEFAULT_JULIA = "@default"


def _default_julia_exe() -> str:
    """
    Return the path of the Julia which is in the path (for internal use).

    This is resolved to the real binary, because ``juliacall`` works out where Julia's system image is from the path of
    the executable it is given, and the directory holding ``juliaup``'s shim has no ``lib/julia`` beside it.
    """
    julia_exe = shutil.which("julia")
    if julia_exe is None:
        raise ValueError(f"PYTHON_JULIACALL_EXE={_DEFAULT_JULIA}: there is no julia in the path")
    return os.path.realpath(julia_exe)


def _default_julia_project(julia_exe: str) -> str:
    """
    Return the path of the default environment of some Julia (for internal use).

    Which environment that is depends on the depot, on the version, and on ``JULIA_PROJECT``, which conda sets to an
    environment named after the conda environment. It is therefore asked of that Julia rather than worked out here.
    """
    import subprocess  # pylint: disable=import-outside-toplevel

    try:
        return subprocess.run(
            [julia_exe, "-e", "print(dirname(Base.active_project()))"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exception:
        raise ValueError(
            f"PYTHON_JULIACALL_PROJECT={_DEFAULT_JULIA}: {julia_exe} did not report its default environment"
        ) from exception


# Expand the ``@default`` we accept in the two variables ``juliacall`` uses to locate Julia. It has no notion of "the
# Julia I already have": ``PYTHON_JULIACALL_EXE`` must be an executable and ``PYTHON_JULIACALL_PROJECT`` a directory
# which exists, and if neither is given then ``juliapkg`` installs a Julia and an environment of its own.
#
# This has to happen before ``juliacall`` is imported, since that is when it reads them. If something else imported it
# first, then it has already rejected ``@default`` as a path which does not exist - which is the point of using a value
# it cannot accept, rather than a variable of our own which it would silently ignore.
if os.environ.get("PYTHON_JULIACALL_EXE") == _DEFAULT_JULIA:
    os.environ["PYTHON_JULIACALL_EXE"] = _default_julia_exe()

if os.environ.get("PYTHON_JULIACALL_PROJECT") == _DEFAULT_JULIA:
    os.environ["PYTHON_JULIACALL_PROJECT"] = _default_julia_project(
        os.environ.get("PYTHON_JULIACALL_EXE") or _default_julia_exe()
    )

# How Julia is run is left as you set it. ``juliacall`` warns by itself when signal handling is unset and Julia has more
# than one thread, which is the combination that crashes; there is nobody to warn you that leaving both unset gives you
# a single-threaded Julia, so we do.
#
# Only when ``juliacall`` has not been imported yet, so this is said once even when several of ``dafpy``,
# ``somegraphspy`` and ``metacellspy`` are imported: the first of them ends by importing ``juliacall``, so the rest stay
# quiet. Recording it in the variable instead would not work - ``juliacall`` reads an empty value as an empty value, and
# refuses it.
if "juliacall" not in sys.modules and "PYTHON_JULIACALL_THREADS" not in os.environ:
    warnings.warn(
        "PYTHON_JULIACALL_THREADS is not set, so Julia will use a single thread. Set it to 'auto' to use the whole "
        "machine, and set PYTHON_JULIACALL_HANDLE_SIGNALS to 'yes' along with it, or Julia and Python will fight over "
        "signals and the process will die with a segfault."
    )

# If ``METACELLS_TRACE_COMPILE`` is set, Julia is started writing its compile trace (see ``compile_tracing``). The module
# is imported only then, so ``python -m dafpy.compile_tracing`` does not find it imported already.
if os.environ.get("METACELLS_TRACE_COMPILE"):
    from .compile_tracing import configure_julia  # pylint: disable=import-outside-toplevel

    configure_julia()

from juliacall import Main  # type: ignore # isort: skip

# Our Julia code is the Julia package ``DafPy``, in ``julia/DafPy`` beside this file. It also caches the code our calls
# compile. Its ``Project.toml`` lists the Julia packages we use.
_GLUE_PROJECT_PATH = os.path.join(os.path.dirname(__file__), "julia", "DafPy", "Project.toml")

# While ``DafPy`` itself is precompiled, Julia may not import anything into ``Main``, so ``DafPy`` hands itself over
# instead, and it binds the same names.
_DAFPY_GLUE: Any = getattr(builtins, "_dafpy_glue", None)

#: The interface to the Julia run-time. This is Julia's ``Main``, except while ``DafPy`` itself is precompiled.
jl = Main if _DAFPY_GLUE is None else _DAFPY_GLUE

if _DAFPY_GLUE is None:
    with open(_GLUE_PROJECT_PATH, "rb") as glue_project_file:
        _GLUE_PACKAGES = tomllib.load(glue_project_file)["deps"]
    # Everything is imported rather than ``using``, so no package's exports leak into Julia's ``Main``. This keeps
    # ``Main`` clear for other Python packages that wrap Julia packages and are used in the same session.
    for package in ["DafPy"] + sorted(_GLUE_PACKAGES):
        if package != "PrecompileTools":
            jl.seval("import " + package)

#: The version of Julia being used.
jl_version = (jl.VERSION.major, jl.VERSION.minor, jl.VERSION.patch)


class JlEnum(Enum):
    """
    A Python base class for a set of named values matching a Julia type.

    Grouping the values in a class (as opposed to listing them in a ``Literal``) is what allows auto-completion to
    list them; a ``Literal`` offers no completions at all.
    """

    def __str__(self) -> str:
        return self.value


class UndefInitializer:
    """
    A Python class to use instead of Julia's ``UndefInitializer``. We need this to allow ``@overload`` to work in the
    presence of ``Undef``.
    """


#: A Python value to use instead of Julia's ``undef``. We need this to allow ``@overload`` to work in the presence of
#: ``undef``.
Undef = UndefInitializer()  # pylint: disable=invalid-name


class DefaultValue:
    """
    A Python class for the value of a parameter that was not specified.

    We can't use ``None`` for this because ``None`` is how one specifies Julia's ``nothing``, and not every parameter
    defaults to ``nothing`` (e.g., the ``packed`` of ``complete_daf`` defaults to ``false``).
    """

    def __repr__(self) -> str:
        return "DEFAULT"

    # A copy is the same object, so a copied ``DEFAULT`` is still ``DEFAULT``.
    def __copy__(self) -> "DefaultValue":
        return self

    def __deepcopy__(self, memo: Any) -> "DefaultValue":
        return self


#: A Python value for a parameter that was not specified, so the Julia default should be used instead.
DEFAULT = DefaultValue()


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

    @classmethod
    def wrap_jl_object(cls, jl_obj):
        """
        Wrap a Julia object (for internal use).
        """
        instance = cls.__new__(cls)
        JlObject.__init__(instance, jl_obj)
        return instance


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

    value = _typed_array(value)

    if isinstance(value, np.ndarray) and value.dtype.type == np.str_:
        value = jl.Vector(value)

    return value


# Julia cannot type the elements of a ``pandas`` ``Series``, ``Index`` or ``Categorical``, or of a ``numpy`` array of
# ``object`` or ``bytes``. Each would arrive as a ``PyIterable{Any}`` or a ``PyArray{Py}``, which a typed parameter
# rejects. So these are converted to a ``numpy`` array of strings or of numbers. An ``object`` array is converted only if
# all its elements are strings, or all are numbers. Anything else (e.g., a ``None`` or a ``NaN`` among strings) is an
# error, rather than silently becoming the string ``"None"`` or ``"nan"``.
def _typed_array(value: Any) -> Any:
    if isinstance(value, (pd.Series, pd.Index, pd.Categorical)):
        value = np.asarray(value)

    if not isinstance(value, np.ndarray):
        return value

    if value.dtype.type == np.bytes_:
        return np.char.decode(value, "utf-8")

    if value.dtype != object:
        return value

    entries = value.ravel()
    if all(isinstance(entry, str) for entry in entries):
        return value.astype(str)

    if all(isinstance(entry, (bool, int, float, np.bool_, np.integer, np.floating)) for entry in entries):
        return np.array(value.tolist())

    first_type = type(entries[0])
    for index, entry in enumerate(entries):
        if type(entry) is not first_type:  # pylint: disable=unidiomatic-typecheck
            raise TypeError(
                f"can't pass to Julia an array whose entry 0 is a {first_type.__name__} "
                f"and whose entry {index} is a {type(entry).__name__}"
            )
    raise TypeError(f"can't pass to Julia an array of {first_type.__name__}")


# A parameter which is either one scalar or a collection of them. A scalar crosses as itself, and so does a string,
# which is a collection of characters as far as Python is concerned but a scalar as far as Julia is. A tuple also
# crosses as itself, so everything else is converted to one: a Python list or set arrives as a ``PyList`` or ``PySet``,
# which is neither an ``AbstractVector`` nor an ``AbstractSet``, and a parameter declared as a union of a scalar and a
# collection of them therefore rejects it.
def _to_julia_scalar_or_collection(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes)):
        return value

    if isinstance(value, Collection):
        return tuple(value)

    return value


# A parameter which is a set of names. A Python set arrives as a ``PySet``, which is not an ``AbstractSet``, so a
# parameter declared as one rejects it; there is no Julia set to convert it to other than by constructing one.
#
# The element type is stated rather than taken from the values, because an empty Python set says nothing about what it
# would have held, and a set of the wrong element type is rejected exactly as a ``PySet`` is. An empty set is not an
# odd case here: it is how one says "convert none of the properties".
def _to_julia_strings_set(value: Any) -> Any:
    if value is None or value is DEFAULT:
        return value

    return jl.Set(jl.Vector[jl.String](list(value)))


def _from_julia_array(julia_array: Any, *, writeable: bool = False) -> np.ndarray | sp.csc_matrix:
    if julia_array is None:
        return None

    julia_array = jl.DafPy._strip_wrappers(julia_array)

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


# A ``pandas`` frame becomes a Julia ``DataFrame`` of the same columns. A numeric column crosses as an array, which
# ``pandas`` may or may not share. A string column is copied, since Python and Julia strings have nothing in common in
# memory.
def _to_julia_frame(frame: pd.DataFrame) -> Any:
    columns = []
    for name in frame.columns:
        column = frame[name].to_numpy()
        if column.dtype.kind in ("O", "U", "T"):
            columns.append(jl.Vector[jl.String](list(column)))
        else:
            columns.append(_to_julia_array(column))
    return jl.DafPy._to_frame([str(name) for name in frame.columns], columns)


def _to_julia(value: Any) -> Any:
    # The value to pass to Julia for a keyword argument. The wrappers convert their arguments themselves, so this passes
    # it as it is.
    return value


def _given(**kwargs: Any) -> Mapping[str, Any]:
    """
    Collect the keyword arguments that were actually specified (for internal use).
    """
    return {name: _to_julia(value) for name, value in kwargs.items() if value is not DEFAULT}


def _jl_pairs(mapping: Mapping | None) -> Sequence[Tuple[str, Any]] | None:
    if mapping is None:
        return None
    return list(mapping.items())
