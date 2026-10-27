# The Julia side of `dafpy`: the helpers its wrappers call, and the precompilation of the calls Python makes.
#
# Python always loads `PythonCall` before anything else, so this loads it first too, and is precompiled the same way.
# Its workload runs `dafpy.workloads.run_workload()` from Python, so this package caches the code those calls compile,
# and a session does not compile it again.
module DafPy

using PythonCall
using DataAxesFormats
using PrecompileTools
using TanayLabUtilities

import DataFrames
import HDF5
import LinearAlgebra
import Logging
import Muon
import NamedArrays
import SparseArrays

# The code this caches depends on the Python code of `dafpy`, which runs the workload. Julia only knows of its own
# sources, so this tells it to compile again when any of those files changes.
const DAFPY_DIRECTORY = joinpath(@__DIR__, "..", "..", "..")
for name in readdir(DAFPY_DIRECTORY)
    if endswith(name, ".py")
        include_dependency(joinpath(DAFPY_DIRECTORY, name); track_content = true)
    end
end

function _inefficient_action_handler(new_handler::AbnormalHandler)::AbnormalHandler
    old_handler = TanayLabUtilities.MatrixLayouts.GLOBAL_INEFFICIENT_ACTION_HANDLER
    TanayLabUtilities.MatrixLayouts.GLOBAL_INEFFICIENT_ACTION_HANDLER = new_handler
    return old_handler
end

function _to_daf_readers(readers::AbstractVector)::Vector{DafReader}
    return Vector{DafReader}(readers)
end

function _to_daf_writers(writers::AbstractVector)::Vector{DafWriter}
    return Vector{DafWriter}(writers)
end

function _to_base_dafs(bases::AbstractVector)::Vector{Union{BaseDaf, DafReader}}
    return Vector{Union{BaseDaf, DafReader}}(bases)
end

function _to_frame(names::AbstractVector, columns::AbstractVector)::DataFrames.DataFrame
    return DataFrames.DataFrame([String(name) => column for (name, column) in zip(names, columns)]; copycols = false)
end

const _DafReadersVector = Vector{DafReader}

function _optional_julia_vector_names(vector::NamedArrays.NamedVector)::AbstractVector
    return names(vector, 1)
end
function _optional_julia_vector_names(array::AbstractVector)::Nothing
    return nothing
end

function _strip_wrappers(array::Union{ReadOnlyArray, NamedArrays.NamedArray})::AbstractArray
    array = parent(array)
    return _strip_wrappers(array)
end
function _strip_wrappers(array::AbstractArray)::AbstractArray
    return array
end

function _pairify_columns(items::Maybe{AbstractVector})::Maybe{DataAxesFormats.FrameColumns}
    if items == nothing
        return nothing
    else
        return [name => query for (name, query) in items]
    end
end

function _pairify_axes(items::Maybe{AbstractVector})::Maybe{DataAxesFormats.ViewAxes}
    if items == nothing
        return nothing
    else
        return [key => query for (key, query) in items]
    end
end

function _pairify_data(items::Maybe{AbstractVector})::Maybe{DataAxesFormats.ViewData}
    if items == nothing
        return nothing
    else
        return [key => query for (key, query) in items]
    end
end

function _pairify_merge(items::Maybe{AbstractVector})::Maybe{DataAxesFormats.MergeData}
    if items == nothing
        return nothing
    else
        return [key => query for (key, query) in items]
    end
end

function _pairify_permutations(items::AbstractVector)::AbstractDict{String, Vector{Int}}
    return Dict{String, Vector{Int}}(name => Vector{Int}(permutation) for (name, permutation) in items)
end

function _sets_vector(items::Maybe{AbstractVector})::Maybe{Vector{Set{String}}}
    if items == nothing
        return nothing
    else
        return [Set{String}(item) for item in items]
    end
end

function pyconvert_rule_jl_object(::Type{T}, x::Py) where {T}
    return PythonCall.pyconvert_return(pyconvert(T, x.jl_obj))
end

function pyconvert_rule_undef(::Type{T}, x::Py) where {T}
    return PythonCall.pyconvert_return(undef)
end

# The conversion rules are registered at load time, since a precompiled package does not keep the side effects of its
# top level.
function __init__()::Nothing
    PythonCall.pyconvert_add_rule("dafpy.julia_import:JlObject", Any, pyconvert_rule_jl_object)
    PythonCall.pyconvert_add_rule("dafpy.julia_import:UndefInitializer", UndefInitializer, pyconvert_rule_undef)
    return nothing
end

# Whether the Python that `PythonCall` runs imports the `dafpy` this package belongs to. A precompilation started from
# plain Julia gets whatever Python `PythonCall` finds, which may lack `dafpy`, or have a version of it without this
# package beside it. A directory named `dafpy` in the working directory is also found, as a namespace with no origin.
function is_matching_dafpy()::Bool
    spec = pyimport("importlib.util").find_spec("dafpy")
    if pyis(spec, pybuiltins.None) || pyis(spec.origin, pybuiltins.None)
        return false
    end
    project_path = joinpath(dirname(pyconvert(String, spec.origin)), "julia", "DafPy", "Project.toml")
    return isfile(project_path) && occursin("uuid = \"$(Base.PkgId(DafPy).uuid)\"", read(project_path, String))
end

@compile_workload begin
    __init__()
    if !is_matching_dafpy()
        error(
            "DafPy could not be precompiled: the Python that PythonCall runs ($(PythonCall.C.CTX.exe_path)) " *
            "does not import the dafpy that DafPy belongs to.\n" *
            "This is harmless: DafPy will be precompiled the first time you import dafpy from Python.\n" *
            "To do it now, run: python -c \"import dafpy\" (with the Python where dafpy is installed).",
        )
    end
    # While this is precompiled, `dafpy` cannot import it into Julia's `Main`, so it is handed over here instead.
    pysetattr(pyimport("builtins"), "_dafpy_glue", Py(DafPy))
    pyexec("from dafpy.workloads import run_workload\nrun_workload()", pydict())
end

end  # module
