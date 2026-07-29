"""
Test the Julia environment set up by ``Daf``.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from dafpy.julia_import import jl

#: Helpers that ``Daf`` defines for its own use. They live in the ``DafPy`` module so that other Python packages
#: wrapping Julia packages can define helpers of their own without clashing with these.
HELPER_NAMES = (
    "_DafReadersVector",
    "_inefficient_action_handler",
    "_optional_julia_vector_names",
    "_pairify_axes",
    "_pairify_columns",
    "_pairify_data",
    "_pairify_merge",
    "_strip_wrappers",
    "_to_daf_readers",
    "pyconvert_rule_jl_object",
    "pyconvert_rule_undef",
)

#: Names exported by the Julia packages we wrap. Importing (rather than ``using``) them keeps these out of ``Main``.
EXPORTED_NAMES = (
    "AbnormalHandler",
    "DafReader",
    "MemoryDaf",
    "ReadOnlyArray",
    "chain_reader",
    "daf_as_anndata",
    "reconstruct_axis_b",
)


def _is_defined_in_main(name: str) -> bool:
    return bool(jl.seval(f"isdefined(Main, :{name})"))


def test_helpers_are_in_their_own_module() -> None:
    assert _is_defined_in_main("DafPy")
    for name in HELPER_NAMES:
        assert not _is_defined_in_main(name), f"the helper {name} leaked into Julia's Main"
        assert jl.seval(f"isdefined(Main.DafPy, :{name})"), f"the helper {name} is missing from Main.DafPy"


def test_wrapped_packages_do_not_leak() -> None:
    for name in EXPORTED_NAMES:
        assert not _is_defined_in_main(name), f"the exported {name} leaked into Julia's Main"
