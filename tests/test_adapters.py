"""
Test ``Daf`` adapters.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from daf import *


def test_adapters() -> None:  # pylint: disable=too-many-statements
    dset = MemoryDaf(name="memory!")
    dset.set_scalar("INPUT", 1)
    with adapter(viewer(dset, data={"input": ": INPUT"}), data={"OUTPUT": ": output"}) as adapted:
        adapted.set_scalar("output", adapted.get_scalar("input"))
    assert dset.get_scalar("OUTPUT") == 1
