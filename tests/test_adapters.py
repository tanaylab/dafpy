"""
Test ``Daf`` adapters.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

import dafpy as dp


def test_adapters() -> None:  # pylint: disable=too-many-statements
    daf = dp.MemoryDaf(name="memory!")
    daf.set_scalar("INPUT", 1)
    with dp.adapter(daf, input_data={"input": ": INPUT"}, output_data={"OUTPUT": ": output"}) as adapted:
        adapted.set_scalar("output", adapted.get_scalar("input"))
    assert daf.get_scalar("OUTPUT") == 1
