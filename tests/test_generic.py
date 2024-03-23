"""
Test ``Daf`` generic module.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from daf import *


def test_generic() -> None:  # pylint: disable=too-many-statements
    assert inefficient_action_handler("ErrorHandler") == "IgnoreHandler"
    assert inefficient_action_handler("IgnoreHandler") == "ErrorHandler"
