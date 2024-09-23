"""
Test ``Daf`` generic module.
"""

# pylint: disable=wildcard-import,unused-wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from sys import stderr
from sys import stdout

import dafpy as dp

from .utilities import assert_raises


def test_generic_functions() -> None:
    assert dp.inefficient_action_handler("ErrorHandler") == "WarnHandler"
    assert dp.inefficient_action_handler("WarnHandler") == "ErrorHandler"


def test_generic_logging() -> None:
    with open("tests/test_generic.py", "r", encoding="utf8") as file:
        with assert_raises("not implemented: logging into anything other than stdout and stderr"):
            dp.setup_logger(file)

    dp.setup_logger(stdout, level="Error")
    dp.setup_logger(stderr, level=0)

    dp.setup_logger()
