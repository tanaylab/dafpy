"""
Test fake function.
"""

# pylint: disable=wildcard-import,missing-function-docstring
# flake8: noqa: F403,F405

from daf import *


def test_fake_function() -> None:
    assert fake_function() == "Hello"
