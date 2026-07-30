"""
All the enumerated types, collected in one place so that auto-completion can list them.

Since ``clear = <TAB>`` does not offer the valid values (no IDE infers the type of an assignment target or of a
keyword argument), the practical way to discover them is ``dafpy.enums.<TAB>`` for the types, and then ``.<TAB>``
again for the values of the chosen one. The same types are also available directly, e.g. ``dafpy.CacheGroup``.

You can also ``from dafpy import enums as e`` and write ``e.CacheGroup.MappedData``, or
``from dafpy.enums import *`` and write ``CacheGroup.MappedData``.
"""

from .concat import MergeAction
from .data import CacheGroup
from .generic_functions import AbnormalHandler
from .generic_functions import LogLevel

__all__ = [
    "AbnormalHandler",
    "CacheGroup",
    "LogLevel",
    "MergeAction",
]
