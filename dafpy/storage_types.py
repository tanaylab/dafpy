"""
Type definitions for the types that can be stored in ``Daf``.
"""

import numpy as np

__all__ = ["StorageNumber", "StorageScalar"]

#: Supported Python number types.
#:
#: The built-in Python ``int`` and ``float`` types are automatically converted to ``np.int64`` and ``np.float64``,
#: respectively. Use the ``numpy`` types if you want to force the use of a specific (smaller) Julia type.
StorageNumber = (
    bool
    | int
    | float
    | np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | np.float32
    | np.float64
)

#: Supported Python scalar types.
StorageScalar = StorageNumber | str
