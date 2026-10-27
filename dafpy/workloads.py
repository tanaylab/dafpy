"""
Make the calls of a typical session (for internal use).

The Julia package ``DafPy`` runs this while it is precompiled, so the code these calls compile is cached, and a session
does not compile it again. It is not part of the API, so the package does not import it.
"""

import os
import tempfile

import numpy as np
import scipy.sparse as sp  # type: ignore

from .copies import copy_all
from .data import DafWriter
from .formats import complete_daf
from .formats import files_daf
from .formats import memory_daf
from .formats import open_daf
from .views import viewer

__all__ = ["run_workload"]


def run_workload() -> None:
    """
    Make the calls of a typical session, on small data in a temporary directory (for internal use).
    """
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "workload")
        files = files_daf(path, "w+", name="workload")
        assert isinstance(files, DafWriter)
        memory = memory_daf(name="workload")

        for daf in (memory, files):
            daf.add_axis("cell", ["A", "B", "C"])
            daf.add_axis("gene", np.array(["X", "Y"]))
            daf.set_vector("cell", "age", np.array([1, 2, 3], dtype="float32"))
            daf.set_vector("cell", "type", ["T", "U", "T"])
            daf.set_matrix("gene", "cell", "UMIs", np.array([[1, 2, 3], [4, 5, 6]], dtype="uint32", order="F"))
            daf.set_matrix(
                "gene", "cell", "is_high", sp.csc_matrix(np.array([[True, False, True], [False, False, True]]))
            )

        for reader in (memory, open_daf(path), complete_daf(path)):
            reader.get_np_vector("cell", "age")
            reader.get_pd_vector("cell", "type")
            reader.get_np_matrix("gene", "cell", "UMIs")
            reader.get_pd_matrix("gene", "cell", "is_high")
            _ = reader["@ cell : age"]
            viewer(reader, name="view").description()
            copy_all(destination=memory_daf(name="copy"), source=reader)
