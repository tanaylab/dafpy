"""
Test recording what Julia compiles while our public functions run.
"""

# pylint: disable=missing-function-docstring

import json
import os
import subprocess
import sys
from textwrap import dedent

from dafpy.compile_tracing import SWITCH
from dafpy.compile_tracing import report

# A call no workload makes (a vector of ``int8``), so Julia compiles for it.
SESSION = dedent("""
    import numpy as np
    import dafpy

    dafpy.memory_daf(name="traced").add_axis("cell", ["A", "B"]).set_vector(
        "cell", "level", np.array([1, 2], dtype="int8")
    )
    dafpy.memory_daf(name="traced").add_axis("cell", ["A", "B"]).set_vector(
        "cell", "level", np.array([3, 4], dtype="int8")
    )
    """)


def test_compile_tracing(tmp_path) -> None:
    subprocess.run([sys.executable, "-c", SESSION], env=dict(os.environ, **{SWITCH: str(tmp_path)}), check=True)

    names = os.listdir(tmp_path)
    assert any(name.endswith(".trace.jl") for name in names)
    records_names = [name for name in names if name.endswith(".compiles.jsonl")]
    assert len(records_names) == 1

    with open(tmp_path / records_names[0], encoding="utf8") as file:
        entries = [json.loads(line) for line in file]
    set_vector_entries = [entry for entry in entries if entry["callable"] == "dafpy.data.DafWriter.set_vector"]
    assert len(set_vector_entries) == 1
    assert set_vector_entries[0]["arguments"]["value"] == "ndarray[int8, (2,)]"
    assert set_vector_entries[0]["call_site"].startswith("<string>:")
    assert set_vector_entries[0]["compiled"]

    assert "dafpy.data.DafWriter.set_vector" in report(str(tmp_path))
