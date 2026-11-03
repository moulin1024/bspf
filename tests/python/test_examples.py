# SPDX-License-Identifier: GPL-3.0-or-later
"""Every example script runs to completion (its own convergence asserts pass)."""
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLES = sorted((ROOT / "examples").glob("*/*.py"))


@pytest.mark.parametrize("script", [p for p in EXAMPLES if p.name != "benchmarks.py"], ids=lambda p: p.stem)
def test_example_runs(script):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "python"), os.environ.get("PYTHONPATH", "")]))
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
