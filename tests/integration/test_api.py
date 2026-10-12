import os
import subprocess
import time
from pathlib import Path


def test_native_binary_exists_after_build():
    root = Path(__file__).resolve().parents[2]
    binary = root / "build/cpp/atlassearch_bm25"
    assert binary.exists()
