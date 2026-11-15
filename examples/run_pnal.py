"""Launch the PnAL evaluation runner using previously saved experiment runs.

Mount Google Drive in Colab, install the project dependencies and select SHARD
at the beginning of src/pnal/pnal_algorithm.py before running this entry point.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "pnal" / "pnal_algorithm.py")],
        cwd=ROOT,
        check=True,
    )
