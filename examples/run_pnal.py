"""Example entry point for the PnAL implementation.

Run from the repository root:

    python examples/run_pnal.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pnal import pnal_algorithm  # noqa: F401


if __name__ == "__main__":
    # The uploaded implementation includes its own main entry point in
    # src/pnal/pnal_algorithm.py. Running that file directly starts the
    # training/evaluation routine implemented by the authors.
    print("PnAL package import succeeded.")
    print("To run the full implementation:")
    print("python src/pnal/pnal_algorithm.py")
