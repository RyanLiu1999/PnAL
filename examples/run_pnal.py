"""Run the reported evaluation-time PnAL supplement using saved experiment runs.

Mount /content/drive/MyDrive in Colab and choose SHARD in
src/pnal/pnal_algorithm.py before launching.
"""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "src" / "pnal" / "pnal_algorithm.py"), run_name="__main__")
