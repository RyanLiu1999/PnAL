"""Check that released aggregate tables and the experiment source are internally consistent."""
from __future__ import annotations
import ast
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def load(path):
    with (ROOT / path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

main = load("data/paper/main_comparisons.csv")
assert len(main) == 40, f"Expected 40 comparison records, found {len(main)}"
settings = {(r["testbed"], r["agents"]) for r in main}
assert len(settings) == 8
assert all(sum((r["testbed"], r["agents"]) == k for r in main) == 5 for k in settings)
expected = {
    ("Intersection","2"):(91.5,8.3), ("Intersection","3"):(78.5,16.0),
    ("Intersection","4"):(62.8,28.7), ("Intersection","5"):(49.3,36.3),
    ("Roundabout","2"):(85.7,10.7), ("Roundabout","4"):(63.8,28.8),
    ("Highway","2"):(98.3,1.7), ("Highway","4"):(97.2,2.8),
}
for row in main:
    assert 0 <= float(row["success_mean_pct"]) <= 100
    assert 0 <= float(row["collision_mean_pct"]) <= 100
    if row["method"] == "PnAL":
        actual = (float(row["success_mean_pct"]),float(row["collision_mean_pct"]))
        assert actual == expected[row["testbed"],row["agents"]]
for row in load("data/paper/frozen_component_ablations.csv"):
    assert 0 <= float(row["success_mean_pct"]) <= 100
    assert 0 <= float(row["collision_mean_pct"]) <= 100
src = (ROOT / "src/pnal/pnal_algorithm.py").read_text(encoding="utf-8")
ast.parse(src)
assert "DEEPSEEK_API_KEY = os.getenv" in src
assert "%%writefile" not in src
print("PASS: eight PnAL settings, 40 comparison records, ablations, and Python source syntax")
