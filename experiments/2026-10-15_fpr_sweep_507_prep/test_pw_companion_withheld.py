"""pw_companion's summary check (2e via 507-prep, 10-17, withheld audit item (a)): a withheld summ() block (finite point, ci95 None,
"withheld") is NOT an available block. The nested summ_ is compiled from the runner's own source (no Modal, no record).

  uv run python experiments/2026-10-15_fpr_sweep_507_prep/test_pw_companion_withheld.py
"""

import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
tree = ast.parse((HERE / "run_peraction311.py").read_text())
pc = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pw_companion")
fn = next(n for n in pc.body if isinstance(n, ast.FunctionDef) and n.name == "summ_")
ns = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "summ_", "exec"), ns)
summ_ = ns["summ_"]
OK = {"point": 0.4, "finite_draws": 1000, "draws": 1000, "ci95": [0.3, 0.5], "degenerate": False}
cases = {
    "a complete block is available": summ_(OK),
    "a withheld block (finite point, ci95 None) is NOT available": not summ_({"point": 0.4, "finite_draws": 900, "draws": 1000, "ci95": None,
                                                                            "withheld": "fewer than 95% complete draws"}),
    "a block with the withheld key alone is NOT available": not summ_(OK | {"withheld": "x"}),
    "a block without ci95 is NOT available": not summ_({k: v for k, v in OK.items() if k != "ci95"}),
    "no finite draw: NOT available (as before)": not summ_(OK | {"finite_draws": 0}),
    "a bool point: NOT available (as before)": not summ_(OK | {"point": True}),
    "not a dict: NOT available": not summ_(None),
}
for name, ok in cases.items():
    print(("PASS " if ok else "FAIL ") + name)
assert all(cases.values()), "pw_companion withheld tests failed"
print(f"{len(cases)} checks pass")
