"""Re-extract the behaviour runs whose extracted attempt was not the counted one (Codex on #401, thread
PRRT_kwDOU9RNXc6rIbC1). The launcher counts the first attempt with no redrawn_because that is not a rerun
(make_phase2_attempt_labels.py); probe_shift.viewer.pairs._log takes the last attempt with an .eval. This runs
extract_behavior.py unchanged, with _log swapped for the counted rule, on the mismatched runs only, and splices them into
the archived CSV (numbers only).

  PYTHONPATH=src uv run --no-project --python 3.14 --with pandas python \
      experiments/2026-10-10_scaleup_behavior/recheck_counted_attempts.py <archived csv> <out csv> <launch dir> [...]
"""
import runpy
import sys
from pathlib import Path

import pandas as pd

import probe_shift.viewer.pairs as P

HERE = Path(__file__).parent
archived, out, launches = Path(sys.argv[1]), Path(sys.argv[2]), [Path(a) for a in sys.argv[3:]]
launches = list(dict.fromkeys(p.resolve() for p in launches))  # each launch once (Codex)
if len({p.name for p in launches}) != len(launches):
    raise ValueError("two launch folders share a name: the archived CSV keys rows by launch name")
last_with_eval, read_launch = P._log, P.read_launch


def counted(run_dir, run):
    """_log's record for the launcher's counted attempt (first, not redrawn, not a rerun), or None if it has no .eval."""
    atts = run.get("attempts") or []
    i = next((i for i, a in enumerate(atts) if not a.get("redrawn_because") and a.get("kind") != "rerun"), None)
    if i is None:
        return None
    rec = last_with_eval(run_dir, run | {"attempts": atts[: i + 1]})
    return rec if rec and rec["attempt"] == i + 1 else None


wrong = {}  # launch name -> keys whose extracted attempt differs from the counted one
for rd in launches:
    for r in read_launch(rd)["runs"]:
        a, b = last_with_eval(rd, r), counted(rd, r)
        if (a or {}).get("attempt") != (b or {}).get("attempt"):
            wrong.setdefault(rd.name, set()).add(r["key"])
            print(f"{rd.name} {r['key']}: extracted attempt {(a or {}).get('attempt')}, counted {(b or {}).get('attempt')}")
P._log = counted
P.read_launch = lambda rd, tries=5: (L := read_launch(rd, tries)) | {"runs": [r for r in L["runs"] if r["key"] in wrong.get(rd.name, ())]}
tmp = out.with_suffix(".fixed_rows.csv")
sys.argv = ["extract_behavior.py", str(tmp)] + [str(rd) for rd in launches if rd.name in wrong]
runpy.run_path(str(HERE / "extract_behavior.py"), run_name="__main__")
old, new = pd.read_csv(archived), pd.read_csv(tmp)
if new.duplicated(["launch", "key"]).any() or len(new) != sum(len(k) for k in wrong.values()):
    raise ValueError("re-extracted rows do not match the mismatched runs one to one")
drop = old.apply(lambda r: r["key"] in wrong.get(r["launch"], ()), axis=1)
print(f"replacing {int(drop.sum())} archived rows with {len(new)} re-extracted rows")
pd.concat([old[~drop], new]).to_csv(out, index=False)
print("->", out)
