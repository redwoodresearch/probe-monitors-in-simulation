"""Coverage check for the SCE recipe with Seed-OSS-36B as the reader (RULE.md Amendment 5): which arms exist in G3's
stored Seed dumps, at G3's chosen layers (40 and 63), and what G3's fit run left on its work Volume. Listings only
(`modal volume ls --json`), $0; no file is read. Prints and writes results/coverage_seed_<stamp>.json.

  uv run --no-project --python 3.12 --with modal==1.6.1 python experiments/2026-10-13_q3_sce_reply_probe/coverage_seed.py
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
CLI = ["uv", "run", "--no-project", "--python", "3.12", "--with", "modal==1.6.1", "modal", "volume", "ls"]
ACTS = "probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12"
WORK = "probe-shift_aswin-surya_sub-q2-g3_work_kill-2026-11-12"
FIT_RUN, DATA_ID = "2026-10-13_111525", "f3cb8687cb12"  # G3's fit run (g3_x_2026-10-13_112514.json) and its packed data


def ls(vol, path):
    r = subprocess.run(CLI + [vol, path, "--json"], capture_output=True, text=True, timeout=300)
    if r.returncode:
        return None  # not there (or the listing failed): reported as missing
    return [(e["filename"].rsplit("/", 1)[-1], e["type"], e["size"]) for e in json.loads(r.stdout)]


out = {"written": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "volume": ACTS, "dumps": {}}
for run, t, _ in ls(ACTS, "/") or []:
    if t != "dir":
        continue
    arms = ls(ACTS, f"{run}/meta") or []
    out["dumps"][run] = {"metas_per_arm": {a: len([e for e in (ls(ACTS, f"{run}/meta/{a}") or []) if e[0].endswith(".npy") is False and e[0].endswith(".json")])
                                           for a, ta, _ in arms if ta == "dir"},
                         "chunk_files": {f"L{l}": len([e for e in (ls(ACTS, f"{run}/by_layer/L{l}") or []) if e[0].endswith(".npy")]) for l in (40, 63)}}
out["g3_work"] = {"volume": WORK, "fit_run": FIT_RUN,
                  "fit_run_files": sorted(f"{n} {s}" for n, _, s in (ls(WORK, f"runs/{FIT_RUN}") or [])),
                  "data_id": DATA_ID, "data_files": sorted(f"{n} {s}" for n, _, s in (ls(WORK, f"data/{DATA_ID}") or [])),
                  "packed_files": sorted(f"{n} {s}" for n, _, s in (ls(WORK, f"data/{DATA_ID}/packed") or []))}
p = HERE / "results" / f"coverage_seed_{out['written']}.json"
p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
