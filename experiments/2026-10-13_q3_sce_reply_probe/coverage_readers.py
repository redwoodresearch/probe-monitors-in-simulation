"""Coverage check for the SCE recipe with Olmo-3.1-32B and Llama-3.3-70B as readers (RULE.md Amendments 7 and 8): which
arms exist in each reader's stored dumps (and the q3rmfa simulated honest dumps), at each reader's chosen layers, and
what each reader's committed fit run left on its work Volume. Listings only (`modal volume ls --json`), $0; no file is
read. Prints and writes results/coverage_readers_<stamp>.json.

  uv run --no-project --python 3.12 --with modal==1.6.1 python experiments/2026-10-13_q3_sce_reply_probe/coverage_readers.py
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
CLI = ["uv", "run", "--no-project", "--python", "3.12", "--with", "modal==1.6.1", "modal", "volume", "ls"]
RMFA = "probe-shift_aswin-surya_sub-q3rmfa_acts_kill-2026-11-13"
READERS = {  # acts Volume, work Volume, fit run, its packed data, chosen layers per test fold (reply_break records)
    "olmo": ("probe-shift_aswin-surya_sub-q2-olmo_acts_kill-2026-11-12", "probe-shift_aswin-surya_sub-q2-olmo_work_kill-2026-11-12",
             "2026-10-12_110549", "356ad1bc0eeb", (32, 56, 56, 56, 40)),
    "llama": ("probe-shift_aswin-surya_sub-q2-llama_acts_kill-2026-11-12", "probe-shift_aswin-surya_sub-q2-llama_work_kill-2026-11-12",
              "2026-10-12_143129", "36a9844b9775", (60, 70, 60, 60, 70)),
}


def ls(vol, path):
    r = subprocess.run(CLI + [vol, path, "--json"], capture_output=True, text=True, timeout=300)
    if r.returncode:
        return None  # not there (or the listing failed): reported as missing
    return [(e["filename"].rsplit("/", 1)[-1], e["type"], e["size"]) for e in json.loads(r.stdout)]


def dumps(vol, layers, only=None):
    out = {}
    for run, t, _ in ls(vol, "/") or []:
        if t != "dir" or (only and only not in run):
            continue
        arms = ls(vol, f"{run}/meta") or []
        out[run] = {"metas_per_arm": {a: len([e for e in (ls(vol, f"{run}/meta/{a}") or []) if e[0].endswith(".json")]) for a, ta, _ in arms if ta == "dir"},
                    "layer_dirs": sorted(n for n, tt, _ in (ls(vol, f"{run}/by_layer") or []) if tt == "dir"),
                    "chunk_files": {f"L{l}": len([e for e in (ls(vol, f"{run}/by_layer/L{l}") or []) if e[0].endswith(".npy")]) for l in sorted(set(layers))}}
    return out


out = {"written": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "readers": {}}
for name, (acts, work, fit, data, layers) in READERS.items():
    out["readers"][name] = {
        "acts_volume": acts, "dumps": dumps(acts, layers), "chosen_layers": layers,
        "simhonest_volume": RMFA, "simhonest_dumps": dumps(RMFA, layers, only={"olmo": "olmo31", "llama": "llama33"}[name]),
        "work_volume": work, "fit_run": fit, "fit_run_files": sorted(f"{n} {s}" for n, _, s in (ls(work, f"runs/{fit}") or [])),
        "data_id": data, "data_files": sorted(f"{n} {s}" for n, _, s in (ls(work, f"data/{data}") or [])),
        "packed_files": sorted(f"{n} {s}" for n, _, s in (ls(work, f"data/{data}/packed") or []))}
p = HERE / "results" / f"coverage_readers_{out['written']}.json"
p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
