"""09 @ c68587df6 ("do 2"), item 4: the readability gate for the POST HOC all-readable-probes pooled RMS row on commands (C) at per-action 1e-2,
per probe x set, from the calibration pools only (the stage B exports' gate fields k / units_above / estimable; no RMS value is read). Rule (25h
item 2 at draw 0): T estimable, and >= 10 pool units strictly above T, on every side the separate line uses (H_B for real arms, H_sim for sims).
Sol (a)'s S_newmains part is the c6023b805 extension export (its own pools). Writes results/readability_commands_<stamp>.json (exclusive create).

  python3 -I experiments/2026-10-15_il_analysis_09/readability_commands_all_readable.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
SPEC = ("specs/il_posthoc_spec_stageB_wtpooled_2026-10-15_221909.json", "26a406c3e8ca96aa83fbb0506d852661b2682bcc88ef376d5997ba4ff790e311")
FIVE = ["Sol (a) C", "FIBS C", "harmfulness C", "refusal C", "high-stakes C"]
PT = "0.01"
git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731

sb = (HERE / SPEC[0]).read_bytes()
assert hashlib.sha256(sb).hexdigest() == SPEC[1], "stage B spec sha256 differs"
hits = {h["set"]: h for h in json.loads(sb)["il"]["posthoc_hits"]}


def gate(pools, pl):  # -> (readable, detail) for the separate line at PT
    pv = pools.get(pl)
    if pv is None:
        return False, "no pool entry"
    if "status" in pv:
        return False, f"export status: {pv['status']}"
    det, ok = {}, True
    for sd in ("H_B", "H_sim"):
        g = pv[sd][PT]
        ua = g.get("units_above")
        r_ = g.get("estimable") is True and isinstance(ua, int) and ua >= 10
        det[sd] = {"estimable": g.get("estimable"), "units_above": ua, "k": g.get("k"), "n_units": pv[sd].get("n_units"), "readable": r_}
        ok &= r_
    return ok, det


out = {"label": "readability gate (09 @ c68587df6 item 4) for the POST HOC all-readable-probes pooled RMS row: commands, per-action 1e-2, separate line; "
                "calibration pools only; POST HOC; PRELIMINARY",
       "rule": "per probe x set: T estimable and >= 10 pool units strictly above T on H_B and on H_sim (25h item 2, draw 0)",
       "spec": {"path": SPEC[0], "sha256": SPEC[1]}, "per_probe": {}}
for pl in FIVE:
    row = {}
    for s_ in ("S_old", "S_new", "S_newmains"):
        h = hits[s_]
        src = "export"
        pools = h["pools"]
        if s_ == "S_newmains" and pl.startswith("Sol (a)"):
            pools, src = h["extension"]["pools"], f"extension ({h['extension']['label']})"
        r_, det = gate(pools, pl)
        row[s_] = {"readable": r_, "source": src, "gate": det, "manifest": h["extension"]["manifest"] if src != "export" else h["manifest"]}
    row["sets_entering"] = [s_ for s_ in ("S_old", "S_new", "S_newmains") if row[s_]["readable"]]
    out["per_probe"][pl] = row
out["meta"] = {"git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "stamp_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")}
b = json.dumps(out, indent=1, ensure_ascii=False).encode()
fn = HERE / "results" / f"readability_commands_{out['meta']['stamp_pt']}.json"
fn.open("xb").write(b)
print(fn.name, hashlib.sha256(b).hexdigest(), json.dumps({p: v["sets_entering"] for p, v in out["per_probe"].items()}))
