"""own_frac100_L48 in the stage B RMS (09 OK via 2e, 10-15 22:5x): eligibility FIRST, before any own_frac RMS value is computed or read.
Mechanical application of the frozen rule, read by key from the registered table; nothing is recomputed:
  - Rule: Append 13 §4 (real TPR in 10–90% on v6's seeded 93-task eligibility half, per readout and run-level α), frozen for every origin by
    Amendment 7b; for the expansion family the frozen table is X_err v7 phase 1 @ 6514cf401 (RULE 21c §3: "Eligibility uses Amendment 7b's
    frozen sets (and v7's phase-1 table @ 6514cf401 for the expansion family), unchanged").
  - Sets: S_old and S_new take the frozen result (frozen eligibility is a property of the probe x readout x α, not of the set);
    S_newmains is withheld ("real-arm coverage not pinned in the frozen build", 09).
Writes results/eligibility_own_frac100_<stamp>.json (exclusive create). POST HOC; PRELIMINARY.

  python3 -I experiments/2026-10-15_il_z_e2e/eligibility_own_frac100.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
TABLE = ("6514cf401", "experiments/2026-10-14_xerr_prelim_09/results/xerr_v7_phase1_2026-10-14_180820.json")
RULE = ("8b9ca793f", "experiments/2026-10-14_priority1_spec_09/RULE.md")
PROBE = "own_frac100_L48"
git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731

tb = git("show", f"{TABLE[0]}:{TABLE[1]}")
rows = [r for r in json.loads(tb)["result"]["phase1_eligibility_record"]["eligibility_table"] if r["variant"] == PROBE]
assert sorted((r["readout"], r["alpha"]) for r in rows) == [("C", "0.01"), ("C", "0.05"), ("R", "0.01"), ("R", "0.05")], rows
for r in rows:  # the rule itself, checked against the table's own status
    assert r["eligible"] == (0.10 <= r["real_tpr"] <= 0.90) and r["status"] == ("eligible" if r["eligible"] else "ineligible"), r
rule_b = git("show", f"{RULE[0]}:{RULE[1]}")
assert b"A probe-readout-rate row enters the RMS pool only if its REAL TPR lies in 10" in rule_b
LABEL = "real run 1 shares the probe's training launch (S_old cells only)"  # 09 @ 95dd0a8bc
out = {"label": "own_frac100_L48 eligibility for the stage B RMS: the frozen Append 13 §4 result, applied per set; POST HOC; PRELIMINARY",
       "rule": "Append 13 §4 (RULE @ 8b9ca793f): real TPR in 10–90% on v6's seeded 93-task eligibility half; frozen for all origins (Amendment 7b); "
               "expansion-family table = X_err v7 phase 1 @ 6514cf401 (21c §3)",
       "table": {"commit": TABLE[0], "path": TABLE[1], "sha256": hashlib.sha256(tb).hexdigest()},
       "frozen_rows": [{k: r[k] for k in ("readout", "alpha", "pool", "eligible", "status", "n", "k")} for r in sorted(rows, key=lambda r: (r["readout"], r["alpha"]))],
       "per_set": {s: {f"{r['readout']}|{r['alpha']}": ("eligible" if r["eligible"] else "ineligible (real TPR outside 10–90% on v6's eligibility half)")
                       for r in rows} for s in ("S_old", "S_new")}
       | {"S_newmains": "withheld: real-arm coverage not pinned in the frozen build (09)"},
       "rms_label": LABEL,
       "consequence": "per-action RMS rows use the run-level 1% row of each readout (as v6's members do): R ineligible -> own_frac100 R rows withheld with "
                      "the same label form as refusal / high-stakes; C eligible -> own_frac100 C is a member at S_old and S_new",
       "note": "real_tpr values are not copied here (eligibility status and counts only)",
       "meta": {"git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "stamp_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")}}
b = json.dumps(out, indent=1, ensure_ascii=False).encode()
fn = HERE / "results" / f"eligibility_own_frac100_{out['meta']['stamp_pt']}.json"
fn.open("xb").write(b)
print(fn.name, hashlib.sha256(b).hexdigest(), json.dumps(out["per_set"], ensure_ascii=False))
