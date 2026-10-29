"""c_asis sign check by key path (RULE.md): reads three committed records (git show, sha256-checked) and writes every value with its path.
Definitions (run_e2e.py @ e61cf75bd lines 531/535; fpr_core.catches_and_gaps): fixed_threshold_vs_real_mean = sim@tau_real - real_mean@tau_real;
transfer_real_mean_minus_sim = real_mean@tau_sim - sim@tau_sim; common-calibration diff sim_minus_* = sim - real at the shared real line.

  uv run python experiments/2026-10-15_panel_noise_matched/check_csigns.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
SRC = {"e2e183": ("e61cf75bd:experiments/2026-10-15_e2e_sim_pipeline/results/e2e_2026-10-15_134139_reduce_2026-10-15_135347.json", "f7eb48b0780104fbf854959ae3e3175b194209b26a4a0d548cc96bcd3d9f03d6"),
       "common455": ("origin/aswin/fpr-sweep-507-prep:experiments/2026-10-15_fpr_sweep_507_prep/results/common_run455_2026-10-15_101451.json", "c7b348575e5e1a2c2a4c97e192659d3a432d7c99721b0ab996c92e36f306d810")}
PATHS = [("e2e183", ["rows", "c_asis R", "unit", "unit0.01", "as_is", "gap", "transfer_real_mean_minus_sim", "point"], "S_old 183, sim line on real (as-is): real - sim at tau_sim"),
         ("e2e183", ["rows", "c_asis R", "unit", "unit0.01", "matched", "gap", "transfer_real_mean_minus_sim", "point"], "S_old 183, sim line on real (matched): real - sim at tau_sim"),
         ("e2e183", ["rows", "c_asis R", "unit", "unit0.01", "matched", "gap", "fixed_threshold_vs_real_mean", "point"], "S_old 183, shared real line: sim - real mean at tau_real"),
         ("common455", ["probes", "c_asis R", "unit", 16, "blocks", "S_old", "diff", "sim_minus_real_mean", "point"], "455 table S_old, shared real line: sim - real mean"),
         ("common455", ["probes", "c_asis R", "unit", 16, "blocks", "S_new", "diff", "sim_minus_real_mean", "point"], "455 table S_new, shared real line: sim - real mean"),
         ("common455", ["probes", "c_asis R", "unit", 16, "blocks", "S_new", "diff", "sim_minus_run1", "point"], "455 table S_new, shared real line: sim - run1"),
         ("common455", ["probes", "c_asis R", "unit", 16, "blocks", "S_new", "diff", "sim_minus_run2", "point"], "455 table S_new, shared real line: sim - run2"),
         ("common455", ["probes", "c_asis R", "unit", 16, "blocks", "combined", "diff", "sim_minus_real_mean", "point"], "455 table combined, shared real line: sim - real mean"),
         ("common455", ["probes", "c_asis R", "unit", 16, "blocks", "combined", "diff", "sim_minus_run1", "point"], "455 table combined, shared real line: sim - run1")]

recs = {}
for k, (spec, sha) in SRC.items():
    b = subprocess.run(["git", "show", spec], capture_output=True, check=True, cwd=HERE).stdout
    assert hashlib.sha256(b).hexdigest() == sha, k
    recs[k] = json.loads(b)
assert recs["common455"]["probes"]["c_asis R"]["unit"][16]["fpr_target"] == 0.01
out = []
for k, kp, what in PATHS:
    v = recs[k]
    for p in kp:
        v = v[p]
    out.append({"record": SRC[k][0], "sha256": SRC[k][1], "key_path": "/".join(map(str, kp)), "what": what, "value_points": round(100 * v, 2),
                "sim_scores_higher": (v < 0) if "real - sim" in what else (v > 0)})
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
dest = HERE / "results" / f"csigns_{stamp}.json"
dest.open("xb").write(json.dumps({"label": "POST HOC; from raw data, by script", "per_action_fpr": 0.01, "readout": "reply (c_asis R)", "values": out}, indent=1).encode())
for o in out:
    print(f"{o['value_points']:+7.2f}  sim higher={o['sim_scores_higher']}  {o['what']}  [{o['key_path']}]")
print(dest.name)
