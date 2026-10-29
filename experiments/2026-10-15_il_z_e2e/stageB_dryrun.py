"""Stage B dry run (no Modal compute, no spend): build every stage-B config from an inputs file (today's stub, or INPUTS_FINAL.json when the 507-prep
fork commits it), check that every pinned Volume path exists (metadata listing only, `modal volume ls`), and write the launch plan with commands and the
SPEND estimate. The stub's roots are the fifth pass's 24 +311 roots and the 19 nm360 roots, with no _rlr replacements.

  python3 experiments/2026-10-15_il_z_e2e/stageB_dryrun.py [--inputs <INPUTS json>]
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
MODAL = "/Users/aswinsurya/.local/bin/modal"
VOL = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
args = sys.argv[1:]
inp_path = next((a.split("=", 1)[1] for a in args if a.startswith("--inputs=")), None)
STUB = {"label": "stage B STUB inputs (today's): +311 = the fifth pass's 24 roots (no _rlr yet); nm360 = the 19 FINAL roots; five-arm cells v2 @ 668a3eb91",
        "simh311_roots": [r["root"] for r in json.loads((HERE / "config/inputs_znew_pass5.json").read_text())["simh_roots"]],
        "pass6_pointers": {"sol_maps": {}, "p2_recs": {}}}
inp = json.loads(Path(inp_path).read_text()) if inp_path else STUB


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    if r.returncode:
        raise SystemExit(f"{' '.join(cmd)}: {r.stderr[-400:]}")
    return r.stdout


plan = {"label": "stage B dry run; no Modal compute; PRELIMINARY", "inputs": inp.get("label"), "configs": {}, "volume_checks": {}, "launch": [], "spend_estimate": {}}
# 1. build the configs into config/dryrun_* (never over the committed configs)
(HERE / "config" / "pass6_pointers.json").write_text(json.dumps(inp["pass6_pointers"], indent=1) + "\n")
run(["python3", "make_config_znew_pass6.py", *inp["simh311_roots"], "--out=dryrun_inputs_znew_pass6.json"])
c6 = json.loads((HERE / "config/dryrun_inputs_znew_pass6.json").read_text())
c6["fold_minus1_rule"] = "s1"
(HERE / "config/dryrun_inputs_znew_pass6.json").write_text(json.dumps(c6, indent=1) + "\n")
plan["configs"]["Z sixth pass"] = {"path": "config/dryrun_inputs_znew_pass6.json", "roots": len(c6["simh_roots"]),
                                   "replacement_roots": [r["root"] for r in c6["simh_roots"] if r.get("replacement_root")],
                                   "co_primary": len(c6["co_primary_cells"]), "M": len(c6["M_cells"]), "fold_minus1_rule": c6.get("fold_minus1_rule")}
for nm, path in (("RMS S_old", "config/inputs_rms_S_old.json"), ("RMS S_new (§1)", "config/inputs_rms_S_new_s1.json"),
                 ("RMS S_newmains (+ deviation)", "config/inputs_rms_S_newmains.json")):
    plan["configs"][nm] = {"path": path, "sha256": sha((HERE / path).read_bytes())}
# 2. Volume existence checks (metadata only): every pinned manifest / root directory
dirs = set()
for r in c6["simh_roots"]:
    dirs.add(r["sol"]["path"].strip("/").rstrip("/"))
    dirs.add(r["p2"]["path"].strip("/"))
for v in list(c6["arms"].values()) + list(c6["il_manifests"].values()):
    for sub in ("sol", "p2"):
        dirs.add(v[sub]["path"].replace("/work/", "").rsplit("/", 1)[0])
for path in ("config/inputs_rms_S_newmains.json", "config/inputs_rms_S_new_s1.json"):
    cc = json.loads((HERE / path).read_text())
    for v in cc["folders"].values():
        for sub in ("sol", "p2"):
            dirs.add(v[sub]["path"].replace("/work/", "").rsplit("/", 1)[0])
    for r in cc["H_sim"]["roots"]:
        dirs.add(r["sol"]["path"].strip("/"))
        dirs.add(r["p2"]["path"].strip("/"))
missing = []
for d in sorted(dirs):
    r = subprocess.run([MODAL, "volume", "ls", VOL, d], capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        missing.append(d)
plan["volume_checks"] = {"directories_checked": len(dirs), "missing": missing}
# 3. launch plan (commands only) and the SPEND estimate
plan["launch"] = [
    "PYTHONPATH=src modal run --detach experiments/2026-10-15_il_z_e2e/run_znew.py --config config/<final pass6 config>",
    "PYTHONPATH=src modal run --detach experiments/2026-10-15_il_z_e2e/run_znew.py --config config/<final pass6 config + draw_seed [20261015, 11], export_arrays>",
    "PYTHONPATH=src modal run experiments/2026-10-15_il_z_e2e/run_zpool.py --new-name <export record> --cohort <final cohort key>",
    "PYTHONPATH=src modal run --detach experiments/2026-10-15_il_z_e2e/run_z.py --mode rms --rms config/inputs_rms_S_old.json",
    "PYTHONPATH=src modal run experiments/2026-10-15_il_z_e2e/run_rms_hits.py --config config/<final RMS S_new §1 config>",
    "PYTHONPATH=src modal run experiments/2026-10-15_il_z_e2e/run_rms_hits.py --config config/<final RMS S_newmains config>",
    "PYTHONPATH=src modal run experiments/2026-10-15_il_z_e2e/run_rms_hits.py --config config/<final RMS S_newmains config> --deviation-mode",
    "renders: run_znew.py --mode render; run_zpool.py --mode render"]
plan["spend_estimate"] = {"Z sixth pass + export": "2 x (12 x 8-core CPU, ~4 min) ~ $0.6", "re-pool": "~$0.05", "RMS S_old": "~$0.5 (run_z loaders)",
                          "RMS S_new / S_newmains / deviation": "~$0.2", "renders": "~$0.01", "total": "~$1.4 raw, ~$1.8 with the 30% buffer"}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
b = json.dumps(plan, indent=1).encode()
(HERE / "results" / f"stageB_dryrun_{stamp}.json").open("xb").write(b)
print(f"stageB_dryrun_{stamp}.json", sha(b), "| dirs", len(dirs), "missing", len(missing))
