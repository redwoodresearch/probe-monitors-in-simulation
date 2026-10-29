"""compare_simh_full_identity.py (09's S_paper493 ruling 1, via d9 ~17:30): do the +311 harness's simulator tree, run.yaml and
sim.yaml hashes EQUAL those recorded for simh_full / primary_v3 (25h §13 identity), and is the gate_settings reference the same
file? Committed records only (no box reads). Writes results/sp493_simh_identity_<stamp>.json.
  python3 compare_simh_full_identity.py <repo>
"""
import datetime as dt
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

repo = sys.argv[1]
show = lambda ref, p: subprocess.run(["git", "-C", repo, "show", f"{ref}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
PUR, PP = "origin/aswin/q3-q35-simhonest-purchase", "experiments/2026-10-14_q3_q35_simhonest_purchase/results/"
pre = json.loads(show(PUR, PP + "sim_settings_prelaunch_2026-10-14_135614.json"))  # simh_full's pre-launch harness check
cont = json.loads(show(PUR, PP + "sim_settings_cont_final_2026-10-14_160932_simh_cont.json"))  # simh_full's final settings check
ref = {"simulator_src_sha256": pre["harness"]["tree"]["src_sha256"], "sim_yaml_sha256": pre["harness"]["inputs_sha256"]["config/sim.yaml"],
       "run_yaml_sha256": pre["harness"]["inputs_sha256"]["config/run.yaml"]}
assert cont["pass"] is True and cont["reference_summary"]["given"]["tree"]["src_sha256"] == ref["simulator_src_sha256"]
gate_ref = hashlib.sha256(show(PUR, PP + "sim_settings_2026-10-14_125923.json")).hexdigest()
SU, SUC = "origin/aswin/q3-scaleup-311", "experiments/2026-10-14_q3_scaleup_311/config/"
harness = {}
for m in ("harness_manifest_fu_q123.tsv", "harness_manifest_fu_q2r.tsv", "harness_manifest_fu_q4.tsv"):
    rows = dict(reversed(x.split("  ", 1)) for x in show(SU, SUC + m).decode().splitlines())
    harness[m] = {"config/run.yaml": rows.get("./config/run.yaml"), "config/sim.yaml": rows.get("./config/sim.yaml"),
                  "gate_settings_reference": rows.get("./config/sim_settings_reference_2026-10-14_125923.json"),
                  "run_yaml_equal": rows.get("./config/run.yaml") == ref["run_yaml_sha256"],
                  "sim_yaml_equal": rows.get("./config/sim.yaml") == ref["sim_yaml_sha256"],
                  "gate_reference_equal": rows.get("./config/sim_settings_reference_2026-10-14_125923.json") == gate_ref}
here = Path(__file__).parent / "results"
mans = []
for f in sorted(here.glob("manifest_simh_*.json")):
    d = json.loads(f.read_text()); s = d.get("simulator") or {}; h = d.get("harness") or {}
    got = {"simulator_src_sha256": s.get("src_sha256") or (s.get("run_yaml_entry") or {}).get("src_sha256"),
           "sim_yaml_sha256": s.get("sim_yaml_sha256"), "run_yaml_sha256": h.get("run_yaml_sha256")}
    mans.append({"manifest": f.name, "host": d.get("host_box"), "frozen_manifest": (h.get("frozen_manifest") or "").rsplit("/", 1)[-1],
                 "equal": {k: got[k] == ref[k] for k in ref}})
fu = [m for m in mans if m["frozen_manifest"] == "harness_manifest_fu_q123.tsv"]
out = {"what": "25h §13 identity: +311 harness (fu_q123) vs simh_full / primary_v3", "written_pt": dt.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%F %T"),
       "simh_full_reference": {"source": f"{PUR}:{PP}sim_settings_prelaunch_2026-10-14_135614.json (harness block) + sim_settings_cont_final_2026-10-14_160932_simh_cont.json (pass, tree)", **ref,
                               "gate_settings_reference_sha256": gate_ref},
       "frozen_harness_manifests": harness,
       "generation_manifests_fu_q123": {"n": len(fu), "all_equal": all(all(m["equal"].values()) for m in fu)},
       "generation_manifests_all": {"n": len(mans), "all_equal": all(all(m["equal"].values()) for m in mans), "rows": mans},
       "verdict": None}
out["verdict"] = "EQUAL" if out["generation_manifests_fu_q123"]["all_equal"] and all(v["run_yaml_equal"] and v["sim_yaml_equal"] and v["gate_reference_equal"] for v in harness.values()) else "DIFFERS"
p = here / f"sp493_simh_identity_{dt.datetime.now(ZoneInfo('America/Los_Angeles')):%Y-%m-%d_%H%M%S}.json"
p.write_text(json.dumps(out, indent=1) + "\n")
print(out["verdict"], {k: v[:12] for k, v in ref.items()}, "gate ref", gate_ref[:12], "| fu_q123 manifests", out["generation_manifests_fu_q123"],
      "| frozen", {m: (v["run_yaml_equal"], v["sim_yaml_equal"], v["gate_reference_equal"]) for m, v in harness.items()}, "->", p.name)
