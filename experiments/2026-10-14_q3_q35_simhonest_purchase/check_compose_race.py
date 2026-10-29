"""Compose-race check (d9, 10-15 ~00:00; READ-ONLY): did the new-mains S5 launch (started 21:26 PT on sims-astra, its own
harness ~/phase2/nm/attack_check_nm) change the per-environment compose file under the shared settings dir while one of
this purchase's runs was active? run_pilot.py writes settings/linuxarena/<env>/codebase/compose.yml at every launch in an
environment (simulated runs too) and staggers launches only within one launcher.

For each attempt of the continuation (2026-10-14_160932_simh_cont), the t43 rerun and S5, from launch.json only: its
environment, start/end, and the inputs the generated compose file depends on (CONTROL_TOWER_SETTINGS_DIR, PHASE2_DIGEST_SET,
SKIP_DOCKER_BUILD, INTERNET_SIMULATOR_IMAGE, ENV_IMAGE_TAG_<ENV>, and the control-tower tree on PYTHONPATH). For every one of
ours active after S5's first attempt started, every S5 attempt in the same environment whose window overlaps it, and whether
the compose inputs are equal. Also the current sha256 of each such environment's compose files on disk (read only).
Keys, times, paths and hashes only. ON THE DEVBOX (system python3):  python3 check_compose_race.py --out <json>
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

RUNS = {"cont": "/home/aswin/phase2/runs/2026-10-14_160932_simh_cont", "t43_rerun": "/home/aswin/phase2/runs/2026-10-14_211500_simh_t43_rerun",
        "s5": "/home/aswin/phase2/nm/runs/2026-10-14_newmains_s5"}
FIXED = ("CONTROL_TOWER_SETTINGS_DIR", "PHASE2_DIGEST_SET", "SKIP_DOCKER_BUILD", "INTERNET_SIMULATOR_IMAGE")


def t(s):
    return datetime.fromisoformat(s) if s else None


def attempts(name):
    d = json.loads(Path(RUNS[name], "launch.json").read_text())
    envs = {c["key"]: c["env"] for c in d["tasks"]} if isinstance(d.get("tasks"), list) else {}
    out = []
    for r in d["runs"]:
        env = (r.get("task") or {}).get("env") if isinstance(r.get("task"), dict) else envs.get(r["task"])
        for i, a in enumerate(r["attempts"]):
            ev = a.get("env_vars") or {}
            out.append({"launch": name, "run": r["key"], "real": r["real"], "attempt": i, "kind": a.get("kind"), "env": env,
                        "start": a.get("start"), "end": a.get("end"),
                        "inputs": {**{k: ev.get(k) for k in FIXED},
                                   "ENV_IMAGE_TAG": ev.get(f"ENV_IMAGE_TAG_{(env or '').upper()}"),
                                   "PYTHONPATH": ev.get("PYTHONPATH")}})
    return d, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ours, s5 = [], []
    meta = {}
    for name in RUNS:
        d, a = attempts(name)
        meta[name] = {"started": d.get("started"), "finished": d.get("finished"), "digest_sets": (d.get("pilot") or {}).get("digest_sets"),
                      "settings_dirs": sorted({x["inputs"]["CONTROL_TOWER_SETTINGS_DIR"] for x in a if x["inputs"]["CONTROL_TOWER_SETTINGS_DIR"]})}
        (s5 if name == "s5" else ours).extend(a)
    s5_first = min(t(x["start"]) for x in s5 if x["start"])
    rows = []
    for o in ours:
        if not o["start"] or (o["end"] and t(o["end"]) < s5_first):
            continue  # ended before S5's first attempt: no overlap possible
        o_end = t(o["end"]) if o["end"] else datetime.max.replace(tzinfo=s5_first.tzinfo)
        same_env = [x for x in s5 if x["env"] == o["env"] and x["start"]]
        overl = [x for x in same_env if t(x["start"]) <= o_end and (not x["end"] or t(x["end"]) >= t(o["start"]))]
        rows.append({"run": f"{o['launch']}:{o['run']}", "env": o["env"], "start": o["start"], "end": o["end"], "our_inputs": o["inputs"],
                     "s5_same_env_overlapping": [{"run": x["run"], "real": x["real"], "start": x["start"], "end": x["end"],
                                                  "inputs_equal": x["inputs"] == o["inputs"],
                                                  "differing": {k: [o["inputs"][k], x["inputs"][k]] for k in o["inputs"] if o["inputs"][k] != x["inputs"][k]}}
                                                 for x in overl],
                     "s5_same_env_any": len(same_env)})
    disk = {}
    for env in sorted({r["env"] for r in rows if r["env"]}):
        cb = Path("/home/aswin/phase2/settings/linuxarena", env, "codebase")
        disk[env] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(cb.glob("*compose*")) if p.is_file()} if cb.is_dir() else None
    clean = all(all(x["inputs_equal"] for x in r["s5_same_env_overlapping"]) for r in rows)
    res = {"launches": meta, "s5_first_attempt_start": s5_first.isoformat(), "our_attempts_active_after_s5_start": rows,
           "compose_files_on_disk_now": disk, "verdict": "CLEAN: every overlapping same-environment S5 attempt had equal compose inputs"
           if clean else "MISMATCH: see differing", "clean": clean}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"settings_dirs": {k: v["settings_dirs"] for k, v in meta.items()}, "digest_sets": {k: v["digest_sets"] for k, v in meta.items()},
                      "s5_first": res["s5_first_attempt_start"], "ours_active_after": [(r["run"], r["env"], r["start"], r["end"],
                      len(r["s5_same_env_overlapping"]), all(x["inputs_equal"] for x in r["s5_same_env_overlapping"])) for r in rows],
                      "verdict": res["verdict"]}, indent=1))


if __name__ == "__main__":
    main()
