"""ON BOX 4, read-only (d9 for Aswin's decision, 10-15 ~09:30 PT): metadata of 4r's main and follow-up launches, keys
only: each launch.json's sha256, started/finished, host, configs (path + sha256 of the file in the harness), policy, real
arms, runs by arm and state, server URLs, the follow-up's references_from and server file, the harness manifests'
sha256, and the follow-up's settings-gate result. No attack text. Prints one JSON object."""

import collections
import hashlib
import json
from pathlib import Path

H = Path("/home/aswin/phase2/nm/attack_check_nm")
R = Path("/home/aswin/phase2/nm/runs")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None


def meta(run: Path) -> dict:
    raw = (run / "launch.json").read_bytes()
    d = json.loads(raw)
    arms = collections.defaultdict(collections.Counter)
    for r in d["runs"]:
        arms[r["key"].split("/")[0] + ("_r2" if r["key"].endswith("_r2") else "")][r["state"]] += 1
    cfg = d.get("pilot_config")
    tasks = d.get("pilot_tasks")
    return {"launch_id": run.name, "launch_json": str(run / "launch.json"), "launch_json_sha256": hashlib.sha256(raw).hexdigest(),
            "started": d.get("started"), "finished": d.get("finished"), "host": d.get("host"),
            "pilot_config": cfg, "pilot_config_sha256_now": sha(H / cfg) if isinstance(cfg, str) else None,
            "pilot_tasks": tasks if isinstance(tasks, str) else None,
            "pilot_tasks_sha256_now": sha(H / tasks) if isinstance(tasks, str) else None,
            "input_sha256": d.get("input_sha256"), "probe_shift_commit": d.get("probe_shift_commit"),
            "policy": d.get("policy"), "real_arms": d.get("real_arms"), "references_from": d.get("references_from"),
            "servers": d.get("servers"), "n_runs": len(d["runs"]),
            "runs_by_arm_and_state": {a: dict(c) for a, c in sorted(arms.items())}}


main, fu = meta(R / "2026-10-14_newmains_s4r"), meta(R / "2026-10-14_newmains_s4r_fu")
fb = Path("/home/aswin/phase2/nm/checks/fu_bases_s4r.txt").read_text().strip().split(",")
gs = H / "results" / "gate_settings_new_2026-10-14_newmains_s4r_fu.json"
print(json.dumps({
    "main": main, "follow_up": fu,
    "follow_up_server_file": {"path": "/home/aswin/phase2/nm/checks/fu_bases_s4r.txt", "urls": fb,
                              "subset_of_main_servers": all(u in (main["servers"] or []) for u in fb)},
    "same_server_urls": sorted(main["servers"] or []) == sorted(fu["servers"] or []),
    "harness": {"path": str(H), "harness_manifest_tsv_sha256": sha(H / "config/harness_manifest.tsv"),
                "harness_manifest_fu_tsv_sha256": sha(H / "config/harness_manifest_fu.tsv")},
    "follow_up_settings_gate": {"path": str(gs), "sha256": sha(gs),
                                "content": json.loads(gs.read_text()) if gs.exists() and gs.stat().st_size < 20000 else None},
    "answer": "TWO launches: separate launch.json files and launch ids, run one after the other by end_main on the same box and harness"}, indent=1))
