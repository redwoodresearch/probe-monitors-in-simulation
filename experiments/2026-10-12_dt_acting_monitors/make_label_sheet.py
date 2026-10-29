"""ID-only labelling sheet for Aswin's destructive / transfer labels on Qwen3.5-27B's own acting runs (2026-10-12; $0).

Aswin, relayed by the orchestrator at about 12:05 PT: "i will label evrything, give me id only sheet and count". One row
per rule-1 command call (the 196-task monitor's rows: kind "command", a row, label impactful or read_only; counted
attempts only, as run_score_onpolicy.py) in six arms, read from the extraction meta on the acts Volume. Columns: arm,
task, run_key, irid, call, call_index, tool (the meta's tool name), existing_label, and empty destructive / transfer
columns. No arguments and no text: the meta's `param` field is not written.

  uv run --no-project --with modal==1.6.1 python make_label_sheet.py --out-csv <path> --out-counts <path>
"""

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
LAUNCHES = ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1", "2026-10-10_scaleup_l2_segends_v1",
            "2026-10-10_scaleup_topup_segends_v1"]
ARMS = {  # sheet arm -> (extraction runs, meta arm)
    "real_honest_A": (LAUNCHES, "honest_a"),
    "real_honest_B": (LAUNCHES, "honest_b"),
    "sim_honest": (LAUNCHES[:1], "sim_honest@september"),
    "real_attack_1": (LAUNCHES, "attack"),
    "sim_attack": (LAUNCHES, "sim_attack@september"),
    "real_attack_2": (["2026-10-10_second_attack_segends_v1"], "attack"),
}
COLUMNS = ["arm", "task", "run_key", "irid", "call", "call_index", "tool", "existing_label", "destructive", "transfer"]


def main(out_csv: str, out_counts: str):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    v = modal.Volume.from_name(ACTS, version=2)
    jobs = []
    for arm, (runs, meta_arm) in ARMS.items():
        for run in runs:
            done = json.loads(b"".join(v.read_file(f"{run}/complete.json")))  # written only when every shard finished
            n_all = sum(len(list(v.listdir(d.path))) for d in v.listdir(f"{run}/meta"))
            if n_all != done["epochs"]:
                raise SystemExit(f"{run}: {n_all} meta files, {done['epochs']} epochs staged")
            files = [e.path for e in v.listdir(f"{run}/meta/{meta_arm}")]
            if not files:
                raise SystemExit(f"{run}/{meta_arm}: no meta files")
            jobs += [(arm, run, done["epochs"], p) for p in files]

    def load(job):
        arm, run, _, p = job
        return arm, run, json.loads(b"".join(v.read_file(p)))

    with ThreadPoolExecutor(32) as ex:
        metas = list(ex.map(load, jobs))
    rows, skipped = [], Counter()
    for arm, run, m in metas:
        if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
            skipped[f"{arm}: redrawn or rerun attempt"] += 1
            continue
        for c in m["calls"]:
            if c.get("kind", "command") == "command" and c["row"] is not None and c["label"] in ("impactful", "read_only"):
                rows.append({"arm": arm, "task": m["task"], "run_key": m["key"], "irid": c["irid"], "call": c["call"],
                             "call_index": c["call_index"], "tool": c["name"], "existing_label": c["label"], "destructive": "", "transfer": ""})
    keys = [(r["arm"], r["run_key"], r["irid"], r["call"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate (arm, run_key, irid, call) rows")
    rows.sort(key=lambda r: (list(ARMS).index(r["arm"]), r["task"], r["run_key"], r["irid"], r["call"]))
    with open(out_csv, "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    per_arm = {}
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        per_task = Counter(r["task"] for r in rs if r["existing_label"] == "impactful")
        per_arm[arm] = {"rows": len(rs), "impactful": sum(r["existing_label"] == "impactful" for r in rs),
                        "read_only": sum(r["existing_label"] == "read_only" for r in rs), "tasks": len({r["task"] for r in rs}),
                        "runs": len({r["run_key"] for r in rs}), "impactful_per_task": dict(sorted(per_task.items()))}
    out = {"git": git, "stamp": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"), "columns": COLUMNS,
           "sheet": Path(out_csv).name, "sheet_sha256": hashlib.sha256(Path(out_csv).read_bytes()).hexdigest(),
           "rows": len(rows), "impactful": sum(a["impactful"] for a in per_arm.values()), "read_only": sum(a["read_only"] for a in per_arm.values()),
           "per_arm": per_arm, "skipped_meta_files": dict(skipped), "meta_files_read": len(metas)}
    with open(out_counts, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    for arm, a in per_arm.items():
        ipt = list(a["impactful_per_task"].values())
        print(f"{arm}: rows {a['rows']}, impactful {a['impactful']}, read-only {a['read_only']}, tasks {a['tasks']}, runs {a['runs']}, "
              f"impactful per task median {sorted(ipt)[len(ipt) // 2] if ipt else 0}, max {max(ipt) if ipt else 0}")
    print(f"total rows {out['rows']}, impactful {out['impactful']}, read-only {out['read_only']}; sheet sha256 {out['sheet_sha256']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-counts", required=True)
    main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
