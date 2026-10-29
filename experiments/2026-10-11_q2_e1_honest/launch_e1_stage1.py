"""E1 stage 1's budget gate and launch (AMENDMENTS.md 4; Aswin's cap: $12.90 for pilot plus stage 1, one deployed app).
Refuses unless: the pilot (bound to this index) passed its cached-replay equivalence check; no stage 1 was reserved before;
no other watcher is running on the app; and the bill so far plus 1.3 x stage 1's reprice is at most $12.90. The bill so
far is the larger of `modal billing report` (hourly, it lags) and the pilot's wall time (launch to its last Volume write,
plus 2 min idle) at $5.64 per H200-hour, which covers start-up, idle, staging and the coordinator because the pilot's two
H200 containers ran one after the other. Stage 1 is repriced from the pilot's measured function times: each container's
load once, per-run work scaled to the index's stage-1 runs (scoring without the one-pass checks), plus 3 min start-up and
1 min idle per container; the larger of that and the bill so far scaled by the same ratio. It also requires
the bill so far plus the worst case the function timeouts allow (20 + 30 H200-min plus a 10-min start-up and 1-min idle limit per container) within the cap,
so stage 1 can't pass $12.90 even if the watcher fails. It then starts watch_e1.py (stop when the whole app's bill
reaches $12.90, or at a deadline inside the remaining dollars) and waits until it polls, writes the gate record and a
reservation to the Volume (the pipeline refuses stage 1 without the record, and runs each record once), and spawns.

  uv run --no-project --python 3.12 --with modal==1.2.6 python experiments/2026-10-11_q2_e1_honest/launch_e1_stage1.py <index name> <pilot stamp> [--dry-run]
"""
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run_e1 import GEN_TIMEOUT, IDLE, SCORE_TIMEOUT, STARTUP  # noqa: E402
APP = "probe-shift_aswin-surya_sub-q2-e1-honest_kill-2026-11-08"
CAP, BUFFER, RATE = 12.90, 1.3, 5.64 / 3600
CLI = ["uv", "run", "--no-project", "--with", "modal==1.6.1", "modal"]


def cli_json(*args):
    out = json.loads(subprocess.run(CLI + list(args) + ["--json"], capture_output=True, text=True, timeout=120, check=True).stdout)
    return out if isinstance(out, list) else out.get("data", out)


def main(index_name: str, pilot: str, dry_run: bool = False):
    vol = modal.Volume.from_name(APP)
    read = lambda p: json.loads(b"".join(vol.read_file(p)))
    if any(e.path.endswith("stage1_reservation.json") for e in vol.listdir("/")):
        raise SystemExit(f"stage 1 is already reserved: {read('stage1_reservation.json')}")
    pp = read(f"{pilot}/pipeline.json")
    if pp.get("part") != "pilot" or pp.get("index") != index_name:
        raise SystemExit(f"{pilot} is not a pilot of {index_name}")
    n1 = sum(json.loads(line)["part"] == "stage1" for line in b"".join(vol.read_file(f"inputs/{index_name}")).decode().splitlines())
    n0 = pp["generate"]["runs"]
    app_id = next(a["app_id"] for a in cli_json("app", "list") if a["description"] == APP and a["state"] == "deployed")
    if subprocess.run(["pgrep", "-f", app_id], capture_output=True).returncode == 0:
        raise SystemExit(f"another watcher is running on {app_id}; stop it first")
    entries = {e.path.rsplit("/", 1)[-1]: e for e in vol.listdir(pilot)}
    t_launch = datetime.strptime(pilot[:17], "%Y-%m-%d_%H%M%S").replace(tzinfo=ZoneInfo("America/Los_Angeles")).timestamp()
    wall = entries["pipeline.json"].mtime - t_launch + 120
    report = sum(float(r["cost"]) for r in cli_json("billing", "report", "--start", "2026-10-11", "--end", "2026-10-13", "--resolution", "h")
                 if r.get("object_id") == app_id)
    so_far = max(report, wall * RATE)
    g, s = pp["generate"], pp["score"]
    scores = [read(e.path) for e in vol.listdir(f"{pilot}/scores") if e.path.endswith(".json")]
    full_s = sum(r.get("full_seconds", 0) for r in scores)
    t_pilot = g["seconds"] + s["seconds"]
    per_seq = (s["seconds"] - s["load_s"] - full_s) / len(scores)  # scoring priced per sequence, 5 a run (the most possible)
    t1 = g["load_s"] + n1 * (g["seconds"] - g["load_s"]) / n0 + s["load_s"] + 5 * n1 * per_seq + 2 * (180 + 60)
    worst = (GEN_TIMEOUT + SCORE_TIMEOUT + 2 * (STARTUP + IDLE)) * RATE  # the configured limits bound stage 1, watcher or not
    projected = max(t1 * RATE, so_far * t1 / t_pilot)
    remaining_s = (CAP - so_far) / RATE - 180
    gate = {"pilot": pilot, "index": index_name, "stage 1 runs": n1, "equivalence": pp.get("equivalence"), "app": app_id,
            "billing report so far": report, "pilot wall s (with idle)": wall, "bill so far": so_far, "stage 1 seconds, repriced": t1,
            "stage 1 cost, repriced": projected, "bill so far + 1.3 x stage 1": so_far + BUFFER * projected,
            "stage 1 worst case (function timeouts)": worst, "bill so far + worst case": so_far + worst, "cap": CAP}
    gate["pass"] = (bool((pp.get("equivalence") or {}).get("pass")) and gate["bill so far + 1.3 x stage 1"] <= CAP
                    and gate["bill so far + worst case"] <= CAP and remaining_s > 1.3 * t1)
    stamp = f"{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}_stage1"
    (rec := HERE / "results" / f"gate_{stamp}{'_dryrun' if dry_run else ''}.json").write_text(json.dumps(gate, indent=1) + "\n")
    print(json.dumps(gate))
    if dry_run:  # the gate and reprice only: no watcher, reservation or spawn
        return
    if not gate["pass"]:
        raise SystemExit(f"gate refused; record {rec}")
    deadline = time.time() + min(1.5 * t1 + 600, remaining_s - 600)
    (log_p := HERE.parents[1] / "data" / "e1_honest" / f"watch_{stamp}.log").parent.mkdir(parents=True, exist_ok=True)
    w = subprocess.Popen([sys.executable, str(HERE / "watch_e1.py"), app_id, str(CAP), str(deadline)], stdout=log_p.open("w"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(36):  # the watcher must have read the bill and the app successfully before anything is reserved or spawned
        if "READY" in log_p.read_text() or w.poll() is not None:
            break
        time.sleep(5)
    if w.poll() is not None or "READY" not in log_p.read_text():
        w.kill()
        raise SystemExit(f"the watcher isn't running with a good read; nothing reserved or spawned (log {log_p})")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "gate.json").write_text(json.dumps(gate))
        (Path(tmp) / "res.json").write_text(json.dumps({"stamp": stamp, "gate": rec.name}))
        with vol.batch_upload() as b:  # fails if either file already exists
            b.put_file(str(Path(tmp) / "gate.json"), f"{stamp}/gate.json")
            b.put_file(str(Path(tmp) / "res.json"), "stage1_reservation.json")
    if w.poll() is not None:
        raise SystemExit(f"the watcher exited before the spawn; stage 1 reserved but not spawned (log {log_p})")
    call = modal.Function.from_name(APP, "pipeline").spawn(index_name, stamp, "stage1")
    launch = {"stamp": stamp, "index": index_name, "part": "stage1", "call_id": call.object_id, "watcher pid": w.pid,
              "watcher stop ($, whole app)": CAP, "deadline": datetime.fromtimestamp(deadline, ZoneInfo("America/Los_Angeles")).isoformat(),
              "gate record": rec.name}
    (HERE / "results" / f"launch_{stamp}_{call.object_id}.json").write_text(json.dumps(launch, indent=1) + "\n")
    print(json.dumps(launch))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--dry-run" in sys.argv)
