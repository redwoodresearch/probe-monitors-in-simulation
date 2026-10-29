"""Spawns Amendment 7's jobs on DEPLOYED apps (a dropped laptop does not stop them), and collects them. Deploy first:
`modal deploy run_read_tests.py` (and `modal deploy run_extract_replay.py` for the extraction).

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml python spawn_read_tests.py refit
  ... stage run1|run2|sim                        (swap stagings for one arm group)
  ... extract <config> [--test] [--attempt N]   (run_extract_replay.py's bookkeeping, shards spawned; manifest saved)
  ... extract_finish <config> [--test] [--attempt N]   (every expected shard must have succeeded)
  ... score <swapped extraction run> <refit result file>
  ... get <call id> <output path>                (waits; writes the result as JSON)
"""

import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP, EXTRACT_APP = "probe-shift_aswin-surya_sub-q3onpol_readtests_kill-2026-11-10", "probe-shift_aswin-surya_sub-q3onpol_extract_kill-2026-11-08"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
SCRUB1 = ["rerun_strict_segends_v2_scrub1", "scaleup_canary_segends_v1_scrub1", "scaleup_l2_segends_v1_scrub1", "scaleup_topup_segends_v1_scrub1"]
GROUPS = {"run1": [(s, "attack", [s]) for s in SCRUB1], "run2": [("second_attack_segends_v1_scrub1", "attack", SCRUB1)],
          "sim": [(s, "sim_attack@september", [s], True) for s in SCRUB1]}  # Amendment 8: the arm keeps its own tool definitions


def main(cmd: str, *args):
    if cmd == "refit":
        print(json.dumps({"refit": modal.Function.from_name(APP, "refit").spawn().object_id}))
    elif cmd == "stage":
        fn = modal.Function.from_name(APP, "swap_stage")
        print(json.dumps({f"{g[0]}|{g[1]}": fn.spawn(*g).object_id for g in GROUPS[args[0]]}))
    elif cmd == "score":
        sel = json.loads(Path(args[1]).read_text())["selections"]
        print(json.dumps({"score": modal.Function.from_name(APP, "score_swapped").spawn(args[0], sel).object_id}))
    elif cmd == "get":
        r = modal.FunctionCall.from_id(args[0]).get()
        Path(args[1]).write_text(json.dumps(r))
        print("->", args[1])
    elif cmd in ("extract", "extract_finish"):  # run_extract_replay.main's bookkeeping, around spawned shards
        import yaml

        sys.path.insert(0, str(HERE.parents[1] / "src"))
        from probe_shift.models import QWEN35_27B

        test = "--test" in args
        attempt = int(args[args.index("--attempt") + 1]) if "--attempt" in args else 1
        cfg = yaml.safe_load((HERE / "config" / args[0]).read_text()) | {"attempt": attempt, "test": test, "revision": QWEN35_27B["revision"]}
        if cfg["model"] != QWEN35_27B["name"]:
            raise ValueError("unpinned model")
        if test:
            cfg["run"] += "_test"
        vol = modal.Volume.from_name(ACTS, version=2)
        git = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
               "dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=HERE).stdout.strip())}
        n = 1 if test else cfg["shards"]
        man = HERE.parents[2] / ".launch_manifests" / f"{cfg['run']}_a{attempt}.json"  # outside git (scratch, per launch)
        if cmd == "extract":
            settings = {k: v for k, v in cfg.items() if k != "attempt"}
            try:
                stored = json.loads(b"".join(vol.read_file(f"{cfg['run']}/config.json")))
            except (FileNotFoundError, modal.exception.NotFoundError):
                stored = None
            if stored is not None and stored != settings:
                raise ValueError(f"{cfg['run']} exists with other settings")
            with vol.batch_upload(force=True) as b:
                if stored is None:
                    b.put_file(io.BytesIO(json.dumps(settings, indent=1).encode()), f"{cfg['run']}/config.json")
                b.put_file(io.BytesIO(json.dumps({"attempt": attempt, "git": git}, indent=1).encode()), f"{cfg['run']}/attempt_{attempt}.json")
            fn = modal.Function.from_name(EXTRACT_APP, "extract_shard")
            man.parent.mkdir(exist_ok=True)
            calls = {}
            with man.open("x") as fh:  # created before any spawn; rewritten after each, so no call id is ever lost
                fh.write(json.dumps({"run": cfg["run"], "attempt": attempt, "config": args[0], "calls": calls}))
            for s in range(n):
                calls[s] = fn.spawn(cfg, s).object_id
                print(f"shard {s}: {calls[s]}", flush=True)
                man.write_text(json.dumps({"run": cfg["run"], "attempt": attempt, "config": args[0], "calls": calls}))
            print(json.dumps({"manifest": str(man), "calls": calls}))
            return
        m = json.loads(man.read_text())
        if m["run"] != cfg["run"] or m["attempt"] != attempt or sorted(map(int, m["calls"])) != list(range(n)):
            raise SystemExit("the launch manifest does not match this run, attempt and shard count")
        res = []
        for s_, cid in sorted(m["calls"].items(), key=lambda x: int(x[0])):
            try:
                r = modal.FunctionCall.from_id(cid).get()
                if r.get("shard") != int(s_):
                    raise ValueError(f"call {cid} returned shard {r.get('shard')}, expected {s_}")
                res.append(r)
            except Exception as e:  # a failed shard is recorded, and the extraction is not marked complete
                res.append({"error": repr(e)[:500], "shard": int(s_)})
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        out = {"git": git, "config": cfg, "app": EXTRACT_APP, "volume": ACTS, "shards": res, "launch_manifest": m}
        path = HERE / "results" / f"extract_replay_{cfg['run']}_a{attempt}_{stamp}.json"
        path.write_text(json.dumps(out, indent=1))
        print(json.dumps(res, indent=1)[:2000], "\n->", path)
        if any("error" in r for r in res):
            raise SystemExit("a shard failed; not marking the extraction complete")
        if not test:
            with vol.batch_upload(force=True) as b:
                b.put_file(io.BytesIO(json.dumps({"epochs": sum(r["epochs"] for r in res), "attempt": attempt,
                                                  "inputs_sha256": cfg["inputs_sha256"]}).encode()), f"{cfg['run']}/complete.json")
            print("complete:", sum(r["epochs"] for r in res), "epochs")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(*sys.argv[1:])
