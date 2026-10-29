"""Q4 (B), RULE.md Amendment 5: the list of .eval files behind the runs d9's cross-read staged (sets `main` and `run2`), from
launch metadata only. This is stage_crossread.counted_attempts (aswin/q3-qwen3-crossread-score @ 815c28b1) without its hashing:
each file's sha256 comes from its launch's S3 manifest and is checked on the Modal Volume after the upload. Reads only
launch.json and the manifests in the local S3 cache, never a transcript. Writes the upload manifest (JSON).

  python3 experiments/2026-10-13_q4_why_weaker_POSTHOC/make_q4b_upload_list.py <out.json>
"""

import json
import sys
from pathlib import Path

CACHE = Path.home() / ".cache/probe-shift/s3/individual-aswin-surya-probe-shift-kill-2027-04-06/phase2/runs"
MAIN = ["2026-10-09_000523_rerun_strict", "2026-10-09_173815_scaleup_canary", "2026-10-09_182505_scaleup", "2026-10-10_001314_scaleup_topup"]
RUN2 = "2026-10-10_001314_second_attack"
SETS = {"main": ["attack", "sim_attack@september", "sim_honest@september", "honest_a", "honest_b"], "run2": ["attack"]}
WORLD = {("main", "attack"): "run1", ("run2", "attack"): "run2", ("main", "sim_attack@september"): "sim",
         ("main", "honest_a"): "honest_a", ("main", "honest_b"): "honest_b", ("main", "sim_honest@september"): "sim_honest"}


def counted(set_name: str) -> dict:
    launches = [RUN2] if set_name == "run2" else MAIN
    out, no_eval = {}, []
    for run in sorted(set(launches) | set(MAIN)):
        man = json.loads((CACHE / f"{run}.manifest_v1.json").read_text())
        want = {f["key"].split(f"phase2/runs/{run}/", 1)[1]: (f["sha256"], f["bytes"]) for f in man["files"] if f["key"].startswith(f"phase2/runs/{run}/")}
        for r in json.loads((CACHE / run / "launch.json").read_text())["runs"]:
            for ai, a in enumerate(r.get("attempts") or []):
                if a.get("redrawn_because") or a.get("kind") == "rerun":
                    continue
                rel_dir = (a.get("log_dir") or "").split(f"/{run}/", 1)[-1]
                arm_dir = rel_dir.split("/", 1)[0]
                if not ((run in launches and arm_dir in SETS[set_name]) or (run in MAIN and arm_dir in ("honest_a", "honest_b"))):
                    continue
                key = (arm_dir, r["task"])
                if key in out:
                    raise ValueError(f"several counted attempts for {key}")
                evs = sorted(e for e in want if e.startswith(rel_dir + "/") and e.endswith(".eval"))
                if not evs:
                    no_eval.append(f"{run}/{rel_dir}")
                    continue
                if len(evs) != 1:
                    raise ValueError(f"{run}/{rel_dir}: {len(evs)} .eval files")
                if not (CACHE / run / evs[0]).exists():
                    raise ValueError(f"{run}/{evs[0]}: not in the local cache")
                out[key] = {"launch": run, "rel": evs[0], "sha256": want[evs[0]][0], "bytes": want[evs[0]][1], "arm_dir": arm_dir,
                            "task": r["task"], "attempt_index": ai, "attempt_kind": a.get("kind"), "policy_called": a.get("policy_called")}
    return out, no_eval


def main(dest: str):
    files, worlds, no_evals = {}, {}, {}
    for set_name in SETS:
        got, no_eval = counted(set_name)
        no_evals[set_name] = no_eval
        for (arm_dir, task), rec in got.items():
            if set_name == "run2" and arm_dir != "attack":
                continue  # honest arms come from `main` (the donors); run2's own pass lists them again
            w = WORLD[(set_name, arm_dir)]
            path = f"{rec['launch']}/{rec['rel']}"
            files.setdefault(path, rec | {"path": path})
            worlds.setdefault(w, []).append({"task": task, "path": path})
    m = {"note": ("Q4 (B): the .eval files behind d9's cross-read sets main and run2, as stage_crossread.counted_attempts selects them "
                  "(no redraws, no reruns); sha256 and bytes from each launch's S3 manifest, to be checked on the Volume after the upload"),
         "source": "s3://individual-aswin-surya-probe-shift-kill-2027-04-06/phase2/runs (local cache copy)",
         "counts": {w: len(v) for w, v in worlds.items()} | {"files": len(files), "bytes": sum(f["bytes"] for f in files.values())},
         "counted_attempts_without_eval": no_evals, "worlds": worlds, "files": sorted(files.values(), key=lambda f: f["path"])}
    Path(dest).write_text(json.dumps(m, indent=1) + "\n")
    print(json.dumps(m["counts"]), {k: len(v) for k, v in no_evals.items()})


if __name__ == "__main__":
    main(sys.argv[1])
