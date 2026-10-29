"""1e-4 real honest runs (RULE.md, Amendments 6 and 9): the staging's upload manifest for this job's two launches, in 2e's format. A copy of 2e's
make_q4f_c_upload_list.py (aswin/q4-why-weaker @ 50a7664ea) whose only changes are its inputs and where it runs: the launch's
launch.json and the devbox-made per-replicate manifests (gate1.py final: path, bytes, sha256 of every attempt's .eval) read on
the Volume by a Modal CPU function, in place of a laptop read of the launch's S3 manifest. The counted-attempt rule is 2e's
(B's): per run, the attempts without `redrawn_because` and not of kind "rerun" are counted (one per task, else it stops); each
counted attempt's single .eval comes from its log_dir. Reads only launch.json and the manifests, never a transcript; returns
counts, keys, paths and hashes, written locally to config/manifest_rh1e4_eval_upload_v1.json.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run experiments/2026-10-14_q3_q35_realhonest_1e4/make_rh_upload_list.py
"""

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q3realhonest_mklist_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-q3realhonest_data_kill-2026-11-14"
RUNS = ["2026-10-14_135334_honest_rh980", "2026-10-14_160335_honest_rh_relaunch"]  # launch 1, then the relaunch (Amendment 9)
SELECTION = HERE / "results" / "relaunch_selection_2026-10-14.json"  # the 74 infra-affected keys, committed at bc192959d
CAUSE = {"a": "infra: server 0 stopped by its laptop billing watcher at 15:19:51 PT",
         "b": "infra: cut by gate 1's Ctrl+C at 15:26:10 PT"}
ARMS = ("honest_b",)
app = modal.App(APP)


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=2, memory=8192, timeout=900, retries=0,
              volumes={"/ev": modal.Volume.from_name(VOL, version=2).read_only()})
def make(sel: dict) -> dict:
    """Amendment 9's records: launch 1's attempts of the 74 affected keys are uncounted, with `redrawn_because` = their
    infrastructure cause (B's tag for a superseded attempt); the relaunch's attempts of those keys are counted, with
    `redraw_of` naming the original. Every other attempt follows B's rule in its own launch."""
    affected = {t: CAUSE["a"] for t in sel["a_server0_down"]} | {t: CAUSE["b"] for t in sel["b_cut_by_ctrl_c"] if t not in sel["a_server0_down"]}
    assert len(affected) == len(sel["affected"]) == 74, len(affected)
    files, no_eval, uncounted, runs_all, original, counted_keys = {}, [], [], [], {}, set()
    for RUN in RUNS:
      meta = Path("/ev/evals_v1") / RUN
      want = {}
      for i in range(1, 6):
          for line in (meta / f"upload_manifest_r{i}.tsv").read_text().splitlines():
              path, size, sha = line.split("\t")
              want[path] = (sha, int(size))
      runs = json.loads((meta / "launch.json").read_text())["runs"]
      runs_all += runs
      for r in runs:
        for ai, a in enumerate(r.get("attempts") or []):
            rel_dir = (a.get("log_dir") or "").split(f"/{RUN}/", 1)[-1]
            if RUN == RUNS[0] and r["task"] in affected:  # every attempt of an affected key, superseded by the relaunch (Amendment 9)
                uncounted.append({"launch": RUN, "task": r["task"], "attempt_index": ai, "attempt_kind": a.get("kind"), "rel_dir": rel_dir,
                                  "redrawn": True, "redrawn_because": affected[r["task"]], "also_prestart_redrawn": bool(a.get("redrawn_because"))})
                original[r["task"]] = f"{RUN}/{rel_dir}"  # the last attempt
                continue
            if a.get("redrawn_because") or a.get("kind") == "rerun":
                uncounted.append({"launch": RUN, "task": r["task"], "attempt_index": ai, "attempt_kind": a.get("kind"), "rel_dir": rel_dir, "redrawn": bool(a.get("redrawn_because"))})
                continue
            arm_dir = rel_dir.split("/", 1)[0]
            if arm_dir not in ARMS:
                continue
            key = (arm_dir, r["task"])
            if key in counted_keys:  # Codex round 13: tracked apart from `files`, so an attempt without an .eval counts too
                raise ValueError(f"several counted attempts for {key}")
            counted_keys.add(key)
            evs = sorted(e for e in want if e.startswith(rel_dir + "/") and e.endswith(".eval"))
            if not evs:
                no_eval.append(f"{RUN}/{rel_dir}")
                continue
            if len(evs) != 1:
                raise ValueError(f"{RUN}/{rel_dir}: {len(evs)} .eval files")
            path = f"{RUN}/{evs[0]}"
            files[key] = {"launch": RUN, "rel": evs[0], "sha256": want[evs[0]][0], "bytes": want[evs[0]][1], "arm_dir": arm_dir, "task": r["task"],
                          "attempt_index": ai, "attempt_kind": a.get("kind"), "policy_called": a.get("policy_called"), "path": path}
            files[key] |= {"world": f"honest_{r['task'].rsplit('_', 1)[1]}", "stratum": "S_old", "launch_id": RUN,  # for the scorers (d9)
                           "rerun_key": r["task"] in affected}
            if RUN == RUNS[1] and r["task"] in affected:
                files[key] |= {"redraw_of": original.get(r["task"]), "redraw_cause": affected[r["task"]]}
    runs = runs_all
    tasks = sorted({r["task"] for r in runs})
    return {"note": ("1e-4 real honest runs: the .eval files behind the counted runs of launch 2026-10-14_135334_honest_rh980 and its "
                     "relaunch 2026-10-14_160335_honest_rh_relaunch (no redraws, no reruns; launch 1's 74 infra-affected attempts uncounted with "
                     "their cause, Amendment 9), by make_q4b_upload_list.py's rules; sha256 and bytes from the devbox-made manifests"),
            "source": [f"{VOL}:/evals_v1/{x}" for x in RUNS], "relaunched_keys": len(affected),
            "counts": {"runs_in_launch": len(runs), "tasks": len(tasks), "attempts": sum(len(r.get("attempts") or []) for r in runs),
                       "counted_with_eval": len(files), "counted_without_eval": len(no_eval), "uncounted_attempts": len(uncounted),
                       "bytes": sum(f["bytes"] for f in files.values())},
            "uncounted_attempts": uncounted, "counted_attempts_without_eval": {"c": no_eval},
            "worlds": {w: [{"task": t, "path": f["path"], "launch_id": f["launch_id"], "rerun_key": f["rerun_key"]}
                           for (_, t), f in sorted(files.items()) if f["world"] == w] for w in sorted({f["world"] for f in files.values()})},
            "files": sorted(files.values(), key=lambda f: f["path"])}


@app.local_entrypoint()
def main():
    m = make.remote(json.loads(SELECTION.read_text()))
    dest = HERE / "config" / "manifest_rh1e4_eval_upload_v1.json"
    assert not dest.exists(), f"{dest} exists"
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(json.dumps(m, indent=1) + "\n")
    tmp.rename(dest)  # Amendment 13: written whole or not at all
    print(json.dumps(m["counts"]), json.dumps([{k: u[k] for k in ("task", "attempt_kind", "redrawn")} for u in m["uncounted_attempts"]]))
    print("->", dest)
