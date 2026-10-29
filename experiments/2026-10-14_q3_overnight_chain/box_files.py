"""Overnight chain (RULE.md here): run ON a scale-up devbox (stdlib only, python3 -), over a launch that has ENDED. It prints one
JSON to stdout: the exact files to relay (path relative to ~/phase2/runs, bytes, sha256) and the staging manifests in
stage_q4b's upload-manifest format, by B's counted-attempt rule (make_rh_upload_list.py's: per run, the attempts without
`redrawn_because` and not of kind "rerun" are counted; one counted attempt per (arm folder, task), else it stops; its single
.eval comes from its log_dir). Reads launch.json and hashes files; never opens a transcript. Writes nothing on the box.

  devbox.sh python3 - main|ms <run> - <earlier snapshots' cells|-> <snapshot id>   < box_files.py   # a main launch: sets main and run2
  devbox.sh python3 - fu|fus <run> <main run> <earlier snapshots' cells|-> <snapshot id>   < box_files.py   # a simulated follow-up
  (a follow-up's simulated arm folder and world: CHAIN_SIM_ARM / CHAIN_SIM_WORLD, default sim_attack@september / sim, e.g. the
  sim-honest launches' sim_honest@september; its <main run> may be @index:<merged reference index>, as an interleaved launch's)

A snapshot of a running launch (ms: main, fus: follow-up) takes only finished work, and never a cell an earlier snapshot took; the
final batch (main, fu) is the remainder. A main snapshot takes WHOLE cells only (every honest_a, honest_b, attack and attack _r2
run of the cell done), so each attack has its honest donor and no honest run is staged twice.

Set run2: the attack arm's `<task>_r2` entries (the interleaved second attack, same launch) under their base task, with
`launch_task` recording the launcher's name, so stage_q4b's donor lookup finds the cell's honest runs.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CHAIN_ROOT") or (Path.home() / "phase2" / "runs"))  # the launch's run root (the new mains: ~/phase2/nm/runs)
STRATUM = os.environ.get("CHAIN_STRATUM") or "S_new"  # 09's Append 21 / 21c: S_new (the +311), S_newmains (the 360)
RECORDS = ["launch.json", "gate1_final.json", "gate1_counts.json", "gate1_watch.log", "pairs_check.json", "attempt_check_final.json", "unpause_onbox.log",
           "gate_pause_watch.log"]
SIBLINGS = [".launcher.log", ".docker_events.jsonl", ".resources.log", ".meta.json", ".il_meta.json",  # beside the run folder
            ".rerun_of.json", ".original_pairs.json", ".supersession_rows.json"]  # a paired rerun's sidecars (design fork; keys and hashes)
# cells.json holds the approach texts: approach and attack files move only via relay_approaches.sh (d9's standing rule), so it is never listed
# here (until 10-15 16:36 it was relayed as a record; record @ 8baac14ad). No file whose name marks it as one is ever listed (FORBIDDEN).
HONEST, SIM = ("honest_a", "honest_b"), "sim_attack@september"
FU_SIM = os.environ.get("CHAIN_SIM_ARM") or SIM  # a follow-up's simulated arm folder (the chain sets it from the family's config)
FU_WORLD = os.environ.get("CHAIN_SIM_WORLD") or "sim"  # its world in the manifests
DONOR_COPIES = set()  # index-donor copies this read relays with the batch (a live snapshot keeps them too)
REAL_ARMS = ("honest_a", "honest_b", "attack")  # the launcher's `arm` of a main launch's real runs (attack holds <cell> and <cell>_r2)


def cell_of(task: str) -> str:
    return task[:-3] if task.endswith("_r2") else task


def gate_exit(msg: str):
    """A validation failure (not a read failure): exit status 3, so the chain stops this batch loudly."""
    print("GATE: " + msg, file=sys.stderr)
    sys.exit(3)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def counted(run: str, live: bool = False, exclude: frozenset = frozenset(), whole_cells: bool = False, exclude_keys: frozenset = frozenset()) -> tuple[dict, list, dict]:
    """{(arm folder, task): file record}, [counted attempts without an .eval], launch facts. `live` (a snapshot of a running
    launch): the launch need not be finished, and only runs the launcher has marked done, with an ended counted attempt, are
    taken; `whole_cells` (a main snapshot): only cells whose every real run is done; `exclude`: cells already in an earlier
    snapshot (never taken twice)."""
    d = json.loads((ROOT / run / "launch.json").read_text())
    if not live and not d.get("finished") and not os.environ.get("CHAIN_DRY"):  # CHAIN_DRY=1: a test read of a live launch, never relayed
        print(f"{run}: launch.json not finished", file=sys.stderr)
        sys.exit(4)
    out, no_eval, seen = {}, [], set()
    d["_members"] = []
    complete = None
    if whole_cells:
        real = [r for r in d["runs"] if r.get("arm") in REAL_ARMS]
        complete = {cell_of(r["task"]) for r in real} - {cell_of(r["task"]) for r in real if r.get("state") != "done"}
    for r in d["runs"]:
        if cell_of(r["task"]) in exclude:
            continue
        if complete is not None and cell_of(r["task"]) not in complete:
            continue
        if live and r.get("state") != "done":  # a snapshot holds only runs the launcher has finished
            continue
        for ai, a in enumerate(r.get("attempts") or []):
            if a.get("redrawn_because") or a.get("kind") == "rerun":
                continue
            if live and (a.get("end") is None or a.get("exit_code") is None):
                gate_exit(f"{run}/{r['task']}: state done but its counted attempt has not ended")
            rel_dir = (a.get("log_dir") or "").split(f"{run}/", 1)[-1]
            arm = rel_dir.split("/", 1)[0]
            if f"{arm}/{r['task']}" in exclude_keys:  # an interleaved launch's run (not cell) already in an earlier snapshot
                break
            key = (arm, r["task"])
            if key in seen:
                gate_exit(f"{run}: several counted attempts for {key}")
            seen.add(key)
            d["_members"].append(r)  # every counted run, with or without an .eval
            evs = sorted((ROOT / run / rel_dir).glob("*.eval"))
            if not evs:
                no_eval.append({"path": f"{run}/{rel_dir}", "arm": arm, "task": r["task"]})
                continue
            if len(evs) != 1:
                gate_exit(f"{run}/{rel_dir}: {len(evs)} .eval files")
            rel = str(evs[0].relative_to(ROOT / run))
            out[key] = {"launch": run, "rel": rel, "sha256": sha(evs[0]), "bytes": evs[0].stat().st_size, "arm_dir": arm,
                        "task": r["task"], "attempt_index": ai, "attempt_kind": a.get("kind"), "policy_called": a.get("policy_called"),
                        "path": f"{run}/{rel}", "state": r.get("state"), "launch_id": run.rsplit("/", 1)[-1], "stratum": STRATUM,  # 09's Append 21 / 21c; the bare id (T196 layout: run = runs/<id>; d9 10-15)
                        "launch_task": r["task"], "first_start": a.get("start")}  # the launcher's task name on every row (run2 rows: <cell>_r2; 4f's scorer keys)
    facts = {"runs": len(d["runs"]), "states": {}, "finished": d.get("finished"), "members": d["_members"], "policy": d.get("policy"),
             "il_order": {t.get("key") or t.get("cell"): i for i, t in enumerate(d.get("tasks") or [])},  # the launch's task order by LAUNCHER task (the task file's `key`)
             "roster": sorted({r["task"][:-3] if r["task"].endswith("_r2") else r["task"] for r in d["runs"]}),  # every cell the launch lists
             "not_counted": [{"arm": (r.get("key") or "").split("/")[0], "task": r["task"], "state": r.get("state")} for r in d["runs"]
                             if not any(not a.get("redrawn_because") and a.get("kind") != "rerun" for a in r.get("attempts") or [])]}
    for r in d["runs"]:
        facts["states"][r.get("state")] = facts["states"].get(r.get("state"), 0) + 1
    return out, no_eval, facts


def index_donors(main_run: str, files: dict) -> dict:
    """@index:<path>: honest_a donors are the reference copies the launch was given (a merged index), each re-hashed against the
    index; they travel with the batch (added to `files`), since no main batch relays them."""
    idx = json.loads((ROOT / main_run[len("@index:"):]).read_text())
    mc = {}
    for cell, ref in sorted(idx["references"].items()):
        cp = Path(ref["copy"])
        if not cp.exists():
            continue
        h = sha(cp)
        if h != ref["sha256"]:
            gate_exit(f"{cp}: reference copy differs from its index sha256")
        rel = str(cp.relative_to(ROOT))
        if "/references/" in rel:  # a copy inside a launch's references/ dir (il_old's refs_old/<launch>/references/<cell>.eval)
            launch, rel_in = rel.split("/references/", 1)[0], "references/" + rel.split("/references/", 1)[1]
        else:  # a copy in a plain folder (the ug index: refs_ug/q<k>/<cell>.eval)
            launch, rel_in = rel.rsplit("/", 1)
        DONOR_COPIES.add(rel)
        mc[("honest_a", cell)] = {"launch": launch, "rel": rel_in, "sha256": h, "bytes": cp.stat().st_size, "arm_dir": "honest_a", "task": cell,
                                  "attempt_index": None, "attempt_kind": "reference", "policy_called": True, "path": rel, "state": "done",
                                  "launch_id": ref.get("source", "").split("/runs/", 1)[-1].split("/", 1)[0], "stratum": STRATUM, "launch_task": cell,
                                  "reference_source": ref.get("source"), "reference_index": main_run[len("@index:"):]}
        files[rel] = (cp.stat().st_size, h)
    return mc


def manifest(files: list, set_name: str, no_eval: list, note: str) -> dict:
    return {"note": note, "counts": {"files": len(files), "by_arm": {a: sum(f["arm_dir"] == a for f in files) for a in sorted({f["arm_dir"] for f in files})}},
            "counted_attempts_without_eval": {set_name: no_eval}, "files": sorted(files, key=lambda f: f["path"])}


def FORBIDDEN(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1].lower()
    return name == "cells.json" or "approach" in name or "attack_file" in name or "attacker_prompt" in name


def main():
    kind, run = sys.argv[1], sys.argv[2]
    files = {}
    for rec in RECORDS + sorted(p.name for p in (ROOT / run).glob("*.txt")) + sorted(p.name for p in (ROOT / run).glob("upload_manifest_*.tsv")):  # records, markers, upload manifests
        p = ROOT / run / rec
        if p.exists():
            files[f"{run}/{rec}"] = (p.stat().st_size, sha(p))
    for suf in SIBLINGS:
        p = ROOT / f"{run}{suf}"
        if p.exists():
            files[f"{run}{suf}"] = (p.stat().st_size, sha(p))
    snap = sys.argv[5] if len(sys.argv) > 5 else None
    exclude = frozenset(x for x in (sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] != "-" else []) if x)
    il = kind in ("il", "ils")  # an interleaved launch: its snapshots exclude earlier RUNS ("<arm folder>/<task>"), not cells
    c, no_eval, facts = counted(run, live=kind in ("fus", "ms", "ils"), exclude=frozenset() if il else exclude, whole_cells=kind == "ms",
                                exclude_keys=exclude if il else frozenset())
    for f in c.values():
        files[f["path"]] = (f["bytes"], f["sha256"])
    out = {"kind": kind, "run": run, "facts": facts, "manifests": {}, "excluded_earlier_snapshots": sorted(exclude)}
    if kind in ("main", "ms"):  # a main launch's snapshot (ms) or its final remainder (main)
        honest = [f | {"world": a, "snapshot_id": snap} for (a, _), f in c.items() if a in HONEST]
        r1 = [f | {"world": "run1", "snapshot_id": snap} for (a, t), f in c.items() if a == "attack" and not t.endswith("_r2")]
        r2 = [f | {"task": t[:-3], "launch_task": t, "world": "run2", "snapshot_id": snap} for (a, t), f in c.items()
              if a == "attack" and t.endswith("_r2")]  # task = the BASE cell
        # cells: a snapshot's = the whole cells it took; the remainder's = every roster cell no snapshot took (never-started cells
        # included, so their labeler rows reach the combined labels)
        out["members"] = (sorted({cell_of(t) for (_, t) in c} | {cell_of(x["task"]) for x in no_eval}) if kind == "ms"
                          else sorted(set(facts["roster"]) - exclude))
        other = sorted({a for (a, _) in c} - {*HONEST, "attack"})
        if other:
            gate_exit(f"{run}: counted attempts in unexpected arms {other} (the main launch's simulated arm stays paused)")
        ne_main = [x["path"] for x in no_eval if not (x["arm"] == "attack" and x["task"].endswith("_r2"))]  # by the launcher's task name
        ne_r2 = [x["path"] for x in no_eval if x["arm"] == "attack" and x["task"].endswith("_r2")]
        out["manifests"]["main"] = manifest(honest + r1, "main", ne_main, f"scale-up main launch {run}: honest_a, honest_b, attack r1")
        out["manifests"]["run2"] = manifest(honest + r2, "run2", ne_r2, f"scale-up main launch {run}: attack r2 (same launch, interleaved) under base tasks; honest donors")
    elif kind in ("fu", "fus"):  # the follow-up: a snapshot of its finished runs so far (fus), or the final remainder (fu)
        main_run = sys.argv[3]
        if FU_SIM.startswith("sim_honest"):  # a simulated HONEST arm renders with its own prompt: no donors, and no read of the main (it may live elsewhere)
            mc = {}
        else:
            mc = index_donors(main_run, files) if main_run.startswith("@index:") else counted(main_run)[0]
        sims = [f | {"world": FU_WORLD, "sim_launch": run, "run1_launch": main_run, "sim_launch_ne_run1_launch": run != main_run, "snapshot_id": snap}
                for (a, _), f in c.items() if a == FU_SIM]
        out["members"] = sorted({f["task"] for f in sims} | {x["task"] for x in no_eval})  # the launcher's task names (= cells)
        other = sorted({a for (a, _) in c} - {FU_SIM})
        if other:
            gate_exit(f"{run}: counted attempts in unexpected arms {other} (a follow-up runs the simulated arm only)")
        donors = [f | {"world": a} for (a, _), f in mc.items() if a in HONEST]
        out["manifests"]["main"] = manifest(donors + sims, "main", [x["path"] for x in no_eval],
                                            f"scale-up simulated follow-up {run} of {main_run}: {FU_SIM}; honest donors from the main launch")
        out["main_run"] = main_run
        refs = ROOT / run / "references"  # the pinned honest_a references the follow-up actually had (names only)
        out["references"] = sorted(x.name for x in refs.glob("*.eval")) if refs.is_dir() else []
        out["markers"] = sorted(x.name for x in (ROOT / run).glob("*.txt"))  # e.g. SIM_CAP_PAUSE.txt
    elif il:  # interleaved: its real attack (world run3) and simulated attack (world sim2) in one launch; donors from the main launch
        main_run = sys.argv[3]
        if main_run.startswith("@index:"):  # the T196 (il_old): honest_a donors are the reference copies its launch was given (a merged index)
            mc = index_donors(main_run, files)
        else:
            mc, _, _ = counted(main_run)
        other = sorted({a for (a, _) in c} - {"attack", SIM})
        if other:
            gate_exit(f"{run}: counted attempts in unexpected arms {other} (an interleaved launch runs attack and {SIM})")
        pauses = []
        gl = ROOT / run / "gate_pause_watch.log"
        if gl.exists():  # the pause gate's events (lines naming a pause), by timestamp
            for line in gl.read_text().splitlines():
                if "PAUSE" in line.upper() and line[:4].isdigit():
                    pauses.append(line[:200])
        base = {"interleaved_launch": True, "sim_attempt_timeout_s": 300, "main_launch": main_run, "snapshot_id": snap}
        run3 = [f | base | {"world": "run3", "il_order": facts["il_order"].get(t)} for (a, t), f in c.items() if a == "attack"]
        sim2 = [f | base | {"world": "sim2", "sim_launch": run, "run1_launch": main_run, "sim_launch_ne_run1_launch": run != main_run, "il_order": facts["il_order"].get(t)}
                for (a, t), f in c.items() if a == SIM]
        donors = [f | {"world": a} for (a, _), f in mc.items() if a in HONEST]
        out["manifests"]["run3"] = manifest(donors + run3, "main", [x["path"] for x in no_eval if x["arm"] == "attack"],
                                            f"interleaved launch {run} of {main_run}: attack (world run3); honest donors from the main launch")
        out["manifests"]["sim2"] = manifest(donors + sim2, "main", [x["path"] for x in no_eval if x["arm"] == SIM],
                                            f"interleaved launch {run} of {main_run}: {SIM} (world sim2); honest donors from the main launch")
        out["members"] = sorted({f"{a}/{t}" for (a, t) in c} | {f"{x['arm']}/{x['task']}" for x in no_eval})  # RUN keys
        out["main_run"], out["pause_events"] = main_run, pauses
        refs = ROOT / run / "references"
        out["references"] = sorted(x.name for x in refs.glob("*.eval")) if refs.is_dir() else []
        out["markers"] = sorted(x.name for x in (ROOT / run).glob("*.txt"))
    else:
        gate_exit(f"kind {kind!r}")
    sub = {"runs": [{k: r[k] for k in ("key", "task", "arm", "state", "attempts") if k in r} for r in facts["members"]]}
    out["launch_subset"] = json.dumps(sub, sort_keys=True)  # the members' launch.json records, for a snapshot's labeler (keys and fields only)
    facts.pop("members", None)
    if kind in ("fus", "ms", "ils"):  # a live launch's records and logs change under the relay: a snapshot moves only its members' .evals
        files = {k: v for k, v in files.items() if (k.startswith(f"{run}/") or "/references/" in k or k in DONOR_COPIES) and k.endswith(".eval")}  # (+ static donor copies)
    bad = sorted(p for p in files if FORBIDDEN(p))
    if bad:  # never listed for relay (approach/attack files move only via relay_approaches.sh)
        gate_exit(f"{run}: forbidden files in the relay list: {bad[:5]}")
    out["files"] = [{"path": p, "bytes": b, "sha256": s} for p, (b, s) in sorted(files.items())]
    out["counted_without_eval"] = no_eval
    print(json.dumps(out))


if __name__ == "__main__":
    main()
