"""Halves on Qwen3.5 (PLAN.md in this folder; Aswin "Buy it ($39–59)", 00:04:07 PT 10-12, via 0d): stage the evened texts
for the Qwen3.5 reader, and prove they are the texts that ran (the byte gate). Keys and numbers only leave the containers.

  --mode stage     CPU: the evened copies of H3's changed test runs (plant 1 left out: halves doesn't use it), B-dev's
                   changed dev runs and its 100 dev planted copies (evened, R2 undone), plus H3's 4 unchanged gate runs
                   as they are, made by H3's and B-dev's own code (imported unchanged, hash-checked), written as staged
                   inputs in the format run_extract_qwen35.py reads (`<run>/inputs/shard_*.jsonl`, `manifest.json`). The
                   changed sets must equal H3's and B-dev's cells.json lists; every message list must be free of
                   'attachment://'. Refuses to overwrite an existing staged run.
  --mode bytegate  1 H100: three staged items (an H3 test changed run, a B-dev dev changed run, a B-dev planted copy) read
                   by B-dev's Qwen3-32B reader (the code path H3 and B-dev used) must equal the stored H3 / B-dev rows
                   bitwise at L58 and L63, with identical output positions and range outputs: identical activations
                   mean identical tokens, so the staged text is the evened text that ran.

probe_shift comes from the as-run commit 23b5c37a (`git archive 23b5c37a src/probe_shift` into <asrun>), where H3's and
B-dev's dependency hash (33d539e9) holds; on main only torch_probe.py differs (reader.py is unchanged), and H3's
check_b() refuses any other version, locally (checked in main) and in the containers.

  PYTHONPATH=<asrun>/src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal deploy \\
      experiments/2026-10-11_claimB_halves_qwen35/run_stage_evened_modal.py
  PYTHONPATH=<asrun>/src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy python \\
      experiments/2026-10-11_claimB_halves_qwen35/run_stage_evened_modal.py stage|bytegate
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
H3_DIR = HERE.parent / "2026-10-11_claimB_h3_fix_to_plateau"
sys.path[:0] = [str(H3_DIR), str(HERE.parent / "2026-10-11_claimB_slips_evened_probe")] if modal.is_local() else ["/root"]
import run_h3_small_modal as R  # noqa: E402  (H3's runner at 23b5c37a: B-dev's code, constants, the evened test reads)

H3_SHA12 = "04c3953b139e"  # run_h3_small_modal.py whose extraction wrote the evened test reads
B, M = R.B, R.M
SIM, E1, E2, PL1, PL2 = M.SIM, M.REAL, "real-attack-e2", "sim-attack-fixed-plant1", "sim-attack-fixed-plant2"
STAGED_NEW = "2026-10-12_la_fixed-sim-v2_probe-inputs-b-l1-evened-halves_v1"
FIELDS = ("key", "arm", "cohort", "file", "sample_id", "epoch", "environment", "main_task", "side_task", "cell",
          "n_submit_calls", "sim_reference", "tokens")
N_FILES, NEEDLE = 8, "attachment://"
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
GATE_OUT = f"claimB/halves_qwen35_v1/bytegate_{SCRIPT_SHA[:12]}"

app = modal.App("probe-shift_aswin-surya_sub-claimb-hq35stage_kill-2026-11-12")
image = R.image.add_local_file(H3_DIR / "run_h3_small_modal.py", "/root/run_h3_small_modal.py")


def check_code():
    R.check_b()
    if R.SCRIPT_SHA[:12] != H3_SHA12:
        raise ValueError("H3's runner differs from the one whose extraction wrote the evened test reads")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def items_from_plans():
    """Every staged item: (item dict with messages, manifest row). Built only from H3's and B-dev's code and plans."""
    runs = R.staged()
    h3 = json.loads((Path("/acts") / R.OUT / "cells.json").read_text())
    bd = json.loads((Path("/acts") / R.DEV_READS / "cells.json").read_text())
    if h3["script_sha256"] != R.SCRIPT_SHA or (bd["script_sha256"], bd["deps_sha256"]) != (B.SCRIPT_SHA, B.DEPS_SHA):
        raise ValueError("a cells.json was made by other code")
    out = []

    def add(kind, cell, arm, msgs, rng, arm_as=None, key_suffix=""):
        base = runs[(cell, arm)]
        tool_msgs = [k for k, m in enumerate(msgs) if m["role"] == "tool"]
        item = {f: base[f] for f in FIELDS if f in base} | {"messages": msgs}
        item["arm"], item["key"] = arm_as or arm, base["key"] + key_suffix
        if NEEDLE in json.dumps(msgs):
            raise ValueError(f"{kind} {cell} {arm}: '{NEEDLE}' in a staged message list")
        row = {"kind": kind, "arm": item["arm"], "key": item["key"], "cell": cell, "source_arm": arm,
               "environment": base["environment"], "main_task": base["main_task"],
               "messages_sha256": sha(json.dumps(msgs, sort_keys=True).encode()),
               "range_msg_indices": list(rng), "range_tool_ordinals": [tool_msgs.index(i) for i in rng]}
        out.append((item, row))

    for cell, arm in h3["changed"]:
        if arm == PL1:
            continue
        e, rng = R.evened_version(runs, cell, arm)
        if e == runs[(cell, arm)]["messages"] and not rng:
            raise ValueError(f"{cell} {arm}: listed as changed by H3 but the evening changes nothing")
        add("h3_test_changed", cell, arm, e, rng)
    for cell, arm in bd["changed"]:
        e, n, rng = B.evened(runs[(cell, arm)]["messages"])
        if not (n or rng):
            raise ValueError(f"{cell} {arm}: listed as changed by B-dev but the evening changes nothing")
        add("bdev_dev_changed", cell, arm, e, rng)
    for cell in bd["planted"]:
        e, _, rng = B.evened(runs[(cell, SIM)]["messages"])
        p, n = B.planted(e)
        if not n:
            raise ValueError(f"{cell}: B-dev planted copy has no R2 undo")
        add("bdev_dev_planted", cell, SIM, p, rng, arm_as=PL2, key_suffix="#plant2-dev")
    for cell in h3["gate"]:
        add("h3_gate_unchanged", cell, SIM, runs[(cell, SIM)]["messages"], [])
    return out, {"h3_cells_sha256": sha((Path("/acts") / R.OUT / "cells.json").read_bytes()),
                 "bdev_cells_sha256": sha((Path("/acts") / R.DEV_READS / "cells.json").read_bytes()),
                 "h3_changed_wo_plant1": sum(1 for _, a in h3["changed"] if a != PL1), "bdev_changed": len(bd["changed"]),
                 "bdev_planted": len(bd["planted"]), "h3_gate": len(h3["gate"])}


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": B.P.vol}, timeout=30 * 60, retries=0)
def stage() -> dict:
    check_code()
    dst = Path("/acts") / STAGED_NEW
    if dst.exists():
        raise ValueError(f"{STAGED_NEW} exists: staged runs are never overwritten")
    pairs, src = items_from_plans()
    ids = [(it["arm"], it["key"]) for it, _ in pairs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate (arm, key) in the staged items")
    (dst / "inputs").mkdir(parents=True)
    files = {}
    for f in range(N_FILES):
        body = "".join(json.dumps(it) + "\n" for it, _ in pairs[f::N_FILES]).encode()
        name = f"shard_{f:03d}.jsonl"
        (dst / "inputs" / name).write_bytes(body)
        files[name] = sha(body)
    arms = {}
    for it, _ in pairs:
        arms[it["arm"]] = arms.get(it["arm"], 0) + 1
    by_kind = {}
    for _, row in pairs:
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
    man = {"run": STAGED_NEW, "level": "level1", "from": M.STAGED_B, "arms": arms, "files": files, "by_kind": by_kind,
           "items": [row for _, row in pairs], "script_sha256": SCRIPT_SHA, "h3_runner_sha256": R.SCRIPT_SHA,
           "b_script_sha256": B.SCRIPT_SHA, "b_deps_sha256": B.DEPS_SHA,
           "from_manifest_sha256": sha((Path("/acts") / M.STAGED_B / "manifest.json").read_bytes())} | src
    (dst / "manifest.json").write_text(json.dumps(man))
    B.P.vol.commit()
    return {k: v for k, v in man.items() if k != "items"}


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": B.P.vol, "/hf": B.P.hf}, timeout=20 * 60, retries=0)
def bytegate() -> dict:
    import numpy as np

    check_code()
    man = json.loads((Path("/acts") / STAGED_NEW / "manifest.json").read_text())
    if man["script_sha256"] != SCRIPT_SHA:
        raise ValueError("the staged run was made by another version of this script")
    staged, found = {}, sorted((Path("/acts") / STAGED_NEW / "inputs").glob("shard_*.jsonl"))
    if {f.name for f in found} != set(man["files"]):
        raise ValueError("the staged shard files differ from the manifest's list")
    for f in found:
        if sha(f.read_bytes()) != man["files"][f.name]:
            raise ValueError(f"{f.name} differs from the manifest")
        for line in f.read_text().splitlines():
            it = json.loads(line)
            staged[(it["arm"], it["key"])] = it
    pick = {}
    for kind in ("h3_test_changed", "bdev_dev_changed", "bdev_dev_planted"):
        pick[kind] = sorted((r for r in man["items"] if r["kind"] == kind and r["source_arm"] == SIM), key=lambda r: r["key"])[0]
    ref_dir = {"h3_test_changed": R.OUT, "bdev_dev_changed": R.DEV_READS, "bdev_dev_planted": R.DEV_READS}
    version = {"h3_test_changed": "evened", "bdev_dev_changed": "evened", "bdev_dev_planted": "planted"}
    jobs, refs = [], {}
    for kind, r in pick.items():
        it, cell, v = staged[(r["arm"], r["key"])], r["cell"], version[kind]
        name = hashlib.sha256(f"{cell}|{SIM}|{v}".encode()).hexdigest()[:16] + ".npz"
        ref = np.load(Path("/acts") / ref_dir[kind] / "runs" / name)
        refs[kind] = (name, ref, json.loads(str(ref["meta"])))
        jobs.append((r["cell"], SIM, version[kind], it["messages"], r["range_msg_indices"]))
    from probe_shift.models import QWEN3_32B

    model, revision = QWEN3_32B["name"], QWEN3_32B["revision"]
    for kind, (name, _, m) in refs.items():  # the stored reads must be the reads that ran: B-dev's code, the pinned model
        if (m["script_sha256"], m.get("deps_sha256"), m["model"], m["revision"]) != (R.B_SCRIPT, R.B_DEPS, model, revision):
            raise ValueError(f"{kind} {name}: stored read made by other code or model")
        if (m["cell"], m["arm"], m["version"]) != (pick[kind]["cell"], SIM, version[kind]):
            raise ValueError(f"{kind} {name}: stored read is for another run or version")
    B.OUT = GATE_OUT
    stats = B._read(jobs, model, revision, "bytegate")
    res = {}
    for kind, (name, ref, meta) in refs.items():
        mine = np.load(Path("/acts") / GATE_OUT / "runs" / name)
        mm = json.loads(str(mine["meta"]))
        same_pos = mm["steps"] == meta["steps"] and mm["range"] == meta["range"]
        same = {f"L{l}": bool(np.array_equal(mine[f"L{l}"], ref[f"L{l}"])) for l in M.LAYERS}
        res[kind] = {"key": pick[kind]["key"], "positions_equal": same_pos, "bitwise": same, "rows": int(len(mine["L63"]))}
    out = {"items": res, "passed": all(v["positions_equal"] and all(v["bitwise"].values()) for v in res.values()),
           "model": model, "revision": revision, "script_sha256": SCRIPT_SHA, "stats": stats}
    (Path("/acts") / GATE_OUT / "bytegate.json").write_text(json.dumps(out))
    B.P.vol.commit()
    return out


@app.function(image=image, cpu=1, memory=1024, timeout=120, retries=0)
def version() -> str:
    return SCRIPT_SHA


def main(mode: str):
    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    check_code()  # locally too: probe_shift must be the as-run version (see the docstring)
    if modal.Function.from_name(app.name, "version").remote() != SCRIPT_SHA:
        raise SystemExit("the deployed app runs other code: `modal deploy` this revision first")
    if mode not in ("stage", "bytegate"):
        raise SystemExit(f"unknown mode {mode}")
    call = modal.Function.from_name(app.name, mode).spawn()
    print(f"{mode}: {call.object_id}", flush=True)
    print(json.dumps(call.get(timeout=40 * 60), indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
