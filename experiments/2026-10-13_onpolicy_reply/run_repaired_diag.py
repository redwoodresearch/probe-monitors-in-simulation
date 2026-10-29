"""Repaired format, cross-pipeline diagnostic (RULE.md step 3 with Amendment 2.3): no calibrated reading. One Modal CPU
container; returns run scores only.

The fixed (repaired) simulated runs exist only in the reader-style dump `2026-10-10_la_september-onpolicy-l1_qwen35-27b_
alllayers_v1` (honest system prompt, times masked, 32k cut), beside the first and second real runs read the same way.
Each swapped cell's probe scores that dump's kept output_end rows: (a) q2's outer model of the task's q2 fold at L63;
(b) claim C's refit weights (plantslip_2e); (c) the swapped cell-(c) grid's saved weights at stage 4's choices. Run
score = mean over kept outputs.
Gate (identity): per arm and task, the reader dump's eval file equals the file of the run the replay pipeline counted
(the swapped extractions' counted metas: run 1 from `*_scrub1_swapsys`, run 2 from the second attack's, simulated from
`*_scrub1_swapsys2`); runs that differ are dropped and counted. Coverage reported per arm: runs, truncated runs, kept
outputs per run (reader) against outputs per run (replay).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run experiments/2026-10-13_onpolicy_reply/run_repaired_diag.py --probes <q2 probes_L63.npz>
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-onpol-reply_repaired_kill-2026-11-13"
Q35, ACTS, SCORE = "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06", "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08", "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"
DUMP = "2026-10-10_la_september-onpolicy-l1_qwen35-27b_alllayers_v1"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
ARMS = {"real-attack-r1": "run1", "real-attack-r2": "run2", "sim-attack-sept-fixed": "sim_fixed"}
BASES = ["2026-10-09_rerun_strict_segends_v1_scrub1", "2026-10-10_scaleup_canary_segends_v1_scrub1", "2026-10-10_scaleup_l2_segends_v1_scrub1",
         "2026-10-10_scaleup_topup_segends_v1_scrub1"]
REPLAY = {"run1": [(f"{b}_swapsys", "attack") for b in BASES], "run2": [("2026-10-10_second_attack_segends_v1_scrub1_swapsys", "attack")],
          "sim_fixed": [(f"{b}_swapsys2", "sim_attack@september") for b in BASES]}
app = modal.App(APP)
q35, acts, score_vol = (modal.Volume.from_name(n, version=2).read_only() for n in (Q35, ACTS, SCORE))
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy").add_local_file(REPO / T196, f"/root/{T196}")


def frozen_folds(d):  # run_claim_c.frozen_folds, unchanged
    import hashlib
    h = lambda s: hashlib.sha256((d["salt"] + s).encode()).hexdigest()  # noqa: E731
    envs, by_env = sorted(d["eligible_mains_by_env"], key=h), {}
    for c in d["cells"]:
        by_env.setdefault(c["env"], []).append(c["main"])
    return {c["key"]: (by_env[c["env"]].index(c["main"]) + envs.index(c["env"])) % 5 for c in d["cells"]}


@app.function(image=image, cpu=8, memory=65536, timeout=1800, retries=0, volumes={"/q35": q35, "/acts": acts, "/score": score_vol})
def diag(q2: dict, q2_fold_of_task: dict, c_choices: list) -> dict:
    import numpy as np

    t196 = json.loads((Path("/root") / T196).read_text())
    fold_of = frozen_folds(t196)
    key_of = {(c["env"], c["main"]): c["key"] for c in t196["cells"]}
    q2 = {k: np.asarray(v, np.float64) for k, v in q2.items()}
    Wb = np.load("/score/plantslip_2e/claimc_refit_reply_end.npz")
    mb = json.loads(str(Wb["meta"]))
    Wc = {f: np.load(f"/score/onpolreply_c_bc89eeb2869c/swapped_L1/W_L{l}_C{C:g}.npz") for f, (l, C) in enumerate(c_choices)}
    cells = {"a": (q2_fold_of_task, {f: 63 for f in range(5)},
                   lambda x, f: ((x - q2[f"output_end__task_outer_{f}__mean"]) / q2[f"output_end__task_outer_{f}__std"]) @ q2[f"output_end__task_outer_{f}__w"] + q2[f"output_end__task_outer_{f}__b"]),
             "b": (fold_of, {f: mb[str(f)]["layer"] for f in range(5)}, lambda x, f: ((x - Wb[f"{f}|mean"]) / Wb[f"{f}|std"]) @ Wb[f"{f}|w"] + mb[str(f)]["b"]),
             "c": (fold_of, {f: c_choices[f][0] for f in range(5)},
                   lambda x, f: (((x - Wc[f]["mean"][f]) / Wc[f]["std"][f]) @ Wc[f]["W"][f]) * float(Wc[f]["sqrtC"]) + Wc[f]["b"][f])}
    counted = {}  # (world, task) -> eval file basename, from the replay pipeline's counted metas
    replay_outputs = {}
    for world, sources in REPLAY.items():
        for run, arm in sources:
            for f in sorted((Path("/acts") / run / "meta" / arm).glob("*.json")):
                m = json.loads(f.read_text())
                if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                    continue
                counted[(world, m["task"])] = Path(m["run"]).name
                replay_outputs[(world, m["task"])] = replay_outputs.get((world, m["task"]), 0) + sum(c.get("kind") == "output_end" and c.get("row") is not None for c in m["calls"])
    root = Path("/q35") / DUMP
    out = {"identity": {}, "coverage": {}, "run_scores": {c: {} for c in cells}}
    for arm, world in ARMS.items():
        metas = [json.loads(f.read_text()) for f in sorted((root / "meta" / arm).glob("*.json"))]
        match, differ, nokey, trunc, kept_n = 0, 0, 0, 0, []
        rows_by_layer = {}
        for m in metas:
            t = key_of.get((m["environment"], m["main_task"]))
            if t is None:
                nokey += 1
                continue
            if counted.get((world, t)) != Path(m["file"]).name:
                differ += 1
                continue
            match += 1
            trunc += bool(m["truncated"])
            ks = [p for p in m["positions"] if p["kind"] == "output_end" and p["kept"] and p["row"] is not None]
            kept_n.append(len(ks))
            for p in ks:
                rows_by_layer.setdefault(m["chunk"], []).append((t, m["row_start"] + p["row"]))
        out["identity"][world] = {"matched": match, "file_differs": differ, "no_task_key": nokey}
        out["coverage"][world] = {"runs": match, "truncated": trunc, "kept_outputs_per_run_mean": float(np.mean(kept_n)) if kept_n else None,
                                  "replay_outputs_per_run_mean": float(np.mean([v for (w, _), v in replay_outputs.items() if w == world]))}
        for name, (fo, layer_of, sc) in cells.items():
            tot = {}
            for chunk, items in rows_by_layer.items():
                for layer in sorted({layer_of[fo[t]] for t, _ in items}):
                    A = np.load(root / "by_layer" / f"L{layer}" / f"{chunk}.npy", mmap_mode="r")
                    for f in range(5):
                        sel = [(t, r) for t, r in items if fo[t] == f and layer_of[f] == layer]
                        if sel:
                            for (t, _), s in zip(sel, sc(np.asarray(A[[r for _, r in sel]], np.float64), f)):
                                tot.setdefault(t, []).append(float(s))
            out["run_scores"][name] |= {f"{world}|{t}": float(np.mean(v)) for t, v in tot.items()}
    return out


@app.local_entrypoint()
def main(probes: str):
    import numpy as np

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=HERE).stdout:
        raise SystemExit("uncommitted tracked changes")
    a = json.loads((HERE / "results" / "cell_a_run_scores_POSTHOC_2026-10-13_121016.json").read_text())
    s4 = json.loads((HERE / "results" / "stage4_cell_c_POSTHOC_2026-10-13_121906.json").read_text())["cells"]
    q2 = {k: v.tolist() for k, v in np.load(probes).items() if k.startswith("output_end__task_outer_")}
    res = diag.remote(q2, a["fold_of_task"], s4["c_swapped_L1"]["choices_real_column"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"repaired_diag_scores_POSTHOC_{stamp}.json"
    dest.open("x").write(json.dumps({"label": "POST HOC (RULE.md step 3, Amendment 2.3): cross-pipeline diagnostic, no repair-effect reading", "git": git} | res))
    print(json.dumps({"identity": res["identity"], "coverage": res["coverage"]}), dest.name)
