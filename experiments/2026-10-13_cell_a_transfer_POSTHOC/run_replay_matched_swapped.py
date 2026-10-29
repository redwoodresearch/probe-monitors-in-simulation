"""Cell (a) transfer, Amendment 3: the SWAPPED replay (claimc_cache_v1: level 1, honest system prompt, with the tools block and
reasoning) truncation-matched to the level-1 reader (POST HOC; one Modal CPU container, scores only). A copy of
run_replay_matched.py with the level-1 swapped cache and the swapped stagings' counted metas; the rest as below.

Amendment 2's truncation-matched replay control, as it was:

Per level L (0, 1) and world (run1, run2, sim, honest_a, honest_b): each run's replay output_end rows (the unswapped cache
retrain_cache_v1/level<L>, as cell (a)) in turn order (`order`), cut to the first n, where n is the number of output_end
positions the reader-style dump kept for the same run (attack arms: 2026-10-10_la_september-onpolicy-l<L>_..., the repaired
simulated run's count for the simulated run; honest arms: 2026-10-13_la_september-onpolicy-honest-l<L>_...), matched by
eval file against the replay's counted metas, as run_score_honest_reader.py. Each output is scored by q2's L63 outer model
of its task's q2 fold; run score = mean over the first n. Also returned per world: runs matched, runs where n equals the
replay's output count, and runs where the reader kept more outputs than the replay has (cut to the replay's count, counted).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-13_cell_a_transfer_POSTHOC/run_replay_matched_swapped.py --probes <q2 probes_L63.npz> --fold-of-task <cell_a json>
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-onpol-reply_xfermatchedsw_kill-2026-11-13"
Q35, ACTS, SCORE = ("probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06", "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08",
                    "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08")
SHA_L63 = "6cf38abe14c5f3badc1d"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
OLD = "2026-10-10_la_september-onpolicy-l{L}_qwen35-27b_alllayers_v1"
NEW = "2026-10-13_la_september-onpolicy-honest-l{L}_qwen35-27b_alllayers_v1"
READER = {"run1": ("old", "real-attack-r1"), "run2": ("old", "real-attack-r2"), "sim": ("old", "sim-attack-sept-fixed"),
          "honest_a": ("new", "real-honest-a"), "honest_b": ("new", "real-honest-b")}
D = {"0": ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1", "2026-10-10_scaleup_l2_segends_v1", "2026-10-10_scaleup_topup_segends_v1"]}
D["1"] = [d + "_scrub1" for d in D["0"]]
REPLAY = {"1": {"run1": [(d + "_swapsys", "attack") for d in D["1"]], "run2": [("2026-10-10_second_attack_segends_v1_scrub1_swapsys", "attack")],
               "sim": [(d + "_swapsys2", "sim_attack@september") for d in D["1"]], "honest_a": [(d, "honest_a") for d in D["1"]],
               "honest_b": [(d, "honest_b") for d in D["1"]]}}
app = modal.App(APP)
q35, acts, score_vol = (modal.Volume.from_name(n, version=2).read_only() for n in (Q35, ACTS, SCORE))
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy").add_local_file(REPO / T196, f"/root/{T196}")


@app.function(image=image, cpu=8, memory=65536, timeout=2400, retries=0, volumes={"/q35": q35, "/acts": acts, "/score": score_vol})
def matched(P: dict, fold_of_task: dict) -> dict:
    import numpy as np

    P = {k: np.asarray(v, np.float64) for k, v in P.items()}
    sc = lambda X, f: ((np.asarray(X, np.float64) - P[f"output_end__task_outer_{f}__mean"]) / P[f"output_end__task_outer_{f}__std"]) @ P[f"output_end__task_outer_{f}__w"] + float(P[f"output_end__task_outer_{f}__b"])  # noqa: E731
    t196 = json.loads((Path("/root") / T196).read_text())
    key_of = {(c["env"], c["main"]): c["key"] for c in t196["cells"]}
    out = {}
    for L in ("1",):
        counted = {}
        for world, sources in REPLAY[L].items():
            for run, arm in sources:
                for fm in sorted((Path("/acts") / run / "meta" / arm).glob("*.json")):
                    m = json.loads(fm.read_text())
                    if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                        continue
                    counted[(world, m["task"])] = Path(m["run"]).name
        kept = {}
        for world, (which, arm) in READER.items():
            root = Path("/q35") / (OLD if which == "old" else NEW).format(L=L)
            for f in sorted((root / "meta" / arm).glob("*.json")):
                m = json.loads(f.read_text())
                t = key_of.get((m["environment"], m["main_task"]))
                if t is None or counted.get((world, t)) != Path(m["file"]).name:
                    continue
                kept[f"{world}|{t}"] = sum(p["kind"] == "output_end" and p["kept"] and p["row"] is not None for p in m["positions"])
        d = Path("/score") / "claimc_cache_v1"
        rows = json.loads((d / "on_rows.L63.json").read_text())
        A = np.load(d / "on_L63.npy", mmap_mode="r")
        by = {}
        for i, r in enumerate(rows):
            if r["kind"] == "output_end" and r["world"] in READER:
                by.setdefault(f"{r['world']}|{r['task']}", []).append((tuple(r["order"]), i))
        scores, stats = {}, {w: {"matched": 0, "n_equal": 0, "reader_more": 0, "no_replay": 0} for w in READER}
        for k, n in kept.items():
            w, t = k.split("|", 1)
            if k not in by:
                stats[w]["no_replay"] += 1
                continue
            idx = [i for _, i in sorted(by[k])]
            stats[w]["matched"] += 1
            stats[w]["n_equal"] += n == len(idx)
            stats[w]["reader_more"] += n > len(idx)
            take = idx[:max(1, min(n, len(idx)))]
            scores[k] = float(sc(np.asarray(A[take], np.float64), fold_of_task[t]).mean())
        out[L] = {"run_scores": scores, "stats": stats}
    return out


@app.local_entrypoint()
def main(probes: str, fold_of_task: str):
    import numpy as np

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=HERE).stdout:
        raise SystemExit("uncommitted tracked changes")
    if not hashlib.sha256(Path(probes).read_bytes()).hexdigest().startswith(SHA_L63):
        raise SystemExit("probes sha256 mismatch")
    P = {k: v.tolist() for k, v in np.load(probes).items() if k.startswith("output_end__task_outer_")}
    res = matched.remote(P, json.loads(Path(fold_of_task).read_text())["fold_of_task"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"replay_matched_swapped_POSTHOC_{stamp}.json"
    dest.open("x").write(json.dumps({"label": "POST HOC (RULE.md Amendment 3): swapped replay, truncation-matched to the level-1 reader", "git": git} | res))
    print(json.dumps({L: r["stats"] for L, r in res.items()}))
    print(dest.name)
