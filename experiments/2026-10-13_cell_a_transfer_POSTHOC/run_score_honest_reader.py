"""Cell (a) transfer, Amendment 1: score Qwen3.5's own runs entirely reader-style with q2's frozen L63 probe (POST HOC;
one Modal CPU container, scores only). Per level (0, 1):
- extraction gate: the 3 gate transcripts' L63 output_end rows in the new dump (arm gate-real-attack-r1) against the
  existing dump's real-attack-r1 rows of the same key: max |difference| <= 1e-2, else that level is not scored;
- attack arms from the existing dump 2026-10-10_la_september-onpolicy-l<L>_qwen35-27b_alllayers_v1 (real-attack-r1 -> run1,
  real-attack-r2 -> run2, sim-attack-sept-fixed -> sim), kept only where the eval file equals the replay pipeline's counted
  run (as run_repaired_diag.py / run_xfer_cpu.py), so the attempt labels apply;
- honest arms from the new dump 2026-10-13_la_september-onpolicy-honest-l<L>_qwen35-27b_alllayers_v1 (real-honest-a ->
  honest_a, real-honest-b -> honest_b);
- run score = mean over kept output_end positions, each scored by q2's outer model of its task's q2 fold.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-13_cell_a_transfer_POSTHOC/run_score_honest_reader.py --probes <q2 probes_L63.npz> --fold-of-task <cell_a json>
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
APP = "probe-shift_aswin-surya_sub-onpol-reply_xferscore_kill-2026-11-13"
Q35, ACTS = "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06", "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
SHA_L63 = "6cf38abe14c5f3badc1d"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
OLD = "2026-10-10_la_september-onpolicy-l{L}_qwen35-27b_alllayers_v1"
NEW = "2026-10-13_la_september-onpolicy-honest-l{L}_qwen35-27b_alllayers_v1"
ATTACK = {"real-attack-r1": "run1", "real-attack-r2": "run2", "sim-attack-sept-fixed": "sim"}
HONEST = {"real-honest-a": "honest_a", "real-honest-b": "honest_b"}
D = {"0": ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1", "2026-10-10_scaleup_l2_segends_v1", "2026-10-10_scaleup_topup_segends_v1"]}
D["1"] = [d + "_scrub1" for d in D["0"]]
REPLAY = {L: {"run1": [(d, "attack") for d in D[L]], "run2": [("2026-10-10_second_attack_segends_v1" + ("_scrub1" if L == "1" else ""), "attack")],
              "sim": [(d, "sim_attack@september") for d in D[L]]} for L in ("0", "1")}
app = modal.App(APP)
q35, acts = (modal.Volume.from_name(n, version=2).read_only() for n in (Q35, ACTS))
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy").add_local_file(REPO / T196, f"/root/{T196}")


@app.function(image=image, cpu=8, memory=65536, timeout=2400, retries=0, volumes={"/q35": q35, "/acts": acts})
def score(P: dict, fold_of_task: dict) -> dict:
    import numpy as np

    P = {k: np.asarray(v, np.float64) for k, v in P.items()}
    sc = lambda X, f: ((np.asarray(X, np.float64) - P[f"output_end__task_outer_{f}__mean"]) / P[f"output_end__task_outer_{f}__std"]) @ P[f"output_end__task_outer_{f}__w"] + float(P[f"output_end__task_outer_{f}__b"])  # noqa: E731
    t196 = json.loads((Path("/root") / T196).read_text())
    key_of = {(c["env"], c["main"]): c["key"] for c in t196["cells"]}

    def rows_of(root, arm):  # {file key: (task, [(chunk, row)])} over kept output_end positions
        out = {}
        for f in sorted((root / "meta" / arm).glob("*.json")):
            m = json.loads(f.read_text())
            t = key_of.get((m["environment"], m["main_task"]))
            ks = [(m["chunk"], m["row_start"] + p["row"]) for p in m["positions"] if p["kind"] == "output_end" and p["kept"] and p["row"] is not None]
            out[m["key"] if "key" in m else f.stem] = (t, ks, Path(m["file"]).name)
        return out

    def load(root, items):
        by = {}
        for i, (c, r) in enumerate(items):
            by.setdefault(c, []).append((i, r))
        X = np.empty((len(items), len(P["output_end__task_outer_0__mean"])), np.float64)
        for c, ir in by.items():
            A = np.load(root / "by_layer" / "L63" / f"{c}.npy", mmap_mode="r")
            X[[i for i, _ in ir]] = np.asarray(A[[r for _, r in ir]], np.float64)
        return X

    res = {}
    for L in ("0", "1"):
        old, new = Path("/q35") / OLD.format(L=L), Path("/q35") / NEW.format(L=L)
        gate_new, gate_old = rows_of(new, "gate-real-attack-r1"), rows_of(old, "real-attack-r1")
        g = {}
        for k, (t, ks, _) in gate_new.items():
            if k not in gate_old:
                g[k] = {"found": False}
                continue
            a, b = load(new, ks), load(old, gate_old[k][1])
            g[k] = {"found": True, "rows": [len(ks), len(gate_old[k][1])],
                    "max_abs_diff": float(np.abs(a - b).max()) if a.shape == b.shape else None}
        ok = len(g) == 3 and all(x.get("found") and x.get("max_abs_diff") is not None and x["max_abs_diff"] <= 1e-2 for x in g.values())
        res[L] = {"gate": g, "gate_passes": ok}
        if not ok:
            continue
        counted = {}
        for world, sources in REPLAY[L].items():
            for run, arm in sources:
                for fm in sorted((Path("/acts") / run / "meta" / arm).glob("*.json")):
                    m = json.loads(fm.read_text())
                    if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                        continue
                    counted[(world, m["task"])] = Path(m["run"]).name
        scores, ident = {}, {}
        for root, arms in ((old, ATTACK), (new, HONEST)):
            for arm, world in arms.items():
                n_ok = n_diff = n_nokey = 0
                for k, (t, ks, file) in rows_of(root, arm).items():
                    if t is None:
                        n_nokey += 1
                        continue
                    if world in ("run1", "run2", "sim") and counted.get((world, t)) != file:
                        n_diff += 1
                        continue
                    if not ks:
                        continue
                    X = load(root, ks)
                    scores[f"{world}|{t}"] = float(sc(X, fold_of_task[t]).mean())
                    n_ok += 1
                ident[world] = {"scored": n_ok, "file_differs": n_diff, "no_task_key": n_nokey}
        res[L] |= {"identity": ident, "run_scores": scores}
    return res


@app.local_entrypoint()
def main(probes: str, fold_of_task: str):
    import numpy as np

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=HERE).stdout:
        raise SystemExit("uncommitted tracked changes")
    if not hashlib.sha256(Path(probes).read_bytes()).hexdigest().startswith(SHA_L63):
        raise SystemExit("probes sha256 mismatch")
    P = {k: v.tolist() for k, v in np.load(probes).items() if k.startswith("output_end__task_outer_")}
    fot = json.loads(Path(fold_of_task).read_text())["fold_of_task"]
    res = score.remote(P, fot)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"honest_reader_scores_POSTHOC_{stamp}.json"
    dest.open("x").write(json.dumps({"label": "POST HOC (RULE.md Amendment 1); run scores only", "git": git} | res))
    print(json.dumps({L: {"gate": r["gate"], "gate_passes": r["gate_passes"], "identity": r.get("identity")} for L, r in res.items()}))
    print(dest.name)
