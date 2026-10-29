"""Cell (a) transfer, the Modal CPU parts (RULE.md items 1-3; POST HOC). One container, numbers only, no text.

- markers (item 1): in the first 4 counted epochs of each staged replay input (levels 0 and 1, honest A and attack), the
  count of "tools>" ([15449, 29]) before the first assistant turn, the number of <think> blocks (248068 ... 248069) and how
  many hold more than the empty "\n\n" (271), i.e. the previous turns' reasoning.
- recenter (item 3): unswapped level 0 at L63. Own outputs (retrain_cache_v1/level0 for run1, run2, sim, honest A, honest
  B; honest C from 2026-10-11_honest_c_segends_v1, counted epochs, as run_t1_sweep.own_data) and Sol's real-honest outputs
  (q2's packed L63 rows). Per-feature mean and SD of own honest C and of Sol's real honest. Each own output is scored by
  q2's outer model of its task's q2 fold, raw, v1 (x - m_own + m_Sol) and v2 ((x - m_own) / s_own * s_Sol + m_Sol); the run
  score is the mean over its outputs. Sol's run scores (raw) are returned per arm for Sol's threshold, gated against q2's
  stored per-output scores (1e-3, as cell (a)'s parity gate).
- reader_l0 (item 2): the level-0 reader-style dump 2026-10-10_la_september-onpolicy-l0_qwen35-27b_alllayers_v1 scored
  with q2's L63 probe exactly as run_repaired_diag.py scored the level-1 dump, eval-file identity against the level-0
  replay's counted metas.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-13_cell_a_transfer_POSTHOC/run_xfer_cpu.py --probes <q2 probes_L63.npz> --q2-runs <q2 runs.json> --fold-of-task <cell_a json>
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
APP = "probe-shift_aswin-surya_sub-onpol-reply_xfercpu_kill-2026-11-13"
ACTS, SCORE, Q2WORK, Q35 = ("probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08", "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08",
                            "probe-shift_aswin-surya_sub-q2-qwen35-reply_work_kill-2026-11-12", "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06")
Q2DATA, HONEST_C, DUMP_L0 = "d403ff8ad9c6", "2026-10-11_honest_c_segends_v1", "2026-10-10_la_september-onpolicy-l0_qwen35-27b_alllayers_v1"
SHA = {"probes": "6cf38abe14c5f3badc1d", "q2_runs": "3c028aca246181950e46"}
BASES = ["rerun_strict_segends_v2", "scaleup_canary_segends_v1", "scaleup_l2_segends_v1", "scaleup_topup_segends_v1"]
MARK = [(b, a) for b in ("scaleup_l2_segends_v1", "scaleup_l2_segends_v1_scrub1") for a in ("honest_a", "attack")]
DUMPS0 = ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1", "2026-10-10_scaleup_l2_segends_v1", "2026-10-10_scaleup_topup_segends_v1"]
REPLAY0 = {"run1": [(d, "attack") for d in DUMPS0], "run2": [("2026-10-10_second_attack_segends_v1", "attack")],
           "sim_fixed": [(d, "sim_attack@september") for d in DUMPS0]}
ARMS = {"real-attack-r1": "run1", "real-attack-r2": "run2", "sim-attack-sept-fixed": "sim_fixed"}
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
app = modal.App(APP)
acts, score_vol, q35 = (modal.Volume.from_name(n, version=2).read_only() for n in (ACTS, SCORE, Q35))
q2_vol = modal.Volume.from_name(Q2WORK).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy").add_local_file(REPO / T196, f"/root/{T196}")


@app.function(image=image, cpu=8, memory=98304, timeout=2400, retries=0, volumes={"/acts": acts, "/score": score_vol, "/q2": q2_vol, "/q35": q35})
def work(P: dict, q2_fold_of_run: list, q2_arm_of_run: list, fold_of_task: dict) -> dict:
    import numpy as np

    out = {"markers": [], "recenter": {}, "reader_l0": {}}
    # --- item 1: markers in the staged replay inputs
    for s, a in MARK:
        n = 0
        with (Path("/acts") / s / "inputs" / f"{a}.jsonl").open() as fh:
            for line in fh:
                ep = json.loads(line)
                if ep.get("redrawn_because") or ep.get("attempt_kind") == "rerun":
                    continue
                ids = ep["ids"]
                first_asst = next((i for i in range(len(ids) - 1) if ids[i] == 248045 and ids[i + 1:i + 2] == [74455]), len(ids))  # <|im_start|>assistant
                tools = sum(ids[i:i + 2] == [15449, 29] for i in range(min(first_asst, len(ids))))
                opens = [i for i, x in enumerate(ids) if x == 248068]
                closes = [i for i, x in enumerate(ids) if x == 248069]
                inner = [c - o - 1 for o, c in zip(opens, closes)]
                out["markers"].append({"staging": s, "arm": a, "epoch_tokens": len(ids), "tools_markers_before_first_assistant": tools,
                                       "think_blocks": len(opens), "think_blocks_with_reasoning": sum(k > 1 for k in inner),
                                       "think_inner_tokens_median": float(np.median(inner)) if inner else None})
                n += 1
                if n == 4:
                    break
    # --- the q2 probe per fold
    P = {k: np.asarray(v, np.float64) for k, v in P.items()}
    W = {f: (P[f"output_end__task_outer_{f}__w"], float(P[f"output_end__task_outer_{f}__b"]), P[f"output_end__task_outer_{f}__mean"], P[f"output_end__task_outer_{f}__std"]) for f in range(5)}
    sc = lambda X, f: ((np.asarray(X, np.float64) - W[f][2]) / W[f][3]) @ W[f][0] + W[f][1]  # noqa: E731
    # --- Sol: run scores (raw) and real-honest feature stats
    d = Path("/q2/data") / Q2DATA
    t = np.load(d / "tables.npz")
    X = np.load(d / "packed" / "L63_output_end.npy", mmap_mode="r")
    ror = t["output_end__run"]
    rf, ra = np.asarray(q2_fold_of_run)[ror], np.asarray(q2_arm_of_run)[ror]
    sol_tot, sol_n = {}, {}
    stored = P["__stored_step"]
    gate = 0.0
    for f in range(5):
        idx = np.flatnonzero(rf == f)
        s = np.concatenate([sc(X[idx[i:i + 4096]], f) for i in range(0, len(idx), 4096)])
        gate = max(gate, float(np.abs(s - stored[idx]).max()))
        for r, v in zip(ror[idx], s):
            sol_tot[int(r)] = sol_tot.get(int(r), 0.0) + float(v)
            sol_n[int(r)] = sol_n.get(int(r), 0) + 1
    out["sol_run_scores"] = {"gate_max_abs_diff_vs_stored": gate, "by_arm": {}}
    for r in sol_tot:
        out["sol_run_scores"]["by_arm"].setdefault(q2_arm_of_run[r], []).append(sol_tot[r] / sol_n[r])
    hon = np.flatnonzero(ra == "real-honest")
    Xh = np.concatenate([np.asarray(X[hon[i:i + 4096]], np.float64) for i in range(0, len(hon), 4096)])
    m_sol, s_sol = Xh.mean(0), Xh.std(0)
    del Xh
    # --- own, unswapped level 0 (as run_t1_sweep.own_data)
    dd = Path("/score") / "retrain_cache_v1" / "level0"
    rows = json.loads((dd / "on_rows.L63.json").read_text())
    kn = [i for i, r in enumerate(rows) if r["kind"] == "output_end" and r["world"] in ("run1", "run2", "sim", "honest_a", "honest_b")]
    A = np.load(dd / "on_L63.npy", mmap_mode="r")
    Xo, ro = np.asarray(A[kn], np.float64), [f"{rows[i]['world']}|{rows[i]['task']}" for i in kn]
    root, parts = Path("/acts") / HONEST_C, []
    for fm in sorted((root / "meta" / "honest_b").glob("*.json")):
        m = json.loads(fm.read_text())
        if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
            continue
        cs = [c for c in m["calls"] if c.get("kind") == "output_end" and c.get("row") is not None]
        if cs:
            B = np.load(root / "by_layer" / "L63" / f"{m['chunk']}.npy", mmap_mode="r")
            parts.append(np.asarray(B[[m["row_start"] + c["row"] for c in cs]], np.float64))
            ro += [f"honest_c|{m['task']}"] * len(cs)
    Xo = np.concatenate([Xo] + parts)
    isc = np.array([u.startswith("honest_c|") for u in ro])
    m_own, s_own = Xo[isc].mean(0), Xo[isc].std(0)
    if (s_own <= 0).any() or (s_sol <= 0).any():
        raise ValueError("a feature with zero SD")
    fold = np.array([fold_of_task[u.split("|", 1)[1]] for u in ro])
    trans = {"raw": Xo, "v1": Xo - m_own + m_sol, "v2": (Xo - m_own) / s_own * s_sol + m_sol}
    for name, Z in trans.items():
        s = np.empty(len(ro))
        for f in range(5):
            s[fold == f] = sc(Z[fold == f], f)
        tot, cnt = {}, {}
        for u, v in zip(ro, s):
            tot[u], cnt[u] = tot.get(u, 0.0) + float(v), cnt.get(u, 0) + 1
        out["recenter"][name] = {u: tot[u] / cnt[u] for u in sorted(tot)}
    out["recenter_meta"] = {"own_honest_c_outputs": int(isc.sum()), "sol_honest_outputs": int(len(hon)),
                            "mean_shift_norm": float(np.linalg.norm(m_own - m_sol)), "sd_ratio_median": float(np.median(s_own / s_sol))}
    del trans, Xo
    # --- item 2, level 0: the reader-style dump scored with q2's L63 probe (run_repaired_diag.py's logic)
    t196 = json.loads((Path("/root") / T196).read_text())
    key_of = {(c["env"], c["main"]): c["key"] for c in t196["cells"]}
    counted = {}
    for world, sources in REPLAY0.items():
        for run, arm in sources:
            for fm in sorted((Path("/acts") / run / "meta" / arm).glob("*.json")):
                m = json.loads(fm.read_text())
                if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                    continue
                counted[(world, m["task"])] = Path(m["run"]).name
    rt = Path("/q35") / DUMP_L0
    out["reader_l0"] = {"identity": {}, "run_scores": {}}
    for arm, world in ARMS.items():
        metas = [json.loads(f.read_text()) for f in sorted((rt / "meta" / arm).glob("*.json"))]
        match = differ = nokey = 0
        items = {}
        for m in metas:
            tk = key_of.get((m["environment"], m["main_task"]))
            if tk is None:
                nokey += 1
                continue
            if counted.get((world, tk)) != Path(m["file"]).name:
                differ += 1
                continue
            match += 1
            for p in m["positions"]:
                if p["kind"] == "output_end" and p["kept"] and p["row"] is not None:
                    items.setdefault(m["chunk"], []).append((tk, m["row_start"] + p["row"]))
        out["reader_l0"]["identity"][world] = {"matched": match, "file_differs": differ, "no_task_key": nokey}
        tot = {}
        for chunk, its in items.items():
            Bm = np.load(rt / "by_layer" / "L63" / f"{chunk}.npy", mmap_mode="r")
            for f in range(5):
                sel = [(tk, r) for tk, r in its if fold_of_task[tk] == f]
                if sel:
                    for (tk, _), v in zip(sel, sc(np.asarray(Bm[[r for _, r in sel]], np.float64), f)):
                        tot.setdefault(tk, []).append(float(v))
        out["reader_l0"]["run_scores"] |= {f"{world}|{tk}": float(np.mean(v)) for tk, v in tot.items()}
    return out


@app.local_entrypoint()
def main(probes: str, q2_runs: str, fold_of_task: str):
    import numpy as np

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=HERE).stdout:
        raise SystemExit("uncommitted tracked changes")
    for k, p in (("probes", probes), ("q2_runs", q2_runs)):
        if not hashlib.sha256(Path(p).read_bytes()).hexdigest().startswith(SHA[k]):
            raise SystemExit(f"{k} sha256 mismatch")
    runs = json.loads(Path(q2_runs).read_text())
    P = {k: v.tolist() for k, v in np.load(probes).items() if k.startswith("output_end__task_outer_")}
    P["__stored_step"] = np.load(Path(q2_runs).parent / "scores_L63.npz")["output_end__step__task_outer"].tolist()
    fot = json.loads(Path(fold_of_task).read_text())["fold_of_task"]
    res = work.remote(P, [r["fold"] for r in runs], [r["arm"] for r in runs], fot)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"xfer_cpu_POSTHOC_{stamp}.json"
    dest.open("x").write(json.dumps({"label": "POST HOC (RULE.md items 1-3); scores and counts only", "git": git} | res))
    print(json.dumps({"markers": res["markers"], "sol_gate": res["sol_run_scores"]["gate_max_abs_diff_vs_stored"], "recenter_meta": res["recenter_meta"],
                      "reader_l0_identity": res["reader_l0"]["identity"]}))
    print(dest.name)
