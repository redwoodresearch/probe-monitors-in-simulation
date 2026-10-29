"""Planted format (RULE.md step 3 with Amendments 1-2): the gutter slip planted into Qwen3.5's own first and second real
runs, scored by every cell's probe. Deployed Modal app. Swapped variant: the P7/P8 stagings (`plantslip_p78_run{1,2}`,
level 1, swapped); unswapped: the P1 stagings (`plantslip_p1_run{1,2}`, level 0). There is no unswapped level-1 plant.

- refit_b (H100): cell (b) unswapped level 0 has no saved weights, so each fold's selected candidate (stage 1) is refit
  exactly as run_plantslip_onpolicy.refit (Amendment 2.6). Gate: every run's out-of-fold mean equals the stored score
  within 1e-6, and the selection rule picks the same candidates. Only then are the weights written (exclusive).
- score (CPU): per cell, every output_end of the variant's cache scored by that cell's probe at its fold's layer.
  Gates: run means equal the cell's stored run scores within 1e-6; every planted readout matches exactly one cache row
  by (world, task, turn, call index), else stop; pre-edit readouts within 1e-3 of the cache. Then per run, post-edit
  outputs take their planted scores and the run score is the mean (run_plantslip_onpolicy.score_p78's splice). Returns
  per-task planted and unplanted run scores; the statistics are local (analyze_planted.py).
  Scorers: (a) q2's outer model of the task's q2 fold at L63; (b) the refit weights (swapped: plantslip_2e/claimc_refit_
  reply_end.npz); (c) the cell-(c) grid's saved column-0 weights at stage 4's choices.

  modal deploy experiments/2026-10-13_onpolicy_reply/run_planted.py
  uv run --no-project --python 3.12 --with "modal==1.6.1" python experiments/2026-10-13_onpolicy_reply/run_planted.py refit_b | score
"""

import hashlib
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-onpol-reply_planted_kill-2026-11-13"
ACTS, SCORE = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08", "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
LAYER_GRID, C_GRID, MAX_ITER, MAX_ITER_CAP, TOL_GATE, DRIFT_TOL = [39, 51, 63], [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 5000, 80000, 1e-6, 1e-3
OUT, CTAG = "onpolreply_planted", "onpolreply_c_bc89eeb2869c"
BASES = ["rerun_strict_segends_v2", "scaleup_canary_segends_v1", "scaleup_l2_segends_v1", "scaleup_topup_segends_v1"]
VARIANTS = {"swapped_L1": {"cache": "claimc_cache_v1", "b_records": "claimc_fb82f67348ac", "b_weights": "plantslip_2e/claimc_refit_reply_end.npz",
                           "runs": {"run1": ["2026-10-12_plantslip_p78_run1", [f"{b}_scrub1_swapsys" for b in BASES]],
                                    "run2": ["2026-10-12_plantslip_p78_run2", ["second_attack_segends_v1_scrub1_swapsys"]]}},
            "unswapped_L0": {"cache": "retrain_cache_v1/level0", "b_records": "retrain_qwen_5eb808288f50/level0/attempted",
                             "b_weights": f"{OUT}/refit_b_unswapped_L0.npz",
                             "runs": {"run1": ["2026-10-12_plantslip_p1_run1", BASES], "run2": ["2026-10-12_plantslip_p1_run2", ["second_attack_segends_v1"]]}}}

app = modal.App(APP)
acts_vol, score_vol = modal.Volume.from_name(ACTS, version=2), modal.Volume.from_name(SCORE, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scipy")
         .add_local_python_source("probe_shift").add_local_file(REPO / T196, f"/root/{T196}"))


def tol_of(C):
    return 1e-4 * min(C**0.5, 1.0)


def frozen_folds(d):  # run_claim_c.frozen_folds, unchanged
    h = lambda s: hashlib.sha256((d["salt"] + s).encode()).hexdigest()  # noqa: E731
    envs, by_env = sorted(d["eligible_mains_by_env"], key=h), {}
    for c in d["cells"]:
        by_env.setdefault(c["env"], []).append(c["main"])
    return {c["key"]: (by_env[c["env"]].index(c["main"]) + envs.index(c["env"])) % 5 for c in d["cells"]}


def staging_index(sources: list) -> dict:  # run_plantslip_onpolicy.staging_index @ main, unchanged
    call, first, run_first, planted_keys, run_of_key, counted, orig = {}, {}, {}, set(), {}, {}, {}
    for s in sources:
        ids = {}
        for line in (Path("/acts") / s / "inputs" / "attack.jsonl").read_text().splitlines():
            ep = json.loads(line)
            ids[ep["key"]] = ep["ids"]
            run_of_key[ep["key"]] = ep["run"]
            counted[ep["key"]] = not (ep.get("redrawn_because") or ep.get("attempt_kind") == "rerun")
            for c in ep["calls"]:
                orig[(c["irid"], c["call"])] = c["readout"]
                if c["readout"] is not None:
                    call[(c["irid"], c["call"])] = (ep["run"], ep["epoch"], c["readout"])
        for line in (Path("/acts") / f"{s}_plantslip" / "inputs" / "attack.jsonl").read_text().splitlines():
            ep = json.loads(line)
            a, b = ids[ep["key"]], ep["ids"]
            p0 = next(i for i in range(min(len(a), len(b))) if a[i] != b[i])
            first[ep["key"]] = p0
            planted_keys.add(ep["key"])
            cur = run_first.get(ep["run"])
            if cur is None or (ep["epoch"], p0) < cur:
                run_first[ep["run"]] = (ep["epoch"], p0)
    return {"call": call, "first": first, "run_first": run_first, "planted_keys": planted_keys, "run_of_key": run_of_key,
            "counted": counted, "orig_readout": orig}


def planted_rows(run: str, layer_of_fold: dict, fold_of: dict, ix: dict):  # run_plantslip_onpolicy.planted_rows, kinds = output_end
    import numpy as np

    root = Path("/acts") / run
    metas = [json.loads(f.read_text()) for f in sorted((root / "meta" / "attack").glob("*.json"))]
    extracted = {m["key"] for m in metas}
    keys, picks = [], []
    for m in metas:
        if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
            continue
        p0 = ix["first"][m["key"]]
        for c in m["calls"]:
            if c.get("kind", "command") == "output_end" and c.get("row") is not None:
                post = c["readout"] >= p0 or c["readout"] != ix["orig_readout"].get((c["irid"], c["call"]))
                keys.append((m["task"], c["turn"], c.get("call_index", 0), post))
                picks.append((m["chunk"], m["row_start"] + c["row"], layer_of_fold[fold_of[m["task"]]]))
    X = np.empty((len(picks), 5120), np.float16)
    for chunk, layer in sorted({(p[0], p[2]) for p in picks}):
        a = np.load(root / "by_layer" / f"L{layer}" / f"{chunk}.npy")
        idx = [i for i, p in enumerate(picks) if p[0] == chunk and p[2] == layer]
        X[idx] = a[[picks[i][1] for i in idx]]
    return keys, X, {k for k in ix["planted_keys"] - extracted if ix["counted"][k]}


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, timeout=3600, retries=0, volumes={"/score": score_vol}, scaledown_window=5)
def refit_b(choices: list, deviation: bool = False) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F

    from probe_shift import torch_probe as tp

    score_vol.reload()
    V = VARIANTS["unswapped_L0"]
    rd, cache = Path("/score") / V["b_records"], Path("/score") / V["cache"]
    cands = [(l, C) for C in C_GRID for l in LAYER_GRID]
    R = {c: dict(np.load(rd / f"L{c[0]}_output_end_C{c[1]:g}.npz")) for c in cands}
    r0 = R[cands[0]]
    consistent = all(np.array_equal(R[c][k], r0[k]) for c in cands for k in ("on_runs", "Y", "on_fold", "train"))
    ok_f = np.stack([R[c]["grad_full"] <= R[c]["tol"] for c in cands])
    ok_v = np.stack([R[c]["grad_val"] <= R[c]["tol"] for c in cands])
    unconverged = float(1 - np.concatenate([ok_f.ravel(), ok_v.ravel()]).mean())
    Vv = np.stack([np.where(ok_f[i] & ok_v[i] & np.isfinite(R[c]["val_auroc"]), R[c]["val_auroc"], -np.inf) for i, c in enumerate(cands)])
    picked = [list(cands[Vv[:, f, 0].argmax()]) for f in range(5)]
    same = bool(consistent and unconverged <= 0.05 and picked == [list(x) for x in choices])
    fold_of = frozen_folds(json.loads((Path("/root") / T196).read_text()))
    weights, per_fold, worst, oof = {}, [], 0.0, {}
    for f, (layer, C) in enumerate(choices):
        rows = json.loads((cache / f"on_rows.L{layer}.json").read_text())
        kn = np.flatnonzero([r["kind"] == "output_end" for r in rows])
        X = torch.from_numpy(np.load(cache / f"on_L{layer}.npy")[kn]).cuda()
        runs, n_run = np.unique([f"{r['world']}|{r['task']}" for r in rows], return_inverse=True)
        n_run = n_run[kn]
        r = R[(layer, C)]
        if not np.array_equal(r["on_runs"].astype(str), runs):
            raise ValueError("the stored record's runs differ from the cache's")
        r_fold = np.array([fold_of[u.split("|", 1)[1]] for u in runs])
        if not np.array_equal(r_fold, r["on_fold"]):
            raise ValueError("folds differ")
        a = np.flatnonzero((r["train"] & (r_fold != f))[n_run])
        b_ = np.flatnonzero(r_fold[n_run] == f)
        xa, xb = X[a].double(), X[b_].double()
        mean = xa.mean(0)
        std = xa.std(0, unbiased=False)
        std = std.where(std > 0, std.new_ones(()))
        Xa, Xb = (xa - mean) / std, (xb - mean) / std
        Xa.mul_(C**0.5), Xb.mul_(C**0.5)
        Ya = torch.from_numpy(r["Y"][n_run[a]]).cuda()  # all 101 columns, as the grid fit them jointly (run_plantslip_onpolicy.refit); column 0 kept
        mi, prev = MAX_ITER, None
        while True:  # as run_claim_c.grid
            W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol_of(C))
            Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
            loss = (F.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
            loss.backward()
            g = torch.maximum(Wg.grad.abs().amax(0), bg.grad.abs()).cpu().numpy()
            if g.max() <= tol_of(C) or evals < mi or evals == prev or mi * 4 > MAX_ITER_CAP:
                break
            mi, prev = mi * 4, evals
        S = tp.decision(Xb, W, b)[:, 0].cpu().numpy()
        u, inv = np.unique(n_run[b_], return_inverse=True)
        M = np.bincount(inv, weights=S) / np.bincount(inv)
        oof |= {runs[k]: float(m) for k, m in zip(u, M)}
        stored = r["on_score"][u, 0]
        fin = np.isfinite(stored)
        d = float(np.abs(M[fin] - stored[fin]).max())
        worst = max(worst, d)
        per_fold.append({"fold": f, "layer": layer, "C": C, "max_abs_diff": d, "evals": int(evals)})
        weights[f] = {"mean": mean.cpu().numpy(), "std": std.cpu().numpy(), "w": W[:, 0].cpu().numpy() * C**0.5, "b": float(b[0]), "layer": layer, "C": C}
        del X, Xa, Xb, xa, xb
        torch.cuda.empty_cache()
    out = {"picked": picked, "pinned": choices, "same_selection": same, "unconverged_share": unconverged, "per_fold": per_fold, "max_abs_diff": worst,
           "gate": bool(same and worst <= TOL_GATE), "deviation": deviation, "oof_run_scores": oof}
    if out["gate"] or (deviation and same):  # Amendment 5: a labelled deviation writes its own file
        p = Path("/score") / (V["b_weights"] if out["gate"] else V["b_weights"].replace(".npz", "_DEVIATION.npz"))
        p.parent.mkdir(exist_ok=True)
        with open(p, "xb") as fh:  # score = ((x - mean) / std) @ w + b, w already scaled by sqrt(C) (as claimc_refit_*)
            np.savez(fh, meta=json.dumps({str(f): {"layer": v["layer"], "C": v["C"], "b": v["b"]} for f, v in weights.items()}),
                     **{f"{f}|{n}": v[n] for f, v in weights.items() for n in ("mean", "std", "w")})
        score_vol.commit()
        out["weights_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


@app.function(image=image, cpu=8, memory=65536, timeout=3600, retries=0, volumes={"/acts": acts_vol.read_only(), "/score": score_vol.read_only()})
def score(variant: str, q2: dict, q2_fold_of_task: dict, a_scores: dict, c_choices: list, b_deviation: dict | None = None) -> dict:
    import numpy as np

    V, root = VARIANTS[variant], Path("/score")
    fold_of = frozen_folds(json.loads((Path("/root") / T196).read_text()))
    q2 = {k: np.asarray(v, np.float64) for k, v in q2.items()}
    bw = V["b_weights"].replace(".npz", "_DEVIATION.npz") if b_deviation is not None else V["b_weights"]  # Amendment 5
    b_ok = (root / bw).exists()  # unswapped level 0: the refit gate failed (3.4e-3 > 1e-6); scored only as Amendment 5's deviation
    Wb = np.load(root / bw) if b_ok else None
    mb = json.loads(str(Wb["meta"])) if b_ok else None
    Wc = {f: np.load(root / CTAG / variant / f"W_L{l}_C{C:g}.npz") for f, (l, C) in enumerate(c_choices)}
    cells = {  # name: (fold of a task, layer of a fold, scorer(x, fold))
        "a": (q2_fold_of_task, {f: 63 for f in range(5)},
              lambda x, f: ((x - q2[f"output_end__task_outer_{f}__mean"]) / q2[f"output_end__task_outer_{f}__std"]) @ q2[f"output_end__task_outer_{f}__w"] + q2[f"output_end__task_outer_{f}__b"]),
        "b": (fold_of, {f: mb[str(f)]["layer"] for f in range(5)} if b_ok else None, lambda x, f: ((x - Wb[f"{f}|mean"]) / Wb[f"{f}|std"]) @ Wb[f"{f}|w"] + mb[str(f)]["b"]),
        "c": (fold_of, {f: c_choices[f][0] for f in range(5)},
              lambda x, f: (((x - Wc[f]["mean"][f]) / Wc[f]["std"][f]) @ Wc[f]["W"][f]) * float(Wc[f]["sqrtC"]) + Wc[f]["b"][f]),
    }
    stored = {"a": a_scores}
    for name, rec_dir, choices in ((("b", V["b_records"], [[mb[str(f)]["layer"], mb[str(f)]["C"]] for f in range(5)]),) if b_ok else ()) + (("c", f"{CTAG}/{variant}", c_choices),):
        st = {}
        for f, (l, C) in enumerate(choices):
            r = np.load(root / rec_dir / f"L{l}_output_end_C{C:g}.npz")
            for u, fo, s in zip(r["on_runs"].astype(str), r["on_fold"], r["on_score"][:, 0]):
                if fo == f and np.isfinite(s):
                    st[u] = float(s)
        stored[name] = st if not (name == "b" and b_deviation is not None) else dict(b_deviation)  # Amendment 5: the refit's own scores
    ixs = {w: staging_index(src) for w, (run, src) in V["runs"].items()}
    out = {}
    for name, (fo, layer_of, sc) in cells.items():
        if layer_of is None:
            out[name] = {"not scored": "the refit gate failed (max 3.43e-3 > 1e-6); no weights were written (RULE.md Amendment 2.6)"}
            continue
        base = {}  # (world, task, turn, call_index) -> unplanted score
        for layer in sorted(set(layer_of.values())):
            rows = json.loads((root / V["cache"] / f"on_rows.L{layer}.json").read_text())
            X = np.load(root / V["cache"] / f"on_L{layer}.npy", mmap_mode="r")
            idx = [i for i, r in enumerate(rows) if r["kind"] == "output_end" and r["world"] in ("run1", "run2") and layer_of[fo[r["task"]]] == layer]
            for f in range(5):
                sub = [i for i in idx if fo[rows[i]["task"]] == f]
                if sub:
                    for i, s in zip(sub, sc(np.asarray(X[sub], np.float64), f)):
                        base[(rows[i]["world"], rows[i]["task"], rows[i]["order"][0], rows[i]["order"][1])] = float(s)
        runs_base = {}
        for (w, t, _, _), s in base.items():
            runs_base.setdefault(f"{w}|{t}", []).append(s)
        common = [u for u in runs_base if u in stored[name]]
        gate = max(abs(float(np.mean(runs_base[u])) - stored[name][u]) for u in common)
        res = {"base_vs_stored": {"runs": len(common), "max_abs_diff": gate, "passes": bool(gate <= TOL_GATE)}}
        if not res["base_vs_stored"]["passes"]:
            out[name] = res
            continue
        for world, (run, _src) in V["runs"].items():
            keys, X, missing = planted_rows(run, layer_of, fo, ixs[world])
            excl_tasks = sorted({k.split("/")[1] for k in {ixs[world]["run_of_key"][m] for m in missing}})
            post, noise, unmatched = {}, [], 0
            for f in range(5):
                sel = [i for i, k in enumerate(keys) if fo[k[0]] == f]
                if not sel:
                    continue
                for i, s in zip(sel, sc(X[sel].astype(np.float64), f)):
                    task, turn, ci, is_post = keys[i]
                    key = (world, task, turn, ci)
                    if key not in base:
                        unmatched += 1
                        continue
                    if is_post:
                        post[key] = float(s)
                    else:
                        noise.append(abs(float(s) - base[key]))
            if unmatched:
                raise ValueError(f"{variant} cell {name} {world}: {unmatched} planted readouts without a cache row")
            per_task = {}
            for (w, t, a1, a2), s in base.items():
                if w == world:
                    d = per_task.setdefault(t, {"unplanted": [], "planted": []})
                    d["unplanted"].append(s)
                    d["planted"].append(post.get((w, t, a1, a2), s))
            res[world] = {"per_task": {t: [float(np.mean(v["unplanted"])), float(np.mean(v["planted"]))] for t, v in per_task.items()},
                          "planted_readouts": len(post), "runs_changed": sum(1 for v in per_task.values() if v["planted"] != v["unplanted"]),
                          "excluded_tasks_failed_extraction": excl_tasks,
                          "drift": {"n": len(noise), "max": float(max(noise)) if noise else None, "passes": bool(noise) and max(noise) <= DRIFT_TOL}}
        out[name] = res
    return out


if __name__ == "__main__":
    import numpy as np

    cmd, fn = sys.argv[1], None
    s1 = json.loads((HERE / "results" / "stage1_cell_b_POSTHOC_2026-10-13_120555.json").read_text())["cells"]
    s4 = json.loads((HERE / "results" / "stage4_cell_c_POSTHOC_2026-10-13_121906.json").read_text())["cells"]
    a = json.loads((HERE / "results" / "cell_a_run_scores_POSTHOC_2026-10-13_121016.json").read_text())
    if cmd in ("refit_b", "refit_b_deviation"):
        r = modal.Function.from_name(APP, "refit_b").remote(s1["b_unswapped_L0"]["choices_real_column"], cmd == "refit_b_deviation")
        if cmd == "refit_b_deviation":
            from datetime import datetime
            from zoneinfo import ZoneInfo
            dest = HERE / "results" / f"refit_b_unswapped_L0_DEVIATION_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
            dest.open("x").write(json.dumps(r))
            print(dest.name)
        print(json.dumps({k: v for k, v in r.items() if k != "oof_run_scores"}))
    elif cmd == "score_deviation":  # Amendment 5: unswapped level 0 only, cell (b) from the deviation refit
        q2 = {k: v.tolist() for k, v in np.load(sys.argv[2]).items() if k.startswith("output_end__task_outer_")}
        dev = json.loads((HERE / "results" / sys.argv[3]).read_text())
        res = modal.Function.from_name(APP, "score").remote("unswapped_L0", q2, a["fold_of_task"], a["variants"]["unswapped_L0"]["run_scores"],
                                                            s4["c_unswapped_L0"]["choices_real_column"], dev["oof_run_scores"])
        from datetime import datetime
        from zoneinfo import ZoneInfo
        dest = HERE / "results" / f"planted_scores_b_L0_DEVIATION_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
        dest.open("x").write(json.dumps({"label": "POST HOC, Amendment 5 DEVIATION (refit within 3.43e-3)", "variants": {"unswapped_L0": {"b": res["b"]}}}))
        print(json.dumps(res["b"]["base_vs_stored"]), dest.name)
    elif cmd == "score":
        q2 = {k: v.tolist() for k, v in np.load(sys.argv[2]).items() if k.startswith("output_end__task_outer_")}
        f = modal.Function.from_name(APP, "score")
        calls = {v: f.spawn(v, q2, a["fold_of_task"], a["variants"][v]["run_scores"], s4[f"c_{v}"]["choices_real_column"]) for v in VARIANTS}
        res = {v: c.get() for v, c in calls.items()}
        from datetime import datetime
        from zoneinfo import ZoneInfo
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        dest = HERE / "results" / f"planted_scores_POSTHOC_{stamp}.json"
        dest.open("x").write(json.dumps({"label": "POST HOC (RULE.md step 3, Amendments 1-2); per-task run scores", "variants": res}))
        print(json.dumps({v: {c: {k: (x if k == "base_vs_stored" else {kk: vv for kk, vv in x.items() if kk != "per_task"}) for k, x in r.items()} for c, r in res[v].items()} for v in res})[:3000])
        print(dest.name)
