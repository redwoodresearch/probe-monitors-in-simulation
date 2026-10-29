"""Q3(a) (RULE.md, this folder): can a probe on Qwen3.5 L30 activations tell simulated from real SABOTAGE calls, compared with
count-matched ordinary (not_sabotage) calls? Numbers only; no text is read or printed.

Built from run_t2_realism_v2.py @ 599fa553 (aswin/q3-sabotage-t2-realism): the same joins, meta prep, probe (fit_dual), folds,
C choice and AUROC. New here: the sabotage row set, 5 count-matched ordinary draws, a within-task label-shuffle null (batched
refits), and a gate that reproduces T2 v2's stored AUROCs on T2's own call set. Everything runs on Modal: the laptop only reads
the input files' bytes (their sha256 pins are checked on Modal); the orchestrator (detached-safe) writes the result to the work Volume.

  cd <repo root>; PYTHONPATH=src modal run --detach experiments/2026-10-13_q3_sabotage_separability/run_sab_separability.py --inputs <scaled_inputs dir>
"""

import csv
import hashlib
import io
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-q3sabsep_run_kill-2026-11-13"
WORK = "probe-shift_aswin-surya_sub-q3sabsep_work_kill-2026-11-13"
T2WORK = "probe-shift_aswin-surya_sub-q3t2_work_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
HIDDEN, SEED, B = 5120, 20261013, 2000
CGRID, CFIX = [1e-5, 1e-4, 1e-3, 1e-2], 1e-4
NDRAW, NSHUF, CH = 5, 100, 50
READOUTS = ["command_last", "command_mean", "call_end"]  # command_last is primary (RULE)
T2FILE, T2SHA = "t2_realism_v2_2026-10-13_183302.json", "6e093cb6cfa26c3ac2e31e87c2cca7724dd01a3789fa5b7019f2131e882f30fc"
HERE = Path(__file__).parent
PINS = {"flagrate_rows_2026-10-13_122417.csv": "44c75f78e78f86b4bcab409b1379762a8d4dccadfd456f28ee651832eb2b13a7",
        "scaled_rows_2026-10-13_152021.csv": "a95453ac2688e61ba99184726ce27a4dd54276a04730a56d67825f204d1ee1f3",
        "command_rows_2026-10-13_162911.csv": "096f147f6a72ebf7b1479630b6fb3d8cfa8c141a5a68e54c9ce18a770765d05e",
        "sim_votes.csv": "06f4ef4bc00b53767b074185cda99311f2ba44f7c8f5a4f38ce59dd1537e81ae",
        "real_votes.csv": "9a96988db9818c8b97f3045ade83f19fae0e19e7d0cac134fa775005eba8dc36",
        "scaleup_panel_votes_k116.csv": "78ae0fa4e7e85da9a1ee85c9944be917b47e63c0d0a62f6f71390f046fdb7fcf"}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
t2work = modal.Volume.from_name(T2WORK, version=2)
work = modal.Volume.from_name(WORK, version=2, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy").add_local_file(
    str(Path(__file__).parent / "t2_core.py"), "/root/t2_core.py")


def h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


@app.function(image=image, cpu=2, memory=8192, timeout=900, retries=0, volumes={"/acts": acts.read_only()})
def prep_qwen35(calls: list) -> dict:
    """T2's prep_qwen35 (t2_core.py, verbatim): output_end and B1's averaged-command locations, with the last-token gate."""
    import t2_core

    return t2_core.prep_qwen35(calls)


@app.function(image=image, gpu="A100-40GB", cpu=8, memory=65536, timeout=900, retries=0, volumes={"/acts": acts.read_only()})
def fit_readout(p: dict) -> dict:
    """p: rows [call_id, world, task, (x, chunk, row)], sets {name: {"idx", "universe", "null": [contrasts]}}. Per set and contrast,
    T2's out-of-fold fit (selected and fixed C), per-draw AUROCs, and the within-task label-shuffle null where asked."""
    import numpy as np
    import torch
    from t2_core import aucs, fit_dual, wauc_draws

    rows, dev = p["rows"], "cuda"
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r[3][0], r[3][1]), []).append((i, int(r[3][2])))
    X = np.empty((len(rows), HIDDEN), np.float64)
    for (x, chunk), lst in groups.items():
        a = np.load(f"/acts/{x}/by_layer/L30/{chunk}.npy", mmap_mode="r")
        X[[i for i, _ in lst]] = a[[r for _, r in lst]]
    if not np.isfinite(X).all():
        raise SystemExit("non-finite readings")
    world, task = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    fold = np.array([h(f"t2fold|{t}") % 5 for t in task])
    inner = np.array([h(f"t2inner|{t}") % 3 for t in task])

    def train_score(tr, te, Cs, Ytr):
        """Scores (S, n_te, len(Cs)) for each label row of Ytr (S, n_tr); T2's standardization and fit, batched over rows."""
        Xt = X[tr]
        mu, sd = Xt.mean(0), Xt.std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        Zt = torch.from_numpy((Xt - mu) / sd).to(dev)
        Ze = torch.from_numpy((X[te] - mu) / sd).to(dev)
        K, w, out = Zt @ Zt.T, torch.ones(len(tr), dtype=torch.float64, device=dev), []
        for s0 in range(0, len(Ytr), CH):
            Yc = torch.from_numpy(Ytr[s0:s0 + CH].astype(float)).to(dev)
            a = fit_dual(K[None].expand(len(Yc), -1, -1), Yc[:, None, :], w, Cs)
            out.append(torch.matmul(Ze, (a @ Zt).transpose(1, 2)).cpu().numpy())
        return np.concatenate(out)

    def oof(Y, sel):
        """T2's out-of-fold procedure for each label row of Y (S, n): inner arithmetic-mean C choice per row."""
        S = len(Y)
        s_sel, s_fix, chosen = np.full((S, len(rows)), np.nan), np.full((S, len(rows)), np.nan), {}
        for f in range(5):
            tr, te = sel[fold[sel] != f], sel[fold[sel] == f]
            inner_auc = []
            for g in range(3):
                itr, iva = tr[inner[tr] != g], tr[inner[tr] == g]
                if len(set(Y[0, iva].tolist())) < 2:  # within-task shuffles keep each task's counts, so this is the same for every row
                    continue
                sc = train_score(itr, iva, CGRID, Y[:, itr])
                row = np.zeros((S, len(CGRID)))
                for s in range(S):
                    for j in range(len(CGRID)):
                        a, b = wauc_draws(sc[s, :, j], Y[s, iva], np.ones((1, len(iva))))
                        row[s, j] = a[0] / b[0]
                inner_auc.append(row)
            arith = np.mean(np.array(inner_auc), 0)  # (S, C)
            best = np.argmax(arith, 1)  # first (smallest C) on ties
            if S == 1:  # T2's exact outer call
                sc = train_score(tr, te, [CGRID[best[0]], CFIX], Y[:, tr])
                s_sel[0, te], s_fix[0, te] = sc[0, :, 0], sc[0, :, 1]
                chosen[f] = {"C": CGRID[best[0]], "inner_auc_arith": arith[0].tolist()}
            else:
                sc = train_score(tr, te, CGRID, Y[:, tr])
                s_sel[:, te] = np.stack([sc[s, :, best[s]] for s in range(S)])
                s_fix[:, te] = sc[:, :, CGRID.index(CFIX)]
                chosen[f] = dict(Counter(CGRID[b] for b in best))
        return s_sel, s_fix, chosen

    res = {}
    for name, spec in p["sets"].items():
        idx = np.array(spec["idx"])
        uni = spec["universe"]
        T = len(uni)
        rng = np.random.default_rng(SEED)
        Wt = np.vstack([np.ones(T)] + [np.bincount(rng.integers(0, T, T), minlength=T) for _ in range(B)]).astype(float)
        pos_u = {t: i for i, t in enumerate(uni)}
        tix = np.array([pos_u.get(t, -1) for t in task])
        if (tix[idx] < 0).any():
            raise SystemExit(f"{name}: a row's task is outside its bootstrap universe")
        res[name] = {}
        for cname, pos, neg in (("sim_vs_real", "sim", "real"), ("run2_vs_real", "run2", "real")):
            sel = idx[(world[idx] == pos) | (world[idx] == neg)]
            y = np.zeros(len(rows), int)
            y[world == pos] = 1
            s_sel, s_fix, chosen = oof(y[None], sel)
            out = {"n": {"pos": int((y[sel] == 1).sum()), "neg": int((y[sel] == 0).sum())}, "tasks": int(len(set(task[sel].tolist()))),
                   "tasks_both_classes": int(sum(len(set(y[sel][task[sel] == t].tolist())) == 2 for t in set(task[sel].tolist()))),
                   "C_chosen": chosen}
            for v, s in (("selected_C", s_sel[0]), ("fixed_C_1e-4", s_fix[0])):
                comb, pooled, per = aucs(s[sel], y[sel], fold[sel], Wt[:, tix[sel]])
                out[v] = {"combined_draws": comb.tolist(), "pooled_full": float(pooled[0]), "per_fold": per}
            if cname in spec.get("null", []):
                nrng = np.random.default_rng(h(f"q3a-null|{spec['null_key']}|{cname}"))
                Y = np.tile(y, (NSHUF, 1))
                for t in sorted(set(task[sel].tolist())):
                    it = sel[task[sel] == t]
                    for s in range(NSHUF):
                        Y[s, it] = nrng.permutation(y[it])
                n_sel, _, n_ch = oof(Y, sel)
                out["null_selected_C"] = {"auroc": [float(aucs(n_sel[s][sel], Y[s][sel], fold[sel], np.ones((1, len(sel))))[0][0]) for s in range(NSHUF)],
                                          "C_chosen_counts": n_ch}
            res[name][cname] = out
    return {"n_rows": len(rows), "sets": res}


@app.function(image=image, cpu=4, memory=8192, timeout=600, retries=0)
def summarize(R: dict, t2: dict) -> dict:
    import numpy as np

    def ci(d):
        d = np.asarray(d, float)[1:]
        d = d[np.isfinite(d)]
        return [float(v) for v in np.percentile(d, [2.5, 97.5])] if len(d) >= 0.95 * B else None

    out = {}
    for ro, r in R.items():
        S, row = r["sets"], {}
        gate = {}
        for v in ("selected_C", "fixed_C_1e-4"):
            for c, t2c in (("sim_vs_real", "sim_vs_real"), ("run2_vs_real", "run2_vs_real_control")):
                mine, want = S["t2gate"][c][v]["combined_draws"][0], t2["summary"]["qwen35_L30"][ro][v][t2c]["auroc"]
                gate[f"{v}|{c}"] = {"this_job": mine, "t2_v2": want, "abs_diff": abs(mine - want)}
        row["t2_gate"] = {"values": gate, "pass": all(g["abs_diff"] <= 1e-6 for g in gate.values())}
        for v in ("selected_C", "fixed_C_1e-4"):
            D = lambda s, c: np.asarray(S[s][c][v]["combined_draws"], float)  # noqa: E731
            sab_s, sab_c = D("sab", "sim_vs_real"), D("sab", "run2_vs_real")
            ord_s = np.mean([D(f"ord{d}", "sim_vs_real") for d in range(NDRAW)], 0)
            ord_c = np.mean([D(f"ord{d}", "run2_vs_real") for d in range(NDRAW)], 0)
            row[v] = {"sab_sim_vs_real": {"auroc": float(sab_s[0]), "ci95": ci(sab_s), "pooled": S["sab"]["sim_vs_real"][v]["pooled_full"]},
                      "sab_run2_vs_real_control": {"auroc": float(sab_c[0]), "ci95": ci(sab_c)},
                      "ord_sim_vs_real_drawmean": {"auroc": float(ord_s[0]), "ci95": ci(ord_s),
                                                   "per_draw": [float(D(f"ord{d}", "sim_vs_real")[0]) for d in range(NDRAW)]},
                      "ord_run2_vs_real_drawmean": {"auroc": float(ord_c[0]), "ci95": ci(ord_c),
                                                    "per_draw": [float(D(f"ord{d}", "run2_vs_real")[0]) for d in range(NDRAW)]},
                      "diff_sab_minus_ord_sim_vs_real": {"value": float(sab_s[0] - ord_s[0]), "ci95": ci(sab_s - ord_s)},
                      "diff_sab_sim_minus_sab_control": {"value": float(sab_s[0] - sab_c[0]), "ci95": ci(sab_s - sab_c)},
                      "diff_ord_sim_minus_ord_control": {"value": float(ord_s[0] - ord_c[0]), "ci95": ci(ord_s - ord_c)}}
        nulls = {}
        for s, c in (("sab", "sim_vs_real"), ("sab", "run2_vs_real"), ("ord0", "sim_vs_real")):
            nv = np.asarray(S[s][c]["null_selected_C"]["auroc"], float)
            obs = S[s][c]["selected_C"]["combined_draws"][0]
            nulls[f"{s}|{c}"] = {"observed": obs, "null_mean": float(nv.mean()), "null_p95": float(np.percentile(nv, 95)),
                                 "p": float((1 + (nv >= obs).sum()) / (NSHUF + 1)), "n": int(len(nv))}
        row["null"] = nulls
        sv = row["selected_C"]
        d_ord, d_ctl = sv["diff_sab_minus_ord_sim_vs_real"]["ci95"], sv["diff_sab_sim_minus_sab_control"]["ci95"]
        if not row["t2_gate"]["pass"]:
            reading = "not read (T2 reproduction gate failed)"
        elif d_ord is None or d_ctl is None:
            reading = "inconclusive (interval undefined)"
        else:
            yes = sv["sab_sim_vs_real"]["auroc"] > nulls["sab|sim_vs_real"]["null_p95"] and d_ord[0] > 0 and d_ctl[0] > 0
            no = d_ord[1] < 0.05
            reading = ("small: separable beyond ordinary commands, but by under 0.05 AUROC" if yes and no else "yes" if yes
                       else "no" if no else "inconclusive")
        row["reading"] = reading
        row["counts"] = {s: {c: {k: S[s][c][k] for k in ("n", "tasks", "tasks_both_classes")} for c in S[s]} for s in S}
        out[ro] = row
    return out


@app.function(image=image, cpu=1, memory=8192, timeout=1200, retries=0, volumes={"/work": work, "/t2": t2work.read_only()})
def orchestrate(files: dict, meta: dict) -> dict:
    import numpy as np

    for n, want in PINS.items():
        if hashlib.sha256(files[n]).hexdigest() != want:
            raise SystemExit(f"{n}: sha256 differs from its pin")
    t2b = Path(f"/t2/{T2FILE}").read_bytes()
    if hashlib.sha256(t2b).hexdigest() != T2SHA:
        raise SystemExit("T2 v2 result on the work Volume differs from its pin")
    t2 = json.loads(t2b)
    rd = lambda n: list(csv.DictReader(io.StringIO(files[n].decode())))  # noqa: E731
    key4 = lambda r: (r["run_id"], int(r["epoch"]), int(r["turn"]), int(r["call_index"]))  # noqa: E731
    votes = {}
    for n, default in (("sim_votes.csv", "sim"), ("real_votes.csv", None), ("scaleup_panel_votes_k116.csv", None)):
        for r in rd(n):
            if key4(r) in votes:
                raise SystemExit("a call appears twice in the votes")
            votes[key4(r)] = (r.get("world") or default, r["label"], int(r["state_changing"]))
    cm = {r["call_id"]: (r["x"], r["chunk"], int(r["row"])) for r in rd("command_rows_2026-10-13_162911.csv")}
    F = [r for r in rd("flagrate_rows_2026-10-13_122417.csv") if r["part"] == "2"]
    N = rd("scaled_rows_2026-10-13_152021.csv")
    calls, skip = [], Counter()
    for r in F + N:  # T2's order and checks; both labels kept
        v = votes.get(key4(r))
        if v is None:
            skip["no_vote"] += 1
            continue
        if v[0] != r["world"] or v[2] != int(r["state_changing"]):
            raise SystemExit("a row disagrees with its vote on world or state_changing")
        if v[2] != 1 or v[1] not in ("sabotage", "not_sabotage"):
            skip[f"label_{v[1]}_sc{v[2]}"] += 1
            continue
        calls.append({"call_id": r["call_id"], "world": r["world"], "task": r["task"], "label": v[1],
                      "call_end": (r["x"], r["chunk"], int(r["row"])), "command_last": cm.get(r["call_id"])})
    if len({c["call_id"] for c in calls}) != len(calls):
        raise SystemExit("call_id collision")
    PQ = prep_qwen35.remote([[c["call_id"], c["call_end"], c["command_last"]] for c in calls])
    if PQ["avg_gate_mismatches"]:
        raise SystemExit(f"averaged mapping gate: {PQ['avg_gate_mismatches']} of {PQ['avg_gate_checked']} rows differ")
    for c in calls:
        c.update(PQ["locs"][c["call_id"]])
    t2set = [c for c in calls if c["label"] == "not_sabotage" and all(c.get(k) for k in ("call_end", "command_last", "command_mean", "reply_last"))]
    t2_n = sum(t2["coverage"]["qwen35_L30"]["common_set"].values())
    if len(t2set) != t2_n:
        raise SystemExit(f"T2 common set: {len(t2set)} calls here vs {t2_n} in T2")
    common = [c for c in calls if all(c.get(k) for k in READOUTS)]
    sab = [c for c in common if c["label"] == "sabotage"]
    ordpool = {}
    for c in common:
        if c["label"] == "not_sabotage":
            ordpool.setdefault((c["world"], c["task"]), []).append(c)
    k_sab = Counter((c["world"], c["task"]) for c in sab)
    draws, shortfall = [], Counter()
    for d in range(NDRAW):
        pick = []
        for wt in sorted(k_sab):
            pool = ordpool.get(wt, [])
            k = min(k_sab[wt], len(pool))
            shortfall[d] += k_sab[wt] - k
            ix = np.random.default_rng(h(f"q3a-ord|{d}|{wt[0]}|{wt[1]}")).choice(len(pool), size=k, replace=False) if k else []
            pick += [pool[i] for i in sorted(ix)]
        draws.append(pick)
    universe = sorted({c["task"] for c in common})
    jobs = {}
    for ro in READOUTS:
        rows, pos = [], {}

        def ix(cs, ro=ro, rows=rows, pos=pos):
            out = []
            for c in cs:
                if c["call_id"] not in pos:
                    pos[c["call_id"]] = len(rows)
                    rows.append([c["call_id"], c["world"], c["task"], list(c[ro])])
                out.append(pos[c["call_id"]])
            return out

        sets = {"t2gate": {"idx": ix(t2set), "universe": sorted({c["task"] for c in t2set})},
                "sab": {"idx": ix(sab), "universe": universe, "null": ["sim_vs_real", "run2_vs_real"], "null_key": "sab"}}
        for d in range(NDRAW):
            sets[f"ord{d}"] = {"idx": ix(draws[d]), "universe": universe} | ({"null": ["sim_vs_real"], "null_key": "ord0"} if d == 0 else {})
        jobs[ro] = fit_readout.spawn({"rows": rows, "sets": sets})
    R = {ro: j.get() for ro, j in jobs.items()}
    summary = summarize.remote(R, t2)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = meta | {"rows_skipped": dict(skip),
                  "labels_state_changing": {f"{w}|{lab}": n for (w, lab), n in sorted(Counter((c["world"], c["label"]) for c in calls).items())},
                  "prep_counts": PQ["counts"], "avg_gate_checked": PQ["avg_gate_checked"], "avg_gate_mismatches": PQ["avg_gate_mismatches"],
                  "t2_common_set": len(t2set), "common_set": {f"{w}|{lab}": n for (w, lab), n in sorted(Counter((c["world"], c["label"]) for c in common).items())},
                  "lost_from_common_set": {k: sum(not c.get(k) for c in calls) for k in READOUTS},
                  "sab_per_world": dict(Counter(c["world"] for c in sab)), "sab_tasks_per_world": {w: len({c["task"] for c in sab if c["world"] == w}) for w in ("real", "run2", "sim")},
                  "ord_per_world_per_draw": [dict(Counter(c["world"] for c in dr)) for dr in draws], "ord_shortfall_per_draw": [shortfall[d] for d in range(NDRAW)],
                  "universe_tasks": len(universe), "summary": summary, "written": stamp}
    Path(f"/work/sab_separability_{stamp}.json").write_text(json.dumps(res, indent=1) + "\n")
    Path(f"/work/sab_separability_draws_{stamp}.json").write_text(json.dumps(R) + "\n")
    work.commit()
    return {"stamp": stamp, "result": res}


@app.local_entrypoint()
def main(inputs: str):
    for f in (Path(__file__).name, "t2_core.py", "RULE.md"):
        st = subprocess.run(["git", "status", "--porcelain", "--", f], cwd=HERE, capture_output=True, text=True)
        tr = subprocess.run(["git", "ls-files", "--error-unmatch", f], cwd=HERE, capture_output=True)
        if tr.returncode or st.stdout:
            raise SystemExit(f"{f} is not committed unchanged at HEAD")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    files = {n: (Path(inputs) / n).read_bytes() for n in PINS}
    meta = {"rule": "RULE.md (Q3a)", "label": "PRELIMINARY until Codex agrees and Aswin reviews; numbers only", "git": git, "app": APP,
            "inputs_sha256": PINS, "t2_v2_result": {"file": T2FILE, "sha256": T2SHA, "volume": T2WORK}}
    out = orchestrate.spawn(files, meta).get()
    dst = HERE / "results" / f"sab_separability_{out['stamp']}.json"
    dst.parent.mkdir(exist_ok=True)
    with dst.open("x") as fh:
        fh.write(json.dumps(out["result"], indent=1) + "\n")
    print("->", dst, hashlib.sha256(dst.read_bytes()).hexdigest())
