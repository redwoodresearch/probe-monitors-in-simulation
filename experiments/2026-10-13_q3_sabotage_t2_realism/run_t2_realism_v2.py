"""T2 v2: run_t2_realism.py @ 88c9f429 with Codex round 1's two code fixes (POST HOC, labelled; v1 results kept beside):
(1) inner C selection by the ARITHMETIC mean of inner-fold AUROCs, as RULE.md says (v1 used the pair-weighted mean); both
    objectives' choices are recorded per fold; (2) the averaged-command join is extraction-scoped, exactly as B1 (v1 pooled the key
    across extractions); cross-extraction key collisions of the v1 key are counted. Orchestration runs in a remote function so the
    app survives a lost laptop client (modal run --detach); the result is also written to a work Volume.

T2 (RULE.md, this folder): a realism probe (simulated vs first real run, and the control second vs first real run) on the
sabotage set's not_sabotage calls, at each sabotage readout. Numbers only; no text is read or printed.

Steps, all on Modal (the laptop only builds the call list from committed ID/label files and writes the returned JSON):
1. prep_qwen35 (CPU) reads the Qwen3.5 extraction metas and maps each call's call_end location to its output_end (same meta, same
   irid and call_index), and each command location to B1's averaged reading (run_grid_avg.py @ 31e4551c's join), with B1's
   last-token equality gate. prep_xread (CPU) reads the cross-read metas: command and output_end by (world, task, turn, call_index).
2. fit (A100, one per reader x readout) loads the rows, fits both contrasts out of fold (5 task folds, inner 3-fold C choice), and
   returns per-fold scores and 2,000 per-draw AUROCs for the common-set task bootstrap.
3. summarize (CPU) computes the AUROCs, intervals, paired differences and readings.

  cd <repo root>; PYTHONPATH=src modal run experiments/2026-10-13_q3_sabotage_t2_realism/run_t2_realism.py --inputs <scaled_inputs dir>
"""

import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-q3t2_realism-v2_kill-2026-11-13"
WORK = "probe-shift_aswin-surya_sub-q3t2_work_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
XACTS = "probe-shift_aswin-surya_sub-q3xread_acts_kill-2026-11-13"
AVG, HIDDEN, SEED, B = "2026-10-13_sabavg_", 5120, 20261013, 2000
CGRID, CFIX = [1e-5, 1e-4, 1e-3, 1e-2], 1e-4
XSRC = [("2026-10-13_xread_main_v1", "attack", "real"), ("2026-10-13_xread_run2_v1", "attack", "run2"), ("2026-10-13_xread_main_v1", "sim_attack@september", "sim")]
HERE = Path(__file__).parent
PINS = {"flagrate_rows_2026-10-13_122417.csv": "44c75f78e78f86b4bcab409b1379762a8d4dccadfd456f28ee651832eb2b13a7",
        "scaled_rows_2026-10-13_152021.csv": "a95453ac2688e61ba99184726ce27a4dd54276a04730a56d67825f204d1ee1f3",
        "command_rows_2026-10-13_162911.csv": "096f147f6a72ebf7b1479630b6fb3d8cfa8c141a5a68e54c9ce18a770765d05e",
        "sim_votes.csv": "06f4ef4bc00b53767b074185cda99311f2ba44f7c8f5a4f38ce59dd1537e81ae",
        "real_votes.csv": "9a96988db9818c8b97f3045ade83f19fae0e19e7d0cac134fa775005eba8dc36",
        "scaleup_panel_votes_k116.csv": "78ae0fa4e7e85da9a1ee85c9944be917b47e63c0d0a62f6f71390f046fdb7fcf"}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
xacts = modal.Volume.from_name(XACTS, version=2)
work = modal.Volume.from_name(WORK, version=2, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy")


def h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def fit_dual(K, y, w, Cs, iters=100, tol=1e-9):
    """run_grid.py's fit_dual (aswin/q3-sabotage-scaleup @ d4be80f6), with tensors created on K's device."""
    import torch
    import torch.nn.functional as F

    L, n, _ = K.shape
    C = torch.tensor(Cs, dtype=K.dtype, device=K.device)[None, :, None]
    Kb, eye = K[:, None], torch.eye(n, dtype=K.dtype, device=K.device)
    a = torch.zeros(L, len(Cs), n, dtype=K.dtype, device=K.device)

    def obj(a):
        f = (Kb @ a[..., None])[..., 0]
        return (C * w * (F.softplus(f) - y * f)).sum(-1) + 0.5 * (a * f).sum(-1)

    for _ in range(iters):
        f = (Kb @ a[..., None])[..., 0]
        p = torch.sigmoid(f)
        r = a - C * w * (y - p)
        res = r.abs().amax(-1)
        if float(res.max()) < tol:
            return a
        delta = torch.linalg.solve(eye + (C * w * p * (1 - p))[..., :, None] * Kb, r[..., None])[..., 0]
        step, j0 = torch.ones(L, len(Cs), 1, dtype=K.dtype, device=K.device), obj(a)
        for _ in range(40):
            worse = obj(a - step * delta) > j0 + 1e-12 * j0.abs()
            if not worse.any():
                break
            step = torch.where(worse[..., None], step / 2, step)
        a = a - step * delta
    raise RuntimeError(f"dual Newton did not converge: max residual {float(res.max()):.3g}")


def wauc_draws(s, y, W):
    """AUROC of scores s (n,) for labels y under each row of task-multiplicity weights W (d, n); ties count half. (d,)"""
    import numpy as np

    u, inv = np.unique(s, return_inverse=True)
    P = np.zeros((W.shape[0], len(u)))
    N = np.zeros((W.shape[0], len(u)))
    np.add.at(P.T, inv[y == 1], W[:, y == 1].T)
    np.add.at(N.T, inv[y == 0], W[:, y == 0].T)
    below = np.cumsum(N, 1) - N
    num = (P * (below + 0.5 * N)).sum(1)
    pairs = P.sum(1) * N.sum(1)
    return num, pairs


def aucs(scores, y, fold, tw):
    """Per-fold-combined (pair-weighted) and pooled AUROC per draw: tw (d, n) row weights. Returns (combined, pooled, per-fold full)."""
    import numpy as np

    num, den, per = 0, 0, {}
    for f in sorted(set(fold.tolist())):
        m = fold == f
        if len(set(y[m].tolist())) < 2:
            continue
        a, b = wauc_draws(scores[m], y[m], tw[:, m])
        num, den = num + a, den + b
        per[int(f)] = float(a[0] / b[0])
    pa, pb = wauc_draws(scores, y, tw)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den, pa / pb, per


@app.function(image=image, cpu=2, memory=8192, timeout=900, retries=0, volumes={"/acts": acts.read_only()})
def prep_qwen35(calls: list) -> dict:
    """calls: [call_id, ce(x, chunk, row), cmd(x, chunk, row) or None]. -> output_end and averaged-command locations."""
    import numpy as np

    xs = sorted({c[1][0] for c in calls} | {c[2][0] for c in calls if c[2]})
    ce_owner, oe, cmd_key, avg, pooled_keys = {}, {}, {}, {}, Counter()
    for x in xs:
        for f in sorted(Path(f"/acts/{x}/meta").glob("*/*.json")):
            m = json.loads(f.read_text())
            if m.get("chunk") is None:
                continue
            fid = f"{x}|{f.parent.name}|{f.name}"
            for c in m["calls"]:
                if c.get("row") is None or "call_index" not in c:
                    continue
                loc, kind = (x, m["chunk"], m["row_start"] + c["row"]), c.get("kind", "command")
                k = (fid, c["irid"], c["call_index"])
                if kind == "call_end":
                    ce_owner[loc] = k
                elif kind == "output_end":
                    oe.setdefault(k, []).append(loc)
                elif kind == "command":
                    cmd_key[loc] = (x, m["arm"], m["key"], c["irid"], c["call_index"], c["readout"])
        for f in sorted(Path(f"/acts/{AVG}{x}/meta").glob("*/*.json")):
            m = json.loads(f.read_text())
            for c in m["calls"]:
                k5 = (m["arm"], m["key"], c["irid"], c.get("call_index", 0), c["readout"])
                pooled_keys[k5] += 1
                avg[(x, *k5)] = (AVG + x, m["chunk"], m["row_start"] + c["row"])
    out, cnt, gate = {}, Counter(), []
    for cid, ce, cmd in calls:
        k = ce_owner.get(tuple(ce))
        lst = oe.get(k, []) if k else []
        cnt["ce_found" if k else "ce_missing"] += 1
        cnt[f"output_end_{min(len(lst), 2)}"] += 1
        a = avg.get(cmd_key.get(tuple(cmd))) if cmd else None
        cnt["cmd_mean_found" if a else "cmd_mean_missing"] += 1
        if a:
            gate.append((tuple(cmd), a))
        out[cid] = {"reply_last": list(lst[0]) if len(lst) == 1 else None, "command_mean": list(a) if a else None}
    by = {}
    for o, a in gate:
        by.setdefault((o[0], o[1], a[0], a[1]), []).append((o[2], a[2]))
    mism = 0
    for (ox, oc, ax, ac), lst in by.items():
        A0 = np.load(f"/acts/{ox}/by_layer/L30/{oc}.npy", mmap_mode="r")
        A1 = np.load(f"/acts/{ax}/last/L30/{ac}.npy", mmap_mode="r")
        mism += sum(not np.array_equal(A0[r], A1[q]) for r, q in lst)
    return {"locs": out, "counts": dict(cnt), "avg_gate_checked": len(gate), "avg_gate_mismatches": int(mism),
            "v1_pooled_key_collisions": int(sum(v > 1 for v in pooled_keys.values()))}


@app.function(image=image, cpu=2, memory=8192, timeout=900, retries=0, volumes={"/acts": xacts.read_only()})
def prep_xread() -> dict:
    loc, dup = {}, Counter()
    for x, arm, world in XSRC:
        for f in sorted(Path(f"/acts/{x}/meta/{arm}").glob("*.json")):
            m = json.loads(f.read_text())
            if m.get("redrawn_because") or m.get("attempt_kind") == "rerun" or m.get("chunk") is None:
                continue
            for c in m["calls"]:
                kind = c.get("kind", "command")
                if kind in ("command", "output_end") and c.get("row") is not None and "call_index" in c:
                    k = f"{world}|{m['task']}|{c['turn']}|{c['call_index']}"
                    r = "command_last" if kind == "command" else "reply_last"
                    if k in loc and r in loc[k]:
                        dup[r] += 1
                    loc.setdefault(k, {})[r] = [x, m["chunk"], m["row_start"] + c["row"]]
    return {"loc": loc, "duplicates": dict(dup)}


def _fit(vol: str, layer: int, rows: list) -> dict:
    """rows: [call_id, world, task, (x, chunk, row)]. Both contrasts, out of fold; returns scores and per-draw AUROCs."""
    import numpy as np
    import torch

    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r[3][0], r[3][1]), []).append((i, int(r[3][2])))
    X = np.empty((len(rows), HIDDEN), np.float64)
    for (x, chunk), lst in groups.items():
        a = np.load(f"{vol}/{x}/by_layer/L{layer}/{chunk}.npy", mmap_mode="r")
        X[[i for i, _ in lst]] = a[[r for _, r in lst]]
    if not np.isfinite(X).all():
        raise SystemExit("non-finite readings")
    world, task = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    tasks = sorted(set(task.tolist()))
    T = len(tasks)
    rng = np.random.default_rng(SEED)
    Wt = np.vstack([np.ones(T)] + [np.bincount(rng.integers(0, T, T), minlength=T) for _ in range(B)]).astype(float)
    tix = np.array([tasks.index(t) for t in task])
    fold = np.array([h(f"t2fold|{t}") % 5 for t in task])
    inner = np.array([h(f"t2inner|{t}") % 3 for t in task])
    dev = "cuda"

    def train_score(tr, te, Cs):
        Xt = X[tr]
        mu, sd = Xt.mean(0), Xt.std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        Zt = torch.from_numpy((Xt - mu) / sd).to(dev)
        Ze = torch.from_numpy((X[te] - mu) / sd).to(dev)
        yt = torch.from_numpy(y[tr].astype(float)).to(dev)
        a = fit_dual((Zt @ Zt.T)[None], yt[None, None, :], torch.ones(len(tr), dtype=torch.float64, device=dev), Cs)
        beta = a[0] @ Zt  # (c, d)
        return (Ze @ beta.T).cpu().numpy()  # (n_te, c)

    res = {}
    for name, pos, neg in (("sim_vs_real", "sim", "real"), ("run2_vs_real", "run2", "real")):
        sel = np.flatnonzero((world == pos) | (world == neg))
        y = np.zeros(len(rows), int)
        y[world == pos] = 1
        s_sel, s_fix, chosen = np.full(len(rows), np.nan), np.full(len(rows), np.nan), {}
        for f in range(5):
            tr, te = sel[fold[sel] != f], sel[fold[sel] == f]
            inner_auc, inner_num, pairs = [], np.zeros(len(CGRID)), 0
            for g in range(3):
                itr, iva = tr[inner[tr] != g], tr[inner[tr] == g]
                if len(set(y[iva].tolist())) < 2:
                    continue
                sc = train_score(itr, iva, CGRID)
                row = []
                for j in range(len(CGRID)):
                    a, b = wauc_draws(sc[:, j], y[iva], np.ones((1, len(iva))))
                    row.append(a[0] / b[0])
                    inner_num[j] += a[0]
                pairs += b[0]
                inner_auc.append(row)
            arith = np.mean(np.array(inner_auc), 0)
            best = int(np.argmax(arith))  # RULE: arithmetic mean over inner folds; argmax takes the first (smallest C) on ties
            chosen[f] = {"C": CGRID[best], "v1_pair_weighted_C": CGRID[int(np.argmax(inner_num / pairs))],
                         "inner_auc_arith": arith.tolist()}
            sc = train_score(tr, te, [CGRID[best], CFIX])
            s_sel[te], s_fix[te] = sc[:, 0], sc[:, 1]
        out = {"n": {"pos": int((y[sel] == 1).sum()), "neg": int((y[sel] == 0).sum())}, "tasks": int(len(set(task[sel].tolist()))), "C_chosen": chosen}
        for v, s in (("selected_C", s_sel), ("fixed_C_1e-4", s_fix)):
            comb, pooled, per = aucs(s[sel], y[sel], fold[sel], Wt[:, tix[sel]])
            out[v] = {"combined_draws": comb.tolist(), "pooled_full": float(pooled[0]), "per_fold": per}
        res[name] = out
    return {"n_rows": len(rows), "n_tasks": T, "contrasts": res}


@app.function(image=image, gpu="A100-40GB", cpu=8, memory=65536, timeout=1500, retries=0, volumes={"/acts": acts.read_only()})
def fit_qwen35(rows: list) -> dict:
    return _fit("/acts", 30, rows)


@app.function(image=image, gpu="A100-40GB", cpu=8, memory=65536, timeout=1500, retries=0, volumes={"/acts": xacts.read_only()})
def fit_xread(rows: list) -> dict:
    return _fit("/acts", 63, rows)


@app.function(image=image, cpu=4, memory=8192, timeout=600, retries=0)
def summarize(results: dict) -> dict:
    import numpy as np

    def ci(d):
        d = np.asarray(d, float)[1:]
        d = d[np.isfinite(d)]
        return [float(v) for v in np.percentile(d, [2.5, 97.5])] if len(d) >= 0.95 * B else None

    out = {}
    for reader, R in results.items():
        out[reader] = {}
        for ro, r in R["readouts"].items():
            row = {"n_rows": r["n_rows"], "n_tasks": r["n_tasks"]}
            for v in ("selected_C", "fixed_C_1e-4"):
                sv, cv = (np.asarray(r["contrasts"][c][v]["combined_draws"]) for c in ("sim_vs_real", "run2_vs_real"))
                a, b = ci(sv), ci(cv)
                near = bool((a and b and a[0] <= b[1] and b[0] <= a[1]) or abs(sv[0] - cv[0]) <= 0.05)
                row[v] = {"sim_vs_real": {"auroc": float(sv[0]), "ci95": a, "pooled": r["contrasts"]["sim_vs_real"][v]["pooled_full"],
                                          "per_fold": r["contrasts"]["sim_vs_real"][v]["per_fold"]},
                          "run2_vs_real_control": {"auroc": float(cv[0]), "ci95": b, "pooled": r["contrasts"]["run2_vs_real"][v]["pooled_full"],
                                                   "per_fold": r["contrasts"]["run2_vs_real"][v]["per_fold"]},
                          "paired_diff_sim_minus_control": {"value": float(sv[0] - cv[0]), "ci95": ci(sv - cv)},
                          "reading": "no world signal at this readout" if near else "world signal present at this readout"}
            for c in ("sim_vs_real", "run2_vs_real"):
                row[f"n_{c}"] = r["contrasts"][c]["n"] | {"tasks": r["contrasts"][c]["tasks"], "C_chosen": r["contrasts"][c]["C_chosen"]}
            out[reader][ro] = row
        if "reply_last" in R["readouts"]:
            rep = np.asarray(R["readouts"]["reply_last"]["contrasts"]["sim_vs_real"]["selected_C"]["combined_draws"])
            out[reader]["reply_minus_command_sim_vs_real"] = {
                ro: {"value": float(rep[0] - np.asarray(r["contrasts"]["sim_vs_real"]["selected_C"]["combined_draws"])[0]),
                     "ci95": ci(rep - np.asarray(r["contrasts"]["sim_vs_real"]["selected_C"]["combined_draws"]))}
                for ro, r in R["readouts"].items() if ro != "reply_last"}
    return out


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@app.function(image=image, cpu=1, memory=4096, timeout=3000, retries=0, volumes={"/work": work})
def orchestrate(calls: list, meta: dict) -> dict:
    """Runs remotely (detached-safe): prep, fits, summary; writes the result and the draws to the work Volume."""
    pq, px = prep_qwen35.spawn([[c["call_id"], c["call_end"], c["command_last"]] for c in calls]), prep_xread.spawn()
    PQ, PX = pq.get(), px.get()
    if PX["duplicates"]:
        raise SystemExit(f"cross-read metas hold duplicate keys: {PX['duplicates']}")
    if PQ["avg_gate_mismatches"]:
        raise SystemExit(f"averaged mapping gate: {PQ['avg_gate_mismatches']} of {PQ['avg_gate_checked']} last-token rows differ")
    q35 = {c["call_id"]: {"call_end": c["call_end"], "command_last": c["command_last"], **PQ["locs"][c["call_id"]]} for c in calls}
    xr = {c["call_id"]: PX["loc"].get(f"{c['world']}|{c['task']}|{c['turn']}|{c['call_index']}", {}) for c in calls}
    readers = {"qwen35_L30": (q35, ["call_end", "command_last", "command_mean", "reply_last"], fit_qwen35),
               "qwen3-32b_L63": (xr, ["command_last", "reply_last"], fit_xread)}
    coverage, jobs = {}, {}
    for name, (L, ros, fn) in readers.items():
        have = [c for c in calls if all(L[c["call_id"]].get(ro) for ro in ros)]
        coverage[name] = {"common_set": dict(Counter(c["world"] for c in have)), "per_readout_missing": {ro: sum(not L[c["call_id"]].get(ro) for c in calls) for ro in ros}}
        jobs[name] = {ro: fn.spawn([[c["call_id"], c["world"], c["task"], list(L[c["call_id"]][ro])] for c in have]) for ro in ros}
    results = {name: {"readouts": {ro: j.get() for ro, j in J.items()}} for name, J in jobs.items()}
    summary = summarize.remote(results)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = meta | {"prep_qwen35_counts": PQ["counts"], "avg_gate_checked": PQ["avg_gate_checked"], "avg_gate_mismatches": PQ["avg_gate_mismatches"],
                  "v1_pooled_key_collisions": PQ["v1_pooled_key_collisions"], "xread_duplicates": PX["duplicates"], "coverage": coverage,
                  "C_choices": {n: {ro: {c: r["contrasts"][c]["C_chosen"] for c in r["contrasts"]} for ro, r in R["readouts"].items()} for n, R in results.items()},
                  "summary": summary, "written": stamp}
    Path(f"/work/t2_realism_v2_{stamp}.json").write_text(json.dumps(res, indent=1) + "\n")
    Path(f"/work/t2_realism_v2_draws_{stamp}.json").write_text(json.dumps(results) + "\n")
    work.commit()
    return {"stamp": stamp, "result": res}


@app.local_entrypoint()
def main(inputs: str):
    for f in (Path(__file__).name, "RULE.md"):
        st = subprocess.run(["git", "status", "--porcelain", "--", f], cwd=HERE, capture_output=True, text=True)
        tr = subprocess.run(["git", "ls-files", "--error-unmatch", f], cwd=HERE, capture_output=True)
        if tr.returncode or st.stdout:
            raise SystemExit(f"{f} is not committed unchanged at HEAD")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    I = Path(inputs)
    for n, want in PINS.items():
        if sha(I / n) != want:
            raise SystemExit(f"{n}: sha256 differs from its pin")
    key4 = lambda r: (r["run_id"], int(r["epoch"]), int(r["turn"]), int(r["call_index"]))  # noqa: E731
    votes = {}
    for p, default in ((I / "sim_votes.csv", "sim"), (I / "real_votes.csv", None), (I / "scaleup_panel_votes_k116.csv", None)):
        for r in csv.DictReader(open(p)):
            if key4(r) in votes:
                raise SystemExit("a call appears twice in the votes")
            votes[key4(r)] = (r.get("world") or default, r["label"], int(r["state_changing"]))
    cm = {r["call_id"]: (r["x"], r["chunk"], int(r["row"])) for r in csv.DictReader(open(I / "command_rows_2026-10-13_162911.csv"))}
    F = [r for r in csv.DictReader(open(I / "flagrate_rows_2026-10-13_122417.csv")) if r["part"] == "2"]
    N = list(csv.DictReader(open(I / "scaled_rows_2026-10-13_152021.csv")))
    calls, skip = [], Counter()
    for r in F + N:
        v = votes.get(key4(r))
        if v is None:
            skip["no_vote"] += 1
            continue
        if v[0] != r["world"] or v[2] != int(r["state_changing"]):
            raise SystemExit("a row disagrees with its vote on world or state_changing")
        if v[2] != 1 or v[1] != "not_sabotage":
            skip[f"label_{v[1]}_sc{v[2]}"] += 1
            continue
        calls.append({"call_id": r["call_id"], "world": r["world"], "task": r["task"], "turn": int(r["turn"]), "call_index": int(r["call_index"]),
                      "call_end": (r["x"], r["chunk"], int(r["row"])), "command_last": cm.get(r["call_id"])})
    if len({c["call_id"] for c in calls}) != len(calls):
        raise SystemExit("call_id collision")
    meta = {"rule": "RULE.md (T2), code fixes POST HOC (Codex round 1)", "label": "PRELIMINARY until Codex agrees; numbers only", "git": git,
            "app": APP, "inputs_sha256": PINS, "calls_considered": len(calls), "rows_skipped": dict(skip)}
    out = orchestrate.spawn(calls, meta).get()
    dst = HERE / "results" / f"t2_realism_v2_{out['stamp']}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out["result"], indent=1) + "\n")
    print("->", dst, sha(dst))
