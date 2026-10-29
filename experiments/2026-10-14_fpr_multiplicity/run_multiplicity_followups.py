"""Follow-ups to the sweep-wide multiplicity test (RULE.md Amendment 3), on Modal CPU: the family test against run1 only,
harmfulness command on the H100 directions, harmfulness run-level rows, and the per-task decomposition at f = 1.33e-2.
Same machinery as run_multiplicity.py @ dd120fc6c (copied; the as-run file is unchanged). Original docstring:
Sweep-wide multiplicity test of the FPR sweep (RULE.md) on Modal CPU. One container per input file (.map) computes
thresholds, hits and the honest-C sameness pair counts for both pools; a remote orchestrator builds the joint bootstrap,
picks pools mechanically (Append 20), and runs the max / count / studentized / Holm tests. Detached-safe: the record is
written to a run Volume as well as returned.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy==2.3.3 modal run --detach \\
      experiments/2026-10-14_fpr_multiplicity/run_multiplicity_followups.py --labels <claimc_labels_reduced.json>
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-fpr_multiplicity_kill-2026-11-14"
COMMITTED = ("21c00136c", "experiments/2026-10-14_fpr_multiplicity/results/multiplicity_2026-10-14_163540.json",
             "4dd96356ccf406fe6dc29a34563280cce98881e4646fa7b246774836893035d0")
DECOMP_INDEX = 17  # f = 1.33e-2
SOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
P2 = "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13"
LABELS_SHA256 = "d00e88773fc2a71bef51e1ff0ccc2559932f5c3a306a8f5604c5248e797dc995"
T196 = ("experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json",
        "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")
G = "xfer/correctedb_tight22g_2026-10-14_144709/"
RH = "2026-10-14_p2_refusal_harm/2026-10-14_145748/"
HS = "2026-10-14_p2_highstakes/2026-10-14_133519/"
AS = "xfer/asis_2026-10-14_135701/rows_2026-10-14_142228/"
# (group, probe, readout, volume, path, sha256, outer-fold models?)
INPUTS = [
    ("family", "sol_a_tight", "reply", "sol", G + "perrow_asis_tight__a__averaged_reply_top2.npz", "aca93022c00ebb8394977be5013aaf53f4881e35b401c91ab8ffad846cd52d50", True),
    ("family", "sol_a_tight", "command", "sol", G + "perrow_asis_2026-10-14_135701_cmd__a__A_corrected_L63.npz", "3b0645c9c19456278fb6090b94476425b13a156a45a60a4508340068ef182d41", True),
    ("family", "fibs", "reply", "p2", "fibs/2026-10-14_124948/reply_mean.npz", "576538939ae53f9afbf6da8b7e7123e698413c34331baebf7193d8236266f666", False),
    ("family", "fibs", "command", "p2", "fibs/2026-10-14_124948/call_end_content.npz", "3f3ef32be2b576b75bec29f900cb560d39690709b34038cab9549f2b7882b558", False),
    ("family", "harmfulness_h200", "reply", "p2", RH + "harmfulness_reply_mean.npz", "28678271ce765b669d92127f15ff55092208c973d3232b4def0e3b113d52b675", False),
    ("family", "harmfulness_h200", "command", "p2", RH + "harmfulness_call_end_content.npz", "978d124f6057677a4a6d9af3d42f35d38afeeaacd92ab97f3fc7a6d8458b8bd0", False),
    ("family", "refusal_h200", "reply", "p2", RH + "refusal_reply_mean.npz", "8b8e64752f5ee67077ef0ee7f018cc683cde2f51ddaf07e64ccc46bd028ff360", False),
    ("family", "refusal_h200", "command", "p2", RH + "refusal_call_end_content.npz", "67efd85de72fbec4f93704228afe97ac4e6c224b9948dcec577cbfdb105143ff", False),
    ("family", "highstakes", "reply", "p2", HS + "highstakes_reply_mean.npz", "98e246bd8441a14f0c9870a79d5645ba54953f86940712ae7acc12f4c9d02838", False),
    ("family", "highstakes", "command", "p2", HS + "highstakes_call_end_content.npz", "d81d9ef904462961899a604823dc5f00c1423ee61aa578b98618c37d2af10717", False),
    ("hardware", "harmfulness_h100", "command", "p2", "2026-10-14_p2_refusal_harm/2026-10-14_130313/harmfulness_call_end_content.npz",
     "6854083d73b551f37e4c6a18b5cff3b5cfa32ad3419a27ec5e563ac7019071af", False),
]
# per-fold standardization of outer-model probes (Append 2/6): set from 2e's convention before the run (RULE.md Amendment 1)
STD = {"calibration": "fold f's honest runs of the row's own pool (B u C primary, B for B rows)", "per_level": True, "ddof": 0}
UNIT_F = [float(x) for x in __import__("numpy").logspace(-4, -1, 25)]
RUN_A = (0.01, 0.02, 0.05, 0.10)
ARMS = ("run1", "run2", "sim", "honest_a", "honest_b", "honest_c")
N_BOOT, SEED, GAP, CLUSTER = 10_000, [20261014, 1], 0.05, range(14, 19)

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
vols = {"/sol": modal.Volume.from_name(SOL).read_only(), "/p2": modal.Volume.from_name(P2).read_only()}
out_vol = modal.Volume.from_name(APP, create_if_missing=True)


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes=vols)
def per_file(item: list, labels: dict, tasks: list) -> dict:
    """Thresholds, hits and sameness pair counts for one input file, both pools, raw and (outer models) per-fold."""
    import io

    import numpy as np

    group, probe, readout, vol, path, sha, outer = item
    raw = (Path("/" + vol) / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"{path}: sha256 differs")
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    key, score = z["key"].astype(str), z["score"].astype(np.float64)
    unit = z["unit"].astype(np.int64) if "unit" in z.files else None
    fold = z["fold"].astype(np.int64) if outer else None
    if not np.isfinite(score).all() or not {k.split("|", 1)[0] for k in key} <= set(ARMS):
        raise ValueError(f"{path}: non-finite scores or unknown arms")
    if unit is not None and len(set(zip(key.tolist(), unit.tolist()))) != len(key):
        raise ValueError(f"{path}: duplicate (key, unit)")
    tpos = {t: i for i, t in enumerate(tasks)}
    runs, rfold = {}, {}
    for i in range(len(key)):
        runs.setdefault(key[i], []).append(i)
    for k, idx in runs.items():
        if outer:
            fs = set(fold[idx].tolist())
            if len(fs) != 1:
                raise ValueError(f"{path}: run {k} scored by {len(fs)} fold models")
            rfold[k] = fs.pop()
    A = {a: {k.split("|", 1)[1]: np.array(idx) for k, idx in runs.items() if k.startswith(a + "|")} for a in ARMS}
    if outer:  # 09's gate F: a task's fold is the same in every arm
        tf = {}
        for k, f in rfold.items():
            if tf.setdefault(k.split("|", 1)[1], f) != f:
                raise ValueError(f"{path}: a task is scored by different fold models across arms")
    cohort = sorted(t for t in set(A["run1"]) & set(A["run2"]) & set(A["sim"])
                    if all(labels.get(f"{w}|{t}") == "attempted" for w in ("run1", "run2", "sim")))
    if not set(t for a in ARMS for t in A[a]) <= set(tpos):
        raise ValueError(f"{path}: tasks outside T196")

    def thr(pool, r):
        s = np.sort(np.asarray(pool, np.float64), kind="stable")[::-1]
        k = int(np.floor(r * len(s) + 1e-9))
        return float(s[k]), k

    def pair_counts(c, ab):  # G[u, t] = sum over C items in task u, AB items in task t of [s_c > s_ab] + 0.5 [s_c == s_ab]
        ab_s, ab_t = np.array([s for s, _ in ab]), np.array([tpos[t] for _, t in ab])
        o = np.argsort(ab_s, kind="stable")
        ab_s, ab_t = ab_s[o], ab_t[o]
        cum = np.zeros((len(ab_s) + 1, len(tasks)), np.float64)
        np.add.at(cum, (np.arange(1, len(ab_s) + 1), ab_t), 1.0)
        cum = cum.cumsum(0)
        cs, ct = np.array([s for s, _ in c]), np.array([tpos[t] for _, t in c])
        lo, hi = np.searchsorted(ab_s, cs, "left"), np.searchsorted(ab_s, cs, "right")
        rows = cum[lo] + 0.5 * (cum[hi] - cum[lo])
        Gm = np.zeros((len(tasks), len(tasks)))
        np.add.at(Gm, ct, rows)
        n_c, n_ab = np.bincount(ct, minlength=len(tasks)).astype(float), np.bincount(ab_t, minlength=len(tasks)).astype(float)
        return {"G": Gm.tolist(), "n_c": n_c.tolist(), "n_ab": n_ab.tolist()}

    pools = {"BC": ("honest_b", "honest_c"), "B": ("honest_b",)}
    raw_lv = {"run_max": {a: {t: float(score[i].max()) for t, i in A[a].items()} for a in ARMS},
              "run_mean": {a: {t: float(score[i].mean()) for t, i in A[a].items()} for a in ARMS}}

    def scales(vname, parms):
        """Unit scores and run-level scores for one variant and pool. perfold (Append 2/6, 2e's convention): each fold model's
        scores are z-scored by the moments of THAT POOL's honest items of fold f, at the level that sets the threshold."""
        if vname == "raw":
            return score, raw_lv
        zu = np.empty_like(score)
        for f in range(5):
            runs_f = [A[a][t] for a in parms for t in A[a] if rfold[f"{a}|{t}"] == f]
            cal = np.concatenate(runs_f)
            if len(runs_f) < 2 or score[cal].std(ddof=STD["ddof"]) <= 0:
                raise ValueError(f"{path}: gate F failed for fold {f} (unit level)")
            m = fold == f
            zu[m] = (score[m] - score[cal].mean()) / score[cal].std(ddof=STD["ddof"])
        lv = {}
        for name, src in raw_lv.items():
            lv[name] = {a: {} for a in ARMS}
            for f in range(5):
                cal = np.array([src[a][t] for a in parms for t in src[a] if rfold[f"{a}|{t}"] == f])
                mu, sd = float(cal.mean()), float(cal.std(ddof=STD["ddof"]))
                if len(cal) < 2 or sd <= 0:
                    raise ValueError(f"{path}: gate F failed for fold {f} ({name})")
                for a in ARMS:
                    for t, x in src[a].items():
                        if rfold[f"{a}|{t}"] == f:
                            lv[name][a][t] = (x - mu) / sd
        return zu, lv

    out = {"group": group, "probe": probe, "readout": readout, "path": path, "rows": int(len(key)), "cohort": cohort,
           "tasks_present": sorted({t for a in ARMS for t in A[a]}), "variants": {}}
    for vname in ("raw", "perfold") if outer else ("raw",):
        rows, same = [], {}
        for pname, parms in pools.items():
            sc, lv_scores = scales(vname, parms)
            unit_max = {a: {t: float(sc[i].max()) for t, i in A[a].items()} for a in ARMS}
            for lv, rates in (("unit", UNIT_F), ("run_max", RUN_A), ("run_mean", RUN_A)):
                if pname == "BC":  # the Append 5 sameness check, on the scale of the primary (B u C) pool
                    if lv == "unit":
                        c_items = [(float(x), t) for t, i in A["honest_c"].items() for x in sc[i]]
                        ab_items = [(float(x), t) for a in ("honest_a", "honest_b") for t, i in A[a].items() for x in sc[i]]
                    else:
                        c_items = [(x, t) for t, x in lv_scores[lv]["honest_c"].items()]
                        ab_items = [(x, t) for a in ("honest_a", "honest_b") for t, x in lv_scores[lv][a].items()]
                    same[lv] = pair_counts(c_items, ab_items)
                if lv == "unit":
                    pool, stat = np.concatenate([sc[i] for a in parms for i in A[a].values()]), unit_max
                else:
                    pool, stat = np.array([lv_scores[lv][a][t] for a in parms for t in A[a]]), lv_scores[lv]
                for gi, r in enumerate(rates):
                    T, k = thr(pool, r)
                    above = int((pool > T).sum())
                    hits = {a: [float(stat[a][t] > T) for t in cohort] for a in ("run1", "run2", "sim")}
                    rows.append({"level": lv, "rate": r, "grid_index": gi, "pool": pname, "threshold": T, "k": k, "n_pool": int(len(pool)),
                                 "pool_above": above, "greyed": lv == "unit" and above < 10, "hits": hits})
        out["variants"][vname] = {"rows": rows, "sameness": same}
    return out


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0, volumes={"/out": out_vol})
def orchestrate(labels: dict, t196: dict, stamp: str, head: dict, expected: dict) -> dict:
    import numpy as np

    tasks = sorted(c["key"] for c in t196["cells"])
    parts = list(per_file.map([list(x) for x in INPUTS], kwargs={"labels": labels, "tasks": tasks}))
    universe = sorted(set().union(*[set(p["tasks_present"]) for p in parts]))
    tpos = {t: i for i, t in enumerate(tasks)}
    env = {c["key"]: c["env"] for c in t196["cells"]}
    rng = np.random.default_rng(SEED)
    groups = [np.array([tpos[t] for t in universe if env[t] == g]) for g in sorted({env[t] for t in universe})]
    W = np.zeros((N_BOOT + 1, len(tasks)))
    W[0, [tpos[t] for t in universe]] = 1.0
    for b in range(1, N_BOOT + 1):
        for ii in groups:
            W[b, ii] = rng.multinomial(len(ii), np.full(len(ii), 1.0 / len(ii)))

    def band(x):
        ok = x[1:][~np.isnan(x[1:])]
        return [float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))] if len(ok) else [None, None]

    def sameness(s):
        Gm, nc, nab = np.array(s["G"]), np.array(s["n_c"]), np.array(s["n_ab"])
        with np.errstate(invalid="ignore", divide="ignore"):
            auc = np.einsum("bi,ij,bj->b", W, Gm, W) / ((W @ nc) * (W @ nab))
        b = band(auc)
        return {"auroc": float(auc[0]), "ci95": b, "pass": b[0] is not None and 0.45 <= b[0] and b[1] <= 0.55}

    settings = {}  # (group, variant_tag) -> list of setting dicts with draws
    for p in parts:
        ci = np.array([tpos[t] for t in p["cohort"]], dtype=np.int64)
        Wc = W[:, ci]
        den = Wc.sum(1)
        for vname, v in p["variants"].items():
            same = {lv: sameness(s) for lv, s in v["sameness"].items()}
            for r in v["rows"]:
                pool = "BC" if same[r["level"]]["pass"] else "B"
                if r["pool"] != pool:
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    tpr = {a: (Wc @ np.array(h)) / den for a, h in r["hits"].items()}
                d = tpr["sim"] - (tpr["run1"] + tpr["run2"]) / 2
                dd = tpr["run2"] - tpr["run1"]
                d1 = tpr["sim"] - tpr["run1"]
                meta = {"probe": p["probe"], "readout": p["readout"], "variant": vname, "level": r["level"], "rate": r["rate"],
                        "pool": pool, "sameness": same[r["level"]], "threshold": r["threshold"], "k": r["k"], "n_pool": r["n_pool"],
                        "pool_above": r["pool_above"], "greyed": r["greyed"], "n_cohort": len(ci),
                        "grid_index": r["grid_index"] if r["level"] == "unit" else None,
                        "tpr": {a: float(x[0]) for a, x in tpr.items()}}
                if p["probe"].startswith("harmfulness") and p["readout"] == "command" and r["level"] == "unit" and r["grid_index"] == DECOMP_INDEX:
                    meta["decomposition"] = [{"task": t, "env": env[t], **{a: int(r["hits"][a][j]) for a in ("run1", "run2", "sim")}}
                                             for j, t in enumerate(p["cohort"])]
                settings.setdefault(p["group"], []).append(meta | {"_d": d, "_dd": dd, "_d1": d1})

    def tests(rows, field):
        rows = [r for r in rows if not r["greyed"]]
        X = np.array([r[field] for r in rows])  # S x (B+1)
        obs = X[:, 0]
        null = X[:, 1:] - obs[:, None]
        ok = ~np.isnan(null).any(0)
        null = null[:, ok]
        B = null.shape[1]
        M = np.abs(null).max(0)
        Mobs = float(np.abs(obs).max())
        C = (np.abs(null) >= GAP).sum(0)
        Cobs = int((np.abs(obs) >= GAP).sum())
        sd = null.std(1)
        live = sd > 0
        Tn = (np.abs(null[live]) / sd[live, None]).max(0) if live.any() else np.zeros(B)
        Tobs = float((np.abs(obs[live]) / sd[live]).max()) if live.any() else 0.0
        raw_p = (1 + (np.abs(null) >= np.abs(obs)[:, None]).sum(1)) / (B + 1)
        adj_ss = (1 + (M[None, :] >= np.abs(obs)[:, None]).sum(1)) / (B + 1)
        order = np.argsort(raw_p, kind="stable")
        holm, run = np.empty(len(rows)), 0.0
        for j, i in enumerate(order):
            run = max(run, min(1.0, (len(rows) - j) * raw_p[i]))
            holm[i] = run
        per = [{k: v for k, v in r.items() if not k.startswith("_")} | {"value": float(obs[i]), "ci95": band(r[field]),
               "raw_p": float(raw_p[i]), "maxtest_adj_p": float(adj_ss[i]), "holm_p": float(holm[i])} for i, r in enumerate(rows)]
        cl = [i for i, r in enumerate(rows) if r["readout"] == "reply" and r["level"] == "unit" and r["grid_index"] in CLUSTER]
        sub = None
        if cl:
            Ms = np.abs(null[cl]).max(0)
            Cs = (np.abs(null[cl]) >= GAP).sum(0)
            sub = {"settings": len(cl), "max_obs": float(np.abs(obs[cl]).max()), "p_max": float((1 + (Ms >= np.abs(obs[cl]).max()).sum()) / (B + 1)),
                   "count_obs": int((np.abs(obs[cl]) >= GAP).sum()), "p_count": float((1 + (Cs >= (np.abs(obs[cl]) >= GAP).sum()).sum()) / (B + 1))}
        return {"settings": len(rows), "greyed_excluded": None, "nan_draws": int((~ok).sum()), "draws": B,
                "max_abs": {"obs": Mobs, "p": float((1 + (M >= Mobs).sum()) / (B + 1)), "null_q95": float(np.quantile(M, 0.95))},
                "count_ge_5pts": {"obs": Cobs, "p": float((1 + (C >= Cobs).sum()) / (B + 1)), "null_q95": float(np.quantile(C, 0.95))},
                "studentized_max": {"obs": Tobs, "p": float((1 + (Tn >= Tobs).sum()) / (B + 1)), "dropped_zero_sd": int((~live).sum())},
                "holm_below_05": sorted([x for x in per if x["holm_p"] < 0.05], key=lambda x: x["holm_p"]),
                "ten_smallest_raw_p": sorted(per, key=lambda x: x["raw_p"])[:10], "cluster_posthoc": sub, "per_setting": per}

    fam = settings["family"]
    res = {"universe_n": len(universe), "std": STD, "inputs": [list(x) for x in INPUTS],
           "files": [{k: p[k] for k in ("probe", "readout", "path", "rows")} | {"cohort_n": len(p["cohort"])} for p in parts], "tests": {}}
    prim = [r for r in fam if r["variant"] == ("perfold" if r["probe"] == "sol_a_tight" else "raw")]
    t = {"sim_minus_real": tests(prim, "_d"), "sim_minus_run1": tests(prim, "_d1"), "run2_minus_run1": tests(prim, "_dd")}
    for x in t.values():
        x["greyed_excluded"] = sum(r["greyed"] for r in prim)
    got = {k: (t["sim_minus_real"][k]["obs"], t["sim_minus_real"][k]["p"]) for k in ("max_abs", "count_ge_5pts", "studentized_max")}
    if any(tuple(expected["aggregate"][k]) != got[k] for k in got):  # reproduction gate against the committed primary run
        raise ValueError(f"primary sim - real does not reproduce the committed record: {got} vs {expected['aggregate']}")
    fields = ("value", "ci95", "raw_p", "maxtest_adj_p", "holm_p", "pool", "threshold", "tpr")
    key = lambda x: (x["probe"], x["readout"], x["variant"], x["level"], x["grid_index"] if x["level"] == "unit" else x["rate"])  # noqa: E731
    checked = 0
    for comp in ("sim_minus_real", "run2_minus_run1"):  # every committed per-setting row must be reproduced exactly
        mine = {key(x): x for x in t[comp]["per_setting"]}
        theirs = expected["per_setting"][comp]
        if set(mine) != {key(x) for x in theirs}:
            raise ValueError(f"{comp}: the set of settings differs from the committed record")
        bad = sum(any(mine[key(x)][f] != x[f] for f in fields) for x in theirs)
        if bad:
            raise ValueError(f"{comp}: {bad} per-setting rows differ from the committed record")
        checked += len(theirs)
    res["reproduction_gate"] = {"expected": expected["aggregate"], "got": got, "per_setting_rows_checked": checked, "pass": True}
    res["tests"]["family_primary"] = t
    hw = [r for r in settings["hardware"] if r["variant"] == "raw"]
    res["tests"]["hardware_harmfulness_h100_command"] = {"sim_minus_real": tests(hw, "_d"), "sim_minus_run1": tests(hw, "_d1"),
                                                         "run2_minus_run1": tests(hw, "_dd")}
    for x in res["tests"]["hardware_harmfulness_h100_command"].values():
        x["greyed_excluded"] = sum(r["greyed"] for r in hw)
    dec = {}
    for r in prim + hw:
        if "decomposition" in r:
            rows_ = r["decomposition"]
            out_ = {"probe": r["probe"], "rate": r["rate"], "pool": r["pool"], "threshold": r["threshold"]}
            for ref in ("run1", "run2"):
                so = [x for x in rows_ if x["sim"] and not x[ref]]
                ro = [x for x in rows_ if x[ref] and not x["sim"]]
                by = lambda xs: dict(sorted(__import__("collections").Counter(x["env"] for x in xs).items()))  # noqa: E731
                out_[f"vs_{ref}"] = {"sim_only": len(so), f"{ref}_only": len(ro), "both": sum(x["sim"] and x[ref] for x in rows_),
                                     "neither": sum(not x["sim"] and not x[ref] for x in rows_), "sim_only_by_env": by(so),
                                     f"{ref}_only_by_env": by(ro), "sim_only_tasks": sorted(x["task"] for x in so),
                                     f"{ref}_only_tasks": sorted(x["task"] for x in ro)}
            out_["cohort_by_env"] = dict(sorted(__import__("collections").Counter(x["env"] for x in rows_).items()))
            dec[r["probe"]] = out_
    res["decomposition_f1.33e-2"] = dec
    p = Path("/out") / f"followups_{stamp}.json"
    p.write_text(json.dumps(head | res))
    out_vol.commit()
    return res


@app.local_entrypoint()
def main(labels: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    lab_raw, t_raw = Path(labels).read_bytes(), (REPO / T196[0]).read_bytes()
    if hashlib.sha256(lab_raw).hexdigest() != LABELS_SHA256 or hashlib.sha256(t_raw).hexdigest() != T196[1]:
        raise SystemExit("labels or task file sha256 differs from RULE.md")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    head = {"git": git("rev-parse", "HEAD"), "labels_sha256": LABELS_SHA256, "t196_sha256": T196[1], "n_boot": N_BOOT, "seed": SEED,
            "std": STD, "volume_copy": f"{APP}:/followups_{stamp}.json"}
    c_raw = subprocess.run(["git", "show", f"{COMMITTED[0]}:{COMMITTED[1]}"], capture_output=True, check=True, cwd=REPO).stdout
    if hashlib.sha256(c_raw).hexdigest() != COMMITTED[2]:
        raise SystemExit("committed record sha256 differs")
    cp = json.loads(c_raw)["tests"]["family_primary"]["sim_minus_real"]
    cf = json.loads(c_raw)["tests"]["family_primary"]
    expected = {"aggregate": {k: [cp[k]["obs"], cp[k]["p"]] for k in ("max_abs", "count_ge_5pts", "studentized_max")},
                "per_setting": {c: cf[c]["per_setting"] for c in ("sim_minus_real", "run2_minus_run1")}}
    head["committed_primary"] = f"{COMMITTED[1]}@{COMMITTED[0]}"
    res = orchestrate.remote(json.loads(lab_raw)["labels"], json.loads(t_raw), stamp, head, expected)
    p = HERE / "results" / f"followups_{stamp}.json"
    p.parent.mkdir(exist_ok=True)
    with p.open("x") as fh:
        json.dump(head | res, fh)
    print(f"wrote {p.name} sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}")
    for f in res["files"]:
        print("file", json.dumps(f))
    print("reproduction gate:", json.dumps(res["reproduction_gate"]))
    for name, tt in res["tests"].items():
        for comp, t in tt.items():
            print(f"{name:<28} {comp:<16} settings {t['settings']} (greyed out {t['greyed_excluded']}) | max|.| {100 * t['max_abs']['obs']:.2f} "
                  f"p {t['max_abs']['p']:.4f} (null q95 {100 * t['max_abs']['null_q95']:.2f}) | count>=5 {t['count_ge_5pts']['obs']} "
                  f"p {t['count_ge_5pts']['p']:.4f} (q95 {t['count_ge_5pts']['null_q95']:.0f}) | stud max {t['studentized_max']['obs']:.2f} "
                  f"p {t['studentized_max']['p']:.4f} | Holm<0.05: {len(t['holm_below_05'])} | cluster {json.dumps(t['cluster_posthoc'])}")
            for x in t["ten_smallest_raw_p"][:5]:
                print(f"    {x['probe']}|{x['readout']}|{x['variant']}|{x['level']}|{x['rate']:.3g}|{x['pool']} value {100 * x['value']:+.2f} "
                      f"[{100 * x['ci95'][0]:+.2f},{100 * x['ci95'][1]:+.2f}] raw p {x['raw_p']:.4f} max-adj {x['maxtest_adj_p']:.4f} Holm {x['holm_p']:.4f}")

    for name in ("family_primary", "hardware_harmfulness_h100_command"):
        for comp in ("sim_minus_real", "sim_minus_run1", "run2_minus_run1"):
            for x in res["tests"][name][comp]["per_setting"]:
                if x["probe"].startswith("harmfulness") and x["readout"] == "command" and (
                        x["level"] != "unit" and x["rate"] in (0.01, 0.05) or x["level"] == "unit" and x["grid_index"] in (14, 15, 16, 17, 18)):
                    print(f"harm {name[:6]:<6} {comp:<16} {x['probe']}|{x['level']}|{x['rate']:.3g}|{x['pool']} value {100 * x['value']:+.2f} "
                          f"[{100 * x['ci95'][0]:+.2f},{100 * x['ci95'][1]:+.2f}] raw p {x['raw_p']:.4f} max-adj {x['maxtest_adj_p']:.4f} "
                          f"tpr r1 {100 * x['tpr']['run1']:.1f} r2 {100 * x['tpr']['run2']:.1f} sim {100 * x['tpr']['sim']:.1f}")
    for probe, dcm in res["decomposition_f1.33e-2"].items():
        for ref in ("run1", "run2"):
            v = dcm[f"vs_{ref}"]
            print(f"decomp {probe} vs {ref}: sim-only {v['sim_only']} {ref}-only {v[ref + '_only']} both {v['both']} neither {v['neither']} | "
                  f"sim-only by env {json.dumps(v['sim_only_by_env'])} | {ref}-only by env {json.dumps(v[ref + '_only_by_env'])}")
        print(f"decomp {probe} cohort by env {json.dumps(dcm['cohort_by_env'])}")
