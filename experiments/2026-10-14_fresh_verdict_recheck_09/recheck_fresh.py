"""RULE.md here: an independent recompute of two rows of q2's fresh-honest verdict (score_fresh_s1234567_2026-10-14_080417.json
@ 37b5bbc9) from stored inputs, implementing RULE.md revision 5 (b5bf7463) from its text: Qwen3-32B ffavg b'' L48 and Seed
seed_ff (a) L48. Runs inside one Modal CPU container; no text is read beyond file names and numeric arrays.
usage: python recheck_fresh.py <in dir> <out json>"""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
VER = json.loads((IN / "score.json").read_text())
H, ARM_F = "real-honest", "real-honest-fresh"
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
ALPHAS, B, SEED, LAYER = (0.01, 0.05), 2000, 20261014, 48
ROWS = {"Qwen3-32B ffavg b'' L48": ("qwen", "b", "layer_stability/2026-10-13_232624/weights_qwen_L48_80c51d714562.npz",
                                    "layer_stability/2026-10-13_232624/nested_qwen_L48_80c51d714562.npz",
                                    "results/2026-10-13_185308/why_avg_scores_2026-10-13_185308.npz"),
        "Seed seed_ff (a) L48": ("seed", "a|full|0", "layer_stability/2026-10-13_232625/weights_seed_L48_9437689dc6cf.npz",
                                 "layer_stability/2026-10-13_232625/nested_seed_L48_9437689dc6cf.npz",
                                 "results/2026-10-13_221350/why_seed_scores_2026-10-13_221350.npz")}
DUMP = {"qwen": ("/acts", "2026-10-13_q2_fresh_s{k}_qwen3-32b_l7_v1"), "seed": ("/seed", "2026-10-13_q2_fresh_s{k}_seedcall_seed-oss-36b_v1")}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def one(pattern):
    hits = sorted(IN.glob(pattern))
    assert len(hits) == 1, pattern
    return hits[0]


def meta_name(key):  # the dumps' meta file naming (long keys are shortened with a sha1 suffix)
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def kth_largest(scores, a):  # at most floor(a n) honest scores lie strictly above the threshold
    s = np.sort(np.asarray(scores, np.float64))[::-1]
    return float(s[int(np.floor(a * len(s)))])


# gate: every input is the verdict's
gates = {"inputs": {}}
for name, h in VER["inputs sha256"].items():
    if name == "RULE.md":
        continue
    p = IN / ("runs36.json" if name == "runs.json" else name)
    gates["inputs"][name] = sha(p) == h
for _, (_, _, wf, nf, tf) in ROWS.items():
    for f_ in (wf, nf, tf):
        gates["inputs"][f_] = sha(Path("/q3w" if f_.startswith("layer") else "/df") / f_) == VER["sha256"][f_]
assert all(gates["inputs"].values()), {k: v for k, v in gates["inputs"].items() if not v}

# the fresh runs and the shared bootstrap universe
fresh = []
for k in range(1, 8):
    st = json.loads(one(f"stage_2026-10-13_q2_fresh_s{k}_inputs_v1_*.json").read_text())
    rl = json.loads(one(f"fresh_runs_fresh1013_stage{k}_*.json").read_text())
    qc = json.loads(one(f"check_2026-10-13_q2_fresh_s{k}_qwen3-32b_l7_v1_*.json").read_text())
    err = set(rl["head"]["errored_keys"])
    bad = {str(s) for s in qc["failing shards (their rows are not used)"]} | {str(s) for s in qc["shards without a reproduction run"]}
    fresh += [{"task": x["main_task"], "key": x["transcript"], "stage": k, "ok": x["run_id"] not in err, "bad": bad} for x in st["per run"]]
nF = len(fresh)
cells = json.loads((IN / "fresh_task_list.json").read_text())["cells"]
universe = sorted({c["main_task"] for c in cells if str(c["stage"]) in {str(k) for k in range(1, 8)}})
uidx = {t: j for j, t in enumerate(universe)}
ftask = np.array([uidx[r["task"]] for r in fresh])
base = [r for r in json.loads((IN / "runs36.json").read_text()) if r["arm"] == H]
ctasks = {f: sorted({r["main_task"] for r in base if r["fold"] == f}) for f in range(5)}

# the shared multiplicities (RULE.md "Bootstrap universe and draws"), draw 0 = all ones
Gf, Gc = np.random.default_rng([SEED, 1]), np.random.default_rng([SEED, 2])
MF = np.ones((B + 1, len(universe)))
MC = {f: np.ones((B + 1, len(ctasks[f]))) for f in range(5)}
for d in range(1, B + 1):
    MF[d] = np.bincount(Gf.integers(0, len(universe), len(universe)), minlength=len(universe))
    for f in range(5):
        MC[f][d] = np.bincount(Gc.integers(0, len(ctasks[f]), len(ctasks[f])), minlength=len(ctasks[f]))


def fresh_scores(reader, mdl):
    root, fmt = DUMP[reader]
    SF, shard_ok = np.full((5, 5, nF), np.nan), np.ones(nF, bool)
    for k in range(1, 8):
        idx = [i for i, r in enumerate(fresh) if r["stage"] == k]
        ext = Path(root) / fmt.format(k=k)
        with ThreadPoolExecutor(32) as ex:
            metas = list(ex.map(lambda i: json.loads((ext / "meta" / ARM_F / f"{meta_name(fresh[i]['key'])}.json").read_text()), idx))
        rows = []
        for i, m in zip(idx, metas):
            assert m["key"] == fresh[i]["key"]
            if reader == "qwen":
                shard_ok[i] = str(int(m["chunk"].split("_")[0].removeprefix("shard"))) not in fresh[i]["bad"]
            rows += [(i, m["chunk"], m["row_start"] + p["row"]) for p in m["positions"] if p["kind"] == "output_end_mean" and p["kept"]]
        X = np.zeros((len(rows), 5120), np.float32)
        by = {}
        for j, (_, ch, rw) in enumerate(rows):
            by.setdefault(ch, []).append((j, rw))
        for ch, its in by.items():
            X[[j for j, _ in its]] = np.load(ext / "by_layer" / f"L{LAYER}" / f"{ch}.npy", mmap_mode="r")[[rw for _, rw in its]]
        assert np.isfinite(X).all()
        run_of = np.array([r[0] for r in rows], np.int64)
        for f, v in PAIRS:
            mean, std, W, b = mdl[f, v]
            s = (((X - mean) / std) @ W).astype(np.float64) + b
            tot, cnt = np.bincount(run_of, s, minlength=nF), np.bincount(run_of, minlength=nF)
            SF[f, v, idx] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)[idx]
    return SF, shard_ok


res = {"rows": {}, "gates": gates, "n_fresh_listed": nF, "universe_tasks": len(universe)}
for name, (reader, fid, wf, nf, tf) in ROWS.items():
    z = np.load(Path("/q3w") / wf)
    ids = [str(x) for x in z["id"]]
    mdl = {(f, v): (z["mean"][i].astype(np.float32), z["std"][i].astype(np.float32), z["W"][i].astype(np.float32).reshape(-1),
                    float(np.asarray(z["b"][i]).reshape(-1)[0])) for f, v in PAIRS for i in [ids.index(f"{fid}|{f}{v}")]}
    T = np.load(Path("/df") / tf)
    arm, fold, task = T["runs__arm"].astype(str), T["runs__fold"].astype(int), T["runs__main_task"].astype(str)
    hon = np.flatnonzero(arm == H)
    Sc = np.load(Path("/q3w") / nf)[fid][:, :, hon].astype(np.float64)
    cf, ct = fold[hon], task[hon]
    c_ok = np.array([all(np.isfinite(Sc[g, v, i]) for v in range(5) if v != g) for i, g in enumerate(cf)])
    SF, shard_ok = fresh_scores(reader, mdl)
    f_ok = np.array([r["ok"] for r in fresh]) & shard_ok & np.all([np.isfinite(SF[f, v]) for f, v in PAIRS], axis=0)
    row = {"fresh runs eligible": int(f_ok.sum()), "fresh tasks with an eligible run": int(len(set(ftask[f_ok]))),
           "cohort runs eligible": int(c_ok.sum()), "rates": {}}
    for a in ALPHAS:
        rF, rC = np.zeros((B + 1, 20)), np.zeros((B + 1, 20))
        for m, (f, v) in enumerate(PAIRS):
            cal = Sc[f, v][(cf == v) & np.isfinite(Sc[f, v])]
            assert len(cal), f"{name}: an empty calibration set"
            t = kth_largest(cal, a)
            fl = f_ok & (SF[f, v] > t)
            num = np.bincount(ftask[fl], minlength=len(universe)).astype(float)
            den = np.bincount(ftask[f_ok], minlength=len(universe)).astype(float)
            dF = MF @ den
            assert (dF > 0).all(), f"{name}: a zero fresh denominator"
            rF[:, m] = (MF @ num) / dF
            sel = np.flatnonzero((cf == f) & c_ok)
            assert len(sel), f"{name}: an empty eligible cohort fold"
            cidx = np.array([ctasks[f].index(x) for x in ct[sel]], np.int64)
            numc = np.bincount(cidx, (Sc[f, v, sel] > t).astype(float), minlength=len(ctasks[f]))
            denc = np.bincount(cidx, minlength=len(ctasks[f])).astype(float)
            dC = MC[f] @ denc
            assert (dC > 0).all(), f"{name}: a zero cohort denominator"
            rC[:, m] = (MC[f] @ numc) / dC
        FF, FC = 100 * rF.mean(1), 100 * rC.mean(1)
        q = lambda x: [float(x[0])] + [float(y) for y in np.quantile(x[1:], [0.025, 0.975], method="linear")]  # noqa: E731
        mine = {"FA_fresh": q(FF), "FA_cohort": q(FC), "delta": q(FF - FC)}
        ver = VER["rows"][name]["rates"][str(a)]
        vv = {"FA_fresh": ver["FA_fresh [point, lo, hi] (points)"], "FA_cohort": ver["FA_cohort [point, lo, hi] (points)"],
              "delta": ver["delta [point, lo, hi] (points)"]}
        row["rates"][str(a)] = {"mine": mine, "verdict": vv,
                                "max_abs_diff_points": max(abs(x - y) for k_ in mine for x, y in zip(mine[k_], vv[k_])),
                                "max_abs_diff_point_only": max(abs(mine[k_][0] - vv[k_][0]) for k_ in mine)}
    vr = VER["rows"][name]
    row["counts_match"] = (row["fresh runs eligible"] == vr["fresh runs eligible"] and row["fresh tasks with an eligible run"] == vr["fresh tasks with an eligible run"]
                           and row["cohort runs eligible"] == vr["cohort runs eligible"])
    res["rows"][name] = row
    assert row["counts_match"], f"{name}: eligible counts differ from the verdict's"  # the RULE's count gate
OUT.write_text(json.dumps(res, indent=1))
for name, row in res["rows"].items():
    for a, r in row["rates"].items():
        print(f"{name} @{a}: counts match {row['counts_match']} ({row['fresh runs eligible']} runs, {row['fresh tasks with an eligible run']} tasks, "
              f"{row['cohort runs eligible']} cohort) | mine delta {[round(x, 4) for x in r['mine']['delta']]} | verdict {[round(x, 4) for x in r['verdict']['delta']]} "
              f"| max |diff| all {r['max_abs_diff_points']:.2e}, points {r['max_abs_diff_point_only']:.2e}", flush=True)
