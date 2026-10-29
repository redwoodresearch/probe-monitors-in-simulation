"""POST HOC (0d, 10-13 about 11:10): does the output category that carries Llama-3.3-70B's last-token gap also carry its
mean-pooled gap? Step 2: local, $0. Not part of RULE.md; RULE.md and the registered analysis are unchanged.

From run_rows_meanpool_POSTHOC.py's per-row logits (layers 60 and 70) and category codes:
1. Self-check of the engine: decompose() here re-runs the registered Llama decomposition (analyze_gap_where.py's stats,
   copied into one function, thresholds redrawn, seed 20261011) on the registered rows and must equal the registered
   result (gap_where_2026-10-13_112825.json) for every Llama category and statistic at 5% and 1%, or nothing is written.
2. Gates: per-row means reproduce the stored run scores within 1e-3, and give the same per-run, per-model decisions at
   avg_vs_last's fixed thresholds; then avg_vs_last's points and intervals (bd4a077e, avg_vs_last_POSTHOC_2026-10-13_110005.json)
   are reproduced exactly (1e-9) for mean-pooled as simulated - e2 and repaired - e2, and last-token as simulated - e2, at
   L60, L70 and per fold (60, 70, 60, 60, 70), at 5% and 1%.
3. The same decomposition (categories, mean shift, A, D, A_ret, shares and their rules) under avg_vs_last's convention:
   thresholds fixed at the point estimate and 2,000 main-task draws with seed 20261012, as avg_vs_last reads these gaps.
   G = catch(real e2) - catch(arm), the registered sign; avg_vs_last prints arm - e2, so its -4.1 is G = +4.1 here.
Differences from the registered analysis, stated in every table: the arm (as simulated or fix939's repaired, not
repaired-plus), thresholds fixed instead of redrawn, and seed 20261012 instead of 20261011.

  uv run --no-project --python 3.12 --with numpy python experiments/2026-10-13_q2_llama_gap_where/analyze_meanpool_POSTHOC.py \\
      --rows <results/rows_meanpool_POSTHOC_<stamp>.json> --manifest <its _manifest.json> --mp-scores <run_scores_2026-10-13_011737> \\
      --mp-fix <fix_scores_mpllama_2026-10-13_012258.json> --lt-scores <run_scores_2026-10-12_143129> \\
      --reg-rows <abs path: results/rows_2026-10-13_110759.json> --reg-manifest <abs path: results/rows_2026-10-13_110759_manifest.json> \\
      --llama-fix <fix_scores_olmo_2026-10-12_153542.json> --llama-plus <plus_scores_llama_2026-10-13_024006.json> \\
      --llama-reading <plus_reading_llama_2026-10-13_024135.json>
"""
import argparse
import importlib.util
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
_spec = importlib.util.spec_from_file_location("registered", HERE / "analyze_gap_where.py")  # read only: its helpers, unchanged
reg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(reg)
RATES, B, MIN_VALID, MIN_DEN, TOL = reg.RATES, reg.B, reg.MIN_VALID, reg.MIN_DEN, reg.TOL
SEED_AVL = 20261012
AVL = ("bd4a077e", "experiments/2026-10-13_q2_avg_vs_last/results/avg_vs_last_POSTHOC_2026-10-13_110005.json")
REG_RESULT = HERE / "results" / "gap_where_2026-10-13_112825.json"
LAYS = {"L60": (60,) * 5, "L70": (70,) * 5, "per-fold 60,70,60,60,70": (60, 70, 60, 60, 70)}
KEY = {"mean-pooled": "output_end_mean__mean__task_nested", "last-token": "output_end__mean__task_nested"}
ARMS = {"mean-pooled": ("as simulated", "repaired"), "last-token": ("as simulated",)}
KEYS = ("mean shift", "A", "G_ret", "population effect G - G_ret", "D", "A_ret", "A/G", "D/G_ret")
sha = reg.sha


def decompose(cats, ar, ce, L0, F, ctix, draws, T_point, T_draw, shares=True):
    """The registered analysis's per-category statistics (analyze_gap_where.py main(): sums, stats and the summary per
    rate, copied without change of formula). ar: row is real e2; ce: row's cell; L0: row logits with the own-fold column 0;
    T_point: thresholds at the point; T_draw(w): thresholds in a draw. Returns (G point, G draws, entry)."""
    def sums(mask):
        n = [np.bincount(ce[mask & a], minlength=939).astype(float) for a in (ar, ~ar)]
        s = [np.stack([np.bincount(ce[mask & a], weights=L0[mask & a, v], minlength=939) for v in range(5)], 1) for a in (ar, ~ar)]
        return n, s
    (n_e, n_r), (s_e, s_r) = sums(np.ones(len(ar), bool))
    if (n_e == 0).any() or (n_r == 0).any():
        raise SystemExit("a run with no rows")
    mean_e, mean_r = s_e / n_e[:, None], s_r / n_r[:, None]
    pre = {c: sums(m) for c, m in cats.items()}
    keeps = {c: (n_e - nce > 0) & (n_r - ncr > 0) for c, ((nce, ncr), _) in pre.items()}

    def stats(T, w, point):
        wc = w[ctix].astype(float)

        def cat(S, keep):
            return (wc[keep] * reg.caught(S[keep], F[keep], T)).sum() / wc[keep].sum()
        allc = np.ones(939, bool)
        G = 100 * (cat(mean_e, allc) - cat(mean_r, allc))
        res = {}
        for c, ((nce, ncr), (sce, scr)) in pre.items():
            def offsets(keep):
                de = np.bincount(F, weights=wc * keep * nce, minlength=5); dr = np.bincount(F, weights=wc * keep * ncr, minlength=5)
                if (de == 0).any() or (dr == 0).any():
                    return None, None
                me = np.stack([np.bincount(F, weights=wc * keep * sce[:, v], minlength=5) for v in range(5)], 1) / de[:, None]
                mr = np.stack([np.bincount(F, weights=wc * keep * scr[:, v], minlength=5) for v in range(5)], 1) / dr[:, None]
                return me - mr, np.minimum(de, dr)
            r, ok_den = {}, (lambda x: x > 0) if point else (lambda x: x >= MIN_DEN)
            O, wt = offsets(allc)
            if O is not None:
                m = ~np.eye(5, dtype=bool)
                r["mean shift"] = float((-O * wt[:, None])[m].sum() / np.broadcast_to(wt[:, None], (5, 5))[m].sum())
                r["A"] = G - 100 * (cat(mean_e, allc) - cat(mean_r + O[F] * (ncr / n_r)[:, None], allc))
                if ok_den(G):
                    r["A/G"] = r["A"] / G
            keep = keeps[c]
            if wc[keep].sum() > 0:
                G_ret = 100 * (cat(mean_e, keep) - cat(mean_r, keep))
                with np.errstate(divide="ignore", invalid="ignore"):
                    de_, dr_ = (s_e - sce) / (n_e - nce)[:, None], (s_r - scr) / (n_r - ncr)[:, None]
                r["G_ret"], r["population effect G - G_ret"] = G_ret, G - G_ret
                r["D"] = G_ret - 100 * (cat(de_, keep) - cat(dr_, keep))
                if O is not None and ok_den(G_ret):
                    r["D/G_ret"] = r["D"] / G_ret
                O2, _ = offsets(keep)
                if O2 is not None:
                    r["A_ret"] = G_ret - 100 * (cat(mean_e, keep) - cat(mean_r + O2[F] * (ncr / n_r)[:, None], keep))
            res[c] = r
        return G, res
    ones = np.ones(len(draws[0]))
    G0, P0 = stats(T_point, ones, True)
    Gb, Pb = [], {c: {k: [] for k in KEYS} for c in cats}
    for w in draws:
        g, p = stats(T_draw(w), w, False)
        Gb.append(g)
        for c in cats:
            for k in KEYS:
                Pb[c][k].append(p[c].get(k))
    lo, hi = (float(x) for x in np.percentile(Gb, [2.5, 97.5]))
    stable = bool(G0 > 0 and lo > 0)
    entry = {"G (points)": {"point": float(G0), "95%": [lo, hi], "valid draws": B}, "G stable (point > 0, interval above 0)": stable, "categories": {}}
    for c, m in cats.items():
        kp = keeps[c]; ex_tasks = set(ctix[~kp].tolist()); kept_tasks = set(ctix[kp].tolist())
        e = {"rows (real e2, repaired)": [int((m & ar).sum()), int((m & ~ar).sum())],
             "share of rows (real e2, repaired)": [float((m & ar).sum() / ar.sum()), float((m & ~ar).sum() / (~ar).sum())],
             "drop: retained cells, excluded cells, excluded cells' tasks": [int(kp.sum()), int((~kp).sum()), len(ex_tasks)],
             "drop: retained cells' tasks, of all tasks": [len(kept_tasks), len(set(ctix.tolist()))]}
        for k in KEYS:
            v = [x for x in Pb[c][k] if x is not None]
            s = {"point": float(P0[c][k]) if k in P0[c] else "unavailable", "valid draws": len(v)}
            s["95%"] = [float(x) for x in np.percentile(v, [2.5, 97.5])] if len(v) >= MIN_VALID else f"not reported: {len(v)} valid draws (unstable)"
            if MIN_VALID <= len(v) < B:
                s["conditional on the valid draws"] = True
            e[k] = s
        gr = e["G_ret"]
        e["D's denominator G_ret stable"] = bool(isinstance(gr["point"], float) and isinstance(gr["95%"], list) and gr["point"] > 0 and gr["95%"][0] > 0)
        for k, why in (("A/G", None if stable else "G unstable"), ("D/G_ret", None if e["D's denominator G_ret stable"] else "G_ret unstable")):
            why = why if shares else "not formed"
            if why is not None:
                e[k] = {"point": f"not formed: {why}", "95%": "not formed", "valid draws": e[k]["valid draws"]}
        entry["categories"][c] = e
    return G0, np.array(Gb), entry


def categories(X, pre):
    cats = {f"{s} {lab}": X[f"{pre}{s}"] == k for s, labs in reg.CATS.items() for k, lab in enumerate(labs)}
    cats |= {f"readout {lab}": X[f"{pre}readout class"] == k for k, lab in enumerate(("whitespace-only", "newline", "letters", "digits", "punctuation")) if k >= 2}
    cats["readout merges whitespace"] = X[f"{pre}merges whitespace"] == 1
    return cats | {j: np.isin(X[f"{pre}{s}"], ks) for j, (s, ks) in reg.JOINT.items()}


def same(a, b, path=""):
    """Every number in a equals b's within 1e-12, and every string is equal (the self-check)."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], f"{path}/{k}") for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-12
    return a == b


def main():
    ap = argparse.ArgumentParser()
    for a in ("rows", "manifest", "mp-scores", "mp-fix", "lt-scores", "reg-rows", "reg-manifest", "llama-fix", "llama-plus", "llama-reading"):
        ap.add_argument(f"--{a}", required=True, type=lambda p: Path(p).resolve())
    args = ap.parse_args()
    for f in (Path(__file__).resolve(), HERE / "analyze_gap_where.py", REG_RESULT, args.manifest, args.reg_manifest):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    avl = json.loads(subprocess.run(["git", "show", f"{AVL[0]}:{AVL[1]}"], capture_output=True, text=True, check=True, cwd=REPO).stdout)
    avl_l = avl["readers"]["Llama-3.3-70B"]
    hs = avl_l["inputs sha256"]
    bound = {"mean-pooled runs.json": args.mp_scores / "runs.json", "last-token runs.json": args.lt_scores / "runs.json",
             f"mean-pooled {args.mp_fix.name}": args.mp_fix, f"last-token {args.llama_fix.name}": args.llama_fix} | \
            {f"{ro} scores_L{l}.npz": d / f"scores_L{l}.npz" for ro, d in (("mean-pooled", args.mp_scores), ("last-token", args.lt_scores)) for l in (60, 70)}
    if any(hs.get(k) != sha(p) for k, p in bound.items()):
        raise SystemExit("an input isn't the one avg_vs_last read")
    out = {"POST HOC": True, "question": "0d, 10-13 about 11:10: does the category carrying Llama's last-token gap carry its mean-pooled gap?",
           "git": head, "avg_vs_last reading reproduced": f"{AVL[0]}:{AVL[1]}",
           "differences from the registered analysis": {"arm": "as simulated, or fix939's repaired (823 re-rendered + 116 as simulated); not repaired-plus",
                                                       "thresholds": "fixed at the point estimate in every draw (avg_vs_last); the registered analysis redraws them",
                                                       "seed": f"{SEED_AVL} (avg_vs_last); the registered analysis uses 20261011",
                                                       "sign": "G = catch(real e2) - catch(arm), as registered; avg_vs_last prints arm - e2"},
           "inputs sha256": {k: sha(p) for k, p in bound.items()} | {p.name: sha(p) for p in (args.rows, args.manifest, args.reg_rows, args.reg_manifest, REG_RESULT)},
           "self-check": {}, "gates": {}, "configurations": {}}

    # 1. the engine reproduces the registered Llama decomposition exactly
    man = json.loads(args.reg_manifest.read_text())
    if sha(args.reg_rows) != man["rows record sha256"]:
        raise SystemExit("the registered rows record isn't its manifest's")
    rrec = json.loads(args.reg_rows.read_text()); Xr = np.load(args.reg_rows.parent / rrec["rows file"])
    if sha(args.reg_rows.parent / rrec["rows file"]) != rrec["rows sha256"]:
        raise SystemExit("the registered rows file isn't its record's")
    lay_reg = tuple(rrec["readers (probe run, layer per test fold)"]["llama"][1])
    R = reg.reader_llama_olmo("llama", args.lt_scores, args.llama_fix, args.llama_plus, json.loads(args.llama_reading.read_text()), lay_reg)
    U = rrec["units (per reader: runs.json index, key sha256, source, fold; arrays' unit indexes this)"]["llama"]
    pos = {int(i): ("e2", c) for c, i in enumerate(R["ei"])} | {int(i): ("rep", c) for c, i in enumerate(R["si"])}
    u_arm = np.array([pos[i][0] == "e2" for i in U["runs.json index"]]); u_cell = np.array([pos[i][1] for i in U["runs.json index"]])
    u = Xr["llama__unit"]; F = R["F"]
    own = np.arange(5)[None, :] == F[u_cell[u]][:, None]
    L0 = np.where(own, 0.0, Xr["llama__logit"].astype(np.float64))
    registered = json.loads(REG_RESULT.read_text())["readers"]["llama"]
    ones = np.ones(len(R["draws"][0]))
    for a in RATES:
        _, _, entry = decompose(categories(Xr, "llama__"), u_arm[u], u_cell[u], L0, F, R["ctix"], R["draws"], reg.thresholds(R, a, ones), lambda w: reg.thresholds(R, a, w))
        want = {k: v for k, v in registered[str(a)].items() if k != "G and its interval equal the committed reading's (1e-9)"}
        if not same(entry, want):
            raise SystemExit(f"self-check failed at {a}: the engine doesn't reproduce the registered Llama decomposition")
        out["self-check"][str(a)] = "the engine reproduces the registered Llama decomposition (every category and statistic, 1e-12)"
    print("self-check passed", flush=True)

    # 2. the POST HOC rows, avg_vs_last's construction, and its readings reproduced
    if sha(args.rows) != json.loads(args.manifest.read_text())["rows record sha256"]:
        raise SystemExit("the POST HOC rows record isn't the one its committed manifest names")
    rec = json.loads(args.rows.read_text()); X = np.load(args.rows.parent / rec["rows file"])
    if sha(args.rows.parent / rec["rows file"]) != rec["rows sha256"] or rec["inputs sha256"]["runs.json"] != "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96":
        raise SystemExit("the POST HOC rows file isn't its record's, or was built from another runs.json")
    runs = json.loads((args.mp_scores / "runs.json").read_text())
    ident = lambda rs: [(x["arm"], x["key"], x["fold"], x["cell"], x["main_task"]) for x in rs]  # noqa: E731
    if ident(runs) != ident(json.loads((args.lt_scores / "runs.json").read_text())):
        raise SystemExit("the readouts' runs differ")
    arm = np.array([x["arm"] for x in runs]); task = np.array([x["main_task"] for x in runs]); cell = np.array([x["cell"] for x in runs])
    fold = np.array([x["fold"] for x in runs]); hon = arm == "real-honest"
    sim_i = {c: i for i, c in enumerate(cell) if arm[i] == "sim-attack"}; e2_i = {c: i for i, c in enumerate(cell) if arm[i] == "real-attack-e2"}
    cells = sorted(set(sim_i) & set(e2_i)); si, ei = np.array([sim_i[c] for c in cells]), np.array([e2_i[c] for c in cells])
    tasks = np.unique(task); tix = np.searchsorted(tasks, task)
    if len(cells) != 939 or not (tix[si] == tix[ei]).all():
        raise SystemExit("not 939 cells sharing tasks")
    g = np.random.default_rng(SEED_AVL); W = [np.bincount(g.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(B)]
    cpos = {int(i): c for c, i in enumerate(si)} | {int(i): c for c, i in enumerate(ei)}
    Uu = rec["units (runs.json index, source, fold; arrays' unit indexes this)"]
    src, ucell = np.array(Uu["source"]), np.array([cpos[int(i)] for i in Uu["runs.json index"]])
    F = fold[si]
    zf_p = args.mp_fix.parent / json.loads(args.mp_fix.read_text())["scores file"]
    if sha(zf_p) != json.loads(args.mp_fix.read_text())["scores sha256"] or hs.get(f"mean-pooled {zf_p.name}") != sha(zf_p):
        raise SystemExit("the repaired mean-pooled scores file isn't its record's or avg_vs_last's")
    Zf = np.load(zf_p)
    fixmap = {int(i): j for j, i in enumerate(Zf["run_index"]) if Zf["has_fix"][j]}
    if len(fixmap) != 823 or set(ucell[src == "fix"]) != {cpos[i] for i in fixmap}:
        raise SystemExit("the re-rendered runs aren't fix939's 823")
    for ro, d in (("mean-pooled", args.mp_scores), ("last-token", args.lt_scores)):
        pre = f"{ro}__"
        u = X[f"{pre}unit"]
        for lname, lay in LAYS.items():
            N = {l: np.load(d / f"scores_L{l}.npz")[KEY[ro]] for l in (60, 70)}
            S = np.stack([N[lay[fold[i]]][fold[i], :, i] for i in range(len(runs))])
            rowF = F[ucell[u]]; own = np.arange(5)[None, :] == rowF[:, None]
            lg = np.where((np.array(lay)[rowF] == 60)[:, None], X[f"{pre}logit L60"], X[f"{pre}logit L70"]).astype(np.float64)
            if not np.isfinite(lg[~own]).all():
                raise SystemExit(f"{ro} {lname}: a non-finite logit")
            L0 = np.where(own, 0.0, lg)
            T = {}
            for a in RATES:  # avg_vs_last's thresholds: unit weights, fixed at the point estimate
                T[a] = np.full((5, 5), np.nan)
                for f in range(5):
                    for v in range(5):
                        if v != f:
                            s_ = N[lay[f]][f, v]; m = hon & (fold == v) & np.isfinite(s_); T[a][f, v] = reg.thresh(s_[m], np.ones(m.sum()), a)
            for arm_name in ARMS[ro]:
                is_rep = arm_name == "repaired"
                # rows: real e2 units, and the arm's units (repaired: the re-rendered run where there is one, else as simulated)
                keep_u = (src == "e2") | ((src == "fix") if is_rep else np.zeros(len(src), bool)) | ((src == "sim") & ~(is_rep & np.isin(ucell, list({cpos[i] for i in fixmap}))))
                rows = keep_u[u]
                ar, ce = (src[u] == "e2")[rows], ucell[u][rows]
                stored = S[si].copy()
                if is_rep:
                    for i, j in fixmap.items():
                        stored[cpos[i]] = Zf[f"L{lay[fold[i]]}"][fold[i], :, j]
                n_e = np.bincount(ce[ar], minlength=939); n_r = np.bincount(ce[~ar], minlength=939)
                me = np.stack([np.bincount(ce[ar], weights=L0[rows][ar, v], minlength=939) for v in range(5)], 1) / n_e[:, None]
                mr = np.stack([np.bincount(ce[~ar], weights=L0[rows][~ar, v], minlength=939) for v in range(5)], 1) / n_r[:, None]
                valid = np.arange(5)[None, :] != F[:, None]
                d2 = max(float(np.abs(me - S[ei])[valid].max()), float(np.abs(mr - stored)[valid].max()))
                key = f"{ro} @ {lname} @ {arm_name} - e2"
                if not np.isfinite(d2) or d2 > TOL:
                    raise SystemExit(f"{key}: row means differ from the stored scores by {d2}")
                for a in RATES:
                    dec = lambda Sx: np.where(valid, Sx > T[a][F], False)  # noqa: E731
                    if not ((dec(me) == dec(S[ei])).all() and (dec(mr) == dec(stored)).all()):
                        raise SystemExit(f"{key} at {a}: a per-run, per-model decision differs from the stored scores'")
                    G0, Gb, entry = decompose(categories({k.removeprefix(pre): X[k][rows] for k in X.files if k.startswith(pre)}, ""), ar, ce, L0[rows], F, tix[si], W, T[a], lambda w, T_=T[a]: T_)
                    want = avl_l["cells read"][f"{ro} @ {lname} @ {a}"]["differences"][f"{arm_name} - e2"]["points, 95%"]
                    lo, hi = np.percentile(-Gb, [2.5, 97.5])
                    if not (abs(-G0 - want[0]) < 1e-9 and abs(lo - want[1][0]) < 1e-9 and abs(hi - want[1][1]) < 1e-9):
                        raise SystemExit(f"{key} at {a}: {-G0} [{lo}, {hi}] doesn't reproduce avg_vs_last's {want}")
                    out["gates"][f"{key} @ {a}"] = {"max |row mean - stored|": d2, "same decisions": True, "avg_vs_last reproduced (arm - e2)": want}
                    out["configurations"][f"{key} @ {a}"] = entry
                    print(f"{key} at {a}: reproduced {want}", flush=True)
    out["side by side"] = side_by_side(out, json.loads(REG_RESULT.read_text())["readers"]["llama"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"meanpool_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    text = table(out)
    with open(HERE / "results" / f"meanpool_POSTHOC_{stamp}_printout.txt", "x") as fh:
        fh.write(text + "\n")
    print(text); print(dest)


def side_by_side(out, registered):
    """Per category: A/G with its range in the registered reading and each POST HOC configuration; and each one's
    leading category (largest available A, the registered definition)."""
    cols = {f"registered last-token, repaired-plus, redrawn @ {a}": registered[str(a)] for a in RATES} | \
           {k: v for k, v in out["configurations"].items()}
    res = {"leading category (largest available A)": {}, "A/G": {}}
    for name, entry in cols.items():
        C = entry["categories"]
        av = {c: e["A"]["point"] for c, e in C.items() if isinstance(e["A"]["point"], float)}
        res["leading category (largest available A)"][name] = [max(av, key=lambda c: (av[c], -list(C).index(c))) if av else None, entry["G (points)"]]
        res["A/G"][name] = {c: [e["A/G"]["point"], e["A/G"]["95%"]] for c, e in C.items()}
    return res


def table(out):
    f2 = lambda s: reg.fmt(s) if isinstance(s, dict) else "n/a"  # noqa: E731
    lines = ["POST HOC (0d, 10-13 about 11:10). Not RULE.md. Differences from the registered analysis: " + json.dumps(out["differences from the registered analysis"]), ""]
    sb = out["side by side"]
    lines.append("== leading category (largest available A) and G (points, 95%) per configuration:")
    for name, (lead, G) in sb["leading category (largest available A)"].items():
        lines.append(f"  {name:60s} G = {f2(G)}   leading: {lead}")
    names = list(sb["A/G"])
    lines.append("")
    lines.append("== A/G per category (point [95%]; n/a = not formed, e.g. G unstable):")
    for name in names:
        lines.append(f"  [{names.index(name)}] {name}")
    cats = list(next(iter(sb["A/G"].values())))
    for c in cats:
        cells = []
        for name in names:
            p, ci = sb["A/G"][name][c]
            cells.append(f"[{names.index(name)}] " + (f"{p:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]" if isinstance(p, float) and isinstance(ci, list) else "n/a"))
        lines.append(f"  {c:38s} " + "  ".join(cells))
    for key in [k for k in out["configurations"] if "per-fold" in k]:
        e = out["configurations"][key]
        lines.append("")
        lines.append(f"== {key}: G = catch(real e2) - catch(arm) = {f2(e['G (points)'])} points; stable {e['G stable (point > 0, interval above 0)']}")
        lines.append(f"{'category':38s} {'rows % e2/arm':>13s} {'mean shift arm-e2':>24s} {'A':>23s} {'D':>23s} {'G_ret':>23s} {'A_ret':>23s} {'A/G':>22s} {'D/G_ret':>22s}")
        for c, x in e["categories"].items():
            sh = x["share of rows (real e2, repaired)"]
            lines.append(f"{c:38s} {100 * sh[0]:5.1f}/{100 * sh[1]:5.1f}  {f2(x['mean shift']):>24s} {f2(x['A']):>23s} {f2(x['D']):>23s} {f2(x['G_ret']):>23s} "
                         f"{f2(x['A_ret']):>23s} {f2(x['A/G']):>22s} {f2(x['D/G_ret']):>22s}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
