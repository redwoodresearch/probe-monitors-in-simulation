"""POST HOC (PLAN_identical_POSTHOC.md here): on Llama-3.3-70B's registered rows, the mean shift (repaired - real e2) on
outputs byte-identical to an output of the same cell's other-arm run, against non-identical rows and all rows. Local, $0.
Identical or not identical only (the format-tell stop): nothing describes how non-identical outputs differ.

Amendment 1's matched-pair shift M (same text in both runs of a cell, per-cell group ids from the container) is the
primary statistic; all readings are descriptive. The engine is analyze_meanpool_POSTHOC.decompose() under the registered conventions (thresholds redrawn, 2,000 draws, seed
20261011, 5% and 1%). It must first reproduce the registered Llama decomposition exactly, or nothing is written. R, the
ratio of the identical-row shift to the all-rows shift, comes from paired draws with the registered mean-shift formula.

  uv run --no-project --python 3.12 --with numpy python experiments/2026-10-13_q2_llama_gap_where/analyze_identical_POSTHOC.py \\
      --flags <results/identical_POSTHOC_<stamp>.json> --reg-rows <results/rows_2026-10-13_110759.json> \\
      --reg-manifest <results/rows_2026-10-13_110759_manifest.json> --lt-scores <Llama run_scores_2026-10-12_143129> \\
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
_spec = importlib.util.spec_from_file_location("mp", HERE / "analyze_meanpool_POSTHOC.py")  # read only: its engine and helpers
mp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mp)
reg, sha, RATES, B = mp.reg, mp.sha, mp.RATES, mp.B
MIN_ROWS, MIN_TASKS, MIN_ALL = 500, 100, 0.02  # PLAN: uninformative below these; R's draw exclusion (logits)


def shift(mask, ar, ce, L0, F, wc):
    """The registered mean shift of the rows in mask (repaired - real e2, logits) for one weighting; None if a fold lacks rows."""
    out = []
    for a in (ar, ~ar):
        m = mask & a
        n = np.bincount(ce[m], minlength=939).astype(float)
        s = np.stack([np.bincount(ce[m], weights=L0[m, v], minlength=939) for v in range(5)], 1)
        d = np.bincount(F, weights=wc * n, minlength=5)
        if (d == 0).any():
            return None
        out.append((np.stack([np.bincount(F, weights=wc * s[:, v], minlength=5) for v in range(5)], 1) / d[:, None], d))
    (me, de), (mr, dr) = out
    O, wt = me - mr, np.minimum(de, dr); m = ~np.eye(5, dtype=bool)
    return float((-O * wt[:, None])[m].sum() / np.broadcast_to(wt[:, None], (5, 5))[m].sum())


def main():
    ap = argparse.ArgumentParser()
    for a in ("flags", "reg-rows", "reg-manifest", "lt-scores", "llama-fix", "llama-plus", "llama-reading"):
        ap.add_argument(f"--{a}", required=True, type=lambda p: Path(p).resolve())
    args = ap.parse_args()
    for f in (Path(__file__).resolve(), HERE / "analyze_meanpool_POSTHOC.py", HERE / "analyze_gap_where.py", HERE / "PLAN_identical_POSTHOC.md",
              mp.REG_RESULT, args.flags, args.reg_manifest):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if any(sha(p) != reg.PINS.get(p.name) for p in (args.llama_reading, args.llama_fix, args.llama_plus)):
        raise SystemExit("a Llama input isn't the pinned committed file")
    man = json.loads(args.reg_manifest.read_text())
    if sha(args.reg_rows) != man["rows record sha256"]:
        raise SystemExit("the registered rows record isn't its manifest's")
    rrec = json.loads(args.reg_rows.read_text()); Xr = np.load(args.reg_rows.parent / rrec["rows file"])
    if sha(args.reg_rows.parent / rrec["rows file"]) != rrec["rows sha256"]:
        raise SystemExit("the registered rows file isn't its record's")
    frec = json.loads(args.flags.read_text()); Z = np.load(args.flags.parent / frec["flags file"])
    if sha(args.flags.parent / frec["flags file"]) != frec["flags sha256"] or frec["inputs sha256"][args.reg_rows.name] != sha(args.reg_rows):
        raise SystemExit("the flags file isn't its record's, or was built from other rows")
    lay = tuple(rrec["readers (probe run, layer per test fold)"]["llama"][1])
    R = reg.reader_llama_olmo("llama", args.lt_scores, args.llama_fix, args.llama_plus, json.loads(args.llama_reading.read_text()), lay)
    U = rrec["units (per reader: runs.json index, key sha256, source, fold; arrays' unit indexes this)"]["llama"]
    pos = {int(i): ("e2", c) for c, i in enumerate(R["ei"])} | {int(i): ("rep", c) for c, i in enumerate(R["si"])}
    u_arm = np.array([pos[i][0] == "e2" for i in U["runs.json index"]]); u_cell = np.array([pos[i][1] for i in U["runs.json index"]])
    u = Xr["llama__unit"]; F = R["F"]; ar, ce = u_arm[u], u_cell[u]
    own = np.arange(5)[None, :] == F[ce][:, None]
    L0 = np.where(own, 0.0, Xr["llama__logit"].astype(np.float64))
    ones = np.ones(len(R["draws"][0]))
    out = {"POST HOC": True, "plan": "PLAN_identical_POSTHOC.md (this folder)", "git": head,
           "inputs sha256": {p.name: sha(p) for p in (args.flags, args.reg_rows, args.reg_manifest, args.llama_reading, args.llama_fix, args.llama_plus, mp.REG_RESULT)},
           "conventions": "registered: real e2 vs repaired-plus by cell, thresholds redrawn, 2,000 main-task draws, seed 20261011, 5% and 1%"}

    # the engine reproduces the registered Llama decomposition (every category and statistic)
    registered = json.loads(mp.REG_RESULT.read_text())["readers"]["llama"]
    for a in RATES:
        _, _, entry = mp.decompose(mp.categories(Xr, "llama__"), ar, ce, L0, F, R["ctix"], R["draws"], reg.thresholds(R, a, ones), lambda w: reg.thresholds(R, a, w))
        if not mp.same(entry, {k: v for k, v in registered[str(a)].items() if k != "G and its interval equal the committed reading's (1e-9)"}):
            raise SystemExit(f"check failed at {a}: the engine doesn't reproduce the registered Llama decomposition")
    out["check"] = "the engine reproduces the registered Llama decomposition at 5% and 1% (every category and statistic, 1e-12), bash's mean shift " \
                   f"{registered['0.05']['categories']['family bash']['mean shift']['point']:.6f} included"
    print("check passed", flush=True)

    # join the flags to the registered rows by (unit, step, action)
    n = len(Z["unit"])
    if any(len(Z[k]) != n for k in ("step", "action", "identical", "group")) or not np.isin(Z["identical"], (0, 1)).all() \
            or ((Z["group"] >= 0) != (Z["identical"] == 1)).any():
        raise SystemExit("the flags arrays are inconsistent")
    fl = {}
    for a_, b_, c_, d_, g_ in zip(Z["unit"].tolist(), Z["step"].tolist(), Z["action"].tolist(), Z["identical"].tolist(), Z["group"].tolist()):
        if (a_, b_, c_) in fl:
            raise SystemExit("a repeated (unit, step, action) in the flags")
        fl[(a_, b_, c_)] = (d_, g_)
    keys = list(zip(u.tolist(), Xr["llama__step"].tolist(), Xr["llama__action"].tolist()))
    if any(k not in fl for k in keys):
        raise SystemExit(f"{sum(k not in fl for k in keys)} registered rows without a flag")
    ident = np.array([fl[k][0] for k in keys], bool); grp = np.array([fl[k][1] for k in keys])
    counts = {arm_: {"rows": int(sel.sum()), "identical": int((ident & sel).sum()), "share": float((ident & sel).sum() / sel.sum())}
              for arm_, sel in (("real e2", ar), ("repaired-plus", ~ar))}
    out["identical rows"] = counts | {"main tasks with an identical row": len(set(R["ctix"][ce[ident]].tolist())), "main tasks": len(set(R["ctix"].tolist()))}

    # matched groups (Amendment 1): a (cell, group id) with kept rows in both arms; per group, repaired mean - real e2 mean
    gk = np.where(grp >= 0, ce.astype(np.int64) * 1_000_000 + grp, -1)
    uq, inv = np.unique(gk[grp >= 0], return_inverse=True)
    gi = np.full(len(u), -1); gi[grp >= 0] = inv
    sums_ = []
    for a in (ar, ~ar):
        m = (gi >= 0) & a
        cnt = np.bincount(gi[m], minlength=len(uq)).astype(float)
        s = np.stack([np.bincount(gi[m], weights=L0[m, v], minlength=len(uq)) for v in range(5)], 1)
        sums_.append((cnt, s))
    both = (sums_[0][0] > 0) & (sums_[1][0] > 0)
    d = (sums_[1][1][both] / sums_[1][0][both][:, None]) - (sums_[0][1][both] / sums_[0][0][both][:, None])  # [group, v]
    gcell = (uq[both] // 1_000_000).astype(int); gF = F[gcell]
    out["matched groups"] = {"groups with kept rows in both arms": int(both.sum()), "their rows (real e2, repaired-plus)": [int(sums_[0][0][both].sum()), int(sums_[1][0][both].sum())],
                             "main tasks with a matched group": len(set(R["ctix"][gcell].tolist()))}

    def matched_shift(wc):
        wg = wc[gcell]
        W = np.bincount(gF, weights=wg, minlength=5)
        if (W == 0).any():
            return None
        avg = np.stack([np.bincount(gF, weights=wg * d[:, v], minlength=5) for v in range(5)], 1) / W[:, None]
        m = ~np.eye(5, dtype=bool)
        return float((avg * W[:, None])[m].sum() / np.broadcast_to(W[:, None], (5, 5))[m].sum())

    sets = {"identical": ident, "non-identical": ~ident, "all rows": np.ones(len(u), bool)}
    for a in RATES:
        _, _, entry = mp.decompose(sets, ar, ce, L0, F, R["ctix"], R["draws"], reg.thresholds(R, a, ones), lambda w: reg.thresholds(R, a, w))
        out[str(a)] = entry
    one_c = np.ones(939)
    pts = {k: shift(m, ar, ce, L0, F, one_c) for k, m in sets.items()}
    for k in sets:  # the same formula as the engine's mean shift
        e = out["0.05"]["categories"][k]["mean shift"]["point"]
        if (pts[k] is None) != (not isinstance(e, float)) or (pts[k] is not None and abs(e - pts[k]) > 1e-12):
            raise SystemExit(f"the mean-shift formula disagrees with the engine on {k}")
    M0 = matched_shift(one_c)
    rb, mb, rmb, miss, small = [], [], [], 0, 0
    for w in R["draws"]:
        wc = w[R["ctix"]].astype(float)
        si_, sa_, m_ = shift(ident, ar, ce, L0, F, wc), shift(sets["all rows"], ar, ce, L0, F, wc), matched_shift(wc)
        if m_ is not None:
            mb.append(m_)
        if si_ is None or sa_ is None or m_ is None:
            miss += 1; continue
        if abs(sa_) < MIN_ALL:
            small += 1; continue
        rb.append(si_ / sa_); rmb.append(m_ / sa_)
    iv = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])] if len(x) >= 1800 else f"not reported: {len(x)} valid draws"  # noqa: E731
    sa0 = pts["all rows"]
    out["M (matched-pair shift, logits)"] = {"point": M0, "95%": iv(mb), "valid draws": len(mb)}
    out["R_M = M / shift(all rows)"] = {"point": None if M0 is None or not sa0 else M0 / sa0, "95%": iv(rmb), "valid draws": len(rmb)}
    out["R = shift(identical) / shift(all rows)"] = {"point": None if pts["identical"] is None or not sa0 else pts["identical"] / sa0, "95%": iv(rb), "valid draws": len(rb)}
    out["draws left out of the ratios"] = {"a fold without rows or groups": miss, f"|all-rows shift| < {MIN_ALL}": small}
    mg = out["matched groups"]; Mi = out["M (matched-pair shift, logits)"]["95%"]; RM = out["R_M = M / shift(all rows)"]["point"]
    if min(c["identical"] for c in counts.values()) < MIN_ROWS or out["identical rows"]["main tasks with an identical row"] < MIN_TASKS \
            or mg["groups with kept rows in both arms"] < MIN_ROWS or mg["main tasks with a matched group"] < MIN_TASKS or RM is None or not isinstance(Mi, list):
        rd = "uninformative"
    elif RM >= 0.6 and (Mi[1] < 0 or Mi[0] > 0):
        rd = "context (descriptive)"
    elif abs(RM) <= 0.3 and Mi[0] <= 0 <= Mi[1]:
        rd = "output text (descriptive)"
    else:
        rd = "mixed (descriptive)"
    out["reading (PLAN, Amendment 1)"] = rd
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"identical_reading_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    f2 = lambda s, p=2: reg.fmt(s, p)  # noqa: E731
    lines = ["POST HOC (PLAN_identical_POSTHOC.md). Llama-3.3-70B, real e2 vs repaired-plus by cell, R2J readout, registered conventions "
             "(thresholds redrawn, 2,000 draws, seed 20261011). Identical = the output's raw content equals some tool output of the same cell's other-arm run.",
             "identical rows: " + json.dumps(out["identical rows"]), ""]
    for a in (0.01, 0.05):  # 1% first (Aswin via 0d, 10-13 about 11:50); 5% beside it as the registered analysis's committed primary rate
        e = out[str(a)]
        lines.append(f"== {float(a):.0%}" + (" (beside 1%: the registered RULE's primary rate)" if a == 0.05 else "") + ": G = catch(real e2) - catch(repaired-plus) = {f2(e['G (points)'])} points")
        lines.append(f"{'rows':16s} {'rows % e2/rep':>13s} {'mean shift rep-e2':>24s} {'A':>23s} {'A/G':>22s}")
        for k in sets:
            x = e["categories"][k]; sh = x["share of rows (real e2, repaired)"]
            lines.append(f"{k:16s} {100 * sh[0]:5.1f}/{100 * sh[1]:5.1f}  {f2(x['mean shift'], 3):>24s} {f2(x['A']):>23s} {f2(x['A/G']):>22s}")
    for k in ("M (matched-pair shift, logits)", "R_M = M / shift(all rows)", "R = shift(identical) / shift(all rows)"):
        r_ = out[k]; p_ = r_["point"]
        lines.append(f"{k} = " + ("n/a" if p_ is None else f"{p_:+.3f}") + (f" [{r_['95%'][0]:+.3f}, {r_['95%'][1]:+.3f}]" if isinstance(r_["95%"], list) else f" ({r_['95%']})")
                     + f"; valid draws {r_['valid draws']}")
    lines.append("matched groups: " + json.dumps(out["matched groups"]) + "; draws left out of the ratios: " + json.dumps(out["draws left out of the ratios"]))
    lines.append(f"reading (PLAN, Amendment 1): {rd}")
    text = "\n".join(lines)
    with open(HERE / "results" / f"identical_reading_POSTHOC_{stamp}_printout.txt", "x") as fh:
        fh.write(text + "\n")
    print(text); print(dest)


if __name__ == "__main__":
    main()
