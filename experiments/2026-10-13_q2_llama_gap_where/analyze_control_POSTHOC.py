"""POST HOC (PLAN_control_POSTHOC.md here): Llama-3.3-70B, R2J readout. (a) output type, (b) the position split (first 5 /
middle / last 5 outputs of each run) and (c) byte-identical vs differing outputs, for the repaired arm (repaired-plus -
second real run) and the real-vs-real control (first real run - second real run), side by side. Local, $0. Identical or not
identical only (the format-tell stop).

The engine is analyze_meanpool_POSTHOC.decompose() under the registered conventions (thresholds redrawn, 2,000 draws,
seed 20261011); it must reproduce the registered Llama decomposition first. Gates (PLAN): the first real run's row means
reproduce its stored nested scores (1e-3) with the same decisions; the control G and its interval equal G2's committed
"real e2 - e1" row (1e-9); each run's n from the flags equals the rows' n_outputs; the second real run's outputs are the
same in both flag files; the repaired M equals the identical test's committed M. 1% leads, 5% beside it.

  uv run --no-project --python 3.12 --with numpy python experiments/2026-10-13_q2_llama_gap_where/analyze_control_POSTHOC.py \\
      --e1-rows <results/rows_e1_POSTHOC_<stamp>.json> --ident-flags <results/identical_POSTHOC_2026-10-13_115613.json> \\
      --ident-reading <results/identical_reading_POSTHOC_2026-10-13_115755.json> --reg-rows <results/rows_2026-10-13_110759.json> \\
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
E2_E1 = "real e2 - e1 (two real runs; e1 trained the detector)"


def matched_shift(gcell, d, F, wc):
    """analyze_identical_POSTHOC.py's M: per matched group (arm mean - second-real-run mean, per nested model), averaged in
    each fold, combined over the 20 models weighted by the fold's group count."""
    wg = wc[gcell]; gF = F[gcell]
    W = np.bincount(gF, weights=wg, minlength=5)
    if (W == 0).any():
        return None
    avg = np.stack([np.bincount(gF, weights=wg * d[:, v], minlength=5) for v in range(5)], 1) / W[:, None]
    m = ~np.eye(5, dtype=bool)
    return float((avg * W[:, None])[m].sum() / np.broadcast_to(W[:, None], (5, 5))[m].sum())


def groups(grp, ar, ce, L0):
    """Matched groups: (cell, group id) with rows in both arms; per group the arm's mean minus the second real run's."""
    gk = np.where(grp >= 0, ce.astype(np.int64) * 1_000_000 + grp, -1)
    uq, inv = np.unique(gk[grp >= 0], return_inverse=True)
    gi = np.full(len(ar), -1); gi[grp >= 0] = inv
    s_ = []
    for a in (ar, ~ar):
        m = (gi >= 0) & a
        s_.append((np.bincount(gi[m], minlength=len(uq)).astype(float), np.stack([np.bincount(gi[m], weights=L0[m, v], minlength=len(uq)) for v in range(5)], 1)))
    both = (s_[0][0] > 0) & (s_[1][0] > 0)
    d = s_[1][1][both] / s_[1][0][both][:, None] - s_[0][1][both] / s_[0][0][both][:, None]
    return (uq[both] // 1_000_000).astype(int), d


def position_split(k, n):
    return np.where(k < 5, 0, np.where(k >= n - 5, 2, 1))  # first 5, middle, last 5


def flag_index(Z, pre=""):
    n = len(Z[f"{pre}unit"])
    if any(len(Z[f"{pre}{c}"]) != n for c in ("step", "action", "identical", "group")) or not np.isin(Z[f"{pre}identical"], (0, 1)).all() \
            or ((Z[f"{pre}group"] >= 0) != (Z[f"{pre}identical"] == 1)).any():
        raise SystemExit("the flags arrays are inconsistent")
    out = {}
    for t in zip(Z[f"{pre}unit"].tolist(), Z[f"{pre}step"].tolist(), Z[f"{pre}action"].tolist(), Z[f"{pre}identical"].tolist(), Z[f"{pre}group"].tolist()):
        if t[:3] in out:
            raise SystemExit("a repeated (unit, step, action) in the flags")
        out[t[:3]] = t[3:]
    return out


def ranks(index):
    """Per unit: (step, action) -> (k, n), k the index among all of the unit's outputs in run order."""
    by = {}
    for (u, s, a) in index:
        by.setdefault(u, []).append((s, a))
    return {(u, s, a): (k, len(v)) for u, v in by.items() for k, (s, a) in enumerate(sorted(v))}


def main():
    ap = argparse.ArgumentParser()
    for a in ("e1-rows", "ident-flags", "ident-reading", "reg-rows", "reg-manifest", "lt-scores", "llama-fix", "llama-plus", "llama-reading"):
        ap.add_argument(f"--{a}", required=True, type=lambda p: Path(p).resolve())
    args = ap.parse_args()
    for f in (Path(__file__).resolve(), HERE / "analyze_meanpool_POSTHOC.py", HERE / "analyze_gap_where.py", HERE / "PLAN_control_POSTHOC.md",
              mp.REG_RESULT, args.e1_rows, args.ident_flags, args.ident_reading, args.reg_manifest):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if any(sha(p) != reg.PINS.get(p.name) for p in (args.llama_reading, args.llama_fix, args.llama_plus)):
        raise SystemExit("a Llama input isn't the pinned committed file")
    if sha(args.reg_rows) != json.loads(args.reg_manifest.read_text())["rows record sha256"]:
        raise SystemExit("the registered rows record isn't its manifest's")
    rrec = json.loads(args.reg_rows.read_text()); Xr = np.load(args.reg_rows.parent / rrec["rows file"])
    frec = json.loads(args.ident_flags.read_text()); Zi = np.load(args.ident_flags.parent / frec["flags file"])
    erec = json.loads(args.e1_rows.read_text()); X1 = np.load(args.e1_rows.parent / erec["rows file"])
    if sha(args.reg_rows.parent / rrec["rows file"]) != rrec["rows sha256"] or sha(args.ident_flags.parent / frec["flags file"]) != frec["flags sha256"] \
            or sha(args.e1_rows.parent / erec["rows file"]) != erec["sha256 of the rows file"] \
            or frec["inputs sha256"][args.reg_rows.name] != sha(args.reg_rows) or erec["sha256 of inputs"][args.reg_rows.name] != sha(args.reg_rows):
        raise SystemExit("a rows or flags file isn't its record's, or was built from other registered rows")
    lay = tuple(rrec["readers (probe run, layer per test fold)"]["llama"][1])
    if tuple(erec["layer per test fold"]) != lay:
        raise SystemExit("the first real run's rows use other layers")
    R = reg.reader_llama_olmo("llama", args.lt_scores, args.llama_fix, args.llama_plus, json.loads(args.llama_reading.read_text()), lay)
    runs = R["runs"]; F, ctix = R["F"], R["ctix"]
    cpos = {runs[int(i)]["cell"]: c for c, i in enumerate(R["si"])}
    if [runs[int(i)]["cell"] for i in R["ei"]] != [runs[int(i)]["cell"] for i in R["si"]]:
        raise SystemExit("the cells' second real runs and repaired runs are not aligned")
    ones = np.ones(len(R["draws"][0]))
    out = {"POST HOC": True, "plan": "PLAN_control_POSTHOC.md (this folder)", "git": head,
           "sha256 of inputs": {p.name: sha(p) for p in (args.e1_rows, args.ident_flags, args.ident_reading, args.reg_rows, args.reg_manifest,
                                                         args.llama_reading, args.llama_fix, args.llama_plus, mp.REG_RESULT)},
           "conventions": "registered: by cell, thresholds redrawn, 2,000 main-task draws, seed 20261011; 1% leads, 5% beside it (the registered RULE's rate)",
           "sign": "mean shift = arm - second real run (logits); G = catch(second real run) - catch(arm), points", "gates": {}, "arms": {}}

    # the registered rows: second real run and repaired-plus
    U = rrec["units (per reader: runs.json index, key sha256, source, fold; arrays' unit indexes this)"]["llama"]
    upos = {int(i): ("e2", c) for c, i in enumerate(R["ei"])} | {int(i): ("rep", c) for c, i in enumerate(R["si"])}
    u_e2 = np.array([upos[i][0] == "e2" for i in U["runs.json index"]]); u_cell = np.array([upos[i][1] for i in U["runs.json index"]])
    u = Xr["llama__unit"]; ar, ce = u_e2[u], u_cell[u]
    own = np.arange(5)[None, :] == F[ce][:, None]
    L0 = np.where(own, 0.0, Xr["llama__logit"].astype(np.float64))
    registered = json.loads(mp.REG_RESULT.read_text())["readers"]["llama"]
    for a in RATES:  # engine check
        _, _, entry = mp.decompose(mp.categories(Xr, "llama__"), ar, ce, L0, F, ctix, R["draws"], reg.thresholds(R, a, ones), lambda w: reg.thresholds(R, a, w))
        if not mp.same(entry, {k: v for k, v in registered[str(a)].items() if k != "G and its interval equal the committed reading's (1e-9)"}):
            raise SystemExit(f"engine check failed at {a}")
    out["gates"]["engine"] = "reproduces the registered Llama decomposition at 1% and 5% (every category and statistic, 1e-12)"
    print("engine check passed", flush=True)
    fi = flag_index(Zi); rk = ranks(fi)
    keys = list(zip(u.tolist(), Xr["llama__step"].tolist(), Xr["llama__action"].tolist()))
    if any(k not in fi for k in keys):
        raise SystemExit("a registered row without a flag")
    n_flags = {uu: n for (uu, _, _), (_, n) in rk.items()}
    if any(n_flags.get(j) != int(x) for j, x in enumerate(Xr["llama__unit n_outputs"])):
        raise SystemExit("gate 4: a run's n from the flags differs from the rows' n_outputs")
    kr = np.array([rk[k][0] for k in keys]); nr = np.array([rk[k][1] for k in keys])
    rep_ident = np.array([fi[k][0] for k in keys], bool); rep_grp = np.array([fi[k][1] for k in keys])
    e2_k = {(int(U["runs.json index"][uu]), s, a): v for (uu, s, a), v in rk.items() if u_e2[uu]}  # by runs.json index, for gate 4

    # the first real run's rows (control arm)
    E = erec["units (runs.json index, arm, fold; arrays' unit indexes this)"]
    e_idx, e_arm, e_fold = np.array(E["runs.json index"]), np.array(E["arm"]), np.array(E["fold"])
    if any(runs[int(i)]["arm"] != {"e1": "real-attack-e1", "e2": "real-attack-e2"}[a_] for i, a_ in zip(e_idx, e_arm)):
        raise SystemExit("the first-real-run job's units aren't the runs.json arms they name")
    e_cell = np.array([cpos[runs[int(i)]["cell"]] for i in e_idx])
    if not (e_fold == F[e_cell]).all():
        raise SystemExit("a first-real-run unit's fold differs from its cell's")
    u1 = X1["e1__unit"]
    if not (e_arm[u1] == "e1").all():
        raise SystemExit("a first-real-run row from a second-real-run unit")
    c1 = e_cell[u1]; own1 = np.arange(5)[None, :] == F[c1][:, None]
    L1 = X1["e1__logit"].astype(np.float64)
    if not np.isfinite(L1[~own1]).all():
        raise SystemExit("a non-finite first-real-run logit")
    L1 = np.where(own1, 0.0, L1)
    # gate 2: the first real run's row means reproduce its stored nested scores, with the same decisions
    N = {l: np.load(args.lt_scores / f"scores_L{l}.npz")["output_end__mean__task_nested"] for l in set(lay)}
    e1_of = {c: int(i) for c, i in zip(e_cell, e_idx) if runs[int(i)]["arm"] == "real-attack-e1"}
    task = np.array([x["main_task"] for x in runs])
    if sorted(e1_of) != list(range(939)) or any(task[e1_of[c]] != task[int(R["ei"][c])] for c in range(939)) \
            or sum(x["arm"] == "real-attack-e1" for x in runs) != 939:
        raise SystemExit("not one first real run per cell, on the cell's main task (G2's e1 weights equal the cell weights only then)")
    S1 = np.stack([N[lay[F[c]]][F[c], :, e1_of[c]] for c in range(939)])
    n1 = np.bincount(c1, minlength=939)
    if (n1 == 0).any():
        raise SystemExit("a first real run with no rows")
    m1 = np.stack([np.bincount(c1, weights=L1[:, v], minlength=939) for v in range(5)], 1) / n1[:, None]
    valid = np.arange(5)[None, :] != F[:, None]
    d2 = float(np.abs(m1 - S1)[valid].max())
    if not np.isfinite(d2) or d2 > reg.TOL:
        raise SystemExit(f"gate 2 failed: max |row mean - stored| {d2}")
    for a in RATES:
        T = reg.thresholds(R, a, ones)
        if not (np.where(valid, m1 > T[F], False) == np.where(valid, S1 > T[F], False)).all():
            raise SystemExit(f"gate 2 failed at {a}: a decision differs")
    out["gates"]["first real run: max |row mean - stored nested score| (tolerance 1e-3), same decisions at 1% and 5%"] = d2
    # control flags (between the two real runs) and gate 4 on the second real run's outputs
    fc = {(int(a_), int(b_), int(c_)): (int(k_), int(i_), int(g_)) for a_, b_, c_, k_, i_, g_ in
          zip(X1["flags__unit"], X1["flags__step"], X1["flags__action"], X1["flags__k"], X1["flags__identical"], X1["flags__group"])}
    if len(fc) != len(X1["flags__unit"]) or not np.isin(X1["flags__identical"], (0, 1)).all() or ((X1["flags__group"] >= 0) != (X1["flags__identical"] == 1)).any():
        raise SystemExit("the control flags are inconsistent")
    e2_new = {(int(e_idx[uu]), s, a): k for (uu, s, a), (k, _, _) in fc.items() if e_arm[uu] == "e2"}
    if set(e2_new) != set(e2_k) or any(e2_new[x] != e2_k[x][0] for x in e2_new):
        raise SystemExit("gate 4: the second real run's outputs differ between the two flag files")
    n_new = {}
    for (uu, _, _) in fc:
        n_new[uu] = n_new.get(uu, 0) + 1
    k1 = X1["e1__k"]; nn1 = X1["e1__n"]
    if any(n_new[int(j)] != int(x) for j, x in zip(u1, nn1)):
        raise SystemExit("gate 4: a first real run's n differs between its rows and its flags")
    keys1 = list(zip(u1.tolist(), X1["e1__step"].tolist(), X1["e1__action"].tolist()))
    if any(fc.get(k, (None,))[0] != int(kk) for k, kk in zip(keys1, k1)):
        raise SystemExit("gate 4: a first-real-run row's k differs from its flags")
    new_of_e2 = {int(i): uu for uu, (i, a_) in enumerate(zip(e_idx, e_arm)) if a_ == "e2"}
    keys2 = [(new_of_e2[int(U["runs.json index"][uu])], s, a) for (uu, s, a), e in zip(keys, ar) if e]
    out["gates"]["joins"] = "each run's n from the flags equals the rows' n_outputs; the second real run's outputs and k agree across the two flag files"

    # the two arms' rows, categories and matched groups
    fam = {f"output type: {lab}": k for k, lab in enumerate(reg.CATS["family"])}
    POS = ("first 5", "middle", "last 5")

    def cats_for(famv, posv, identv):
        c = {name: famv == k for name, k in fam.items()}
        c["output type: {create, str_replace, insert}"] = np.isin(famv, (2, 3, 4))
        c |= {f"position: {p}": posv == k for k, p in enumerate(POS)}
        c |= {"identical": identv, "differing": ~identv, "all rows": np.ones(len(famv), bool)}
        return c
    arms = {}
    rep = ar  # registered rows: second real run (True) and repaired-plus (False)
    arms["repaired-plus"] = {"ar": rep, "ce": ce, "L0": L0, "cats": cats_for(Xr["llama__family"], position_split(kr, nr), rep_ident), "grp": rep_grp}
    e2 = ar
    ctl_ident_e2 = np.array([fc[k][1] for k in keys2], bool); ctl_grp_e2 = np.array([fc[k][2] for k in keys2])
    ctl_ident_e1 = np.array([fc[k][1] for k in keys1], bool); ctl_grp_e1 = np.array([fc[k][2] for k in keys1])
    arms["first real run"] = {"ar": np.r_[np.ones(e2.sum(), bool), np.zeros(len(u1), bool)], "ce": np.r_[ce[e2], c1], "L0": np.r_[L0[e2], L1],
                              "cats": cats_for(np.r_[Xr["llama__family"][e2], X1["e1__family"]], np.r_[position_split(kr[e2], nr[e2]), position_split(k1, nn1)],
                                               np.r_[ctl_ident_e2, ctl_ident_e1]), "grp": np.r_[ctl_grp_e2, ctl_grp_e1]}
    committed_ctl = json.loads(args.llama_reading.read_text())["repaired-plus, by part 2's rules (its 'fixed' slot is repaired-plus)"]
    ident_committed = json.loads(args.ident_reading.read_text())["M (matched-pair shift, logits)"]["point"]
    for name, A in arms.items():
        res = {"identical rows (second real run, arm)": [int((A["cats"]["identical"] & A["ar"]).sum()), int((A["cats"]["identical"] & ~A["ar"]).sum())],
               "rows (second real run, arm)": [int(A["ar"].sum()), int((~A["ar"]).sum())]}
        for a in RATES:
            G0, Gb, entry = mp.decompose(A["cats"], A["ar"], A["ce"], A["L0"], F, ctix, R["draws"], reg.thresholds(R, a, ones), lambda w: reg.thresholds(R, a, w))
            if name == "first real run":  # gate 3: the control G equals G2's committed real e2 - e1 row
                want = committed_ctl[str(a)]["rows (points; 95% task bootstrap, thresholds redrawn)"][E2_E1]
                lo, hi = np.percentile(Gb, [2.5, 97.5])
                if not (abs(G0 - want[0]) < 1e-9 and abs(lo - want[1][0]) < 1e-9 and abs(hi - want[1][1]) < 1e-9):
                    raise SystemExit(f"gate 3 failed at {a}: control G {G0} [{lo}, {hi}] vs committed {want}")
                out["gates"][f"control G equals G2's committed '{E2_E1}' @ {a}"] = want
            res[str(a)] = entry
        gcell, d = groups(A["grp"], A["ar"], A["ce"], A["L0"])
        M0 = matched_shift(gcell, d, F, np.ones(939))
        if name == "repaired-plus" and (M0 is None or abs(M0 - ident_committed) > 1e-12):
            raise SystemExit("the repaired M differs from the identical test's committed M")
        sa0 = res["0.01"]["categories"]["all rows"]["mean shift"]["point"]
        mb = []
        for w in R["draws"]:
            wc = w[ctix].astype(float)
            m_ = matched_shift(gcell, d, F, wc)
            if m_ is not None:
                mb.append(m_)
        res["matched groups"] = {"groups with rows in both arms": int(len(gcell)), "main tasks with a matched group": len(set(ctix[gcell].tolist()))}
        res["M (matched-pair shift, logits)"] = {"point": M0, "95%": [float(x) for x in np.percentile(mb, [2.5, 97.5])] if len(mb) >= 1800 else "not reported",
                                                 "valid draws": len(mb)}
        res["M / all-rows shift"] = None if M0 is None or not isinstance(sa0, float) or abs(sa0) < 0.02 else M0 / sa0  # PLAN_identical's 0.02 guard
        out["arms"][name] = res
        print(f"{name}: done", flush=True)
    out["gates"]["repaired M equals the identical test's committed M (1e-12)"] = ident_committed

    # the reading (PLAN): 1% point estimates; the mean shift doesn't depend on the rate
    rs, cs = (out["arms"][n]["0.01"]["categories"]["all rows"]["mean shift"] for n in ("repaired-plus", "first real run"))
    rx = isinstance(rs["95%"], list) and (rs["95%"][1] < 0 or rs["95%"][0] > 0)
    ci = isinstance(cs["95%"], list) and cs["95%"][0] <= 0 <= cs["95%"][1]
    if rx and ci and abs(cs["point"]) < abs(rs["point"]) / 3:
        rd = "simulation-specific (descriptive)"
    elif np.sign(cs["point"]) == np.sign(rs["point"]) and abs(cs["point"]) >= abs(rs["point"]) / 2:
        rd = "run-to-run (descriptive)"
    else:
        rd = "mixed (descriptive)"
    out["reading (PLAN)"] = rd
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"control_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    text = table(out)
    with open(HERE / "results" / f"control_POSTHOC_{stamp}_printout.txt", "x") as fh:
        fh.write(text + "\n")
    print(text); print(dest)


def table(out):
    f = lambda s, p=2: reg.fmt(s, p) if isinstance(s, dict) else "n/a"  # noqa: E731
    A, C = out["arms"]["repaired-plus"], out["arms"]["first real run"]
    lines = ["POST HOC (PLAN_control_POSTHOC.md). Llama-3.3-70B, R2J readout, by cell. Repaired = repaired-plus - second real run; "
             "control = first real run - second real run (the first real run trained the detector, in other folds).",
             "Registered conventions: thresholds redrawn, 2,000 main-task draws, seed 20261011. Mean shift = arm - second real run (logits); "
             "G = catch(second real run) - catch(arm) (points). Identical = byte-identical to some output of the other run of the cell.",
             "gates: " + json.dumps(out["gates"]), ""]
    for a in ("0.01", "0.05"):
        why = "" if a == "0.01" else " (beside 1%: the registered RULE's primary rate)"
        lines.append(f"== {float(a):.0%}{why}: G repaired = {f(A[a]['G (points)'])}; G control = {f(C[a]['G (points)'])}")
        lines.append(f"{'category':40s} {'rows % e2/arm (rep)':>19s} {'mean shift (rep)':>24s} {'A (rep)':>22s} {'A/G (rep)':>21s} | "
                     f"{'rows % e2/arm (ctl)':>19s} {'mean shift (ctl)':>24s} {'A (ctl)':>22s} {'A/G (ctl)':>21s}")
        for c in A[a]["categories"]:
            x, y = A[a]["categories"][c], C[a]["categories"][c]
            sx, sy = x["share of rows (real e2, repaired)"], y["share of rows (real e2, repaired)"]
            lines.append(f"{c:40s} {100 * sx[0]:8.1f}/{100 * sx[1]:5.1f}     {f(x['mean shift'], 3):>24s} {f(x['A']):>22s} {f(x['A/G']):>21s} | "
                         f"{100 * sy[0]:8.1f}/{100 * sy[1]:5.1f}     {f(y['mean shift'], 3):>24s} {f(y['A']):>22s} {f(y['A/G']):>21s}")
        lines.append("")
    for name, X in (("repaired", A), ("control", C)):
        m = X["M (matched-pair shift, logits)"]
        lines.append(f"{name}: identical rows (second real run, arm) {X['identical rows (second real run, arm)']} of {X['rows (second real run, arm)']}; "
                     f"matched groups {X['matched groups']}; M = " + ("n/a" if m["point"] is None else f"{m['point']:+.3f}")
                     + (f" [{m['95%'][0]:+.3f}, {m['95%'][1]:+.3f}]" if isinstance(m["95%"], list) else f" ({m['95%']})")
                     + "; M / all-rows shift = " + ("n/a" if X["M / all-rows shift"] is None else f"{X['M / all-rows shift']:+.2f}"))
    lines.append(f"reading (PLAN): {out['reading (PLAN)']}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
