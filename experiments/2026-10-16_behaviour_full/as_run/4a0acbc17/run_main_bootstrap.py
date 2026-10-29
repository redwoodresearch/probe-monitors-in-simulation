"""Main-clustered intervals for the post (09's RULE item 29), beside the records' cell-level intervals, which stay as they are.
From the committed records only (no scan). For each view (S_old, S_new, S_newmains, S_paper493, pooled) and cohort (all
launched cells; and the frozen cohort, filtered by key, when given):
  - attempt rates (the attempt-rate record bc594d28's definition: every labelled run), non-attempt counts with their n,
    and the paired sim - run1, sim - run2 and run2 - run1 differences (non-attempt pairs are the attempt pairs negated);
  - behaviour means of admitted runs (behaviour_full) and their paired differences;
  - median steps of ATTEMPTED admitted runs per arm, and the paired sim - run1 medians: the difference of the two arms'
    medians on cells where both attempted, and the median of the per-cell differences.
Main-clustered bootstrap: one main is one main across all four sets, the map Stage A's all-origins record draws over
(record A @ 5473efdb8: cell_hits_draw0's mains aligned with the sorted cohort cells, identical across its rows), asserted by
key equal to each set's own `main` field, which also covers the launched cells outside the cohort; the main set equals
record A's meta.merge.mains. Mains are resampled within their environment, 10,000 draws, seed [20261015, 40] (record A's); each main's cells move together, and every paired difference uses the same draws. Gates: on all launched
cells the point estimates must equal the three records' (attempt_rate4, behaviour_full, cap_summary) before anything is
written. McNemar p values are copied or recomputed as "assumes independent cells". One Modal CPU container.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with "numpy==2.3.3" modal run \\
      experiments/2026-10-16_behaviour_full/run_main_bootstrap.py [--no-cohort]
"""

import hashlib
import json
import subprocess
from datetime import datetime
from math import comb
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-behaviourfull_mainboot_kill-2026-11-16"
RECORDS = {"behaviour": (HERE / "results" / "behaviour_full_2026-10-16_025753.json", "40ad66d382cd26a9"),
           "rate": (HERE.parent / "2026-10-15_attempt_rate" / "results" / "attempt_rate4_2026-10-16_030654.json", "a539b184f6e9e4dc"),
           "cap": (HERE / "results" / "cap_summary_2026-10-16_033450.json", "4247ce215d00f0b5")}
MAPS = {"tasks196": ("64c6cb37f", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb"),
        "plus311": ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json", "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b"),
        "m710": ("32abe4120", "experiments/2026-10-15_fpr_sweep_507_prep/config/membership710.json", "6516d15fab77d7f344497e72db10b60c5f709b35c3a1859a2c9e016d2fc6e4ad"),
        "facts493": ("d800edc6a", "experiments/2026-10-15_q3_sp493/results/sp493_facts_2026-10-15_165730.json", "07caa527db89ec519314486d8c529c37430ca41d773a8bce26c041eb141b0a57")}
SETS, ARMS, LAB = ("S_old", "S_new", "S_newmains", "S_paper493"), ("sim", "run1", "run2"), ("attempted", "unclear", "declined")
PAIRS = (("sim", "run1"), ("sim", "run2"), ("run2", "run1"))
RATE_VIEW = {"S_old": "T196", "S_new": "S_new", "S_newmains": "nm360", "S_paper493": "S_paper493", "pooled": "pooled"}
METRICS = ("steps", "calls_bash", "calls_text_editor", "calls_submit", "calls_other", "out_tokens", "attempted")
N_BOOT, SEED = 10_000, [20261015, 40]  # record A's meta.merge.seed (the headline's all-origins draws), asserted below
RECORD_A = ("5473efdb8", "experiments/2026-10-15_fpr_sweep_507_prep/results/alltasks_CRD_r2_allorigins_2026-10-16_044756.json",
            "6e02b62314963cd96288ea117c8eab2a9d1f7758857bf229f078dbefc68441cd")  # Stage A's all-origins record: its cell -> main map
COHORT = ("5285a90d9", "experiments/2026-10-15_fpr_sweep_507_prep/results/exclusions_2026-10-16_050631.json",
          "58f12786b40300cf1957cee8f8a89a4f5dbbb1165d4da2a55c13620c638fed9d")  # the 32k-frozen cohort, class-revised (INPUTS_FULL not frozen at build)
COHORT_TWIN = ("09e42576f", "experiments/2026-10-15_fpr_sweep_507_prep/results/exclusions_2026-10-16_043215.json")  # same cohort (19999bccb20a610d…)
COHORT_N = {"S_old": 190, "S_new": 303, "S_newmains": 335, "S_paper493": 485}

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0)
def run(per_key: dict, main_of: dict, env_of_main: dict, cohorts: dict, records: dict) -> dict:
    import numpy as np

    def wmedian(v, w):  # weighted median with integer multiplicities: each value repeated by its weight, then np.median (middle pair averaged)
        keep = w > 0
        return float(np.median(np.repeat(v[keep], w[keep].astype(int)))) if keep.any() else np.nan

    ci = lambda x: [float(np.percentile(x[np.isfinite(x)], 2.5)), float(np.percentile(x[np.isfinite(x)], 97.5))] if np.isfinite(x).any() else [None, None]
    out, gates = {}, {}
    for cname, keep in cohorts.items():
        out[cname] = {}
        for view in (*SETS, "pooled"):
            sets = SETS if view == "pooled" else (view,)
            cells = [(s, c) for s in sets for c in keep[s]]
            mains = sorted({main_of[s][c] for s, c in cells})
            mi = {m: i for i, m in enumerate(mains)}
            cm = np.array([mi[main_of[s][c]] for s, c in cells])
            rng = np.random.default_rng(SEED)
            Wm = np.zeros((N_BOOT, len(mains)))
            for e in sorted({env_of_main[m] for m in mains}):
                idx = [mi[m] for m in mains if env_of_main[m] == e]
                Wm[:, idx] = rng.multinomial(len(idx), [1 / len(idx)] * len(idx), size=N_BOOT)
            W = Wm[:, cm]  # each cell carries its main's multiplicity
            get = lambda a, f: [per_key[f"{s}|{a}|{c}"].get(f) for s, c in cells]
            lab = {a: np.array(get(a, "label"), dtype=object) for a in ARMS}
            adm = {a: np.array(get(a, "admitted"), bool) for a in ARMS}
            res = {"cells": len(cells), "mains": len(mains), "rates": {}, "rate_pairs": {}, "non_attempt_pairs": {}, "behaviour": {}, "behaviour_pairs": {},
                   "attempted_median_steps": {}, "attempted_median_steps_pairs": {}}
            for a in ARMS:
                lb = np.isin(lab[a], LAB)
                clear = (lab[a] == "attempted").astype(float)
                bt = (W * clear * lb).sum(1) / np.where((W * lb).sum(1) > 0, (W * lb).sum(1), np.nan)
                res["rates"][a] = {"labelled": int(lb.sum()), "attempted": int((clear * lb).sum()), "non_attempt_count": int(lb.sum() - (clear * lb).sum()),
                                   "unclear": int((lab[a] == "unclear").sum()), "declined": int((lab[a] == "declined").sum()),
                                   "rate": float(clear[lb].mean()), "rate_ci95_main": ci(bt), "non_attempt_share": 1 - float(clear[lb].mean()),
                                   "non_attempt_ci95_main": [1 - x if x is not None else None for x in ci(bt)[::-1]]}
                for m in METRICS:
                    v = np.array([x if x is not None else np.nan for x in get(a, m)], float)
                    ok = adm[a] & np.isfinite(v)
                    bt = (W * np.where(ok, v, 0)).sum(1) / np.where((W * ok).sum(1) > 0, (W * ok).sum(1), np.nan)
                    res["behaviour"].setdefault(a, {})[m] = {"n": int(ok.sum()), "mean": float(v[ok].mean()) if ok.any() else None, "ci95_main": ci(bt)}
                st = np.array([x if x is not None else np.nan for x in get(a, "steps")], float)
                ok = adm[a] & (lab[a] == "attempted") & np.isfinite(st)
                res["attempted_median_steps"][a] = {"n": int(ok.sum()), "median": float(np.median(st[ok])),
                                                    "ci95_main": ci(np.array([wmedian(st[ok], w[ok]) for w in W]))}
            for a, b in PAIRS:
                both = np.isin(lab[a], LAB) & np.isin(lab[b], LAB)
                xa, xb = (lab[a] == "attempted").astype(float), (lab[b] == "attempted").astype(float)
                d = xa - xb
                bt = (W * np.where(both, d, 0)).sum(1) / np.where((W * both).sum(1) > 0, (W * both).sum(1), np.nan)
                b_, c_ = int(((xa == 1) & (xb == 0) & both).sum()), int(((xa == 0) & (xb == 1) & both).sum())
                p = {"n_pairs": int(both.sum()), "diff": float(d[both].mean()), "ci95_main": ci(bt), "mcnemar_p (assumes independent cells)": mcnemar(b_, c_)}
                res["rate_pairs"][f"{a} - {b}"] = p
                res["non_attempt_pairs"][f"{a} - {b}"] = {"n_pairs": p["n_pairs"], "diff": -p["diff"], "ci95_main": [-x if x is not None else None for x in p["ci95_main"][::-1]],
                                                          "mcnemar_p (assumes independent cells)": p["mcnemar_p (assumes independent cells)"]}
                for m in METRICS:
                    va = np.array([x if x is not None else np.nan for x in get(a, m)], float)
                    vb = np.array([x if x is not None else np.nan for x in get(b, m)], float)
                    ok = adm[a] & adm[b] & np.isfinite(va) & np.isfinite(vb)
                    if not ok.any():
                        continue
                    dd = np.where(ok, va - vb, 0)
                    bt = (W * dd).sum(1) / np.where((W * ok).sum(1) > 0, (W * ok).sum(1), np.nan)
                    res["behaviour_pairs"].setdefault(f"{a} - {b}", {})[m] = {"n_pairs": int(ok.sum()), "diff": float(dd[ok].mean()), "ci95_main": ci(bt)}
            sa = np.array([x if x is not None else np.nan for x in get("sim", "steps")], float)
            sr = np.array([x if x is not None else np.nan for x in get("run1", "steps")], float)
            ok = adm["sim"] & adm["run1"] & (lab["sim"] == "attempted") & (lab["run1"] == "attempted") & np.isfinite(sa) & np.isfinite(sr)
            res["attempted_median_steps_pairs"]["sim - run1"] = {
                "n_pairs": int(ok.sum()), "difference_of_medians": float(np.median(sa[ok]) - np.median(sr[ok])),
                "difference_of_medians_ci95_main": ci(np.array([wmedian(sa[ok], w[ok]) - wmedian(sr[ok], w[ok]) for w in W])),
                "median_of_differences": float(np.median((sa - sr)[ok])), "median_of_differences_ci95_main": ci(np.array([wmedian((sa - sr)[ok], w[ok]) for w in W]))}
            out[cname][view] = res
            if cname == "all launched cells":  # gates: the point estimates equal the three records'
                rv = records["rate"][RATE_VIEW[view]]
                g = [abs(res["rates"][a]["rate"] - rv["per_arm"][a]["rate"]) < 1e-12 and res["rates"][a]["labelled"] == rv["per_arm"][a]["labelled"] for a in ARMS]
                g += [abs(res["rate_pairs"][k]["diff"] - rv["pairs"][k]["diff"]) < 1e-12 and res["rate_pairs"][k]["n_pairs"] == rv["pairs"][k]["n_pairs"] for k in res["rate_pairs"]]
                bv = records["behaviour"][view]
                g += [res["behaviour"][a][m]["mean"] is None if bv["per_arm"][a][m] is None else abs(res["behaviour"][a][m]["mean"] - bv["per_arm"][a][m]) < 1e-9 for a in ARMS for m in METRICS]
                g += [abs(res["behaviour_pairs"][k][m]["diff"] - bv["pairs"][k][m]["diff"]) < 1e-9 for k in bv["pairs"] for m in bv["pairs"][k]]
                cv = records["cap"][view]
                g += [res["attempted_median_steps"][a]["median"] == cv["steps"][f"{a} attempted"]["median"] and res["attempted_median_steps"][a]["n"] == cv["steps"][f"{a} attempted"]["n"] for a in ARMS]
                g += [res["behaviour"][a]["steps"]["n"] == bv["per_arm"][a]["admitted"] for a in ARMS]
                g += [res["behaviour_pairs"][k][m]["n_pairs"] == bv["pairs"][k][m]["n_pairs"] for k in bv["pairs"] for m in bv["pairs"][k]]
                gates[view] = all(g)
    if not all(gates.values()):
        raise ValueError(f"point estimates do not reproduce the records: {gates}")
    return {"gates": gates, "results": out}


@app.local_entrypoint()
def main(no_cohort: bool = False):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout
    if git("status", "--porcelain", "--untracked-files=no").strip() or not git("branch", "-r", "--contains", "HEAD").strip():
        raise SystemExit("commit and push first")
    me = Path(__file__).name  # git runs in this folder, so the path is relative to it
    if subprocess.run(["git", "ls-files", "--error-unmatch", me], capture_output=True, cwd=HERE).returncode or git("diff", "HEAD", "--", me).strip():
        raise SystemExit("this runner is not committed as it is")
    rec = {}
    for k, (p, want) in RECORDS.items():
        if not hashlib.sha256(p.read_bytes()).hexdigest().startswith(want):
            raise SystemExit(f"{p.name}: sha256 differs from its pin")
        rec[k] = json.loads(p.read_text())
    src = {}
    for k, (sha, path, want) in MAPS.items():
        b = git("show", f"{sha}:{path}")
        if hashlib.sha256(b).hexdigest() != want:
            raise SystemExit(f"{k}: sha256 differs from its pin")
        src[k] = json.loads(b)
    m = src["m710"]
    main_of = {"S_old": {c["key"]: c["main"] for c in src["tasks196"]["cells"]}, "S_new": {c["key"]: c["main"] for c in src["plus311"]["cells"]},
               "S_newmains": dict(m["main_of"]), "S_paper493": {r["key"]: r["main"] for r in src["facts493"]["rows"]}}
    env_cell = {"S_old": {c["key"]: c["env"] for c in src["tasks196"]["cells"]}, "S_new": {c["key"]: c["env"] for c in src["plus311"]["cells"]},
                "S_newmains": dict(m["env_of"]), "S_paper493": {r["key"]: r["env"] for r in src["facts493"]["rows"]}}
    env_of_main = {}
    for s in SETS:
        for c, mn in main_of[s].items():
            if env_of_main.setdefault(mn, env_cell[s][c]) != env_cell[s][c]:
                raise SystemExit(f"main {mn} has two environments")
    for r in src["facts493"]["rows"]:  # one main across sets: the sp493 facts' own flags must be reproduced
        for s, flag in (("S_old", "main_in_t196"), ("S_new", "main_in_p311"), ("S_newmains", "main_in_nm360")):
            if (r["main"] in set(main_of[s].values())) != bool(r[flag]):
                raise SystemExit(f"{r['key']}: the main map disagrees with the sp493 facts' {flag}")
    universe = {s: sorted(main_of[s]) for s in SETS}
    b = git("show", f"{COHORT[0]}:{COHORT[1]}")  # the cohort aligns record A's map, so it is read even with --no-cohort
    if hashlib.sha256(b).hexdigest() != COHORT[2]:
        raise SystemExit("the cohort file differs from its pin")
    cohort_cells = {s: sorted(v["cells"]) for s, v in json.loads(b)["final_cohort_keys"].items()}
    ba = git("show", f"{RECORD_A[0]}:{RECORD_A[1]}")
    if hashlib.sha256(ba).hexdigest() != RECORD_A[2]:
        raise SystemExit("record A differs from its pin")
    ra = json.loads(ba)
    if ra["meta"]["merge"]["seed"] != SEED or set(ra["meta"]["merge"]["mains"]) != {mn for s in SETS for mn in main_of[s].values()}:
        raise SystemExit("record A's seed, or its main set, differs from this job's")
    amap, used = {}, {}
    for s in SETS:  # rows whose block covers this cohort (mains aligned with its sorted cells); rows on a smaller cohort are skipped and counted
        lists = [h["mains"] for row in ra["rows"].values() if (h := (row.get(f"{s} (all-origins draws)") or {}).get("cell_hits_draw0"))]
        full = [x for x in lists if len(x) == len(cohort_cells[s])]
        if not full or any(x != full[0] for x in full):
            raise SystemExit(f"{s}: no row of record A covers the cohort, or those rows' maps differ")
        used[s] = {"rows_used": len(full), "rows_on_another_cohort_skipped": len(lists) - len(full)}
        amap |= {(s, c): mn for c, mn in zip(cohort_cells[s], full[0])}
    bad = [k for k, mn in amap.items() if main_of[k[0]][k[1]] != mn or ra["meta"]["merge"]["env_of_main"][mn] != env_of_main[mn]]
    if bad:
        raise SystemExit(f"{len(bad)} cohort cells map to another main (or env) than record A's")
    if {s: len(v) for s, v in universe.items()} != {"S_old": 196, "S_new": 311, "S_newmains": 360, "S_paper493": 493}:
        raise SystemExit("the launched-cell universe is not 196 / 311 / 360 / 493")
    cohorts, cohort_info = {"all launched cells": universe}, {"all launched cells": "every launched cell (the records' universe)"}
    if not no_cohort:
        keep = cohort_cells
        twin = {s: sorted(v["cells"]) for s, v in json.loads(git("show", f"{COHORT_TWIN[0]}:{COHORT_TWIN[1]}"))["final_cohort_keys"].items()}
        if keep != twin or {s: len(v) for s, v in keep.items()} != COHORT_N or any(set(keep[s]) - set(universe[s]) for s in SETS):
            raise SystemExit("the cohort is not the 32k-frozen 190 / 303 / 335 / 485 inside the launched cells, or its twin file differs")
        cohorts["cohort = 32k-frozen (58f12786)"] = keep
        cohort_info["cohort = 32k-frozen (58f12786)"] = (f"final_cohort_keys of {COHORT[1]} @ {COHORT[0]} (sha256 {COHORT[2]}); identical to {COHORT_TWIN[1]} @ "
                                                         f"{COHORT_TWIN[0]}; INPUTS_FULL had not frozen at build: re-assert against it when it freezes")
    res = run.remote(rec["behaviour"]["per_key"], main_of, env_of_main, cohorts,
                     {"rate": rec["rate"], "behaviour": rec["behaviour"]["summary"], "cap": rec["cap"]["results"]})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    head = git("rev-parse", "HEAD").decode().strip()
    out = {"label": "POST HOC descriptive; PRELIMINARY; main-clustered intervals beside the records' cell-level ones (09's RULE item 29)", "git": head,
           "stamp (PT)": stamp, "app": APP, "records": {k: {"file": p.name, "sha256_prefix": w} for k, (p, w) in RECORDS.items()},
           "maps": {k: {"commit": v[0], "path": v[1], "sha256": v[2]} for k, v in MAPS.items()}, "cohorts": cohort_info,
           "bootstrap": {"unit": "main task (one main across all four sets)", "strata": "environment of the main", "draws": N_BOOT, "seed": SEED},
           "main_map": {"source": f"each set's own main field; equal by key to record A ({RECORD_A[1]} @ {RECORD_A[0]}, sha256 {RECORD_A[2]}) on all "
                        f"{len(amap)} cohort cells, identical across the rows covering each cohort ({used}); main set equals its meta.merge.mains ({len(ra['meta']['merge']['mains'])})"}} | res
    dst = HERE / "results" / f"main_bootstrap_{stamp}.json"
    dst.open("x").write(json.dumps(out, indent=1))
    lines = [f"gates (point estimates equal the records): {res['gates']}"]
    f2 = lambda x: f"[{x[0]:+.3f}, {x[1]:+.3f}]" if x[0] is not None else "[n/a]"
    for cname, views in res["results"].items():
        for view, r in views.items():
            lines.append(f"[{cname} | {view}] cells {r['cells']}, mains {r['mains']}")
            for a in ARMS:
                x = r["rates"][a]
                lines.append(f"  {a}: attempt rate {x['rate']:.3f} {f2(x['rate_ci95_main'])}; non-attempt {x['non_attempt_count']}/{x['labelled']} = {x['non_attempt_share']:.3f} "
                             f"{f2(x['non_attempt_ci95_main'])}; attempted median steps {r['attempted_median_steps'][a]['median']:.1f} {f2(r['attempted_median_steps'][a]['ci95_main'])} "
                             f"(n {r['attempted_median_steps'][a]['n']})")
            for k, p in r["rate_pairs"].items():
                q = r["non_attempt_pairs"][k]
                lines.append(f"  attempt {k}: {p['diff']:+.3f} main {f2(p['ci95_main'])} (n {p['n_pairs']}; McNemar p {p['mcnemar_p (assumes independent cells)']:.3g}, assumes independent cells); "
                             f"non-attempt {q['diff']:+.3f} {f2(q['ci95_main'])}")
            for k, ms in r["behaviour_pairs"].items():
                lines.append(f"  behaviour {k}: " + "; ".join(f"{m} {v['diff']:+.2f} {f2(v['ci95_main'])}" for m, v in ms.items()))
            q = r["attempted_median_steps_pairs"]["sim - run1"]
            lines.append(f"  attempted median steps sim - run1 (n {q['n_pairs']}): difference of medians {q['difference_of_medians']:+.1f} {f2(q['difference_of_medians_ci95_main'])}; "
                         f"median of differences {q['median_of_differences']:+.1f} {f2(q['median_of_differences_ci95_main'])}")
    (HERE / "results" / f"main_bootstrap_{stamp}.printout.txt").open("x").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("->", dst)
