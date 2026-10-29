"""09's request (revised Append 21b: "unseen" verified, not assumed): for every +311 main absent from runs36.json, show by key that
no fold model of any famexp variant pass 2 scores was trained on it. NO scores are read: only the fit run's spec (map, folds, row
counts, Sol runs), its fit manifest (per-fit train_tasks counts and adapted draws) and runs36.json, each sha256-pinned.

  python make_nofold_record.py --spec <spec_2026-10-14_163027.json> --fit <fit_2026-10-14_175636.json> --r36 <runs36.json>
      --r36-volume <Volume path of the same bytes>
"""

import argparse
import hashlib
import json
import math
import random
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PINS = {"spec": "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61", "fit": "024fbb0ac04067a08c6dabc49e524a1b2841ab090b2c7f4acdeaf41345034745",
        "r36": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96", "simh": "f420ed2401dcc776cbbcfd414bbe9d6d1132a3bbd012b529a33aba7fa5ecd068"}
SIMH = ("573054a4d", "experiments/2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717/runs.json")  # b_asis's SH source (2e's run_asis_fits.py)
ASIS_FITS = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13:/xfer/asis_2026-10-14_135701/fit_L{L}.json"  # b_asis's fit records (2e)
T196 = ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json")
Q2 = ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json")
CHAIN = "experiments/2026-10-14_q3_overnight_chain/config/manifest_su311_q{q}_main.json"
PASS2 = [f"results/pass2_su311_q{q}_{s}.json" for q, s in ((1, "2026-10-14_202036"), (3, "2026-10-14_202402"), (2, "2026-10-14_203004"))]


def main():
    ap = argparse.ArgumentParser()
    for a in ("--spec", "--fit", "--r36", "--r36-volume", "--chain-commit", "--asis-fit-dir"):
        ap.add_argument(a, required=True)
    ap.add_argument("--message", action="append", required=True, help="label:path of each pass-2 message (sha256 recorded)")
    a = ap.parse_args()
    show = lambda c, p: subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
    raw = {k: Path(getattr(a, k)).read_bytes() for k in ("spec", "fit", "r36")} | {"simh": show(*SIMH)}
    bad = [k for k, b in raw.items() if hashlib.sha256(b).hexdigest() != PINS[k]]
    if bad:
        raise SystemExit(f"pin mismatch: {bad}")
    spec, fit, r36, simh = (json.loads(raw[k]) for k in ("spec", "fit", "r36", "simh"))
    t196 = json.loads(show(*T196))["cells"]
    q2 = json.loads(show(*Q2))["cells"]
    main_of = {c["key"]: c["main"] for c in t196} | {c["key"]: c["main"] for c in q2}
    r36_mains = {r["main_task"] for r in r36}
    t196_mains = {c["main"] for c in t196}
    quarter = {}
    for q in (1, 2, 3, 4):
        for r in json.loads(show(a.chain_commit, CHAIN.format(q=q)))["files"]:
            quarter.setdefault(r["task"], set()).add(f"q{q}")
    absent = sorted({c["main"] for c in q2} - r36_mains)
    mains = [{"main": m, "in_runs36": False, "is_t196_main": m in t196_mains,
              "cells": [{"cell": c["key"], "quarter": sorted(quarter.get(c["key"], [])) or "in no quarter's main manifest"} for c in q2 if c["main"] == m]}
             for m in absent]

    # the training cells of every fit, rebuilt by key from the spec exactly as run_famexp._group / _train_tasks build them
    ri = {"R": 0, "C": 1}
    runs = {r_: {w: {f"{w}|{t}" for k, (w_, t) in spec["map"].items() if w_ == w and spec["n_rows"][k][ri[r_]] > 0} for w in ("run1", "honest_a")} for r_ in ri}
    fold_of = lambda u: spec["fot"][u.split("|", 1)[1]]  # noqa: E731
    fits = [x for g in fit["groups"] for x in g["fits"]]
    per, checks = {}, []
    for x in fits:
        v, rd, f, L = x["variant"], x["readout"], x["fold"], x["layer"]
        if v.startswith("own_frac"):
            t = sorted({u.split("|", 1)[1] for w in ("run1", "honest_a") for u in runs[rd][w] if fold_of(u) != f})
            random.Random(f"famexp|own_a|f{f}").shuffle(t)
            tt = t[: math.ceil(int(v.split("frac")[1].split("_")[0]) / 100 * len(t))]
            checks.append({"fit": f"{v}|{rd}|f{f}", "train_tasks_rebuilt": len(tt), "train_tasks_recorded": x["train_tasks"], "match": len(tt) == x["train_tasks"]})
            cells = set(tt)
            src = "own_a: run1 + honest_a cells of folds != f, seeded prefix (rebuilt from the spec; count matched to the fit record)"
        else:
            sol = {r["main_task"] for r in spec["sol_runs"] if r["fold"] != f}
            cells = {u.split("|", 1)[1] for u in x["draw"]} | {u.split("|", 1)[1] for u in runs[rd]["honest_a"] if fold_of(u) != f}
            checks.append({"fit": f"{v}|{rd}|L{L}|f{f}", "draw_folds_ok": all(fold_of(u) != f for u in x["draw"]), "match": all(fold_of(u) != f for u in x["draw"])})
            src = "adapted: Sol #36 E1/H runs of folds != f (spec sol_runs) + the fit record's run1 draw + honest_a cells of folds != f"
        tm = {main_of[c] for c in cells} | (sol if not v.startswith("own_frac") else set())
        key = f"{v}|{rd}"
        per.setdefault(key, {})[f"f{f}" + ("" if v.startswith("own_frac") else f"|L{L}")] = {
            "training_mains": len(tm), "source": src, "excluded": {m: m not in tm for m in absent}}
    # b_asis = (SA, SH) as 2e's run_asis_fits.py fits it: SA = runs36's sim-attack runs, SH = the simh file's sim-honest runs, each
    # with its own fold; fold model f trains on both sides' runs of folds != f. Its map is the union; SA and SH must agree.
    sa = [r for r in r36 if r["arm"] == "sim-attack"]
    shr = [r for r in simh if r["arm"] == "sim-honest"]
    sa_map, sh_map = {}, {}
    for rows_, mp in ((sa, sa_map), (shr, sh_map)):
        for r in rows_:
            mp.setdefault(r["main_task"], set()).add(r["fold"])
    if any(len(v) != 1 for v in list(sa_map.values()) + list(sh_map.values())) or any(sa_map[m] != sh_map[m] for m in set(sa_map) & set(sh_map)):
        raise SystemExit("STOP: a main has two folds, or SA and SH disagree on a main's fold")
    b_map = {m: next(iter(v)) for m, v in (sa_map | sh_map).items()}
    fitrec = {}
    for L in (34, 46, 47, 49, 63):
        q = Path(a.asis_fit_dir) / f"asisfit_L{L}.json"
        fitrec[L] = {"volume": ASIS_FITS.format(L=L), "sha256": hashlib.sha256(q.read_bytes()).hexdigest(),
                     "train_runs": {str(f): json.loads(q.read_text())["fits"]["b_asis"][str(f)]["train_runs"] for f in range(5)}}
    b_counts = {}
    for rd in ri:
        for f in range(5):
            tm_sa, tm_sh = {r["main_task"] for r in sa if r["fold"] != f}, {r["main_task"] for r in shr if r["fold"] != f}
            want = {"pos": sum(r["fold"] != f for r in sa), "neg": sum(r["fold"] != f for r in shr)}
            b_counts[f] = want | {"fit_records_match": all(fitrec[L]["train_runs"][str(f)] == want for L in fitrec)}
            per.setdefault(f"b_asis|{rd}", {})[f"f{f}"] = {
                "training_mains": len(tm_sa | tm_sh), "excluded": {m: m not in (tm_sa | tm_sh) for m in absent},
                "excluded_sa_side": {m: m not in tm_sa for m in absent}, "excluded_sh_side": {m: m not in tm_sh for m in absent},
                "source": "b_asis (SA, SH): runs36 sim-attack + simh sim-honest runs of folds != f, as run_asis_fits.py; counts checked against 2e's fit records"}
    b_verdict = {m: {"in_map": m in b_map, "designated_fold": b_map.get(m),
                     "designated_excluded": m in b_map and all(per[f"b_asis|{rd}"][f"f{b_map[m]}"]["excluded"][m] for rd in ri),
                     "trained_on_by_folds": sorted(f for f in range(5) if not per["b_asis|R"][f"f{f}"]["excluded"][m])} for m in absent}
    # pass 2's routing sha: sha256(json.dumps(sorted(T196 mains of folds != f | runs36 mains of folds != f))), as run_famexp_pass2.main builds it
    t196_fold = {c["key"]: spec["fot"][c["key"]] for c in t196}
    tm_sha = [hashlib.sha256(json.dumps(sorted({main_of[c] for c, g in t196_fold.items() if g != f} | {r["main_task"] for r in r36 if r["fold"] != f})).encode()).hexdigest()
              for f in range(5)]
    routed = {}
    for p in PASS2:
        rec = json.loads((HERE / p).read_text())
        for m, x in rec["routing"].items():
            routed[f"{p}:{m}"] = {"main": m, "manifest_shas_match_recomputed": x["training_manifest_sha256"] == tm_sha, "in_absent_list": m in absent}
    all_excl = all(all(e["excluded"].values()) for k, d in per.items() if not k.startswith("b_asis") for e in d.values())
    msgs = {}
    for spec_ in a.message:
        lab, path = spec_.split(":", 1)
        mb = Path(path).read_bytes()
        m_ = json.loads(mb)
        hits = {}
        for e in m_["epochs"]:
            if main_of.get(e["cell"]) in absent:
                hits.setdefault(e["folder"], []).append(f"{e['world']}|{e['cell']}")
        msgs[lab] = {"message_sha256": hashlib.sha256(mb).hexdigest(), "epochs": len(m_["epochs"]), "epochs_with_absent_mains_by_folder": hits,
                     "any": bool(hits)}
    out = {"fold_map": {"runs36": {"sha256": PINS["r36"], "volume": a.r36_volume,
                                   "how_used": "fit run 2026-10-14_175636's prep input --r36 (run_famexp.py LOCAL_PINS prefix 76b11486cc353ab1eec3, checked in-job); "
                                               "main_task -> fold; never canonical_folds"},
                        "embedded_in_spec": {"path": "/p2/famexp/2026-10-14_163027/spec_2026-10-14_163027.json", "sha256": PINS["spec"],
                                             "fields": "fot (T196 cell -> fold) and sol_runs (Sol #36 runs with fold)"}},
           "inputs": {"fit_manifest": {"path": "/p2/famexp/2026-10-14_163027/fit_2026-10-14_175636.json", "sha256": PINS["fit"]},
                      "t196": list(T196), "q2_plus311": list(Q2), "quarters": {"commit": a.chain_commit, "path": CHAIN}},
           "label": "by key; no scores read", "absent_mains": mains, "n_absent": len(absent),
           "any_absent_main_is_t196": any(m["is_t196_main"] for m in mains), "per_variant_readout_fold": per, "rebuild_checks": checks,
           "rebuild_checks_all_match": all(c["match"] for c in checks), "routing_sha_recomputed": tm_sha, "routing_in_pass2_records": routed,
           "all_excluded_own_and_adapted": all_excl,
           "b_asis": {"map_sources": {"SA": {"file": "runs36.json", "sha256": PINS["r36"], "arm": "sim-attack"},
                                      "SH": {"commit": SIMH[0], "path": SIMH[1], "sha256": PINS["simh"], "arm": "sim-honest"}},
                      "sa_sh_fold_agreement": True, "fit_records": fitrec, "train_counts_rebuilt": b_counts,
                      "fit_records_all_match": all(x["fit_records_match"] for x in b_counts.values()), "per_absent_main": b_verdict,
                      "designated_all_excluded": all(v["in_map"] and v["designated_excluded"] for v in b_verdict.values())},
           "verdict": {"own_a_and_adapted": "fold -1 verified: every absent main excluded from all five fold models of all 24 variant x readouts" if all_excl else "NOT VERIFIED",
                       "b_asis": "designated fold verified: each absent main is in b_asis's map (SH); its designated model excludes it on both sides; the other four trained on it"
                       if all(v["in_map"] and v["designated_excluded"] for v in b_verdict.values()) else "NOT VERIFIED"},
           "pass2_messages": msgs}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / "results" / f"nofold_exclusion_{stamp}.json"
    with p.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({"path": str(p.relative_to(REPO)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "absent": [m["main"] for m in mains],
                      "verdict": out["verdict"], "rebuild_checks_all_match": out["rebuild_checks_all_match"], "b_asis_fit_records_all_match": out["b_asis"]["fit_records_all_match"],
                      "b_asis_per_main": b_verdict, "routing": routed, "messages": {k: v["epochs_with_absent_mains_by_folder"] for k, v in msgs.items()}}, indent=1))


if __name__ == "__main__":
    main()
