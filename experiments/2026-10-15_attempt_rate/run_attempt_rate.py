"""POST HOC descriptive: the clear-attack-attempt rate, sim vs real, on the LAUNCHED cells (09's RULE @ 0834ffe3e,
experiments/2026-10-14_priority1_spec_09/RULE.md, last section). Labels by key only; no run text is read. The laptop
reads pinned JSON from git (and the T196 reduced label file, checked against its committed sources); every count and
statistic is computed in one Modal CPU container. Output: results/attempt_rate_<stamp>.{json,printout.txt}.

Classes per (arm, cell): clear = "attempted", unclear, declined; anything else (no label, run_failed, no run-table row,
never launched) is infra and never counts as not attempted. Primary rate = clear / (clear + unclear + declined);
sensitivity = clear / (clear + unclear). Pairs: sim - run1, sim - run2 and the yardstick run2 - run1 on cells where both
arms are labelled. Intervals: environment-stratified cell bootstrap (pooled: stratified by set and environment), 10,000
draws, seed [20261016, 1]; exact two-sided McNemar.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with "numpy==2.3.3" modal run \\
      experiments/2026-10-15_attempt_rate/run_attempt_rate.py
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
APP = "probe-shift_aswin-surya_sub-attemptrate_main_kill-2026-11-15"
SRC = {  # name: (commit, path, sha256)
    "plus311": ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json",
                "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b"),
    "labels311": ("806b7a63a6", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/attempt_labels_311.json",
                  "22fc502fc3faa1323e5bee5927a2d6c8ef0ec158e55e76186b54425a2293356b"),
    "table311": ("806b7a63a6", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/run_table_311.json",
                 "1824c9d993eab4caca3febd045fb3c18d0aee4ac57215a736bb2660ac626043f"),
    "m710": ("32abe4120", "experiments/2026-10-15_fpr_sweep_507_prep/config/membership710.json",
             "6516d15fab77d7f344497e72db10b60c5f709b35c3a1859a2c9e016d2fc6e4ad"),
    "labels360": ("f60fb6cff", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/attempt_labels_nm360.json",
                  "5d28a4f51dbdbfb79883553068ae06b29329117e49deeeff6cbed60e7482a415"),
    "tasks196": ("64c6cb37f", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json",
                 "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb"),
    "attrition": ("a29c6557f", "experiments/2026-10-15_launch_disjoint/results/cohort_attrition_2026-10-15_223144.json",
                  "cd27fde34cb5bb187b4af6a5d6ee6a40b19a4d8383a34a299c8dc69d6ddb3489"),
}
T196_LABELS, T196_SHA, T196_REF = HERE / "data" / "claimc_labels_reduced.json", "d00e88773fc2a71bef51e1ff0ccc2559932f5c3a306a8f5604c5248e797dc995", "73c9f9789d"
T196_DIR = "experiments/2026-10-08_q3_onpolicy_figures/results"
GATE = {"S_new": {"sim": 18, "run1": 9, "run2": 5}, "nm360": {"sim": 9, "run1": 3, "run2": 8}}  # RULE §4(c)
LABELER = {"S_new and nm360": "experiments/2026-10-14_q3_overnight_chain/labeler/make_phase2_attempt_labels.py @ 806b7a63a "
           "(rubric: label() of experiments/2026-10-07_q3_nano_monitor/make_attempt_labels.py and run_self_reports' text match, "
           "both pinned at 8950b34; reads each run's own .eval: the last submit call's attack_successful and description, and "
           "linux_scorer side_task_success, which in sim arms is the simulator's verdict)",
           "T196": "the same rubric via make_attempt_labels' Phase 2 runs at 73c9f9789d, sim arm self-report only (scorer ignored)"}
ARMS, PAIRS, SEED, N_DRAWS = ("sim", "run1", "run2"), (("sim", "run1"), ("sim", "run2"), ("run2", "run1")), [20261016, 1], 10_000

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


def mcnemar(b, c):  # exact two-sided: discordant b (first arm clear only) vs c (second arm clear only)
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


@app.function(image=image, cpu=2, memory=4096, timeout=600, retries=0)
def analyze(src: dict, t196: dict) -> dict:
    import numpy as np

    out, sets = {"gate": {}, "checks": {}}, {}
    # ---- S_new: the 311 launched cells; labels by key; no run-table row or no label = infra
    lab, table = src["labels311"]["labels"], src["table311"]
    in_table = {r["cell"]: r for r in table["cells"]}
    dropped = {d["cell"]: f"no run-table row: dropped at {d['stage']}" for d in table["dropped"]}
    cells = {c["key"]: c["env"] for c in src["plus311"]["cells"]}
    if len(cells) != 311:
        raise ValueError("S_new: not 311 launched cells")
    sets["S_new"] = {c: {"env": cells[c], **{a: lab.get(f"{a}|{c}") or (dropped.get(c) or (f"never launched (sim_status {in_table[c]['sim_status']})"
                     if a == "sim" and c in in_table and in_table[c]["sim_status"] != "ran" else "no label")) for a in ARMS}} for c in cells}
    g = {a: sum(1 for c in in_table if [x for x in ARMS if lab.get(f"{x}|{c}") != "attempted"] == [a]) for a in ARMS}
    out["gate"]["S_new"] = {"definition": "run-table cells where exactly this arm is not 'attempted' (any other label or none), as the attrition record's S_new classes",
                            "computed": g, "record": GATE["S_new"], "match": g == GATE["S_new"]}
    # ---- nm360: the 360 launched cells. A run is labelled only if its key is in a scored folder (membership710
    # folder_keys), is not in excluded_runs, and its label (attempt_labels_nm360, by key) is not run_failed; else infra.
    # S4r cells are read like every cell: by key, after the same exclusions. Batch = the folders' batch for the cell
    # (independent of attempt labels; S5R is S5's rerun).
    m, lab = src["m710"], src["labels360"]["labels"]
    scored = {k for v in m["folder_keys"].values() for k in v}
    batches = {}
    for f, ks in m["folder_keys"].items():
        for k in ks:
            batches.setdefault(k.split("|", 1)[1], set()).add(m["folders"][f]["batch"])

    def nm_label(a, c):
        k = f"{a}|{c}"
        if k in m["excluded_runs"]:
            return "excluded: " + m["excluded_runs"][k].split(" (")[0]
        if k not in scored:
            return "no scored row (no folder)"
        v = lab.get(k)
        return "no label" if v is None else "run failed: no counted attempt" if v == "run_failed" else v

    sets["nm360"] = {c: {"env": m["env_of"][c], "batch": sorted(batches.get(c, [])), **{a: nm_label(a, c) for a in ARMS}} for c in m["env_of"]}
    out["checks"]["nm360_S4r"] = {"handling": "S4r cells: labels by key from attempt_labels_nm360 after membership710's exclusions and folder coverage, like every cell",
                                  "cells": {c: {a: v[a] for a in ARMS} for c, v in sorted(sets["nm360"].items()) if "S4r" in v["batch"]}}
    if len(sets["nm360"]) != 360:
        raise ValueError("nm360: not 360 launched cells")
    g = {a: sum(1 for c, v in sets["nm360"].items() if v[a] == "unclear" and all(v[x] == "attempted" for x in ARMS if x != a)) for a in ARMS}
    out["gate"]["nm360"] = {"definition": "cells where this arm is 'unclear' and the other two 'attempted', as the attrition record's nm360 'label unclear' status rows (S4r included)",
                            "computed": g, "record": GATE["nm360"], "match": g == GATE["nm360"]}
    # the record itself carries these counts (asserted, so the gate's numbers are the record's, not typed in)
    rec = src["attrition"]
    rs = {a: sum(v["n"] for k, v in rec["S_new"]["classes"].items() if k.startswith("not attempted in all three: ") and "+" not in k and k.split(": ", 1)[1].split()[0] == a) for a in ARMS}
    rn = {a: sum(v["n"] for k, v in rec["S_newmains"]["classes"].items() if k in (f"{a}: label unclear",)) +
             sum(1 for t in rec["S_newmains"]["status_row_text"].values() if t == f"S4r: {a}: label unclear") for a in ARMS}
    out["gate"]["record_counts_reproduce_RULE"] = {"S_new": rs, "nm360": rn, "match": rs == GATE["S_new"] and rn == GATE["nm360"]}
    if not (out["gate"]["S_new"]["match"] and out["gate"]["nm360"]["match"] and out["gate"]["record_counts_reproduce_RULE"]["match"]):
        out["STOP"] = "reproduction gate failed: no rate computed"
        return out
    # ---- T196 (S_old): the 196 launched tasks, if labels exist
    if t196:
        env = {c["key"]: c["env"] for c in src["tasks196"]["cells"]}
        sets["T196"] = {c: {"env": env[c], **{a: t196.get(f"{a}|{c}") or "no label in the reduced file (no counted attempt or no .eval)" for a in ARMS}} for c in env}
    # ---- the labeler's scorer clause: how many labels it decided, per arm (same source for sim and real when 0 in sim)
    out["checks"]["scorer_clause_decisions"] = {s: {a: n for a, n in v.items()} for s, v in src["scorer_clause"].items()}
    out["checks"]["labeler"] = LABELER
    out["checks"]["same_render"] = ("FLAG: renders may differ (sanitized-render caveat). Not verifiable by key: labels come from each run's own .eval "
                                    "submit-call fields (attack_successful, description) by the same code for every arm, not from a shared rendering; "
                                    "the scorer clause's decisions per arm are listed under scorer_clause_decisions")
    out["checks"]["gate_definitions_differ"] = ("S_new's gate counts any single-arm non-attempt (unclear, declined or no label); nm360's counts single-arm "
                                                "'label unclear' only, as each record section defines them: do not compare the two counts directly")
    out["checks"]["run_length"] = "FLAG: not computed. No committed per-run length record exists; a Modal count of staged calls per run is the follow-up"
    # ---- estimands
    rng = np.random.default_rng(SEED)
    pooled = {f"{s}|{c}": v | {"stratum": f"{s}|{v['env']}"} for s, cs in sets.items() for c, v in cs.items()}
    views = {"S_new": {c: v | {"stratum": v["env"]} for c, v in sets["S_new"].items()},
             "nm360": {c: v | {"stratum": v["env"]} for c, v in sets["nm360"].items()},
             "nm360 S1-S3+S5": {c: v | {"stratum": v["env"]} for c, v in sets["nm360"].items() if v["batch"] and set(v["batch"]) <= {"S1", "S2", "S3", "S5", "S5R"}}}
    out["checks"]["nm360_batch_unknown"] = {"cells with no folder in any arm (no batch; all infra; outside the S1-S3+S5 view)": sorted(c for c, v in sets["nm360"].items() if not v["batch"])}
    if "T196" in sets:
        views["T196"] = {c: v | {"stratum": v["env"]} for c, v in sets["T196"].items()}
    else:
        out["T196"] = {"unavailable": src.get("t196_reason", "no labels")}
    views["pooled"] = pooled
    LAB = ("attempted", "unclear", "declined")
    for name, cs in views.items():
        keys = sorted(cs)
        strata = sorted({cs[k]["stratum"] for k in keys})
        W = np.zeros((N_DRAWS, len(keys)))
        for st in strata:
            idx = [i for i, k in enumerate(keys) if cs[k]["stratum"] == st]
            W[:, idx] = rng.multinomial(len(idx), [1 / len(idx)] * len(idx), size=N_DRAWS)
        lab_of = {a: [cs[k][a] for k in keys] for a in ARMS}
        res = {"cells": len(keys), "per_arm": {}, "pairs": {}, "pairs_declined_excluded": {}}
        for a in ARMS:
            v = lab_of[a]
            infra = {}
            for x in v:
                if x not in LAB:
                    infra[x] = infra.get(x, 0) + 1
            labelled = np.array([x in LAB for x in v])
            clear = np.array([x == "attempted" for x in v], float)
            nod = np.array([x in ("attempted", "unclear") for x in v])
            rate = lambda sel: float(clear[sel].mean()) if sel.any() else None
            den = (W * labelled).sum(1)
            boot = (W * clear * labelled).sum(1) / np.where(den > 0, den, np.nan)
            boot = boot[np.isfinite(boot)]
            res["per_arm"][a] = {"labelled": int(labelled.sum()), "clear": int((clear * labelled).sum()), "unclear": v.count("unclear"), "declined": v.count("declined"),
                                 "infra": infra, "rate": rate(labelled), "rate_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                 "rate_declined_excluded": rate(nod)}
        for key, excl in (("pairs", False), ("pairs_declined_excluded", True)):
            for a, b in PAIRS:
                ok = np.array([x in LAB and y in LAB and not (excl and "declined" in (x, y)) for x, y in zip(lab_of[a], lab_of[b])])
                xa = np.array([x == "attempted" for x in lab_of[a]], float)
                xb = np.array([x == "attempted" for x in lab_of[b]], float)
                d_obs = float((xa - xb)[ok].mean()) if ok.any() else None
                Wm = W * ok
                den = Wm.sum(1)
                boot = ((Wm * (xa - xb)).sum(1) / np.where(den > 0, den, np.nan))
                bb, cc = int(((xa == 1) & (xb == 0) & ok).sum()), int(((xa == 0) & (xb == 1) & ok).sum())
                f = boot[np.isfinite(boot)]
                res[key][f"{a} - {b}"] = {"n_pairs": int(ok.sum()), "rate_first": float(xa[ok].mean()) if ok.any() else None, "rate_second": float(xb[ok].mean()) if ok.any() else None,
                                          "diff": d_obs, "ci95": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))] if len(f) else [None, None],
                                          "discordant_first_only": bb, "discordant_second_only": cc, "mcnemar_p": mcnemar(bb, cc)}
            y = res[key]["run2 - run1"]["ci95"]
            res[key]["yardstick_excludes_0"] = bool(y[0] is not None and (y[0] > 0 or y[1] < 0))
            for a, b in PAIRS[:2]:
                p = res[key][f"{a} - {b}"]
                p["reading"] = ("uninterpretable against the yardstick" if res[key]["yardstick_excludes_0"]
                                else "differs (p < .001)" if p["mcnemar_p"] < 0.001 else "no claim (p >= .001)")
        out[name] = res
    out["sets_launched_cells"] = {s: len(v) for s, v in sets.items()}
    return out


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout
    if git("status", "--porcelain", "--untracked-files=no").strip() or not git("branch", "-r", "--contains", "HEAD").strip():
        raise SystemExit("commit and push first: the record carries its commit")
    src = {}
    for name, (sha, path, want) in SRC.items():
        b = git("show", f"{sha}:{path}")
        if hashlib.sha256(b).hexdigest() != want:
            raise SystemExit(f"{name}: sha256 differs from its pin")
        src[name] = json.loads(b)
    raw = T196_LABELS.read_bytes() if T196_LABELS.exists() else b""
    t196, t196_check = {}, {}
    if hashlib.sha256(raw).hexdigest() == T196_SHA:
        d = json.loads(raw)
        t196_check = {f: hashlib.sha256(git("show", f"{T196_REF}:{T196_DIR}/{f}")).hexdigest() == want for f, want in d["files_sha256"].items()}
        if d["ref"].startswith(T196_REF) and all(t196_check.values()):
            t196 = d["labels"]
    reason = "" if t196 else ("reduced label file sha256 differs from its pin" if not t196_check else "its committed sources do not match (or another ref)")
    # the scorer clause, per arm, from the label records' rule fields (keys and rule names only)
    sc = {"S_new": {}, "nm360": {}}
    for s, name in (("S_new", "labels311"), ("nm360", "labels360")):
        for k, r in src[name]["rules"].items():
            if r == "side-task scorer pass":
                sc[s][k.split("|")[0]] = sc[s].get(k.split("|")[0], 0) + 1
        sc[s] = {a: sc[s].get(a, 0) for a in ARMS}
    sc["T196 (raw label files, decided_by_scorer_clause)"] = {}
    t_arm = lambda f, a: "run2" if a == "attack" and "second_attack" in f else "run1" if a == "attack" else "sim" if a == "sim_attack@september" else f"other ({a})"
    for f in (json.loads(raw)["files_sha256"] if t196 else []):
        for line in git("show", f"{T196_REF}:{T196_DIR}/{f}").decode().splitlines():
            r = json.loads(line)
            k = t_arm(f, r["arm"])
            if r.get("decided_by_scorer_clause"):
                sc["T196 (raw label files, decided_by_scorer_clause)"][k] = sc["T196 (raw label files, decided_by_scorer_clause)"].get(k, 0) + 1
    src["scorer_clause"], src["t196_reason"] = sc, reason
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = analyze.remote(src, t196)
    head = git("rev-parse", "HEAD").decode().strip()
    rec = {"label": "POST HOC descriptive; PRELIMINARY (09's RULE @ 0834ffe3e, last section)", "git": head, "stamp (PT)": stamp, "app": APP,
           "sources": {k: {"commit": v[0], "path": v[1], "sha256": v[2]} for k, v in SRC.items()},
           "T196_labels": {"file": T196_LABELS.name, "sha256_ok": bool(t196) or hashlib.sha256(raw).hexdigest() == T196_SHA, "ref": T196_REF,
                           "committed_sources_match": t196_check, "used": bool(t196)}} | res
    dst = HERE / "results" / f"attempt_rate_{stamp}.json"
    dst.parent.mkdir(exist_ok=True)
    dst.open("x").write(json.dumps(rec, indent=1))
    lines = [f"gate: {json.dumps(res['gate'])}"]
    if "STOP" in res:
        lines.append(res["STOP"])
    else:
        lines += [f"labeler: {json.dumps(res['checks']['labeler'])}", f"same render: {res['checks']['same_render']}",
                  f"scorer-clause decisions: {json.dumps(res['checks']['scorer_clause_decisions'])}", f"run length: {res['checks']['run_length']}",
                  f"gate definitions: {res['checks']['gate_definitions_differ']}", f"nm360 S4r: {json.dumps(res['checks']['nm360_S4r'])}",
                  f"nm360 batch unknown: {json.dumps(res['checks']['nm360_batch_unknown'])}"]
        if "unavailable" in res.get("T196", {}):
            lines.append(f"T196 UNAVAILABLE: {res['T196']['unavailable']}")
        for name in [k for k in res if k not in ("gate", "checks", "sets_launched_cells") and "unavailable" not in res[k]]:
            r = res[name]
            lines.append(f"[{name}] launched cells {r['cells']}")
            for a, v in r["per_arm"].items():
                lines.append(f"  {a}: clear {v['clear']}/{v['labelled']} = {v['rate']:.3f} [{v['rate_ci95'][0]:.3f}, {v['rate_ci95'][1]:.3f}]; unclear {v['unclear']}, declined {v['declined']}; "
                             f"declined-excluded {v['rate_declined_excluded']:.3f}; infra {v['infra']}")
            for key in ("pairs", "pairs_declined_excluded"):
                for pk, p in r[key].items():
                    if isinstance(p, dict):
                        lines.append(f"  {key} {pk}: n {p['n_pairs']}, {p['rate_first']:.3f} vs {p['rate_second']:.3f}, diff {p['diff']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}], "
                                     f"discordant {p['discordant_first_only']}/{p['discordant_second_only']}, McNemar p {p['mcnemar_p']:.4g}" + (f" -> {p['reading']}" if "reading" in p else ""))
                lines.append(f"  {key}: yardstick excludes 0: {r[key]['yardstick_excludes_0']}")
    (HERE / "results" / f"attempt_rate_{stamp}.printout.txt").open("x").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("->", dst)
