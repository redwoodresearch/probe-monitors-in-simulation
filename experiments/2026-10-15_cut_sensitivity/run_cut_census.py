"""32,768-token cut census (0d's boring-explanation check, 10-15; POST HOC, PRELIMINARY). Counts only: reads the stager's
per-run records (parts/part*.recs.json: tokens_before, truncated, commands <label> [inside], outputs shown / with an
output_end inside) of every staged set behind the common comparison, on Modal CPU, read-only. No transcript text, no scores.

Sets: T196 corrected B (q4f_main_v1: run1, sim, honest_a, honest_b; q4f_run2_v1: run2; q4f_c_v1: honest_c) and the A25
sim-honest 149 (simh_q4f_v1_full_b1/_b2); +311 (su311_q{1-4}_{main,run2}_v1 and the sim sets; q2r supersedes q2 by task);
nm360 (the 41 folders of the ~710 membership, keys restricted to membership710.folder_keys).

  uv run --no-project --with modal==1.6.1 modal run experiments/2026-10-15_cut_sensitivity/run_cut_census.py
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-cutsens_census_kill-2026-11-14"
CHAIN, T196V, SIMHV = ("probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13",
                       "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13")
MEMBERSHIP = HERE / "config" / "membership710_0f9d96158.json"  # aswin/fpr-sweep-507-prep @ 0f9d96158 (sha256 6516d15f...)
WORLD = {"attack": "run1", "sim_attack@september": "sim", "sim_honest@september": "sim_honest", "honest_a": "honest_a", "honest_b": "honest_b"}
KEYS = ["commands impactful", "commands impactful inside", "commands read_only", "commands read_only inside", "outputs shown",
        "outputs with an output_end inside"]

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")
VOLS = {"/chain": modal.Volume.from_name(CHAIN, version=2).read_only(), "/t196": modal.Volume.from_name(T196V, version=2).read_only(),
        "/simh": modal.Volume.from_name(SIMHV).read_only()}


@app.function(image=image, cpu=4, memory=8192, timeout=1200, retries=0, volumes=VOLS)
def census(nm_sets: list) -> dict:
    import glob

    def recs(root, set_dir):
        out = []
        files = sorted(glob.glob(f"{root}/{set_dir}/parts/part*.recs.json"))
        if not files:
            raise FileNotFoundError(f"{root}/{set_dir}: no part records")
        for f in files:
            out += json.loads(Path(f).read_text())["recs"]
        return out

    sets = [("T196", "/t196", "q4f_main_v1", {}), ("T196", "/t196", "q4f_run2_v1", {"attack": "run2"}),
            ("T196", "/t196", "q4f_c_v1", {"honest_b": "honest_c"}),
            ("T196", "/simh", "simh_q4f_v1_full_b1", {}), ("T196", "/simh", "simh_q4f_v1_full_b2", {})]
    su = [f"su311_q{q}_{w}_v1" for q in (1, 2, 3, 4) for w in ("main", "run2")]
    su += ["su311_q1_sim_s1_v1", "su311_q1_sim_s2_v1", "su311_q2_sim_v1", "su311_q2r_sim_v1", "su311_q3_sim_s1_v1", "su311_q3_sim_s2_v1",
           "su311_q4_sim_s1_v1", "su311_q4_sim_s2_v1"]
    for s in su:
        sets.append(("+311", "/chain", s, {"attack": "run2"} if "_run2_" in s else {}))
    for s in nm_sets:
        sets.append(("new mains", "/chain", s, {"attack": "run2"} if "_run2_" in s else {}))
    rows, files_read = [], {}
    for tset, root, s, remap in sets:
        rs = recs(root, s)
        files_read[s] = len(rs)
        for r in rs:
            world = remap.get(r["arm_dir"], WORLD.get(r["arm_dir"], r["arm_dir"]))
            if ("_sim" in s or "_run2" in s) and r["arm_dir"] in ("honest_a", "honest_b"):
                continue  # honest donors repeated in sim and run2 sets
            if s == "q4f_main_v1" and r["arm_dir"] == "sim_honest@september":
                continue  # the 20-run pilot; the A25 149 come from simh_q4f_v1_full_b1/_b2
            rows.append({"task_set": tset, "set": s, "world": world, "task": r.get("task"), "run": r.get("run"),
                         "excluded": r.get("excluded"), "tokens_before": r.get("tokens_before"), "tokens_after": r.get("tokens_after"),
                         "truncated": r.get("truncated"), **{k: r.get(k, 0) for k in KEYS}})
    return {"rows": rows, "sets": [s for _, _, s, _ in sets], "recs_per_set": files_read}


@app.local_entrypoint()
def main():
    import hashlib
    from collections import defaultdict

    mem = json.loads(MEMBERSHIP.read_text())
    folders = sorted(mem["folders"])
    nm_sets = sorted({f.split("__")[0] for f in folders})
    out = census.remote(nm_sets)
    counted = {}  # nm360: (set, world|task) keys the membership counts
    for f, keys in mem["folder_keys"].items():
        for k in keys:
            counted[(f.split("__")[0], k)] = True
    cohort_nm = set(mem["cohort"]) | set(mem["cohort_4r_sol_only"])
    rows = []
    for r in out["rows"]:
        if r["excluded"]:
            continue  # staging exclusions have no readout at all; listed by the staging records
        if r["task_set"] == "new mains":
            if (r["set"], f"{r['world']}|{r['task']}") not in counted:
                continue
            r["in_cohort"] = r["task"] in cohort_nm
        rows.append(r)
    # +311 sims: q2r supersedes q2 for its tasks; every other (world, task) must be unique
    seen = defaultdict(list)
    for i, r in enumerate(rows):
        seen[(r["task_set"], r["world"], r["task"])].append(i)
    drop = set()
    dups = {}
    for k, ix in seen.items():
        if len(ix) > 1:
            sets = [rows[i]["set"] for i in ix]
            if k[0] == "+311" and k[1] == "sim" and "su311_q2r_sim_v1" in sets:
                drop |= {i for i in ix if rows[i]["set"] != "su311_q2r_sim_v1"}
            else:
                dups["|".join(map(str, k))] = sets
    if dups:
        raise ValueError(f"{len(dups)} (task_set, world, task) appear in several sets: {list(dups.items())[:5]}")
    rows = [r for i, r in enumerate(rows) if i not in drop]
    agg = defaultdict(lambda: defaultdict(int))
    toks = defaultdict(list)
    for r in rows:
        for key in [(r["task_set"], r["world"]), ("all", r["world"] if r["world"] in ("run1", "run2", "sim") else "honest (a/b/c)")]:
            a = agg[key]
            a["runs"] += 1
            a["truncated"] += int(r["truncated"] or 0)
            for k in KEYS:
                a[k] += r[k] or 0
            toks[key].append(r["tokens_before"] or 0)
    table = []
    for (t, w), a in sorted(agg.items()):
        tb = sorted(toks[(t, w)])
        sh = lambda x, y: round(1 - a[x] / a[y], 4) if a[y] else None
        table.append({"task_set": t, "world": w, "runs": a["runs"], "truncated": a["truncated"],
                      "truncated_share": round(a["truncated"] / a["runs"], 4),
                      "impactful_calls": a["commands impactful"], "impactful_inside": a["commands impactful inside"],
                      "impactful_after_share": sh("commands impactful inside", "commands impactful"),
                      "read_only_calls": a["commands read_only"], "read_only_inside": a["commands read_only inside"],
                      "read_only_after_share": sh("commands read_only inside", "commands read_only"),
                      "outputs_shown": a["outputs shown"], "outputs_inside": a["outputs with an output_end inside"],
                      "outputs_after_share": sh("outputs with an output_end inside", "outputs shown"),
                      "median_tokens_full_run": tb[len(tb) // 2]})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "PRELIMINARY; POST HOC; counts only, from the stager's per-run records (no transcript text, no scores)",
           "cut": 32768, "stager_lines": {"main": "experiments/2026-10-13_q4_why_weaker_POSTHOC/stage_q4b.py:68",
                                          "chain": "experiments/2026-10-14_q3_overnight_chain/stage_q4b.py:71 (aswin/q3-overnight-chain)",
                                          "q35_repaired": "experiments/2026-10-14_q35_sol_avg_POSTHOC/stage_q35_repaired.py:76"},
           "membership": {"path": str(MEMBERSHIP.relative_to(HERE.parents[1])), "sha256": hashlib.sha256(MEMBERSHIP.read_bytes()).hexdigest()},
           "sets": out["sets"], "recs_per_set": out["recs_per_set"], "q2_rows_superseded_by_q2r": len(drop),
           "notes": ["T196 and +311: every staged run of each set (not restricted to the 183/272 cohorts)",
                     "new mains: keys restricted to membership710.folder_keys; in_cohort marks the 261 + 51 cohort cells",
                     "staging exclusions (no readout) are dropped; zero-call runs stay with zero counts",
                     "sim_honest = the A25 149 (simh_q4f_v1_full_b1/_b2): truncation counts only (09's blind: no scores)"],
           "table": table, "per_run": rows}
    p = HERE / "results" / f"cut_census_{stamp}.json"
    with p.open("x") as fh:
        fh.write(json.dumps(rec, indent=1))
    for t in table:
        print(t)
    print("->", p)
