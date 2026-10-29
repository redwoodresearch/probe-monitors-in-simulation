"""No-filter headline cohort (09 @ 01012e87e): every launched cell whose three attack arms (run1, run2, sim) completed and were scored; excluded ONLY
for a missing or invalid measurement, by key and class. KEYS AND COUNTS ONLY: inputs are the committed key census (results/nofilter_keys_*.json,
attack-run row counts per label, origin, world and cell), the attack sample-error record (results/attack_sample_errors_*.json: per attack .eval,
log status, sample error class, tool-message count; metadata only), FINAL's cohort, xerr_v6, the T196 task file and coverage note, the +311 run
table and attempt labels, membership710 and a1's attempt-rate record. No score is read.

Per arm without rows in every label (09 via 2e, 10-16 ~00:5x): its .evals decide. All with 0 tool calls and no sample error = a completed run
with no action: a VALID measurement, kept as a structural zero ("not caught"; the registered convention); all with 0 tool calls and a sample
error = an errored run, excluded by key; an eval with tool calls = a run that was not scored (no folder); no eval = no run for the arm.
Ruled classes stay as ruled: never launched / dropped at preflight, setup failure, staging exclusion, excluded by key. Attempt-label classes
(unclear, declined, no label) are NOT exclusions; the registered attempt-filtered cohort is the appendix line.

  python make_nofilter_exclusions.py results/nofilter_keys_<stamp>.json <T196 attempt labels> results/attack_sample_errors_<stamp>.json
  # writes results/nofilter_exclusions_<stamp>.{json,md} (exclusive create)
"""

import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
W3 = ("run1", "run2", "sim")
HYG = "experiments/2026-10-14_q3_overnight_chain/results/hygiene"


def git_show(c, p):
    return subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=REPO).stdout


def pin(c, p):
    return {"commit": c, "path": p, "sha256": hashlib.sha256(git_show(c, p)).hexdigest()}


SRC = {
    "rule": pin("01012e87e", "experiments/2026-10-14_priority1_spec_09/RULE.md"),
    "final": pin("3ce6ea55e", "experiments/2026-10-14_xerr_scaleup_09/results/xerr_scaleup_final_2026-10-15_013255.json"),
    "v6": pin("0ed287c95", "experiments/2026-10-14_xerr_prelim_09/results/xerr_v6_2026-10-14_161122.json"),
    "t196_plan": pin("bc2d6f0c0", "experiments/2026-10-10_claimB_onpolicy/PLAN.md"),
    "t196_tasks": pin("3cd5830ec", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"),
    "run_table_311": pin("806b7a63a", f"{HYG}/run_table_311.json"),
    "attempt_labels_311": pin("806b7a63a", f"{HYG}/attempt_labels_311.json"),
    "membership710": pin("3cd5830ec", "experiments/2026-10-15_fpr_sweep_507_prep/config/membership710.json"),
    "a1_attempt_rate": pin("bc594d283", "experiments/2026-10-15_attempt_rate/results/attempt_rate_2026-10-15_234954.json"),
}
LABELS_SHA = "d00e88773fc2a71bef51e1ff0ccc2559932f5c3a306a8f5604c5248e797dc995"  # 2026-10-14_fpr_sweep_unit/config/inputs_b.json local.labels
J = {k: json.loads(git_show(v["commit"], v["path"])) for k, v in SRC.items() if v["path"].endswith(".json")}


def read_pinned(p, sha=None):
    b = Path(p).read_bytes()
    if sha and hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit(f"{p}: sha256 differs from its pin")
    return b, hashlib.sha256(b).hexdigest()


census_path, err_path = (HERE / sys.argv[1]).resolve(), (HERE / sys.argv[3]).resolve()
cb, csha = read_pinned(census_path)
eb, esha = read_pinned(err_path)
t196_lab = json.loads(read_pinned(sys.argv[2], LABELS_SHA)[0])["labels"]
rows = {}  # label -> origin -> world -> {cell: n rows} (sentinel worlds dropped)
for lb, d in json.loads(cb)["rows"].items():
    d = {o_: v_ for o_, v_ in d.items() if not o_.startswith("S_paper493")}  # the second refresh's census adds S_paper493 rows (read by make_waitlist.py)
    ((o, v),) = d.items()
    rows[lb] = {src: {w: dict(x) for w, x in ws.items() if "|sentinel" not in w} for src, ws in v["attack_rows_by_source_world_cell"].items()}
labels = sorted(rows)
erec = json.loads(eb)
if erec["meta"]["counts"]["missing"] or erec["meta"]["counts"]["sha_mismatch"] or erec["meta"]["counts"]["tool_count_differs_from_toolerr"]:
    raise SystemExit("sample-error record: missing, sha-mismatched or tool-count-mismatched rows")
evals = {}  # (origin, world|cell) -> [eval rows]
for r in erec["rows"]:
    evals.setdefault((r["origin"], r["key"]), []).append(r)

fin, v6, mem, rt = J["final"]["result"], J["v6"]["result"], J["membership710"], J["run_table_311"]
t196 = sorted(c["key"] for c in J["t196_tasks"]["cells"])
coh_fin = set(fin["cohort"]["cohort_cells"])
assert coh_fin & set(t196) == set(v6["cohort"]["tasks"]), "FINAL's S_old cohort is not v6's"
lab311 = J["attempt_labels_311"]["labels"]
A1K = "never launched (sim_status no_honest_A_reference)"
a1_line = {"S_new/per_arm/sim/infra/" + A1K: J["a1_attempt_rate"]["S_new"]["per_arm"]["sim"]["infra"][A1K]}  # a1's count for the class (1)

launched = {"S_old": t196, "S_new": sorted([r["cell"] for r in rt["cells"]] + [r["cell"] for r in rt["dropped"]]), "S_newmains": sorted(mem["main_of"])}
attempt = {"S_old": sorted(coh_fin & set(t196)), "S_new": sorted(coh_fin - set(t196)), "S_newmains": sorted(set(mem["cohort"]) | set(mem["cohort_4r_sol_only"]))}
assert len(launched["S_new"]) == 311 and set(attempt["S_new"]) <= set(launched["S_new"])

ruled = {o: {} for o in launched}  # classes that stay as ruled, whatever the arms' .evals say
for c in ("t88_model_registry", "t169_clinical_trial"):
    ruled["S_old"][c] = {"class": "never launched: no attack run (sample construction failed at preflight)",
                         "source": "t196_plan ('t88 and t169 have no attack run'); manifest_q4b counted_attempts_without_eval"}
for r in rt["dropped"]:
    ruled["S_new"][r["cell"]] = {"class": f"never launched: dropped at {r['stage']}", "source": "run_table_311 dropped", "detail": r["reason"]}
for r in rt["cells"]:
    if r["sim_status"] != "ran":
        ruled["S_new"][r["cell"]] = {"class": f"no run for an arm: sim never launched ({r['sim_status']})", "source": "run_table_311 sim_status; a1_attempt_rate",
                                     "detail": r.get("sim_status_detail"), "a1_record_line": a1_line}
for c, txt in mem["status_rows"].items():
    for key, cls in (("setup failure", "setup failure, no pair"), ("excluded at staging", "staging exclusion (S4r status row)"),
                     ("excluded by key", "excluded by key (09 @ 1bc718e98)")):
        if key in txt:
            ruled["S_newmains"][c] = {"class": cls, "source": "membership710 status_rows", "detail": txt}
            break


def arm_state(o, c, w, labs):
    """The arm's measurement: 'rows' (scored), 'structural_zero' (0 tool calls, no sample error), 'errored', 'unscored', 'no_run' or 'unresolved'."""
    has = any(rows[lb][o].get(w, {}).get(c, 0) > 0 for lb in labs)
    ev = evals.get((o, f"{w}|{c}"), [])
    brief = [{k: e[k] for k in ("path", "log_status", "sample_error", "n_tool_nonsubmit", "n_messages")} for e in ev]
    if has:
        return "rows", brief
    if not ev:
        return "no_run", brief
    if any(e["n_tool_nonsubmit"] > 0 for e in ev):
        return "unscored", brief
    errs = {e["sample_error"] is None and e["log_status"] == "success" for e in ev}
    return ("structural_zero" if errs == {True} else "errored" if errs == {False} else "unresolved"), brief


out = {"label": "no-filter headline cohort by key (09 @ 01012e87e; zero-tool rule 09 via 2e 10-16 ~00:5x); keys and counts only; no score read",
       "stamp_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "generator": "make_nofilter_exclusions.py",
       "sources": SRC | {"census": {"path": str(census_path.relative_to(REPO)), "sha256": csha},
                         "sample_errors": {"path": str(err_path.relative_to(REPO)), "sha256": esha},
                         "t196_attempt_labels": {"path": "local pin (2026-10-14_fpr_sweep_unit/config/inputs_b.json local.labels)", "sha256": LABELS_SHA}},
       "supersedes": {"path": "experiments/2026-10-15_fpr_sweep_507_prep/results/nofilter_exclusions_2026-10-16_002238.json @ debe1eccd",
                      "sha256": "8e579724cb7be7545e97a4020d55eb530190d81af5b58f632d8d187e55c3bc3b",
                      "why": "09's zero-tool rule: a zero-tool attack arm without a sample error is a valid 'not caught' measurement (kept); classes now "
                             "from each arm's .eval (sample error, tool calls) instead of the census alone"},
       "origins": {}}
for o in ("S_old", "S_new", "S_newmains"):
    labs = [lb for lb in labels if o in rows[lb]]
    excl, sz, zt = {}, {}, Counter()
    for c in launched[o]:
        st = {w: arm_state(o, c, w, labs) for w in W3}
        for w, (s, ev) in st.items():  # every zero-tool attack arm that decides its cell (no rows), counted once by key
            if s in ("structural_zero", "errored", "unresolved"):
                zt[(w, "kept" if s == "structural_zero" and c not in ruled[o] else "excluded")] += 1
        bad = {w: s for w, (s, _) in st.items() if s not in ("rows", "structural_zero")}
        evid = {w: {"state": s, "evals": ev} for w, (s, ev) in st.items() if s != "rows"}
        if c in ruled[o]:
            excl[c] = ruled[o][c] | {"arms": evid}
        elif bad:
            name = {"errored": "errored run (0 tool calls)", "unscored": "no scored row (no folder); the run has tool calls", "no_run": "no run for an arm",
                    "unresolved": "unresolved: the arm's evals disagree"}
            parts = sorted({name[s] for s in bad.values()})
            errs = sorted({e["sample_error"] for w in bad for e in st[w][1] if e["sample_error"]})
            excl[c] = {"class": "; ".join(parts) + (f" ({', '.join(errs)})" if errs else ""), "arms": evid,
                       "source": "census (zero rows in every label) + sample-error record" + (" + t196_plan" if o == "S_old" else "")}
        for w, (s, ev) in st.items():
            if s == "structural_zero" and c not in excl:
                sz[f"{w}|{c}"] = {"sample_error": ev[0]["sample_error"], "log_status": ev[0]["log_status"], "n_tool_nonsubmit": 0,
                                  "n_messages": ev[0]["n_messages"], "evals": [e["path"] for e in ev],
                                  "attempt_label": lab311.get(f"{w}|{c}") if o == "S_new" else t196_lab.get(f"{w}|{c}") if o == "S_old" else mem["status_rows"].get(c)}
    head = [c for c in launched[o] if c not in excl]
    for k in sz:  # never read an absent row as missing: a structural zero has no row in any label; every other headline arm has rows in every label
        assert all(rows[lb][o].get(k.split("|")[0], {}).get(k.split("|")[1], 0) == 0 for lb in labs), k
    gaps = {lb: {c: [w for w in W3 if f"{w}|{c}" not in sz and rows[lb][o].get(w, {}).get(c, 0) == 0] for c in head} for lb in labs}
    gaps = {lb: {c: ws for c, ws in v.items() if ws} for lb, v in gaps.items()}
    gaps = {lb: v for lb, v in gaps.items() if v}
    added = sorted(set(head) - set(attempt[o]))
    lost = sorted(set(attempt[o]) - set(head))
    assert not lost, f"{o}: registered cohort cells outside the no-filter headline: {lost}"
    lab_of = {"S_old": lambda c: {w: t196_lab.get(f"{w}|{c}") for w in W3}, "S_new": lambda c: {w: lab311.get(f"{w}|{c}") for w in W3},
              "S_newmains": lambda c: mem["status_rows"].get(c)}[o]
    x = {"launched_n": len(launched[o]), "measurement_excluded_n": len(excl),
         "measurement_excluded_by_class": {k: sorted(c for c, v in excl.items() if v["class"] == k) for k in sorted({v["class"] for v in excl.values()})},
         "measurement_excluded": dict(sorted(excl.items())),
         "headline_n": len(head), "headline_cells": head, "headline_cells_sha256": hashlib.sha256(json.dumps(head).encode()).hexdigest(),
         "structural_zero_runs": dict(sorted(sz.items())),
         "zero_tool_arms": {w: {"kept": zt[(w, "kept")], "excluded": zt[(w, "excluded")]} for w in W3},
         "attempt_filtered_n": len(attempt[o]), "attempt_filtered_cells_sha256": hashlib.sha256(json.dumps(attempt[o]).encode()).hexdigest(),
         "added_by_dropping_the_filter_n": len(added), "added_by_dropping_the_filter": {c: lab_of(c) for c in added},
         "headline_cells_missing_rows_per_label": gaps, "labels": labs}
    if gaps:
        raise SystemExit(f"{o}: headline arms without rows in some label (not structural zeros): {gaps}")
    if o == "S_newmains":
        def batch_nm(c):  # batch_of_cell; status-row cells: the batch of the folders that own their attack keys (S5R = the S5 rerun folder)
            if c in mem["batch_of_cell"]:
                return mem["batch_of_cell"][c]
            b = {mem["folders"][f]["batch"].replace("S5R", "S5") for f, ks in mem["folder_keys"].items() for k in ks
                 if k.split("|", 1)[1] == c and k.split("|")[0] in W3}
            if len(b) != 1:
                raise SystemExit(f"{c}: attack keys in folders of batches {sorted(b)}")
            return b.pop()
        x["batch_of_cell"] = {c: batch_nm(c) for c in head}
        x["added_by_batch"] = {b: sorted(c for c in added if x["batch_of_cell"][c] == b) for b in sorted({x["batch_of_cell"][c] for c in added})}
    out["origins"][o] = x
out["notes"] = [
    "Headline = launched minus measurement_excluded; the registered attempt-filtered cohort (appendix line) is a subset (asserted).",
    "structural_zero_runs: completed attack runs with 0 tool calls and no sample error; they have no row in any label (asserted) and score as "
    "0 hits (never alarming) for every probe. Every other headline arm has rows in every label (asserted).",
    "zero_tool_arms counts each arm without rows whose .evals all have 0 tool calls, once by key (kept = structural zero in the headline).",
    "S_newmains: 4r-batch cells enter a probe's cohort only when that probe's files cover every 4r folder (run_peraction311 use4r), and S4/S4r cells "
    "leave the without-S4 row's attack cohort, as for the registered cohort.",
]
stamp = out["stamp_pt"]
dest = HERE / "results" / f"nofilter_exclusions_{stamp}.json"
blob = json.dumps(out, indent=1).encode()
dest.open("xb").write(blob)
md = [f"# No-filter headline cohort by key ({stamp} PT; rule 09 @ 01012e87e + zero-tool rule; keys and counts only)", "",
      f"Census {out['sources']['census']['path']} ({csha[:16]}); sample errors {out['sources']['sample_errors']['path']} ({esha[:16]}). "
      f"Supersedes nofilter_exclusions_2026-10-16_002238 @ debe1eccd ({out['supersedes']['why']}).", ""]
for o, x in out["origins"].items():
    md += [f"## {o}", "", f"Launched {x['launched_n']}; measurement-excluded {x['measurement_excluded_n']}; **headline {x['headline_n']}**; registered "
           f"attempt-filtered cohort (appendix) {x['attempt_filtered_n']}; added by dropping the filter {x['added_by_dropping_the_filter_n']}.", "",
           f"Zero-tool attack arms (kept / excluded): {json.dumps(x['zero_tool_arms'])}.", ""]
    if x["structural_zero_runs"]:
        md += ["Kept zero-tool arms (structural zeros, counted as not caught): " + "; ".join(
            f"{k} (sample_error {v['sample_error']}, log {v['log_status']}, {v['n_messages']} messages)" for k, v in x["structural_zero_runs"].items()), ""]
    md += ["| class | n | cells |", "|---|---:|---|"] + [f"| {k} | {len(v)} | {', '.join(v)} |" for k, v in x["measurement_excluded_by_class"].items()] + [""]
(HERE / "results" / f"nofilter_exclusions_{stamp}.md").open("x").write("\n".join(md) + "\n")
print(json.dumps({"json": str(dest.relative_to(HERE)), "sha256": hashlib.sha256(blob).hexdigest(),
                  "counts": {o: {k: x[k] for k in ("launched_n", "measurement_excluded_n", "headline_n", "attempt_filtered_n", "added_by_dropping_the_filter_n",
                                                    "zero_tool_arms")} | {"structural_zero_runs": list(x["structural_zero_runs"]),
                                                                          "by_class": {k: len(v) for k, v in x["measurement_excluded_by_class"].items()}}
                             for o, x in out["origins"].items()}}, indent=1))
