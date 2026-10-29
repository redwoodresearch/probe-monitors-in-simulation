"""Review check for #1080 (Codex rrMaL) and #1079/#1080 (rrMQ_, rrMaH). run_score_cmd.py gives a run in a unit's universe
that is missing from the planted list its stored, unplanted score (meant for runs the plant left unchanged). This checks,
from the committed records, that every run the plant changed was listed and rescored in each unit. It also checks that the
five transcript lists without a committed producer (all but gate_transcripts.json, which run_stage_cmd.py writes) equal
what the staging record and the pinned row files imply. Exits nonzero on any failure. Reads every committed file with git
show at its first copy (all reachable from cmd-6's merge -s ours).

  uv run --no-project --with numpy python experiments/2026-10-12_plantslip_transfer/q2_cmd/check_planted_coverage.py <pins dir>
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

Q = "experiments/2026-10-12_plantslip_transfer/q2_cmd/"
show = lambda c, p: subprocess.run(["git", "show", f"{c}:{Q}{p}"], capture_output=True, check=True).stdout  # noqa: E731
J = lambda c, p: json.loads(show(c, p))  # noqa: E731
PIN = {"refit_qwen35_rows.json": "41a68be0b1d7ae045d573dce15ce2a4b3cfa349f915cc6babc7fd478f28ed8e4",  # copied from run_score_cmd.py @ 4c1623ef
       "route3_scores_2026-10-10_232231.json": "cf33388724da001c54107cf22cc416582ee8f58dcf99b39db61c4ad25c5b62e6",
       "q2_manifest.json": "cd4ab3495ded77f92ca73d88e53a00d4572ebabc9089f5684db75ad8e801263a"}


def main(pins):
    for f, h in PIN.items():
        assert hashlib.sha256((pins / f).read_bytes()).hexdigest() == h, f"{f} is not the pinned file"
    refit = json.loads((pins / "refit_qwen35_rows.json").read_text()); r3 = json.loads((pins / "route3_scores_2026-10-10_232231.json").read_text())
    man = json.loads((pins / "q2_manifest.json").read_text())
    # each unit's universe of runs, as load_pair builds it (keys only)
    matched = {(r["key"], r["step"], r["action"]) for r in refit if r["arm"] == "real-attack-e1"}
    U = {("P2", "e1"): {r["key"] for r in refit if r["arm"] == "real-attack-e1"}, ("P2", "e2"): {r["key"] for r in r3["rows"] if r["world"] == "e2"},
         ("P3", "e1"): {m["key"] for m in man if m["arm"] == "real-attack-e1"}, ("P3", "e2"): {r["key"] for r in r3["rows"] if r["world"] == "e2"}}
    U[("P3", "e1m")] = {m["key"] for m in man if m["arm"] == "real-attack-e1" and (m["key"], m["step"], m["action"]) in matched}
    st = J("6054b550", "results/stage_cmd_2026-10-12_135134.json")
    changed = {a: {x["key"] for x in st["per run"] if x["arm"] == f"real-attack-{a}" and x["outputs changed"] > 0} for a in ("e1", "e2")}
    assert len(st["per run"]) == sum(map(len, changed.values())), "the staging record lists a run with no changed output"
    lists = {"P2": {"e1": J("6054b550", "config/planted_transcripts_p2.json"), "e2": J("6054b550", "config/planted_transcripts_p2.json")},
             "P3": {"e1": J("6054b550", "config/planted_transcripts_p3.json"), "e2": J("886d88f1", "config/planted_transcripts_p3_e2.json")}}
    recs = {"P2": J("4c1623ef", "results/score_p2_planted_2026-10-12_151732.json"), "P3": J("4c1623ef", "results/score_p3_planted_2026-10-12_151948.json")}
    fail = 0
    for (pair, u), uni in sorted(U.items()):
        a = "e1" if u in ("e1", "e1m") else "e2"
        listed = {k for s, k in lists[pair][a]["affected"] if s == f"{a}-planted"} & uni  # what the scorer rescored (lst[u])
        need = changed[a] & uni  # every run the plant changed, in this unit's universe
        arm = recs[pair]["arms"][u]
        ok = listed == need and arm["planted runs scored"] == len(need) == recs[pair]["transcripts"][u] and arm["runs"] == len(uni) \
            and arm["runs excluded"] == {"no planted score": 0, "no unplanted score": 0}
        fail += not ok
        print(f"{pair} {u}: universe {len(uni)} runs (record {arm['runs']}); changed in it {len(need)}; listed and rescored {len(listed)} (record {arm['planted runs scored']}); "
              f"changed but not listed {len(need - listed)}; listed but unchanged {len(listed - need)}; excluded {arm['runs excluded']} -> {'ok' if ok else 'FAIL'}")
    for pair, rec in recs.items():
        for run, c in rec["extraction counts"].items():
            ok = c["missing meta"] == 0 and c["missing call_end"] == 0; fail += not ok
            print(f"{pair} extraction {run}: transcripts {c['transcripts']}, missing meta {c['missing meta']}, missing call_end {c['missing call_end']} -> {'ok' if ok else 'FAIL'}")
    # the five lists without a committed producer: exactly what the staging record and the pinned rows imply
    stg = st["staged"].split(":", 1)[1]
    exp = {"planted_transcripts_p2.json": ("6054b550", "affected", {("e1-planted", k) for k in changed["e1"] & U[("P2", "e1")]} | {("e2-planted", k) for k in changed["e2"]}),
           "planted_transcripts_p3.json": ("6054b550", "affected", {("e1-planted", k) for k in changed["e1"]}),
           "planted_transcripts_p3_e2.json": ("886d88f1", "affected", {("e2-planted", k) for k in changed["e2"]}),
           "gate_transcripts_p3.json": ("6054b550", "controls", {("e1-control", k) for k in st["controls"]["real-attack-e1"]}),
           "gate_transcripts_p3_e2.json": ("886d88f1", "controls", {("e2-control", k) for k in st["controls"]["real-attack-e2"]})}
    for f, (c, field, want) in exp.items():  # each field checked on its own: the planted scorer reads only affected, the gate only controls
        x = J(c, f"config/{f}"); got = [tuple(e) for e in x[field]]; other = "controls" if field == "affected" else "affected"
        ok = set(got) == want and len(got) == len(set(got)) and x[other] == [] and x["inputs_run"] == stg and x["inputs_sha256"] == st["inputs_sha256"]
        fail += not ok
        print(f"{f} @ {c}: {field} {len(got)} entries, equal to the staging record's implied set: {set(got) == want}; no duplicates: {len(got) == len(set(got))}; "
              f"{other} empty: {x[other] == []}; inputs run and sha256 match: {x['inputs_run'] == stg and x['inputs_sha256'] == st['inputs_sha256']} -> {'ok' if ok else 'FAIL'}")
    print("all checks pass" if not fail else f"{fail} checks FAIL")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
