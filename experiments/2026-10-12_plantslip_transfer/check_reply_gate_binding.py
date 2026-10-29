"""Review check for #1087 rw4fw (q2_reply_score.py:134, "Bind planted scoring to the gated artifact hashes"): planted scoring
accepts any --gate record for the pair that says "pass" with the same staging sha256, and the planted record doesn't name it.
The as-run P4-P6 records are in git (first copies; each must be unchanged at HEAD). Per pair this checks that git has only one
controls (gate) record and one planted record, that the gate passed on 20 distinct controls x 4 nested models before the planted
scoring, and that both records name the same frozen run, staging file and sha256, probe sha256 and stored-score sha256; the
probe and stored-score sha256 must also equal the pins record (q2_reply_pins_2026-10-12_133535.json, written before staging;
it pins no staging). git log --all, so a gate record on any branch counts. Prints file names, hashes and pass/fail only: no
score. Exits nonzero on any mismatch.
  uv run --no-project python experiments/2026-10-12_plantslip_transfer/check_reply_gate_binding.py
"""
import json
import re
import subprocess
import sys

REC = "experiments/2026-10-12_plantslip_transfer/results/"
bad = []


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout


def chk(label, ok):
    print(("match   " if ok else "MISMATCH"), label); ok or bad.append(label)


def first_copy(name):
    """the record as first committed, which must equal HEAD's copy"""
    adds = git("log", "--all", "--reverse", "--diff-filter=A", "--format=%H", "--", REC + name).split()
    first = git("show", f"{adds[0]}:{REC + name}")
    return json.loads(first), adds[0][:8], first == git("show", f"HEAD:{REC + name}")


ever = sorted({p.rsplit("/", 1)[1] for p in git("log", "--all", "--name-only", "--format=", "--", REC).split()
               if re.fullmatch(r"q2_reply_scores_P[456]_(controls|planted)_[0-9_-]+\.json", p.rsplit("/", 1)[1])})
pins, pc, same = first_copy("q2_reply_pins_2026-10-12_133535.json")
chk(f"pins record: unchanged at HEAD since its first copy ({pc})", same)
stages = set()
for P in ("P4", "P5", "P6"):
    rd = pins["readers"][P]
    c_name = [n for n in ever if n.startswith(f"q2_reply_scores_{P}_controls_")]; p_name = [n for n in ever if n.startswith(f"q2_reply_scores_{P}_planted_")]
    chk(f"{P}: git has only one controls record {c_name} and one planted record {p_name}", len(c_name) == len(p_name) == 1)
    (c, cc, c_same), (p, pcm, p_same) = first_copy(c_name[0]), first_copy(p_name[0])
    chk(f"{P}: both are unchanged at HEAD since their first copies ({cc}, {pcm})", c_same and p_same)
    print(f"finding  {P}: the planted record has no field naming its gate record" if not any("gate" in k for k in p) else f"note     {P}: the planted record names a gate field")
    chk(f"{P}: the gate passed on 20 distinct controls x 4 nested models, and its stamp precedes the planted record's",
        c.get("gate") == "pass" and c["runs"] == c["gate: distinct controls"] == 20 and c["gate: comparisons (20 controls x 4 nested models)"] == 80
        and c["gate: max |re-scored - stored| nested score (tolerance 1e-3)"] <= 1e-3 and c_name[0][-22:] < p_name[0][-22:])
    chk(f"{P}: same pair, reader and frozen run ({p['frozen run']}) in both records and the pins record",
        c["pair"] == p["pair"] == P and c["reader"] == p["reader"] == rd["reader"] and c["frozen run"] == p["frozen run"] == rd["run"])
    chk(f"{P}: same staging file and sha256 in both records ({p['stage file']}, {p['stage sha256'][:12]})", c["stage file"] == p["stage file"] and c["stage sha256"] == p["stage sha256"])
    stages.add(p["stage sha256"])
    chk(f"{P}: the pins record's stamp (2026-10-12_133535) precedes the staging record's", "2026-10-12_133535" < p["stage file"].rsplit("_", 2)[-2] + "_" + p["stage file"].rsplit("_", 1)[-1][:6])
    probes = {str(l): pins["probes (work Volume)"][f"{P} {rd['work']}:runs/{rd['run']}/probes_L{l}.npz"] for l in rd["layers"]}
    chk(f"{P}: both records' probe sha256 equal the pins at layers {sorted(probes, key=int)}", c["probes sha256"] == p["probes sha256"] == probes)
    stored = {f: pins["stored scores (local)"][f"{P} run_scores_{rd['run']}/{f}"] for f in ["runs.json"] + [f"scores_L{l}.npz" for l in rd["layers"]]}
    chk(f"{P}: both records' stored-score sha256 (runs.json and scores) equal the pins", c["stored scores sha256"] == p["stored scores sha256"] == stored)
chk(f"P4-P6 share one staging record sha256 {sorted(s[:12] for s in stages)}", len(stages) == 1)
print(f"{len(bad)} mismatches")
sys.exit(1 if bad else 0)
