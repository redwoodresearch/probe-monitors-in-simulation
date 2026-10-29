"""make_waitlist.py check on synthetic keys (no real data): a fully scored cell plus a terminal-classed cell pass; a missing label row STOPs;
a sha mismatch STOPs; a non-affirmative class (recoverable, slow, not scored, no scored row, empty, no usable donor) STOPs; an unclassed
terminal cell STOPs; a structural-zero run counts as scored.  python test_waitlist_neg.py"""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_waitlist as M

LABELS = [f"{p} {r}" for p in M.HEADLINE["probes"] for r in M.HEADLINE["readouts"]]


def put(d, name, obj):
    b = json.dumps(obj).encode()
    (d / name).write_bytes(b)
    return f"{d / name}@{hashlib.sha256(b).hexdigest()}"


def run(w, x, c, extra=()):
    cmd = [sys.executable, str(HERE / "make_waitlist.py"), "check", "--waitlist", w, "--exclusions", x, "--census", c, *extra]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def main():
    d = Path(tempfile.mkdtemp())
    W = {"rule": M.RULE, "terminal_cells": {"tT": {"run1": {}}}, "waitlist": [{"origin": "S_paper493", "cell": c, "arms": list(M.WORLDS)} for c in ("tA", "tB")]}
    X = {"label": "TEST synthetic", "origins": {"S_paper493": {"measurement_excluded": {"tT": {"class": "sample error (EmptyPolicyCompletionError)"},
                                                             "tB": {"class": "setup failure (no policy call)"}}, "structural_zero_runs": {}}}}
    rows = {lb: {"S_paper493": {"label": lb, "attack_rows_by_source_world_cell": {"S_paper493": {w: {"tA": 3} for w in M.WORLDS}}}} for lb in LABELS}
    w, x, c = put(d, "w.json", W), put(d, "x.json", X), put(d, "c.json", {"rows": rows})
    r = run(w, x, c)
    assert r.returncode == 0 and json.loads(r.stdout)["pending"] == [], r.stdout + r.stderr
    rows2 = json.loads(json.dumps(rows))
    del rows2["own_all_rs_L48 R"]["S_paper493"]["attack_rows_by_source_world_cell"]["S_paper493"]["sim"]["tA"]
    r = run(w, x, put(d, "c2.json", {"rows": rows2}))
    assert r.returncode != 0 and "own_all_rs_L48 R x sim" in r.stdout, "a missing label row must STOP"
    r = run(w.rsplit("@", 1)[0] + "@" + "0" * 64, x, c)
    assert r.returncode != 0 and "sha256 differs" in r.stderr, "a sha mismatch must STOP"
    X2 = json.loads(json.dumps(X))
    X2["origins"]["S_paper493"]["measurement_excluded"]["tB"]["class"] = "not scored by cut-off (recoverable)"
    r = run(w, put(d, "x2.json", X2), c)
    assert r.returncode != 0 and '"tB"' in r.stdout, "a recoverable class must not resolve a cell"
    for bad in ("slow", "not scored yet", "no scored row (no folder); the run has tool calls", "", "staging exclusion: no usable donor",
                "no usable scores", "no sample error", "awaiting registered gate", "sample error-free run", "EPCE-like", "sample error: null", "sample error: None", "sample error: false",
                "sample error: 0", "sample error", "sample error: yes", "sample error: EmptyPolicyCompletionError (not present)", "EPCE: no",
                "other", "other (cause known)", "other (cause unknown) pending", "sample error: missing",
                "cause unknown", "sample error: cause unknown", "attempt timeout pending", "attempt timeouts", "attempt timeout (cause unknown) pending",
                "time-limit end (7,200 s); messages not saved, so it can't be scored; pending", "time-limit end", "messages not saved"):
        X3 = json.loads(json.dumps(X))
        X3["origins"]["S_paper493"]["measurement_excluded"]["tB"]["class"] = bad
        r = run(w, put(d, "x3.json", X3), c)
        assert r.returncode != 0 and '"tB"' in r.stdout, f"class {bad!r} must not resolve a cell"
    for ok in ("EPCE", "OutputContract", "sim never launched (no usable honest_a reference)", "setup failure (no policy call)",
               "no run for an arm: run2 missing", "other (cause unknown)", "other (cause unknown): RetryError wrapper",
               "attempt timeout", "attempt timeout (cause unknown)", "time-limit end (7,200 s); messages not saved, so it can't be scored"):
        X6 = json.loads(json.dumps(X))
        X6["origins"]["S_paper493"]["measurement_excluded"]["tT"]["class"] = ok
        X6["origins"]["S_paper493"]["measurement_excluded"]["tB"]["class"] = ok
        r = run(w, put(d, "x6.json", X6), c)
        assert r.returncode == 0, f"class {ok!r} is terminal"
    X4 = json.loads(json.dumps(X))
    del X4["origins"]["S_paper493"]["measurement_excluded"]["tT"]
    r = run(w, put(d, "x4.json", X4), c)
    assert r.returncode != 0 and "terminal cells not classed" in r.stderr, "an unclassed terminal cell must STOP"
    X5 = json.loads(json.dumps(X))
    X5["origins"]["S_paper493"]["structural_zero_runs"] = {"sim|tA": {"sample_error": None}}
    c2b = put(d, "c2b.json", {"rows": rows2})
    r = run(w, put(d, "x5.json", X5), c2b)
    assert r.returncode == 0, "a structural-zero run counts as scored"
    FL = ["sol_a_uncut", "FIBS|uncut"]
    rows3 = {f"{p} {r}": {"S_paper493": {"attack_rows_by_source_world_cell": {"S_paper493": {w: {"tA": 1} for w in M.WORLDS}}}} for p in FL for r in ("C", "R")}
    c3 = put(d, "c3.json", {"rows": rows3})
    r = run(w, x, c3)
    assert r.returncode != 0, "the default (32k) labels must not be satisfied by full-length rows"
    r = run(w, x, c3, ["--probes", ",".join(FL)])
    assert r.returncode == 0 and json.loads(r.stdout)["labels"][1] == "sol_a_uncut R", "--probes selects the full-length labels"
    for empty in ("", ",", "sol_a_uncut,"):
        r = run(w, x, c, ["--probes", empty])
        assert r.returncode != 0 and "empty item" in r.stderr, f"--probes {empty!r} must STOP"
    r = run(w, x, c, ["--readouts", "C"])
    assert r.returncode == 0 and json.loads(r.stdout)["labels"][:2] == ["Sol (a) C", "FIBS C"], "--readouts C selects C labels only"
    r = run(w, x, c, ["--origin", "S_new"])
    assert r.returncode != 0 and "origin" in r.stderr, "a waitlist of another origin must STOP"
    XF = json.loads(json.dumps(X)) | {"label": "FROZEN synthetic"}
    xf = put(d, "xf.json", XF)
    assert run(w, xf, c).returncode != 0, "a non-TEST exclusions file without --final must STOP"
    assert run(w, x, c, ["--final"]).returncode != 0, "a TEST exclusions file with --final must STOP"
    assert run(w, xf, c, ["--final"]).returncode == 0, "final file with --final passes when all resolve"
    X9 = json.loads(json.dumps(X))
    X9["origins"]["S_paper493"]["measurement_excluded"]["tB"]["class"] = M.PENDING_LABELS[1]
    r = run(w, put(d, "x9.json", X9), c)
    assert r.returncode == 0 and json.loads(r.stdout)["pending_09_class"] == ["tB"], "TEST passes the renamed pending label"
    r = run(w, put(d, "x9f.json", X9 | {"label": "FROZEN"}), c, ["--final"])
    assert r.returncode != 0 and "tB" in json.loads(r.stdout)["pending"], "final STOPs on the renamed pending label"
    for where in ("cell", "run"):
        X7 = json.loads(json.dumps(X))
        if where == "cell":
            X7["origins"]["S_paper493"]["measurement_excluded"]["tB"]["class"] = M.PENDING_09
            want = "tB"
        else:
            X7["origins"]["S_paper493"]["structural_zero_runs"] = {"sim|tA": {"class": M.PENDING_09}}
            want = "tA"
        r = run(w, put(d, "x7.json", X7), c2b if where == "run" else c)
        assert r.returncode == 0 and json.loads(r.stdout)["pending_09_class"] == [want], f"TEST passes a pending-09 {where}: " + r.stdout + r.stderr
        r = run(w, put(d, "x7f.json", X7 | {"label": "FROZEN"}), c2b if where == "run" else c, ["--final"])
        assert r.returncode != 0 and want in json.loads(r.stdout)["pending"], f"final STOPs on a pending-09 {where}"
    X8 = json.loads(json.dumps(X)) | {"label": "FROZEN"}
    X8["origins"]["S_paper493"]["structural_zero_runs"] = {"run1|tT": {"class": M.PENDING_09}}
    r = run(w, put(d, "x8.json", X8), c, ["--final"])
    assert r.returncode != 0 and "tT" in r.stderr, "a terminal cell marked pending-09 must STOP a final"
    r = run(w, put(d, "x8t.json", X8 | {"label": "TEST"}), c)
    assert r.returncode == 0 and json.loads(r.stdout)["pending_09_class"] == ["tT"], "TEST reports it"
    # full-length: two origins, the FULL_SIX labels, IL extra arms, the uncut-census guard
    FLB = M.labels_of(list(M.FULL_SIX), ["C", "R"])
    WF = {"rule": M.RULE, "mode": "full-length", "labels": FLB, "terminal_cells": {},
          "waitlist": [{"origin": "S_new", "cell": "nA", "arms": list(M.WORLDS), "extra_arms": list(M.EXTRA_WORLDS)},
                       {"origin": "S_paper493", "cell": "tA", "arms": list(M.WORLDS)}]}
    XF2 = {"label": "TEST", "origins": {o: {"measurement_excluded": {}, "structural_zero_runs": {}} for o in ("S_new", "S_paper493")}}
    cen = lambda worlds_n, meta=True: {"meta": {"full_length": {"overrides": "x"}} if meta else {}, "rows": {lb: {"S_newmains": {
        "attack_rows_by_source_world_cell": {"S_new": {w: {"nA": 1} for w in worlds_n}}}, "S_paper493": {
        "attack_rows_by_source_world_cell": {"S_paper493": {w: {"tA": 1} for w in M.WORLDS}}}} for lb in FLB}}
    wf, xf2 = put(d, "wf.json", WF), put(d, "xf2.json", XF2)
    r = run(wf, xf2, put(d, "cf1.json", cen(M.WORLDS + M.EXTRA_WORLDS)))
    assert r.returncode == 0 and json.loads(r.stdout)["pending"] == [], "full: all arms scored on both origins passes: " + r.stdout + r.stderr
    r = run(wf, xf2, put(d, "cf2.json", cen(M.WORLDS + M.EXTRA_WORLDS, meta=False)))
    assert r.returncode != 0 and "uncut census" in r.stderr, "full: a census without meta.full_length must STOP"
    cf3 = put(d, "cf3.json", cen(M.WORLDS))
    r = run(wf, xf2, cf3)
    o0 = json.loads(r.stdout)["origins"][0]
    assert r.returncode == 0 and o0["extra_arm_pending"] == ["nA"] and o0["extra_arms"]["sim2"]["not_scored"] == 1, "default: extra arms report-only"
    r = run(wf, xf2, cf3, ["--extra-arms-gate"])
    assert r.returncode != 0 and json.loads(r.stdout)["pending"] == ["S_new|nA"], "--extra-arms-gate: an unscored extra arm gates"
    XF3 = json.loads(json.dumps(XF2))
    XF3["origins"]["S_new"]["extra_arm_excluded"] = {"sim2|nA": {"class": "sample error: EmptyPolicyCompletionError"}}
    r = run(wf, put(d, "xf3.json", XF3), cf3, ["--extra-arms-gate"])
    assert r.returncode != 0 and "x run3" in r.stdout and "x sim2" not in r.stdout, "a failed sim2 resolves sim2 only; run3 still waits"
    XF3["origins"]["S_new"]["extra_arm_excluded"]["run3|nA"] = {"class": "attempt timeout (cause unknown)"}
    r = run(wf, put(d, "xf4.json", XF3), cf3, ["--extra-arms-gate"])
    o0 = json.loads(r.stdout)["origins"][0]
    assert r.returncode == 0 and o0["resolved_scored"] == 1 and o0["extra_arms"]["run3"]["terminal"] == 1, "terminal extra arms resolve (cell stays)"
    cf7 = cen(M.WORLDS)
    del cf7["rows"][FLB[0]]["S_newmains"]["attack_rows_by_source_world_cell"]["S_new"]["run1"]["nA"]
    r = run(wf, put(d, "xf4b.json", XF3), put(d, "cf7.json", cf7))
    assert r.returncode != 0 and "x run1" in r.stdout, "terminal extra arms never resolve a cell whose primary arm lacks a row"
    r = run(wf, xf2, put(d, "cf8.json", cen(M.WORLDS + M.EXTRA_WORLDS)), ["--readouts", "C,R"])
    assert r.returncode == 0, "--readouts alone inherits the waitlist's probes"
    r = run(wf, xf2, put(d, "cf5.json", cen(M.WORLDS + M.EXTRA_WORLDS)), ["--probes", "Sol (a)"])
    assert r.returncode != 0 and "built for" in r.stderr, "labels other than the waitlist's must STOP"
    XF5 = {"label": "TEST", "origins": {"S_new": {"measurement_excluded": {}}}}
    r = run(wf, put(d, "xf5.json", XF5), put(d, "cf6.json", cen(M.WORLDS + M.EXTRA_WORLDS)))
    assert r.returncode != 0 and "no S_paper493 origin" in r.stderr, "every waitlist origin needs an exclusions block"
    XF6 = json.loads(json.dumps(XF3)) | {"label": "FROZEN"}
    XF6["origins"]["S_new"]["extra_arm_excluded"]["run3|nA"] = {"class": M.PENDING_LABELS[1]}
    cf9 = put(d, "cf9.json", cen(M.WORLDS + M.EXTRA_WORLDS))
    r = run(wf, put(d, "xf6.json", XF6), cf9, ["--final", "--extra-arms-gate"])
    assert r.returncode != 0 and "S_new|nA" in json.loads(r.stdout)["pending"], "--extra-arms-gate: a pending label on an extra arm STOPs a final"
    r = run(wf, put(d, "xf6b.json", XF6), cf9, ["--final"])
    o0 = json.loads(r.stdout)["origins"][0]
    assert r.returncode == 0 and o0["extra_arms"]["run3"]["pending_label"] == 1, "default: a pending extra-arm label is reported, never gates"
    XF7 = json.loads(json.dumps(XF2)) | {"label": "FROZEN"}
    XF7["origins"]["S_new"]["structural_zero_runs"] = {"run1|nA": {"class": M.PENDING_09}}
    r = run(wf, put(d, "xf7.json", XF7), cf9, ["--final"])
    assert r.returncode != 0 and "S_new|nA" in json.loads(r.stdout)["pending"], "a pending label on a PRIMARY arm still STOPs a final"
    XF8 = json.loads(json.dumps(XF2))
    XF8["origins"]["S_new"]["measurement_excluded"] = {"nA": {"class": "setup failure (no policy call)"}}
    r = run(wf, put(d, "xf8.json", XF8), cf3)
    o0 = json.loads(r.stdout)["origins"][0]
    assert r.returncode == 0 and o0["extra_arms"]["sim2"]["admitted"] == 1 and o0["extra_arm_pending"] == ["nA"], "extras counted on terminal cells too"
    WE = {"rule": M.RULE, "terminal_cells": {"tT": {"run1": {}}}, "waitlist": []}
    XE = {"label": "FROZEN", "origins": {"S_paper493": {"measurement_excluded": {"tT": {"class": "sample error: EmptyPolicyCompletionError"}},
                                                        "structural_zero_runs": {"run1|tT": {"class": M.PENDING_09}}}}}
    we, xe = put(d, "we.json", WE), put(d, "xe.json", XE)
    assert run(we, xe, c, ["--final"]).returncode != 0, "an empty waitlist without --origin must STOP"
    r = run(we, xe, c, ["--final", "--origin", "S_paper493"])
    assert r.returncode != 0 and "tT" in r.stderr, "an empty waitlist still checks its terminal cells"
    print("test_waitlist_neg: 84 cases pass")


if __name__ == "__main__":
    main()
