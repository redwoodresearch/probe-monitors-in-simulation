"""Synthetic tests of route C §0.3 (10-17): the wrapper's dryrun_plan reducer and allowlisting, run_step on a dryrun_plan step, and
make_dryrun_sidecar.py. A throwaway git repo and fabricated stdout / logs only; no Modal call, no real record. Each case asserts its exact
outcome. The reviews' items each have their own cases: R-S1 / R-S1b (the string rule, glued numbers included), ruling (a) (no hash: a
refused value is None, a refused key refused_key_<i>), R-S2 (anti-downgrade negatives), O-S1 / O-S2 / O-S3, O2-1 (pin allowlist), O2-3
(paths as segment lists), O2-4 (the sidecar loads the wrapper as committed).

  uv run --no-project --python 3.12 python -I test_dryrun_sidecar.py
"""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = "experiments/2026-10-15_fpr_sweep_507_prep/run_peraction311.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W = load("recstep_w", HERE / "run_records_step_in_container.py")
S = load("dryrun_sidecar", HERE / "make_dryrun_sidecar.py")
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
EMBEDDED = ["threshold 0.0123", "auroc=0.97", "fpr_0.0123", "12.5%", "{0.01, 0.05}", "catch-0.41", "t309:0.0123", "points_1e-3",
            # R-S1b: glued numbers
            "fpr0.01", "auroc0.97", "0.97auc", "tau_0.0123x", "-0.5x", "x1e-3", "lr1e-5", "1.5x", "32.5k", "x_0." + "1" * 40,
            # 507-prep / sidecar3: leading-dot decimals; B3: embedded nan / inf
            "catch_.41", "t309:.0123", "catch-.41", "auroc_.97", "x.5", "k_.0123x", ".41_catch", "auc.97", "a/.5/b", "x_." + "1" * 40,
            "auc_nan", "x_inf"]
GLUED_KEYS = {"fpr0.01": 1, "0.97auc": 2, "lr1e-5": 3, "32.5k": 4, "x_0." + "2" * 40: 5, "catch_.41": 6, "t309:.0123": 7, "catch-.41": 8,
              "auroc_.97": 9, "x.5": 10, "k_.0123x": 11, ".41_catch": 12, "auc.97": 13, "a/.5/b": 14, "x_." + "3" * 40: 15, "auc_nan": 16, "x_inf": 17}
H64, H40 = "3e5a" * 16, "9f1c" * 10
SHA_NAMED = ["x/" + H64 + ".json", H40 + ".json", "name_" + H64 + ".npz"]  # sidecar3 O3-2: sha-named files pass
PLAN = {"dryrun": True, "levels": [0.01, 0.05], "n_labels": 2, "labels": ["a C", "b R"], "label": "filled free text for the record",
        "full_length": {"inputs_full": {"path": "config/INPUTS_FULL_TEST_x.json", "sha256": "a" * 64, "status": "TEST built 2026-10-17 09:00 PT",
                                        "test": True},
                        "jobs_per_variant": 3, "ratio_str": "0.5", "overrides_pending": {"S_new|C": "pending: 4f scorer at 0.97"},
                        "scope": {"auroc_subset": "the 0.6 AUROC subset"}, "label_notes": {"a C": "note with 32k and 1e-3"},
                        "embedded": EMBEDDED, "fpr_0.0123": 7, "plain": ["S_paper493", "t309", "2026-10-17_090000", "b" * 40, "c" * 64] + SHA_NAMED,
                        "glued_keys": GLUED_KEYS, "roots": ["rh1e4_stage_v1__honest_b"]},
        "inputs_final": {"path": "config/f.json", "sha256": "b" * 64, "status": "FROZEN 2026-10-16 04:41 PT"}, "seed": 7, "npz_shards": None}
PINNED = sorted(["a C", "b R", "rh1e4_stage_v1__honest_b"])
PKEYS = sorted(PLAN)
HEAD = json.dumps({"read_started_pt": "2026-10-17_090102", "rows": ["a C", "b R"]})
REFUSED_TEXT = EMBEDDED + list(GLUED_KEYS) + ["0.5", "fpr_0.0123", "pending: 4f scorer at 0.97", "filled free text for the record",
                                              "TEST built 2026-10-17 09:00 PT", "the 0.6 AUROC subset", "note with 32k and 1e-3"]


def out_of(plan=PLAN, head=HEAD, extra=""):
    return "a warning line\n" + (head + "\n" if head else "") + (json.dumps(plan, indent=1) + "\n" if plan is not None else "") + extra


def red_of(plan=PLAN, **kw):
    return W.reduce_dryrun_plan(out_of(plan, **kw), PINNED, PKEYS)


def stops(f, *a):
    try:
        f(*a)
    except (SystemExit, W.Stop):
        return True
    return False


def stop_msg(f, *a):
    try:
        f(*a)
    except (SystemExit, W.Stop) as e:
        return str(e)
    return ""


def any_form(s, dump):  # the text, or any common hash of it (ruling (a): no hash of any kind may relay)
    b = s.encode()
    return any(f in dump for f in (s, hashlib.sha256(b).hexdigest(), hashlib.sha1(b).hexdigest(), hashlib.md5(b).hexdigest()))


cases = {}
r = red_of()
fl_ = r["plan"]["full_length"]
dump_ = json.dumps(r)
rp_ = r["refused_paths"]
cases["reducer: a dry run passes; floats dropped, each path a LIST of segments with int list indices (O2-3)"] = (
    r["pass"] is True and r["plan"]["levels"] == [None, None] and ["plan", "levels", 0] in rp_ and ["plan", "levels", 1] in rp_
    and all(type(p) is list for p in rp_) and r["rows_n"] == 2 and r["plan"]["inputs_final"]["sha256"] == "b" * 64
    and r["read_started_pt"] == "2026-10-17_090102")
cases["R-S1 / R-S1b / (a): every embedded OR glued number string is dropped (None), its path listed"] = (
    fl_["embedded"] == [None] * len(EMBEDDED) and all(["plan", "full_length", "embedded", i] in rp_ for i in range(len(EMBEDDED))))
cases["R-S1 / (a): a whole number-like string is dropped (ratio_str)"] = fl_["ratio_str"] is None and ["plan", "full_length", "ratio_str"] in rp_
cases["R-S1 / (a) / O3-1: a refused dict KEY becomes refused_key_<i>, i counted among REFUSED siblings only (here the only one: 0)"] = (
    fl_.get("refused_key_0") == 7 and "fpr_0.0123" not in fl_ and ["plan", "full_length", "refused_key_0"] in rp_
    and not any(k.startswith("refused_key_") and k != "refused_key_0" for k in fl_))
cases["R-S1b / (a): GLUED-number keys become refused_key_<i> (values kept: ints)"] = (
    fl_["glued_keys"] == {f"refused_key_{i}": v for i, v in enumerate(GLUED_KEYS.values())}
    and all(["plan", "full_length", "glued_keys", f"refused_key_{i}"] in rp_ for i in range(len(GLUED_KEYS))))
cases["R-S1 / (a): free text is dropped unless pinned (overrides_pending reason, label filled, status, scope.auroc_subset, label_notes value)"] = (
    fl_["overrides_pending"] == {"S_new|C": None} and r["plan"]["label"] is None and fl_["inputs_full"]["status"] is None
    and fl_["scope"]["auroc_subset"] is None and fl_["label_notes"] == {"a C": None})
cases["pins: pinned labels relay as is (values, keys and path segments); the allowlisted root relays pinned"] = (
    r["plan"]["labels"] == ["a C", "b R"] and "a C" in fl_["label_notes"] and ["plan", "full_length", "label_notes", "a C"] in rp_
    and fl_["roots"] == ["rh1e4_stage_v1__honest_b"])
cases["R-S1b: the real root rh1e4_stage_v1__honest_b is refused when NOT pinned (its known casualty)"] = (
    W.reduce_dryrun_plan(out_of(), [x for x in PINNED if x != "rh1e4_stage_v1__honest_b"], PKEYS)["plan"]["full_length"]["roots"] == [None])
cases["O3-2: sha-named files (x/<sha64>.json, <sha40>.json, name_<sha64>.npz) pass as values and keys"] = (
    all(W._str_ok(s) for s in SHA_NAMED) and not stops(W._no_float, {s: s for s in SHA_NAMED}, True))
cases["noted contrived passes (accepted): nan5, 5nan, 0x1p-3"] = all(W._str_ok(s) for s in ("nan5", "5nan", "0x1p-3"))
cases["R-S1: plain identifiers, paths, dates with integers and 40/64-hex shas pass unchanged"] = (
    fl_["plain"] == PLAN["full_length"]["plain"] and fl_["inputs_full"]["path"] == "config/INPUTS_FULL_TEST_x.json" and fl_["inputs_full"]["sha256"] == "a" * 64)
cases["(a) blinding: no refused string appears in the reducer's output in ANY form (raw, sha256, sha1, md5); no 'sha256:' marker"] = (
    not any(any_form(s, dump_) for s in REFUSED_TEXT) and "sha256:" not in dump_)
cases["R-S1: the reducer's output passes the wrapper's whitelist with values=True and the pinned strings"] = not stops(W._no_float, r, True, frozenset(PINNED))
cases["R-S1 / R-S1b: _no_float(values=True) STOPs on embedded and glued numbers, as values AND keys; passes pinned; passes placeholders"] = (
    all(stops(W._no_float, {"x": s}, True) for s in EMBEDDED) and all(stops(W._no_float, {k: 1}, True) for k in list(GLUED_KEYS) + ["auroc=0.97"])
    and stops(W._no_float, ["Sol (a) C"], True) and not stops(W._no_float, ["Sol (a) C"], True, frozenset(["Sol (a) C"]))
    and not stops(W._no_float, ["refused_key_3", "a" * 64, "1" * 64, "t309", ["plan", "levels", 0]], True))
cases["impact: the other reducers' real output shapes still pass (paths, shas, counts, booleans; fix_v2, inputs_full, json_line_keys)"] = all(
    not stops(W._no_float, o, True) for o in (
        {"written": [{"path": "experiments/2026-10-14_q3_overnight_chain/results/cutfull/x_2026-10-17_090000.json", "sha256": "c" * 64}]},
        {"parsed": True, "complete": True, "dry": False, "pilot_rows": {"sha256": "d" * 64, "n": 20}, "fix_v2": {"sha256": "e" * 64, "n": 4353},
         "n_stops": 0, "n_post_write_checks": 3},
        {"pass": True, "n_problems": 0}, {"sha256": "f" * 64, "n_pending": 0, "n_problems": 0}, {"json_lines": 1, "keys_match": True, "n_keys": 3}))
cases["reducer: the output names no top-level path / sha that bind_reduced would bind"] = not stops(W.bind_reduced, r, [])
cases["reducer: no plan object -> not a pass"] = W.reduce_dryrun_plan(out_of(plan=None), PINNED, PKEYS)["pass"] is False
cases["reducer: two plan objects -> not a pass"] = W.reduce_dryrun_plan(out_of(extra=json.dumps(PLAN, indent=1) + "\n"), PINNED, PKEYS)["pass"] is False
cases["reducer: dryrun not true -> not a pass"] = W.reduce_dryrun_plan(out_of(plan=PLAN | {"dryrun": False}), PINNED, PKEYS)["pass"] is False
cases["reducer: no {read_started_pt, rows} line -> not a pass"] = W.reduce_dryrun_plan(out_of(head=None), PINNED, PKEYS)["pass"] is False
cases["reducer: two {read_started_pt, rows} lines -> not a pass"] = W.reduce_dryrun_plan(out_of(head=HEAD + "\n" + HEAD), PINNED, PKEYS)["pass"] is False
cases["reducer: more refused paths than MAX_PLAN_REFUSED -> not a pass, and no plan returns"] = (
    lambda x: x["pass"] is False and x["plan"] is None)(W.reduce_dryrun_plan(out_of(plan=PLAN | {"many": [0.5] * (W.MAX_PLAN_REFUSED + 1)}), PINNED, sorted(PKEYS + ["many"])))
cases["reducer: a non-pass returns no plan and no refused paths"] = (lambda x: x["plan"] is None and x["refused_paths"] == [])(
    W.reduce_dryrun_plan(out_of(plan=PLAN | {"dryrun": False}), PINNED, PKEYS))
cases["(a): a refused_key placeholder colliding with a real key of the same dict drops the dict"] = (
    lambda x: x["plan"]["full_length"]["coll"] is None and ["plan", "full_length", "coll"] in x["refused_paths"])(
    W.reduce_dryrun_plan(out_of(plan=PLAN | {"full_length": PLAN["full_length"] | {"coll": {"0.5x 1.5": 1, "refused_key_0": 2}}}), PINNED, PKEYS))
cases["O-S1: a plan whose top-level keys differ from expect_plan_keys -> not a pass (extra key; missing key; none pinned)"] = (
    W.reduce_dryrun_plan(out_of(plan=PLAN | {"extra": 1}), PINNED, PKEYS)["pass"] is False
    and W.reduce_dryrun_plan(out_of(plan={k: v for k, v in PLAN.items() if k != "seed"}), PINNED, PKEYS)["pass"] is False
    and W.reduce_dryrun_plan(out_of(), PINNED, None)["pass"] is False)
cases["O-S2: full_length.inputs_full.test not true -> not a pass (false; absent)"] = all(
    W.reduce_dryrun_plan(out_of(plan=PLAN | {"full_length": PLAN["full_length"] | {"inputs_full": ifu}}), PINNED, PKEYS)["pass"] is False
    for ifu in (PLAN["full_length"]["inputs_full"] | {"test": False}, {k: v for k, v in PLAN["full_length"]["inputs_full"].items() if k != "test"}))

# preflight on a throwaway repo: dryrun_plan is allowed only for the runner, verify_only, main with dryrun (no spec_only), a TEST INPUTS_FULL
d = Path(tempfile.mkdtemp(prefix="dryrun_side_"))
g = lambda *a: subprocess.run(["git", "-C", str(d), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
g("init", "-q")
g("config", "user.email", "t@t")
g("config", "user.name", "t")
OTHER = "experiments/2026-10-17_other/other.py"
FAKE_RUNNER = r'''
import json


def _main(dryrun=False, inputs_full="", plan_json="{}"):
    print("a warning line")
    print(json.dumps({"read_started_pt": "2026-10-17_090102", "rows": ["a C", "b R"]}))
    print(json.dumps(json.loads(plan_json), indent=1), flush=True)


class _Info:
    def __init__(self, f):
        self.raw_f = f


class _Fn:
    def __init__(self, f):
        self.info = _Info(f)


main = _Fn(_main)
'''
for f_, body in ((RUNNER, FAKE_RUNNER), ("experiments/2026-10-15_fpr_sweep_507_prep/inputs_full.py", "x = 1\n"), (OTHER, "y = 2\n")):
    (d / f_).parent.mkdir(parents=True, exist_ok=True)
    (d / f_).write_text(body)
GOOD = {"script": RUNNER, "reducer": "dryrun_plan", "memory_mib": 4096, "verify_only": True, "outputs": [], "n_outputs": 0, "commits": [],
        "entry": {"attr": "main", "flags": {"dryrun": True, "inputs_full": "config/INPUTS_FULL_TEST_2026-10-17_090000.json@" + "c" * 64}},
        "expect_strings": PINNED, "expect_plan_keys": PKEYS}


def spec_commit(spec, name="s.json"):
    p = d / "steps" / name
    p.parent.mkdir(exist_ok=True)
    blob = g("hash-object", spec["script"]) if spec.get("script") in (RUNNER, OTHER) else None
    p.write_text(json.dumps(spec | {"script_blob": blob} if blob else spec))
    g("add", "-A")
    g("commit", "-q", "-m", "s", "--allow-empty")
    return g("rev-parse", "HEAD")


def pre(spec):
    c = spec_commit(spec)
    return W.preflight(d, "steps/s.json", c)


ALLOW = "is not the allowlisted one"
try:
    pre(GOOD)
    GOOD_STOP = None
except (SystemExit, W.Stop) as e_:
    GOOD_STOP = str(e_)[:200]
cases["preflight: a good dryrun_plan step passes (verify_only, main, dryrun, TEST INPUTS_FULL, pinned strings and plan keys)"] = GOOD_STOP is None
cases["preflight: dryrun_plan without dryrun true STOPs"] = stops(pre, GOOD | {"entry": {"attr": "main", "flags": {"inputs_full": GOOD["entry"]["flags"]["inputs_full"]}}})
cases["preflight: dryrun_plan with spec_only STOPs"] = stops(pre, GOOD | {"entry": {"attr": "main", "flags": GOOD["entry"]["flags"] | {"spec_only": "/tmp/x.pkl"}}})
cases["preflight: dryrun_plan not verify_only STOPs"] = stops(pre, GOOD | {"verify_only": False, "outputs": [{"dir": "results", "regex": "x"}], "n_outputs": 1})
cases["preflight: dryrun_plan on a non-TEST INPUTS_FULL STOPs"] = stops(pre, GOOD | {"entry": {"attr": "main", "flags": {"dryrun": True, "inputs_full": "config/INPUTS_FULL_2026.json@" + "c" * 64}}})
cases["preflight: dryrun_plan on another allowlisted script (inputs_full.py) STOPs"] = stops(pre, GOOD | {"script": "experiments/2026-10-15_fpr_sweep_507_prep/inputs_full.py"})
SCRIPTS0 = dict(W.SCRIPTS)
try:  # R-S2: both negatives reach the allowlist check itself (the blob matches; verify_only and the reducer's own rules are satisfied)
    W.SCRIPTS[OTHER] = (None, "inputs_full")
    m7_ = stop_msg(pre, GOOD | {"reducer": "inputs_full", "expect_strings": None, "expect_plan_keys": None})
    m8_ = stop_msg(pre, GOOD | {"script": OTHER})
    m8ok_ = stop_msg(pre, GOOD | {"script": OTHER, "reducer": "inputs_full", "expect_strings": None, "expect_plan_keys": None, "entry": None})
finally:
    W.SCRIPTS.clear()
    W.SCRIPTS.update(SCRIPTS0)
cases["R-S2 (M7): run_peraction311 with a FOREIGN allowlisted verify reducer (inputs_full) STOPs at the allowlist check"] = ALLOW in m7_
cases["R-S2 (M8): ANOTHER allowlisted script with dryrun_plan STOPs at the allowlist check (its own reducer passes: control)"] = ALLOW in m8_ and m8ok_ == ""
cases["preflight: an unknown reducer on the runner STOPs; json_line_keys stays allowed (its own rules)"] = (
    stops(pre, GOOD | {"reducer": "fix_v2"}) and "dryrun_plan" in W.ALT_REDUCERS[RUNNER] and W.SCRIPTS[RUNNER][1] == "json_line_keys")
cases["O-S1 / R-S1: a dryrun_plan step without expect_strings, or without expect_plan_keys, STOPs"] = (
    stops(pre, {k: v for k, v in GOOD.items() if k != "expect_strings"}) and stops(pre, {k: v for k, v in GOOD.items() if k != "expect_plan_keys"}))
cases["O-S1 / R-S1: unsorted, duplicate or empty pins STOP; a non-identifier plan key STOPs"] = (
    stops(pre, GOOD | {"expect_strings": ["b R", "a C"]}) and stops(pre, GOOD | {"expect_strings": ["a C", "a C"]}) and stops(pre, GOOD | {"expect_strings": []})
    and stops(pre, GOOD | {"expect_plan_keys": sorted(PKEYS + ["bad key"])}))
cases["O-S1 / R-S1: another step type carrying expect_strings or expect_plan_keys STOPs"] = stops(
    pre, GOOD | {"reducer": "json_line_keys", "verify_only": False, "outputs": [{"dir": "results", "regex": "x"}], "n_outputs": 1, "expect_keys": ["a"]})
cases["O2-1: a pin with an embedded or glued number STOPs in preflight (no membership oracle); only PIN_ALLOW's root may be pinned"] = (
    all("PIN_ALLOW" in stop_msg(pre, GOOD | {"expect_strings": sorted(PINNED + [s])}) for s in ("fpr0.01", "auroc 0.97", "0.5", "rh1e5_stage_v1__honest_b"))
    and W.PIN_ALLOW == ("rh1e4_stage_v1__honest_b",) and "rh1e4_stage_v1__honest_b" in PINNED and GOOD_STOP is None)

# run_step on a dryrun_plan step (closes N11 / N12): the fabricated runner at the runner's path prints the head line and the plan; the
# container body runs it from a real bundle, dispatches the dryrun_plan reducer with the payload's pins, and whitelists the result
PLAN_RUN = PLAN | {"full_length": {k: v for k, v in PLAN["full_length"].items()}}
GOOD_RUN = GOOD | {"entry": {"attr": "main", "flags": GOOD["entry"]["flags"] | {"plan_json": json.dumps(PLAN_RUN)}}}


def run_step_of(spec):
    c = spec_commit(spec, "run.json")
    payload, _ = W.preflight(d, "steps/run.json", c)
    tmp = Path(tempfile.mkdtemp(prefix="dryrun_rs_"))
    broot = tmp / "bundlevol"
    (broot / "bundles").mkdir(parents=True)
    (tmp / "bt").mkdir()
    f, bsha = W.make_bundle(d, payload["commits"], tmp / "bt")
    shutil.copy(f, broot / "bundles" / f"{bsha}.bundle")
    (tmp / "stagevol").mkdir()
    (tmp / "outvol").mkdir()
    p = dict(payload, bundle=f"bundles/{bsha}.bundle", bundle_sha256=bsha, run_id=uuid.uuid4().hex)
    return W.run_step(p, Path(tempfile.mkdtemp(prefix="dryrun_rs_work_", dir=tmp)), broot, tmp / "stagevol", sys.executable, shutil.which("git"),
                      tmp / "outvol", lambda: None)


rs_ = run_step_of(GOOD_RUN)
rsd_ = json.dumps(rs_)
cases["run_step (N11 / N12): a dryrun_plan step runs OK, the reducer gets the payload's pins, the plan comes back blinded"] = (
    rs_.get("status") == "OK" and rs_["reduced"]["pass"] is True and rs_["reduced"]["plan"]["labels"] == ["a C", "b R"]
    and rs_["reduced"]["plan"]["full_length"]["roots"] == ["rh1e4_stage_v1__honest_b"] and rs_["outputs"] == []
    and not any(any_form(s, rsd_) for s in REFUSED_TEXT))
rs2_ = run_step_of(GOOD_RUN | {"entry": {"attr": "main", "flags": GOOD["entry"]["flags"] | {"plan_json": json.dumps(
    PLAN_RUN | {"full_length": PLAN_RUN["full_length"] | {"inputs_full": PLAN_RUN["full_length"]["inputs_full"] | {"test": False}}})}}})
cases["run_step: a plan whose inputs_full.test is false is not confirmed (STOP; no plan relayed)"] = rs2_.get("status") == "STOP" and (rs2_.get("reduced") or {}).get("plan") is None

# the sidecar writer (O2-4: build takes the wrapper; main() loads it from `git show C:<wrapper>`)
SPEC_B = json.dumps(GOOD).encode()
RED = red_of()
C, APP = "f" * 40, "ap-ABCDEFGHIJKLMN"


def line_of(spec_b, path="steps/s.json", commit=C):
    return f"spec {path} sha256 {hashlib.sha256(spec_b).hexdigest()} commit {commit}"


def log_of(red=RED, spec_line=line_of(SPEC_B), extra=b""):
    return ((spec_line + "\n" if spec_line else "") + "bundle sha256 " + "d" * 64 + " commits 9 bytes 123 run_id " + "e" * 32 + "\nreduced "
            + json.dumps(red, sort_keys=True) + " git_version \"git version 2.39.5\"\nRECSTEP_OK\n").encode() + extra


LOG = log_of()
sd = S.build(LOG, SPEC_B, "steps/s.json", C, APP, W)
cases["sidecar: 110447's keys + refused_paths + run_id; plan and paths carried; the log's sha"] = (
    set(sd) == {"what", "app", "client_log_sha256", "command", "git", "inputs_full", "plan", "ran_pt", "refused_paths", "run_id"}
    and sd["plan"] == RED["plan"] and sd["refused_paths"] == RED["refused_paths"] and sd["run_id"] == "e" * 32 and sd["git"] == C
    and sd["inputs_full"] == GOOD["entry"]["flags"]["inputs_full"] and len(sd["client_log_sha256"]) == 64 and type(sd["command"]) is str)
cases["(a) blinding: no refused string appears in the sidecar in ANY form (raw, sha256, sha1, md5)"] = not any(any_form(s, json.dumps(sd)) for s in REFUSED_TEXT)
cases["sidecar: two RECSTEP_OK lines STOP"] = "exactly one RECSTEP_OK" in stop_msg(S.build, LOG + b"RECSTEP_OK\n", SPEC_B, "steps/s.json", C, APP, W)
cases["sidecar: a reduced view that did not pass STOPs"] = "did not pass" in stop_msg(
    S.build, LOG.replace(b'"pass": true', b'"pass": false'), SPEC_B, "steps/s.json", C, APP, W)
NOTDR = json.dumps(GOOD | {"reducer": "json_line_keys"}).encode()
NOTTEST = json.dumps(GOOD | {"entry": {"attr": "main", "flags": {"dryrun": True, "inputs_full": "config/INPUTS_FULL_x.json@y"}}}).encode()
cases["sidecar (S1, spec line regenerated): a spec that is not a dryrun_plan step STOPs on the spec check, not the binding"] = (
    "is not a dryrun_plan step" in stop_msg(S.build, log_of(spec_line=line_of(NOTDR)), NOTDR, "steps/s.json", C, APP, W))
cases["sidecar (spec line regenerated): a non-TEST INPUTS_FULL in the spec STOPs on the spec check"] = (
    "is not a dryrun_plan step" in stop_msg(S.build, log_of(spec_line=line_of(NOTTEST)), NOTTEST, "steps/s.json", C, APP, W))
cases["sidecar: a bad app id or a bad commit STOPs (spec line regenerated for the commit)"] = (
    "ap- id" in stop_msg(S.build, LOG, SPEC_B, "steps/s.json", C, "app-1", W)
    and "40 hex" in stop_msg(S.build, log_of(spec_line=line_of(SPEC_B, commit="f" * 39)), SPEC_B, "steps/s.json", "f" * 39, APP, W))
cases["sidecar (S6): no bundle line STOPs on the line count"] = "exactly one RECSTEP_OK, bundle" in stop_msg(
    S.build, b"\n".join(x for x in LOG.split(b"\n") if not x.startswith(b"bundle")), SPEC_B, "steps/s.json", C, APP, W)
cases["O-S3: no spec line, two spec lines, or a spec line naming another commit / sha / path STOPs"] = (
    stops(S.build, log_of(spec_line=None), SPEC_B, "steps/s.json", C, APP, W)
    and stops(S.build, log_of(extra=(line_of(SPEC_B) + "\n").encode()), SPEC_B, "steps/s.json", C, APP, W)
    and "O-S3" in stop_msg(S.build, log_of(spec_line=line_of(SPEC_B, commit="9" * 40)), SPEC_B, "steps/s.json", C, APP, W)
    and "O-S3" in stop_msg(S.build, log_of(spec_line=f"spec steps/s.json sha256 {'0' * 64} commit {C}"), SPEC_B, "steps/s.json", C, APP, W)
    and "O-S3" in stop_msg(S.build, log_of(spec_line=line_of(SPEC_B, path="steps/t.json")), SPEC_B, "steps/s.json", C, APP, W))
cases["O-S1 (sidecar): a passed plan whose keys differ from the spec's expect_plan_keys STOPs"] = "O-S1" in stop_msg(
    S.build, log_of(red=RED | {"plan": RED["plan"] | {"extra": 1}}), SPEC_B, "steps/s.json", C, APP, W)
cases["O-S2 (sidecar): a passed plan whose inputs_full.test is not true STOPs"] = "O-S2" in stop_msg(
    S.build, log_of(red=RED | {"plan": RED["plan"] | {"full_length": RED["plan"]["full_length"] | {"inputs_full": RED["plan"]["full_length"]["inputs_full"] | {"test": False}}}}),
    SPEC_B, "steps/s.json", C, APP, W)
cases["O-S3 / R-S1b (sidecar): an unrefused embedded or glued string, key or path segment in the reduced plan STOPs before writing"] = all(
    "string rule" in stop_msg(S.build, log_of(red=x), SPEC_B, "steps/s.json", C, APP, W) for x in (
        RED | {"plan": RED["plan"] | {"label": "fpr0.01"}}, RED | {"plan": RED["plan"] | {"full_length": RED["plan"]["full_length"] | {"lr1e-5": 1}}},
        RED | {"refused_paths": RED["refused_paths"] + [["plan", "x", "0.97auc"]]}))
_run0, git_calls = subprocess.run, []
subprocess.run = lambda *a, **k: git_calls.append(a) or (_ for _ in ()).throw(AssertionError("a git call before the commit check"))
argv0 = sys.argv
try:
    sys.argv = ["make_dryrun_sidecar.py", "--log", "x.log", "--spec", "steps/s.json", "--commit=--output=/tmp/x", "--app", APP, "--out", "o.json"]
    m33_ = stop_msg(S.main)
    sys.argv = ["make_dryrun_sidecar.py", "--log", "x.log", "--spec", "steps/s.json", "--commit", "main", "--app", APP, "--out", "o.json"]
    m33b_ = stop_msg(S.main)
    sys.argv = ["make_dryrun_sidecar.py", "--log", "x.log", "--spec", "steps/s.json", "--commit", C, "--app", "app-1", "--out", "o.json"]
    m33c_ = stop_msg(S.main)
finally:
    subprocess.run, sys.argv = _run0, argv0
cases["O3-3: a bad --commit (an option-like string, a branch name) or app id STOPs before ANY git call or wrapper load"] = (
    all("before any git call" in m for m in (m33_, m33b_, m33c_)) and git_calls == [])
WSRC = (HERE / "run_records_step_in_container.py").read_bytes()
Wc = S.load_wrapper(WSRC, Path(tempfile.mkdtemp(prefix="wrapper_at_c_")))
cases["O2-4: the sidecar's wrapper loader builds the rule from the given bytes (main() passes `git show C:<wrapper>`)"] = (
    Wc.PIN_ALLOW == W.PIN_ALLOW and not stops(S.build, LOG, SPEC_B, "steps/s.json", C, APP, Wc)
    and S.WRAPPER == "experiments/2026-10-17_records_step_container/run_records_step_in_container.py"
    and 'show", f"{a.commit}:{WRAPPER}"' in (HERE / "make_dryrun_sidecar.py").read_text())

for name, ok in cases.items():
    print(("PASS " if ok else "FAIL ") + name)
print(f"{sum(cases.values())} of {len(cases)} checks pass")
sys.exit(0 if all(cases.values()) else 1)
