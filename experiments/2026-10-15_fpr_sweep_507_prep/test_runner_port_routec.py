"""Route C runner port (09, 10-17): lean checks on run_peraction311.py after the kappa line (704122edf..540ae554d) is ported.

1. POSITIVE CONTROL for 5d6320546's STOP: row()'s S_paper493 structural-zero block, run on synthetic units, STOPs when a LISTED
   structural-zero key has attack rows, passes when it has none, and (mutant) stops nothing once the STOP lines are removed.
2. 5d6320546's hunk is in the file byte-equal, right after the line that builds SZK.
3. The kappa D1 reference pin is unset until route C's A lands, so a pooled --kappa fails closed at its reference check.
4. The one merge conflict's resolution: --spec-only and the kappa --dryrun both skip the input upload.
5. 72ee86f30's resume guard: the port keeps route C's stricter form, whose condition is the kappa tip's verbatim.
6. RQ-P1 (a63478): main pins the resume spec before the guard; main never parses or prints the reference bytes; route C's
   generalized mode lines are pinned by exact text.
7. --spec-only stdout (2e 10-17): main's read_started line is printed only without --spec-only, so a spec-only build's stdout holds exactly
   ONE JSON object, dispatch's, as the records wrapper's json_line_keys reducer requires (real runner code on stubs; a mutant fails).
Synthetic data only; the runner is read as text, never imported.
  uv run --offline --no-project --python 3.12 --with numpy==2.3.3 python experiments/2026-10-15_fpr_sweep_507_prep/test_runner_port_routec.py
"""
import ast
import contextlib
import hashlib
import io
import json
import re
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

SRC = (Path(__file__).resolve().parent / "run_peraction311.py").read_text()
STOP_LINES = (
    '        haverows_ = sorted(SZK & set(U["run"][isN].tolist()))\n'
    "        if haverows_:  # a structural zero has no rows by definition; rows here would be overwritten by the never-alarming sentinel\n"
    '            raise SystemExit(f"{label} {vtag}: S_paper493 structural-zero keys with attack rows: {haverows_[:5]}")\n')
SZK_LINE = '        SZK = sz_ & {f"{w}|{c}" for w in A3 for c in cohN}\n'
BLOCK_START = "    SZK = set()  # structural zeros (no-filter list; S_paper493's block)"
BLOCK_END = '        out.setdefault("s_paper493", {})["structural_zero_runs"] = sorted(SZK)\n'


def spo_block(src: str):
    """row()'s S_paper493 structural-zero block (from SZK = set() through the structural_zero_runs line) as a function of its inputs."""
    a = src.index(BLOCK_START)
    b = src.index(BLOCK_END, a) + len(BLOCK_END)
    body = textwrap.indent(textwrap.dedent(src[a:b]), "    ")
    code = ("def f(SPO, cfg, p, cohN, cell, isN, world, A3, U, fold_model, label, vtag, out):\n" + body + "    return SZK\n")
    ns = {"np": np}
    exec(compile(code, "<spo_block>", "exec"), ns)
    return ns["f"]


def units(keys):
    """Synthetic S_paper493 attack units: one finite row per key '<world>|<cell>' (fold 0)."""
    run = np.array(keys)
    return {"run": run, "score": np.zeros(len(keys)), "fold": np.zeros(len(keys), np.int64), "sbm": np.zeros((len(keys), 5)),
            "source": np.array(["S_paper493"] * len(keys))}


def call(f, keys, listed):
    U = units(keys)
    world = np.array([k.split("|", 1)[0] for k in keys])
    cell = np.array([k.split("|", 1)[1] for k in keys])
    cfg = {"sp493": {"structural_zero_keys": {"C": sorted(listed)}}}
    out = {}
    try:
        return "ok", f(True, cfg, {"readout": "C"}, ["c1", "c2"], cell, U["source"] == "S_paper493", world, ("run1", "run2", "sim"), U, True,
                       "LBL", "S_paper493", out), out
    except SystemExit as e:
        return "STOP", str(e), out


res = {}
f = spo_block(SRC)
base = ["run1|c1", "run2|c1", "run2|c2", "sim|c1", "sim|c2"]  # run1|c2 has no rows
pc = call(f, base + ["run1|c2"], {"run1|c2"})  # listed structural zero WITH attack rows
res["positive control: a listed structural-zero key with attack rows STOPs"] = (
    pc[0] == "STOP" and pc[1] == "LBL S_paper493: S_paper493 structural-zero keys with attack rows: ['run1|c2']")
ok = call(f, base, {"run1|c2"})
res["a listed structural-zero key without rows passes; SZK and the record field name it"] = (
    ok[0] == "ok" and ok[1] == {"run1|c2"} and ok[2] == {"s_paper493": {"structural_zero_runs": ["run1|c2"]}})
wh = call(f, base, set())  # unlisted and rowless: WITHHELD (unchanged behaviour)
res["an unlisted cell without rows is WITHHELD, as before"] = wh[0] == "ok" and isinstance(wh[1], dict) and "WITHHELD" in wh[1].get("withheld", "")
mut = spo_block(SRC.replace(STOP_LINES, "", 1))
mm = call(mut, base + ["run1|c2"], {"run1|c2"})
res["mutant (the STOP lines removed): the positive control no longer STOPs"] = SRC.count(STOP_LINES) == 1 and mm[0] == "ok"
res["5d6320546's hunk is byte-equal, right after the SZK line"] = SRC.count(SZK_LINE + STOP_LINES) == 1
res["the kappa D1 reference is unset (None) and the reference check is present (fail closed)"] = (
    "\nKAPPA_D1_REF_COMMIT = None  # 09 (10-17, route C port)" in SRC
    and 'if calib_pooled_world and kref_.split(":", 1)[0] != KAPPA_D1_REF_COMMIT:' in SRC and "x".split(":", 1)[0] != None)  # noqa: E711
res["the conflict resolution: --spec-only and the kappa --dryrun skip the upload"] = SRC.count("    if spec_only:  # --spec-only: the same content-addressed refs; nothing is listed or sent\n        todo_ = {}\n    elif dryrun:  # the kappa dry run") == 1

# 72ee86f30 (a merge; its resume guard is the route line's 704122edf wording): the port keeps route C's stricter form. Its condition is
# the kappa tip's, verbatim (KAPPA_GUARD, ast.unparse of 540ae554d's), after FO.resume_spec_pin also requires <path>@<sha256>.
KAPPA_GUARD = "not (resume_spec.startswith('xfer/fpr507prep/drivers/') and resume_spec.endswith('.pkl') and re.fullmatch('r[0-9]+', resume_tag))"
main_ = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "main")
gi_ = [n for n in ast.walk(main_) if isinstance(n, ast.If) and any(isinstance(b, ast.Raise) and "needs --resume-tag" in ast.unparse(b) for b in n.body)]
pin_src = next(ast.unparse(n) for n in ast.parse((Path(__file__).resolve().parent / "full_override.py").read_text()).body
               if isinstance(n, ast.FunctionDef) and n.name == "resume_spec_pin")
ns_ = {}
exec(pin_src, ns_)


def guard_stops(spec_, tag_):  # the port's two steps in order: the pin parse, then the guard
    try:
        p_, _ = ns_["resume_spec_pin"](spec_)
    except SystemExit:
        return True
    return eval(compile(ast.Expression(gi_[0].test), "<guard>", "eval"), {"re": re, "resume_spec": p_, "resume_tag": tag_})


pkl_ = "xfer/fpr507prep/drivers/x.pkl@" + "a" * 64
res["72ee86f30's resume guard: the port's condition is the kappa tip's verbatim; a resume spec without a resume tag STOPs (and without a sha)"] = (
    len(gi_) == 1 and ast.unparse(gi_[0].test) == KAPPA_GUARD and guard_stops(pkl_, "") and guard_stops(pkl_, "rX")
    and guard_stops("xfer/fpr507prep/drivers/x.pkl", "r1") and guard_stops("", "r1") and not guard_stops(pkl_, "r1"))

# RQ-P1 (a63478's second review of the port): checks that kill its mutants PM5 (the pin call dropped), PM4 (the kappa reference parsed
# and printed in main) and PM2/PM2b (route C's generalized mode lines reverted). Each syntax-tree check also runs on an in-file mutant.
PIN = "resume_spec, resume_sha_ = FO.resume_spec_pin(resume_spec)"
REFS = {"kb_", "pb_"}  # the kappa and pertask reference records' bytes (git_record): hashed with C.sha_b, then carried in meta only


def main_of(src):
    return next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main")


def pin_before_guard(src):  # main's one resume branch calls FO.resume_spec_pin before the resume guard's If
    br_ = [n for n in ast.walk(main_of(src)) if isinstance(n, ast.If) and ast.unparse(n.test) == "resume_spec or resume_tag"]
    if len(br_) != 1:
        return False
    ip_ = [i for i, b in enumerate(br_[0].body) if isinstance(b, ast.Assign) and ast.unparse(b) == PIN]
    ig_ = [i for i, b in enumerate(br_[0].body) if isinstance(b, ast.If)
           and any(isinstance(x, ast.Raise) and "needs --resume-tag" in ast.unparse(x) for x in b.body)]
    return len(ip_) == 1 and len(ig_) == 1 and ip_[0] < ig_[0]


def ref_bytes_inert(src):  # in main, kb_/pb_ are read only as C.sha_b's argument or a dict value; never inside json.loads, pickle.loads or print
    m_ = main_of(src)
    up_ = {c: p for p in ast.walk(m_) for c in ast.iter_child_nodes(p)}
    rd_ = [n for n in ast.walk(m_) if isinstance(n, ast.Name) and n.id in REFS and isinstance(n.ctx, ast.Load)]
    bad_ = [c for c in ast.walk(m_) if isinstance(c, ast.Call) and ast.unparse(c.func) in ("json.loads", "pickle.loads", "print")
            and any(isinstance(x, ast.Name) and x.id in REFS for x in ast.walk(c))]
    return {n.id for n in rd_} == REFS and not bad_ and all(
        (isinstance(up_[n], ast.Call) and ast.unparse(up_[n].func) == "C.sha_b") or (isinstance(up_[n], ast.Dict) and n in up_[n].values) for n in rd_)


res["O-P1: main calls FO.resume_spec_pin(resume_spec) before the resume guard; the mutant without the call fails"] = (
    pin_before_guard(SRC) and SRC.count(PIN) == 1 and not pin_before_guard(SRC.replace(PIN, 'resume_sha_ = ""', 1)))
res["main never parses or prints the kappa/pertask reference bytes (kb_, pb_); the mutant that parses kb_ fails"] = (
    ref_bytes_inert(SRC) and SRC.count("C.sha_b(kb_)") == 1 and not ref_bytes_inert(SRC.replace("C.sha_b(kb_)", "C.sha_b(json.loads(kb_))", 1)))
ROUTEC_PINS = {  # route C's generalized lines, each anchored at a line start and present exactly once (code text; comments free)
    "drive's read-back is the all-tasks record's only": '\n    rb_ = pw_read_back(dest, spec) if pwm_ and spec["mode"] == "alltasks" else None\n',
    "the pooled read-back is the all-tasks record's only": '\n            and spec["mode"] == "alltasks" else {})  #',
    "the dry-run gate covers kappa, pertask and didscore": "\n        if not (kappa or pertask or didscore):\n            return\n",
    "the resumable modes": '\n    resumable = spec["mode"] in ("alltasks", "kappa", "pertask", "didscore")  #',
    "the calib_ref branch": '\n        if "calib_ref" in pw_dir:  #',
    "the manifest-pins branch (per-task modes read the pooled reference)": '\n            if spec["mode"] in ("kappa", "pertask"):  #',
    "the manifest-pins branch (route C's pool_inputs check)": '\n            elif MF_.get("pool_inputs") != pw_pool_inputs(spec["meta"]):  #',
    "merge_meta on the all-tasks record": '\n        if MERGE:\n            meta["merge"] = merge_meta(MERGE)\n        if IN and IN.get("schema") == IF2.SCHEMA:',
    "merge_meta on the PERTASK record": '\n                meta["merge"] = merge_meta(MERGE)\n            dispatch("pertask", pj_, meta, IN, cfg, ',
}
for k, v in ROUTEC_PINS.items():
    res[f"route C pin: {k}"] = SRC.count(v) == 1


# --spec-only stdout (2e 10-17; the census and package 3's spec-only steps): the records wrapper's json_line_keys reducer needs exactly ONE
# stdout JSON object with the pinned keys. Real runner code on stubs: main's read_started statement, then dispatch()'s spec-only branch.
def reduce_json_line_keys(stdout: str, expect: list) -> dict:  # verbatim from the records wrapper (blob ed31995, L175-187)
    """Route C runner (spec-only): exactly ONE stdout line that is a JSON object; its keys are compared with the step spec's pinned
    expect_keys (O19). Only a boolean and counts return: no key text, no value."""
    objs = []
    for x in stdout.splitlines():
        try:
            o = json.loads(x)
        except ValueError:
            continue
        if isinstance(o, dict):
            objs.append(o)
    one = objs[0] if len(objs) == 1 else None
    return {"json_lines": len(objs), "keys_match": one is not None and sorted(one) == expect, "n_keys": len(one) if one is not None else None}


SPEC_KEYS = ["record_would_be", "spec_mb", "spec_only", "spec_path_would_be", "spec_sha256"]  # dispatch()'s spec-only line
T_ = ast.parse(SRC)
FN_ = {n.name: n for n in T_.body if isinstance(n, ast.FunctionDef)}
RSP_ = [s for s in FN_["main"].body if "{'read_started_pt': stamp, 'rows': labels}" in ast.unparse(s)]


def rp_stdout(spec_only, guard=True):
    """stdout of main's read_started statement (guard=False: its print alone, the old unguarded line), then, on --spec-only, dispatch()."""
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(out):
        ns = {"json": json, "stamp": "2026-10-17_120000", "labels": ["Sol (a) C"], "spec_only": spec_only, "Path": Path, "HERE": Path(td),
              "SPEC_ONLY": {"path": str(Path(td) / "s.pkl") if spec_only else ""}, "C": type("C", (), {"sha_b": staticmethod(lambda b: hashlib.sha256(b).hexdigest())})}
        exec(compile(ast.Module(body=RSP_ if guard else RSP_[0].body, type_ignores=[]), "<read_started>", "exec"), ns)
        if spec_only:
            exec(compile(ast.Module(body=[FN_["dispatch"]], type_ignores=[]), "<dispatch>", "exec"), ns)
            ns["dispatch"]("keys", [("Sol (a) C", {"keys_only": True}, None)], {}, {}, {}, "census", "results/nofilter_keys_FULL_TEST_2026-10-17_120000.json",
                           "2026-10-17_120000", True)
    return out.getvalue()


so_ = reduce_json_line_keys(rp_stdout("s.pkl"), SPEC_KEYS)
res["--spec-only: main's read_started print is guarded by `not spec_only`; stdout holds exactly ONE JSON object, dispatch's keys"] = (
    len(RSP_) == 1 and isinstance(RSP_[0], ast.If) and ast.unparse(RSP_[0].test) == "not spec_only" and so_ == {"json_lines": 1, "keys_match": True, "n_keys": 5})
res["without --spec-only the read_started line is still printed"] = [sorted(json.loads(x)) for x in rp_stdout("").splitlines()] == [["read_started_pt", "rows"]]
res["mutant (the old unguarded print): --spec-only stdout holds TWO JSON objects and the reducer fails"] = reduce_json_line_keys(
    rp_stdout("s.pkl", guard=False), SPEC_KEYS)["json_lines"] == 2
for k, v in res.items():
    print(("PASS " if v else "FAIL ") + k)
print(f"PASS {all(res.values())} ({sum(res.values())}/{len(res)})")
sys.exit(0 if all(res.values()) else 1)
