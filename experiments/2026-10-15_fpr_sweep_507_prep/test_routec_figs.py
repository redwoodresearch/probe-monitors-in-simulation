"""Route C figure renders (10-18): each figure's label from the drawn record's OWN sidecar (postfig.routec_label, 09 via 2e), the record's
meta re-assert (postfig.check_record_meta, 09 S2 item 10) and Figure 4's gate (fig4_gate_count.py). Synthetic bytes, temp folders and stubs
only; no record, no Modal call (the gate's remote function is stubbed), nothing written under this folder.
  (1) routec_label: sha before parse, exactly {record, record_sha256, label}, record_sha256 = the record drawn, the label's sha256 = 09's
      registered one (pinned by the plan), PRELIMINARY; another record's sidecar, another label (A's on B), a sha mismatch STOP;
  (2) check_record_meta: git, script_sha256, FROZEN inputs_full (sha, status, test), each miss STOPs; two MUTANTS (the script_sha256 check
      dropped; the record_sha256 tie dropped) are each caught by a check here;
  (3) post_fig4_sweep, run as render_all runs it (the scripts copied into a fresh folder with results/): the three route C flags go together
      and exclude --text-note; a wrong sidecar or label STOPs; with a valid sidecar, the record's meta is checked before the gate (a wrong
      meta STOPs there; the right one reaches the gate, which STOPs on the synthetic record's missing structure);
  (4) fig4_gate_count (the PLANS map edit): the committed route C plan STOPs on its fill tokens; a filled one without any of the three flags,
      with a malformed --record-meta, with --text-note or a non-git sidecar STOPs; a filled one reaches the (stubbed) remote call with record
      type pooled_world_B, intervals fixed+joint and the record meta; app names <= 63 characters in the route C form.

  uv run --no-project --python 3.12 --with numpy==2.3.3 --with modal==1.6.1 --with matplotlib==3.10.6 python test_routec_figs.py   (from this folder)
"""

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig4_gate_count as FG  # noqa: E402  (modal objects only; nothing runs)
import postfig as PF  # noqa: E402
import render_all as RA  # noqa: E402

bad = []
B = "6" * 64  # the drawn record's JSON sha256 (synthetic)
LAB = "route C pooled B (synthetic test label; not 09's text); PRELIMINARY until S2"
LSHA = hashlib.sha256(LAB.encode()).hexdigest()
GIT, RPS, IFS = "a" * 40, "d" * 64, "f" * 64
META = {"git": GIT, "script_sha256": RPS, "full_length": {"inputs_full": {"path": "p", "sha256": IFS, "status": "FROZEN 2026-10-17", "test": False}}}


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        bad.append(name)


def stops(fn, *a):
    try:
        fn(*a)
    except SystemExit:
        return True
    return False


def sidecar(d, label=LAB, rec=B, extra=None):
    b = json.dumps({"record": "xfer/fpr507prep/records/results/b.json", "record_sha256": rec, "label": label} | (extra or {})).encode()
    (d / "sc.json").write_bytes(b)
    return ("sc.json", hashlib.sha256(b).hexdigest(), "path@commit")


def mutant(old, new):
    """postfig with one check removed (a mutant), loaded as its own module."""
    src = (HERE / "postfig.py").read_text()
    assert src.count(old) == 1, old
    m = types.ModuleType("postfig_mutant")
    exec(compile(src.replace(old, new), "postfig_mutant", "exec"), m.__dict__)  # noqa: S102 -- this folder's own postfig, one line changed
    return m


tmp = Path(tempfile.mkdtemp())
# (1) routec_label, with RES pointed at a temp folder
PF.RES = tmp
good = sidecar(tmp)
check("routec_label: the registered label composes", PF.routec_label(good, B, "Figure 4 on B", LSHA) == f"{LAB} | Figure 4 on B")
check("routec_label: sidecar sha mismatch STOPs", stops(PF.routec_label, (good[0], "e" * 64, good[2]), B, "w", LSHA))
check("routec_label: another record's sidecar STOPs", stops(PF.routec_label, good, "e" * 64, "w", LSHA))
check("routec_label: another registered label (e.g. A's on a B record) STOPs", stops(PF.routec_label, good, B, "w", "c" * 64))
check("routec_label: a malformed label sha STOPs", stops(PF.routec_label, good, B, "w", "TBD"))
nolab = LAB.replace("PRELIMINARY", "final")
check("routec_label: no PRELIMINARY STOPs (even with its own sha)", stops(PF.routec_label, sidecar(tmp, nolab), B, "w", hashlib.sha256(nolab.encode()).hexdigest()))
check("routec_label: an extra key STOPs", stops(PF.routec_label, sidecar(tmp, extra={"x": 1}), B, "w", LSHA))

# (2) check_record_meta, and two mutants
check("check_record_meta: the pinned meta passes", not stops(PF.check_record_meta, META, GIT, RPS, IFS))
fin = META["full_length"]["inputs_full"]
for nm, m_ in (("git", META | {"git": "b" * 40}), ("script_sha256", META | {"script_sha256": "e" * 64}),
               ("inputs_full sha", META | {"full_length": {"inputs_full": fin | {"sha256": "e" * 64}}}),
               ("status TEST", META | {"full_length": {"inputs_full": fin | {"status": "TEST x"}}}),
               ("test true", META | {"full_length": {"inputs_full": fin | {"test": True}}}), ("no meta", None)):
    check(f"check_record_meta: {nm} STOPs", stops(PF.check_record_meta, m_, GIT, RPS, IFS))
m1 = mutant('("script_sha256", (meta or {}).get("script_sha256") == script_sha256)', '("script_sha256", True)')
check("MUTANT 1 (script_sha256 check dropped) is caught: it lets a wrong runner sha through", stops(PF.check_record_meta, META | {"script_sha256": "e" * 64}, GIT, RPS, IFS)
      and not stops(m1.check_record_meta, META | {"script_sha256": "e" * 64}, GIT, RPS, IFS))
m2 = mutant(' or sc["record_sha256"] != record_sha:', ' or False:')
m2.RES = tmp
good = sidecar(tmp)  # rewritten: the checks above overwrote sc.json
check("MUTANT 2 (record_sha256 tie dropped) is caught: it accepts another record's sidecar", stops(PF.routec_label, good, "e" * 64, "w", LSHA)
      and not stops(m2.routec_label, good, "e" * 64, "w", LSHA))

# (3) post_fig4_sweep's route C steps, run as render_all runs it (fresh folder, records under results/)
exp = tmp / "exp"
(exp / "results").mkdir(parents=True)
for n in ("post_fig4_sweep.py", "postfig.py", "figio.py"):
    shutil.copy(HERE / n, exp / n)


def record(meta):
    b = json.dumps({"meta": meta}).encode()
    (exp / "results" / "r.json").write_bytes(b)
    return hashlib.sha256(b).hexdigest()


rsha = record(META)
sc = sidecar(exp / "results", rec=rsha)


def run(extra, env=None):
    cmd = [sys.executable, "post_fig4_sweep.py", "--stamp", "_t", "--record", "r.json", rsha, "src", "--record-type", "pooled_world_B",
           "--rows", "a C", "b C", "c C", "d C", "e C", "f C"] + extra
    r = subprocess.run(cmd, cwd=exp, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "MPLBACKEND": "Agg"} | (env or {}))
    return r.returncode, (r.stderr or "") + (r.stdout or "")


ok3 = ["--label-sidecar", *sc, "--label-sha", LSHA, "--record-meta", GIT, RPS, IFS]
EV = {"FIG_COMPARATOR": "run_mean", "FIG_ESTIMATOR": "A"}
rc, o = run(["--label-sidecar", *sc, "--record-meta", GIT, RPS, IFS])
check("post_fig4_sweep: without --label-sha STOPs", rc != 0 and "go together" in o)
rc, o = run(["--label-sidecar", *sc, "--label-sha", LSHA])
check("post_fig4_sweep: without --record-meta STOPs", rc != 0 and "go together" in o)
rc, o = run(ok3 + ["--text-note", "pooled; preliminary"])
check("post_fig4_sweep: with --text-note STOPs", rc != 0 and "go together" in o)
rc, o = run(["--label-sidecar", *sc, "--label-sha", "c" * 64, "--record-meta", GIT, RPS, IFS])
check("post_fig4_sweep: another registered label STOPs", rc != 0 and "registered string" in o)
rc, o = run(["--label-sidecar", *sc, "--label-sha", LSHA, "--record-meta", GIT, "e" * 64, IFS], EV)
check("post_fig4_sweep: a wrong record meta STOPs before the gate", rc != 0 and "record's meta differs" in o and "script_sha256" in o)
rc, o = run(ok3, EV)
check("post_fig4_sweep: the right sidecar and meta reach the gate (which STOPs on the synthetic record)", rc != 0 and "gate failures" in o
      and "record's meta differs" not in o)

# (4) fig4_gate_count: the route C plan and the PLANS map edit
committed = (HERE / FG.PLANS["pooled_world_B_routec"]).read_bytes()
check("gate: the committed route C plan STOPs on its fill tokens", stops(FG.gate_main.info.raw_f, "_unused", "pooled_world_B_routec"))
TOK4 = {"B_RECORD_BASENAME": "b.json", "B_RECORD_JSON_SHA256": "b" * 64, "B_RECORD_COMMITTED_PATH": "x/b.json.gz", "B_RECORD_COMMIT": "e" * 40,
        "B_LAUNCH_COMMIT": "a" * 40, "B_SIDECAR_COMMIT": "c" * 40, "B_SIDECAR_PATH": "x/b_sidecar.json", "B_SIDECAR_SHA256": "5" * 64}
filled = committed
for k_, v_ in TOK4.items():  # a5a7d010 R3: token by token (a whole-file substitution would also rewrite a token left in the note)
    filled = filled.replace(f"<FILL:{k_}>".encode(), v_.encode())
check("gate: filling the plan's named tokens leaves no <FILL anywhere (the note included)", b"<FILL" not in filled)
calls = []


class _Stub:
    def remote(self, *a):
        calls.append(a)
        return {"all_pass": True, "n_fails": 0, "peak_rss_mib": 1, "info": {"not_readable": {}}}


class _Run:
    def __init__(self, out):
        self.stdout = out


FG.HERE = tmp / "gate"
(FG.HERE / "results").mkdir(parents=True)
real_gate_count = FG.gate_count  # the container function itself, for (7)
FG.gate_count = _Stub()
FG.subprocess = type("S", (), {"run": staticmethod(lambda *a, **k: _Run("h" * 40 if "rev-parse" in a[0] else ""))})


def gate_with(plan_bytes):
    p = tmp / "gate" / "plan.json"
    p.write_bytes(plan_bytes)
    FG.PLANS["pooled_world_B_routec"] = str(p)
    for f in (FG.HERE / "results").iterdir():
        f.unlink()
    return stops(FG.gate_main.info.raw_f, "_t", "pooled_world_B_routec")


check("gate: the filled route C plan reaches the remote call", not gate_with(filled) and len(calls) == 1)
check("gate: record type pooled_world_B, intervals fixed+joint and the record meta go to the container",
      calls and calls[-1][6] == "pooled_world_B" and calls[-1][8] == "fixed+joint" and list(calls[-1][9]) == ["a" * 40, "dcd265d77088ccc517308680567158f9ad6ebbc497e3a8632c5de2ed2b1f12aa",
                                                                                                 "1e1868586fe8a786f6e1094b9d5b62e362ec0e2ce241e9b0103d6584af351126"])


def variant(fn):
    q = json.loads(filled)
    fn(q)
    return json.dumps(q).encode()


def drop(q, flag, n):
    a = q["figures"][0]["args"]
    i = a.index(flag)
    del a[i:i + 1 + n]


check("gate: no --label-sidecar STOPs", gate_with(variant(lambda q: drop(q, "--label-sidecar", 3))))
check("gate: no --label-sha STOPs", gate_with(variant(lambda q: drop(q, "--label-sha", 1))))
check("gate: no --record-meta STOPs", gate_with(variant(lambda q: drop(q, "--record-meta", 3))))
check("gate: a malformed --record-meta STOPs", gate_with(variant(lambda q: q["figures"][0]["args"].__setitem__(q["figures"][0]["args"].index("--record-meta") + 1, "TBD"))))
check("gate: a --text-note STOPs", gate_with(variant(lambda q: q["figures"][0]["args"].extend(["--text-note", "x"]))))
check("gate: a non-git sidecar source STOPs", gate_with(variant(lambda q: q["records"].update({"routec_B_label_sidecar.json": "sol:x"}))))
check("gate: another record type STOPs", gate_with(variant(lambda q: q["figures"][0]["args"].__setitem__(q["figures"][0]["args"].index("--record-type") + 1, "sweep15_perset"))))
check("gate: the old pooled_world_B key still maps to its own plan and type", FG.RECORD_TYPE.get("pooled_world_B", "pooled_world_B") == "pooled_world_B"
      and FG.PLANS["pooled_world_B"] == "config/render_plan_fig4_pooledB.json")
check("plan: B's own sidecar and 09's B label sha (READ_LOG @ 598169ded L2226), no A anchor",
      b"routec_B_label_sidecar.json" in committed and b"b84a38b75f63751b9f8d603497236cad3feb74ee6ccd569060bec276cb9c2766" in committed
      and b"--label-a-sha" not in committed and b"routec_A_label" not in committed)
RX = re.compile(r"probe-shift_aswin-surya_sub-[a-z0-9-]+_[a-z0-9-]+_kill-2026-11-14")
check("app names: <= 63 characters, route C form", all(len(n) <= 63 and RX.fullmatch(n) for n in (FG.app.name, RA.app.name)))

# (5) Figures 1 / 5 / 6: the sidecar of EVERY drawn record (09 via 2e, 10-18); derived records' labels are COMPOSED (09's byte rule)
PF.RES = tmp
G, AS = "7" * 64, "4" * 64  # a derived input's sha and the A record's sha (synthetic); LAB / LSHA stand in for A's registered label and its sha


def isc(name, rec, label):
    b = json.dumps({"record": name, "record_sha256": rec, "label": label}).encode()
    (tmp / name).write_bytes(b)
    return [name, hashlib.sha256(b).hexdigest(), "src"]


def comp(step, a_sha=AS, a_label=LAB):
    return f"{a_label} | {step} on route C pooled A @ {a_sha}"


ins = {"group": ("g.json", G, "src")}
gsc = ["group"] + isc("gsc.json", G, comp("run_group_catch")) + ["run_group_catch"]
check("routec_inputs: a derived input whose label is A's label + ' | <step> on route C pooled A @ <A>' passes",
      PF.routec_inputs(ins, [gsc], LAB, LSHA, AS) == {"group": comp("run_group_catch")})
check("routec_inputs: a drawn input without a sidecar STOPs", stops(PF.routec_inputs, ins | {"pw": ("p.json", "8" * 64, "src")}, [gsc], LAB, LSHA, AS))
check("routec_inputs: a sidecar for no drawn input STOPs", stops(PF.routec_inputs, {}, [gsc], LAB, LSHA, AS))
check("routec_inputs: a repeated role STOPs", stops(PF.routec_inputs, ins, [gsc, gsc], LAB, LSHA, AS))
check("routec_inputs: an input sidecar naming another record STOPs",
      stops(PF.routec_inputs, ins, [["group"] + isc("gsc2.json", "9" * 64, comp("run_group_catch")) + ["run_group_catch"]], LAB, LSHA, AS))
bad_prefix = ["group"] + isc("gbp.json", G, comp("run_group_catch", a_label="X" + LAB[1:])) + ["run_group_catch"]
bad_asha = ["group"] + isc("gba.json", G, comp("run_group_catch", a_sha="9" * 64)) + ["run_group_catch"]
bad_suffix = ["group"] + isc("gbs.json", G, comp("run_group_excess")) + ["run_group_catch"]
check("routec_inputs: a label not starting with A's registered bytes STOPs", stops(PF.routec_inputs, ins, [bad_prefix], LAB, LSHA, AS))
check("routec_inputs: a label naming another A record STOPs", stops(PF.routec_inputs, ins, [bad_asha], LAB, LSHA, AS))
check("routec_inputs: a label with another step's suffix STOPs", stops(PF.routec_inputs, ins, [bad_suffix], LAB, LSHA, AS))
nolab = "route C pooled A (synthetic test label without the status word)"
check("routec_inputs: a composed label without PRELIMINARY STOPs",
      stops(PF.routec_inputs, ins, [["group"] + isc("gnp.json", G, comp("run_group_catch", a_label=nolab)) + ["run_group_catch"]], nolab,
            hashlib.sha256(nolab.encode()).hexdigest(), AS))
check("routec_inputs: A's label not matching its sha STOPs", stops(PF.routec_inputs, ins, [gsc], LAB + "x", LSHA, AS))
m3 = mutant('    if set(got) != set(inputs):', '    if False:')
m3.RES = tmp
check("MUTANT 3 (the every-drawn-input check dropped) is caught: it lets a drawn input go without its sidecar",
      not stops(m3.routec_inputs, ins | {"pw": ("p.json", "8" * 64, "src")}, [gsc], LAB, LSHA, AS))
SUF = '        if lbb[len(ab):] != f" | {step} on route C pooled A @ {a_sha}".encode("utf-8"):'
for nm_, (old_, new_), case_ in (("MUTANT P (prefix check dropped)", ('        if hashlib.sha256(lbb[:len(ab)]).hexdigest() != a_label_sha256:', '        if False:'), bad_prefix),
                                 ("MUTANT AS (the <A sha> in the suffix not compared)",
                                  (SUF, '        if not lbb[len(ab):].startswith(f" | {step} on route C pooled A @ ".encode("utf-8")):'), bad_asha),
                                 ("MUTANT S (suffix check dropped)", (SUF, '        if False:'), bad_suffix)):
    mm_ = mutant(old_, new_)
    mm_.RES = tmp
    check(f"{nm_} is caught: it accepts the bad case its control rejects", not stops(mm_.routec_inputs, ins, [case_], LAB, LSHA, AS))
for n in ("post_main.py", "post_excess.py"):
    shutil.copy(HERE / n, exp / n)
gsha = hashlib.sha256(b'{"g": 1}').hexdigest()
(exp / "results" / "g.json").write_bytes(b'{"g": 1}')
(exp / "results" / "p.json").write_bytes(b'{"p": 1}')
psha = hashlib.sha256(b'{"p": 1}').hexdigest()


def esc(name, rec, step):
    b = json.dumps({"record": "x", "record_sha256": rec, "label": comp(step, a_sha=rsha)}).encode()
    (exp / "results" / name).write_bytes(b)
    return [name, hashlib.sha256(b).hexdigest(), "src", step]


sc_g, sc_p, sc_f = ["group"] + esc("gsc.json", gsha, "run_group_catch"), ["pw"] + esc("psc.json", psha, "pw_step"), ["four"] + esc("xsc.json", gsha, "run_group_excess")
LABEL_STOPS = ("label sidecar", "does not start with", "suffix", "lacks PRELIMINARY", "registered string", "not the checked one")


def run_main(extra, env=EV):
    cmd = [sys.executable, "post_main.py", "--stamp", "_t", "--record", "r.json", rsha, "src", "--name", "f1", "--rows", "a", "--readout", "C",
           "--right-source", "pooled_record", "--group-catch", "g.json", gsha, "src", "--floor-pw", "p.json", psha, "src"] + extra
    r = subprocess.run(cmd, cwd=exp, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "MPLBACKEND": "Agg"} | (env or {}))
    return r.returncode, (r.stderr or "") + (r.stdout or "")


a3 = ["--label-sidecar", *sc, "--label-sha", LSHA, "--record-meta", GIT, RPS, IFS]
rc, o = run_main(a3 + ["--input-sidecar", *sc_g])
check("post_main: the pw-shares input without its sidecar STOPs", rc != 0 and "without their label sidecar" in o and "pw" in o)
rc, o = run_main(["--label-sidecar", *sc, "--record-meta", GIT, RPS, IFS, "--input-sidecar", *sc_g, "--input-sidecar", *sc_p])
check("post_main: without --label-sha STOPs", rc != 0 and "go together" in o)
rc, o = run_main(["--label-sidecar", *sc, "--label-sha", LSHA, "--record-meta", "b" * 40, RPS, IFS, "--input-sidecar", *sc_g, "--input-sidecar", *sc_p])
check("post_main: a wrong A meta STOPs after the records load, before drawing", rc != 0 and "record's meta differs" in o and "'git'" in o)
rc, o = run_main(a3 + ["--input-sidecar", *sc_g, "--input-sidecar", *sc_p])
check("post_main: every sidecar and the meta pass (the run then STOPs on the synthetic record's keys)", rc != 0 and "record's meta differs" not in o
      and not any(w_ in o for w_ in LABEL_STOPS) and "STOP" in o)


def run_exc(extra, env=EV):
    cmd = [sys.executable, "post_excess.py", "--stamp", "_t", "--source", "pooled_record", "--record", "r.json", rsha, "src", "--four-record",
           "g.json", gsha, "src", "--name", "f5", "--rows", "a", "b", "c", "d", "--variant", "excess", "--subset-members", "a", "b", "c", "d"] + extra
    r = subprocess.run(cmd, cwd=exp, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "MPLBACKEND": "Agg"} | (env or {}))
    return r.returncode, (r.stderr or "") + (r.stdout or "")


rc, o = run_exc(a3)
check("post_excess: the four-probe input without its sidecar STOPs", rc != 0 and "without their label sidecar" in o and "four" in o)
rc, o = run_exc(a3 + ["--input-sidecar", *sc_f, "--text-note", "x"])
check("post_excess: with --text-note STOPs", rc != 0 and "go together" in o)
rc, o = run_exc(["--label-sidecar", *sc, "--label-sha", LSHA, "--record-meta", GIT, RPS, "e" * 64, "--input-sidecar", *sc_f])
check("post_excess: a wrong A meta STOPs after the records load", rc != 0 and "record's meta differs" in o and "inputs_full.sha256" in o)
CITE = f"experiments/x/results/auroc_table.json@{'1' * 40} sha256 {'2' * 64} (synthetic)"
rc, o = run_exc(a3 + ["--input-sidecar", *sc_f, "--auroc-citation", CITE])
check("post_excess: every sidecar, the meta and the E4 citation pass (then a STOP on the synthetic record)", rc != 0 and "record's meta differs" not in o
      and not any(w_ in o for w_ in LABEL_STOPS) and "needs --auroc-citation" not in o and "STOP" in o)

# 09 via 2e (10-18): Figure 5's four-probe (E4) row: RULE l.4409's registered label verbatim in the caption, and its AUROC source cited
import importlib.util as _ilu  # noqa: E402
import inspect as _insp  # noqa: E402

_sx = _ilu.spec_from_file_location("post_excess_t", HERE / "post_excess.py")
PX = _ilu.module_from_spec(_sx)
_sx.loader.exec_module(PX)
E4_SHA = "8912ccab6c5140493a57469eb30ab73981b891801222cd5cac9ffaf48b01e9af"  # sha256 of RULE.md L4409's string @ 91eefa7ca (no typed copy here)
check("E4: post_excess's E4_LABEL is RULE l.4409's registered string, byte-exact (by sha256)", hashlib.sha256(PX.E4_LABEL.encode("utf-8")).hexdigest() == E4_SHA)
check("E4: the caption builder appends E4_LABEL byte-exact and keeps the caption before it unchanged",
      PX.with_e4("Cap.").startswith("Cap. ") and PX.E4_LABEL in PX.with_e4("Cap.") and PX.with_e4("Cap.").count(PX.E4_LABEL) == 1)
_pe = _insp.getsource(PX.pooled_excess)
check("E4: the pooled_record figure (where the four-probe row always renders) passes its caption through e4_caption, route C by routec_labels",
      "e4_caption(cap, bool(a.routec_labels))" in _pe and "four_record" in _pe and " Post hoc." not in _pe)
_cap = f"Cap. {PX.SUBSET_TXT}"  # the real caption ends with SUBSET_TXT (its 0.6 cut-off), then the saturated sentence if any
check("E4 (Aswin, 10-18): a route C caption ends with E4_LABEL exactly once, keeps SUBSET_TXT's 0.6 and has no generic ' Post hoc.'",
      PX.e4_caption(_cap, True) == f"{_cap} {PX.E4_LABEL}." and PX.e4_caption(_cap, True).count(PX.E4_LABEL) == 1
      and " Post hoc." not in PX.e4_caption(_cap, True) and "0.6" in PX.e4_caption(_cap, True))
check("E4 (Aswin, 10-18): any other pooled_record caption keeps ' Post hoc.' and carries no E4 label",
      PX.e4_caption(_cap, False) == f"{_cap} Post hoc." and PX.E4_LABEL not in PX.e4_caption(_cap, False))
check("E4 (Aswin, 10-18): _source.json's four_probe_label is set inside the route C block only",
      _pe.count('o.src["four_probe_label"] = E4_LABEL') == 1
      and '        o.src["four_probe_label"] = E4_LABEL' in _pe[_pe.index("    if a.routec_labels:"):_pe.index("    want_rows = ")])
_mx = types.ModuleType("post_excess_mutant")
_src = (HERE / "post_excess.py").read_text()
assert _src.count('    return f"{cap} {E4_LABEL}."') == 1
exec(compile(_src.replace('    return f"{cap} {E4_LABEL}."', "    return cap"), "post_excess_mutant", "exec"), _mx.__dict__)  # noqa: S102 -- one line changed
check("E4 mutant (the label dropped from the caption) is caught by the presence check", PX.E4_LABEL not in _mx.with_e4("Cap."))
rc, o = run_exc(a3 + ["--input-sidecar", *sc_f])
check("E4: route C without --auroc-citation STOPs (after the meta and label checks)", rc != 0 and "needs --auroc-citation" in o)
cite_ = next(f["args"][f["args"].index("--auroc-citation") + 1] for f in json.loads((HERE / "config" / "render_plan_figs156_routecA.json").read_text())["figures"]
             if f["script"] == "post_excess.py")
rc, o = run_exc(a3 + ["--input-sidecar", *sc_f, "--auroc-citation", "HELD: placeholder"])
check("E4: a placeholder citation (no path@commit and sha256) STOPs", rc != 0 and "needs --auroc-citation" in o)
rc, o = run_exc(a3 + ["--input-sidecar", *sc_f, "--auroc-citation", cite_])
check("E4: the committed plan cites 09's confirmed source (the AUROC table and record @ 19ba8f563 by sha256, own_all_rs_L48 via own_all_L48 C) "
      "and passes the citation check", PX.CITE_RX.fullmatch(cite_) is not None and rc != 0 and "needs --auroc-citation" not in o
      and all(s_ in cite_ for s_ in ("@19ba8f5634bac60c54c1d426415fa0fdb4d65ae3", "efd0c2d2b74a5294c79f71451fbebb6c8b02f5e9b4b9c820b08c8b548d6f9250",
                                     "ff5d690d40e790af6994a54331d3e4086f52561fb4e21355e4c9a8b603eb378e", "own_all_rs_L48 by RULE l.4419 (via own_all_L48 C)",
                                     ".pooled_all.auroc_real_both.point")))

# (6) the committed Figs 1/5/6 plan and render_all's template STOP
p156 = json.loads((HERE / "config" / "render_plan_figs156_routecA.json").read_text())
A_LSHA = "6100557b57d8ca57699aa71619231eb6fdcbdf103199faa1298c3474cd4ab15c"  # 09's registered A label (READ_LOG L2213 text), relayed by 2e


def roles(args_):
    return sorted(args_[i + 1] for i, x in enumerate(args_) if x == "--input-sidecar")


fs_ = p156["figures"]
check("plan 156: Figs 1 / 6 are post_main C / R, Fig 5 is post_excess; no --text-note, no R37",
      [(f["script"], f["args"][f["args"].index("--fig-number") + 1]) for f in fs_] == [("post_main.py", "1"), ("post_excess.py", "5"), ("post_main.py", "6")]
      and [f["args"][f["args"].index("--readout") + 1] for f in fs_ if f["script"] == "post_main.py"] == ["C", "R"]
      and not any("--text-note" in f["args"] or "r37" in json.dumps(f["args"]).lower() for f in fs_))
check("plan 156: each figure has A's sidecar, A's label sha, the record meta, and a sidecar for every other drawn input",
      all(f["args"][f["args"].index("--label-sha") + 1] == A_LSHA and "--label-sidecar" in f["args"] and "--record-meta" in f["args"] for f in fs_)
      and [roles(f["args"]) for f in fs_] == [["group", "pw"], ["four"], ["group", "pw"]])
check("plan 156: the derived inputs name their steps (09's composed rule): run_group_catch, run_group_excess, analyze_pw_shares",
      [sorted((a_[i + 1], a_[i + 5]) for i, x in enumerate(a_) if x == "--input-sidecar") for a_ in (f["args"] for f in fs_)]
      == [[("group", "run_group_catch"), ("pw", "analyze_pw_shares")], [("four", "run_group_excess")], [("group", "run_group_catch"), ("pw", "analyze_pw_shares")]])
check("plan 156: the pw-shares record is read from its committed copy (git form), never from the solavg Volume",
      p156["records"]["<FILL:PW_BASENAME>"] == "git:<FILL:PW_COMMIT>:<FILL:PW_PATH>"
      and all(f["args"][f["args"].index("--floor-pw") + 3] == "<FILL:PW_PATH>@<FILL:PW_COMMIT>" for f in fs_ if f["script"] == "post_main.py"))
for pl_ in ("config/render_plan_figs156_routecA.json", "config/render_plan_fig4_pooledB_routec.json"):
    check(f"render_all refuses the committed {Path(pl_).name} while it holds fill tokens", stops(RA.main.info.raw_f, str(HERE / pl_)))
    check(f"{Path(pl_).name}: its note names no fill token (a5b40fd5 R1: a token there would STOP a fully filled plan)",
          "<FILL" not in json.loads((HERE / pl_).read_text())["note"])


def clean_stop(fn, *a):
    """True only for a STOP (SystemExit); any other exception (e.g. a parse before a sha check) is False."""
    try:
        fn(*a)
    except SystemExit:
        return True
    except Exception:  # noqa: BLE001 -- the control: a mutant fails here with a non-STOP error
        return False
    return False


(tmp / "bad.json").write_bytes(b"\x00not json")
check("M04 control (figures): a sidecar that is not JSON and fails its sha STOPs before any parse (routec_label)",
      clean_stop(PF.routec_label, ("bad.json", "0" * 64, "src"), B, "w", LSHA))
emp = json.dumps({"record": "", "record_sha256": B, "label": LAB}).encode()
(tmp / "emp.json").write_bytes(emp)
check("routec_label: an empty record field STOPs (a5a7d010 optional)", stops(PF.routec_label, ("emp.json", hashlib.sha256(emp).hexdigest(), "src"), B, "w", LSHA))

# (7) the gate CONTAINER function, run locally on synthetic records (a5a7d010: mutants M5 (fixed-interval check off) and M7 (meta check off)
#     survived the stubbed tests); the record path is absolute, so Path("/sol", path) is the temp file
from post_fig4_sweep import REG15  # noqa: E402

LV = [f"{y:.6g}" for y in REG15]
ROWS = [f"p{i} C" for i in range(6)]
CI = {"point": 0.1, "ci95": [0.0, 0.2], "draws": 10, "finite_draws": 10}


def node():
    return {lv: {"readable": {"all": True, "S_old": {"B": True, "A": True, "S": True}}, "readable_real_line": True, "gap": {"delta_full": CI}} for lv in LV}


def grec(with_fixed=True, meta=META):
    pooled = {r: {"pooled_all_origins": {"members": ["S_old"], "joint": node()} | ({"fixed": node()} if with_fixed else {})} for r in ROWS}
    avg = {lv: {"headline_six_full|C": {"rows": ROWS, "S": CI, "Y": CI}} for lv in LV}
    return {"meta": meta | {"nos4_twin_withhold": {"allowlist": []}, "without_S4": {"twin_withheld": {"combos_flagged": [], "combos_unavailable": []}}},
            "pooled": pooled, "coverage": {r: {"S_old": {"coverage": {"cohort_cells": 5}}} for r in ROWS},
            "coheadline_signed": {"pooled_all_origins": {"joint": avg} | ({"fixed": avg} if with_fixed else {})}}


def gc(rec, intervals, rmeta=None):
    p = tmp / "grec.json"
    b = json.dumps(rec).encode()
    p.write_bytes(b)
    return real_gate_count.local(str(p), hashlib.sha256(b).hexdigest(), ROWS, "pooled_all_origins", "headline_six_full|C", "A", "pooled_world_B",
                                 "none", intervals, rmeta)


r_ok = gc(grec(), "fixed+joint", [GIT, RPS, IFS])
check("gate container: a complete synthetic record passes (joint, fixed and the meta)", r_ok["all_pass"] is True and r_ok["info"]["record_meta"]["git"] == GIT)
r_m5 = gc(grec(with_fixed=False), "fixed+joint")
check("M5 control: fixed+joint with no fixed intervals fails in the container", r_m5["all_pass"] is False and any("[fixed]" in f for f in r_m5["fails"]))
r_m7 = gc(grec(), "joint", [GIT, "e" * 64, IFS])
check("M7 control: a record meta that differs fails in the container", r_m7["all_pass"] is False and any(f.startswith("record meta:") for f in r_m7["fails"]))
r_m7b = gc(grec(meta={}), "joint", [GIT, RPS, IFS])
check("M7 control: a record with no meta fails in the container", r_m7b["all_pass"] is False and r_m7b["fails"][0].startswith("record meta:"))

shutil.rmtree(tmp)
print(f"{'FAIL' if bad else 'OK'}: {len(bad)} failing")
sys.exit(1 if bad else 0)
