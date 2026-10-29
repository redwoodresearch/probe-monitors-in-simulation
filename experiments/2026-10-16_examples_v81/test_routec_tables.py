"""Route C fixes to the v81 tables (run_tables_v81.py, run_examples_v81.py, run_outcome_meta_v81.py; 10-17): synthetic bytes and code only;
no record, no Modal call (modal objects are created at import and never hydrated).
  (1) build_code ships copies.py when the build commit has it (route C's runner imports it at top level), never otherwise, and STOPs on a
      runner that imports copies without it; the real route C runner at f9e6e4139 (code, via git) gets copies.py;
  (2) load_lead: fill tokens STOP (HELD); inputs_full and record_meta go together, record_meta.git = build_commit and its inputs_full_sha256 =
      inputs_full.sha256; the committed route C pins file STOPs (still tokens); the old shape (no inputs_full) keeps the 110447 default;
  (3) check_record_meta (09 S2 item 10, in the container): git, script_sha256, FROZEN inputs_full (sha, status, test), each miss STOPs;
  (4) compose_label (09's sidecar rule): sha before parse, exact keys, record_sha256 = this A, the route C prefix and PRELIMINARY; a 171-pool
      label, a sidecar of another A and a sha mismatch STOP (positive controls);
  (5) 0d (10-17): render and add-on text stay under q2's SCRATCH only; the tables and add-on prints carry no table number;
  (6) the add-on's arms_from_record (container): two arms per drawn cell, "none" cells skipped, sha re-checked, ambiguous / disagreeing keys STOP;
  (7) app names: <= 63 characters, probe-shift_aswin-surya_sub-<sub>_<label>_kill-2026-11-14;
  (8) route C A's own selection record (09 RULE item 40): make_routec_selection.py reads (A) with item 40's NOTE verbatim, cites A by both shas,
      STOPs on an extra (IL) override, a TEST file or an unreadable pin; run_tables requires (A) for exactly this record;
  (9) launch package (10-18): volume_name maps §7 S7RC's committed copy (routec_<TAG>_<name>.json.gz) to the Volume name the spec's dest ends with.

  uv run --no-project --python 3.12 --with numpy==2.3.3 --with modal==1.6.1 python test_routec_tables.py   (from this folder)
"""

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_examples_v81 as X  # noqa: E402
import run_outcome_meta_v81 as OM  # noqa: E402
import run_tables_v81 as T  # noqa: E402

bad = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        bad.append(name)


def stops(fn, *a, **k):
    try:
        fn(*a, **k)
    except SystemExit:
        return True
    return False


def fake_git(files: dict, head: str = "f" * 40):
    """git show <c>:<path> / ls-tree --name-only <c>:<dir> / rev-parse from a dict {(commit, path): text}."""
    def g(*a):
        if a[0] == "rev-parse":
            return ((head if a[-1] == "HEAD" else a[-1].split("^")[0]) + "\n").encode()
        if a[0] == "ls-tree":
            c, d = a[-1].split(":", 1)
            return "\n".join(sorted(p.rsplit("/", 1)[1] for (cc, p) in files if cc == c and p.startswith(d + "/"))).encode()
        c, p = a[-1].split(":", 1)
        if (c, p) not in files:
            raise SystemExit(f"fake git: no {c}:{p}")
        return files[(c, p)].encode()
    return g


real_git = T.git

# (1) build_code
D = T.BUILD_DIR
base = {(c, f"{D}/{n}"): "x = 1\n" for c in ("a" * 40, "b" * 40, "c" * 40) for n in T.BUILD_FILES}
base[("a" * 40, f"{D}/copies.py")] = "y = 2\n"
base[("a" * 40, f"{D}/run_peraction311.py")] = "import json\nimport copies as CP  # noqa\n"
base[("c" * 40, f"{D}/run_peraction311.py")] = "import json\nimport copies as CP  # noqa\n"
T.git = fake_git(base)
code_a, _ = T.build_code("a" * 40)
code_b, _ = T.build_code("b" * 40)
check("build_code: copies.py shipped when the build has it", sorted(code_a) == sorted((*T.BUILD_FILES, "copies.py")))
check("build_code: no copies.py for an older build", sorted(code_b) == sorted(T.BUILD_FILES))
check("build_code: a runner importing copies without copies.py STOPs", stops(T.build_code, "c" * 40))
T.git = real_git
code_rc, _ = T.build_code("f9e6e4139fa67b34e8a1cd81d8cef02087f6c3b6")  # code only (git show of .py files), read-only
check("build_code: the route C runner (71633875 at f9e6e4139) gets copies.py and the anchor once",
      "copies.py" in code_rc and code_rc["run_peraction311.py"].count(T.PATCH_ANCHOR) == 1)

# (2) load_lead
FROZEN = ("1" * 40, "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FULL_x.json", "2" * 64)
good = {"calibration": "pooled", "stub": False, "build_commit": "3" * 40,
        "record": {"path": "r.json.gz", "commit": "4" * 40, "blob_sha256": "5" * 64, "json_sha256": "6" * 64}, "spec": "s.pkl", "spec_sha256": "7" * 64,
        "selection": {"path": "RULE.md", "commit": "8" * 40, "sha256": "9" * 64, "line": 4873, "kind": "RULE item 40"},
        "inputs_full": dict(zip(("commit", "path", "sha256"), FROZEN)),
        "record_meta": {"git": "3" * 40, "script_sha256": "d" * 64, "inputs_full_sha256": FROZEN[2]}}


def lead_of(j):
    T.git = fake_git({("HEAD", "p.json"): json.dumps(j)})
    try:
        return T.load_lead("pooled", "p.json")
    finally:
        T.git = real_git


L, _, _ = lead_of(good)
check("load_lead: inputs_full and record_meta pinned", L["inputs_full"] == FROZEN and L["record_meta"] == good["record_meta"])
check("load_lead: a fill token STOPs", stops(lead_of, good | {"spec": "<FILL:A_DRIVER_SPEC_PATH>"}))
check("load_lead: TBD STOPs", stops(lead_of, good | {"record": good["record"] | {"commit": "TBD"}}))
check("load_lead: inputs_full without record_meta STOPs", stops(lead_of, {k: v for k, v in good.items() if k != "record_meta"}))
check("load_lead: record_meta.git != build_commit STOPs", stops(lead_of, good | {"record_meta": good["record_meta"] | {"git": "e" * 40}}))
check("load_lead: record_meta.inputs_full_sha256 != inputs_full.sha256 STOPs",
      stops(lead_of, good | {"record_meta": good["record_meta"] | {"inputs_full_sha256": "e" * 64}}))
check("load_lead: an extra record_meta key STOPs", stops(lead_of, good | {"record_meta": good["record_meta"] | {"x": "y"}}))
old = {k: v for k, v in good.items() if k not in ("inputs_full", "record_meta")}
check("load_lead: the old shape keeps the 110447 default (no inputs_full in L)", "inputs_full" not in lead_of(old)[0])
committed = json.loads((HERE / "config" / "lead_pins_pooled_routec.json").read_text())  # a hand-committed config (fill tokens, no value)
check("load_lead: the committed route C pins file STOPs (tokens)", stops(lead_of, committed))
check("route C pins: record_meta.script_sha256 is the sha256 of run_peraction311.py blob 71633875",
      committed["record_meta"]["script_sha256"] == hashlib.sha256(real_git("cat-file", "-p", "71633875188c57ca1ceb8271a3f3ca0ed0989452")).hexdigest())

# (3) check_record_meta
RM = {"git": "3" * 40, "script_sha256": "d" * 64, "inputs_full_sha256": FROZEN[2]}
M = {"git": "3" * 40, "script_sha256": "d" * 64, "full_length": {"inputs_full": {"path": FROZEN[1], "sha256": FROZEN[2], "status": "FROZEN 2026-10-17_x", "test": False}}}
check("check_record_meta: the pinned meta passes", not stops(T.check_record_meta, M, RM, FROZEN[2]))
for nm, m_ in (("git", M | {"git": "e" * 40}), ("script_sha256", M | {"script_sha256": "e" * 64}),
               ("inputs_full sha", M | {"full_length": {"inputs_full": M["full_length"]["inputs_full"] | {"sha256": "e" * 64}}}),
               ("status TEST", M | {"full_length": {"inputs_full": M["full_length"]["inputs_full"] | {"status": "TEST x"}}}),
               ("test true", M | {"full_length": {"inputs_full": M["full_length"]["inputs_full"] | {"test": True}}}), ("no full_length", {k: v for k, v in M.items() if k != "full_length"})):
    check(f"check_record_meta: {nm} STOPs", stops(T.check_record_meta, m_, RM, FROZEN[2]))
check("check_record_meta: the laptop's INPUTS_FULL differing from the pin STOPs", stops(T.check_record_meta, M, RM, "e" * 64))

# (4) compose_label
LAB = "route C pooled A (full-length; synthetic test label, not 09's text); PRELIMINARY"  # no typed copy of 09's label (a5a7d010, 10-18)
A = "6" * 64


def sc(d):
    b = json.dumps(d).encode()
    return b, hashlib.sha256(b).hexdigest()


b0, h0 = sc({"record": "xfer/fpr507prep/records/results/a.json", "record_sha256": A, "label": LAB})
got = T.compose_label(b0, h0, A)
check("compose_label: the registered label composes", got == f"{LAB} | v81 tables 3/4 (run_tables_v81) on route C pooled A @ {A}")
check("compose_label: sha mismatch STOPs", stops(T.compose_label, b0, "e" * 64, A))
check("compose_label: a sidecar of another A STOPs", stops(T.compose_label, b0, h0, "e" * 64))
check("compose_label: an extra key STOPs", stops(T.compose_label, *sc({"record": "r", "record_sha256": A, "label": LAB, "x": 1}), A))
check("compose_label: a 171-pool label STOPs", stops(T.compose_label, *sc({"record": "r", "record_sha256": A,
                                                                         "label": "171-pool; pooled A (pilot excluded; S_old tau_sim 171); superseded when the fix lands; PRELIMINARY"}), A))
check("compose_label: no PRELIMINARY STOPs", stops(T.compose_label, *sc({"record": "r", "record_sha256": A, "label": LAB.replace("PRELIMINARY", "final")}), A))
check("sidecar pointer: the group steps' form only", bool(T.SIDECAR_RX.fullmatch(f"git:{'a' * 40}:x/LABEL.json@{A}"))
      and not T.SIDECAR_RX.fullmatch(f"git:{'a' * 40}:x/LABEL.md@{A}") and not T.SIDECAR_RX.fullmatch(f"{'a' * 40}:x/LABEL.json@{A}"))

# (5) 0d (10-17): text stays on q2's 10-16 path; no table number leaves the container in a printed line (source checks)
import inspect  # noqa: E402

check("text path: q2's SCRATCH only (no --scratch option)", "scratch" not in inspect.signature(T.main.info.raw_f).parameters
      and "scratch" not in inspect.signature(OM.main.info.raw_f).parameters and not hasattr(X, "scratch_dir"))
check("prints: the tables counts carry no dedupe or drift count", '"duplicates_dropped_count": res' not in inspect.getsource(T._full)
      and '"rows_with_dT_nonzero": res' not in inspect.getsource(T._full))
check("prints: the add-on returns and prints no count", "counts" not in inspect.getsource(OM._run) and "counts" not in inspect.getsource(OM.main.info.raw_f))

# (6) arms_from_record
OUT = {"both caught": ("S_old|t1", "S_old|t1"), "real caught, sim missed": ("S_new|t2", None), "real missed, sim caught": None, "both missed": ("S_paper493|t3", None)}


def table(outcomes):
    oc = {}
    for o, v in outcomes.items():
        oc[o] = {"n": 1 if v else 0, "drawn": "none" if v is None else {"cell": v[0], "real (run 1)": {"run_key": f"{v[0].split('|')[1]}/r1#e0", "z": 1.0},
                                                                        "sim (sim 1)": "no command"}}
    return json.dumps({"what": "v81 threshold tables (synthetic)", "label": "L", "run_label": "RL", "outcome_table": {"outcomes": oc}}).encode()


rows = [{"key": f"{t}/{w[0]}1#e0" if w == "run1" else f"{t}/s1#e0", "task": t, "world": w, "origin": o, "origin_any": False, "eval_sha256s": [hashlib.sha256((t + w).encode()).hexdigest()]}
        for o, t in (("S_old", "t1"), ("S_new", "t2"), ("S_paper493", "t3")) for w in ("run1", "sim")]
tb = table(OUT)
arms, lab = OM.arms_from_record(tb, hashlib.sha256(tb).hexdigest(), rows)
check("arms_from_record: two arms per drawn cell, the 'none' cell skipped", len(arms) == 6 and lab == {"label": "L", "run_label": "RL"})
check("arms_from_record: sha re-check STOPs", stops(OM.arms_from_record, tb, "e" * 64, rows))
check("arms_from_record: no drawn cell STOPs", stops(lambda: OM.arms_from_record(t0 := table({k: None for k in OUT}), hashlib.sha256(t0).hexdigest(), rows)))
check("arms_from_record: two physical keys for one arm STOP", stops(OM.arms_from_record, tb, hashlib.sha256(tb).hexdigest(), rows + [rows[0] | {"key": "t1/other#e0"}]))
check("arms_from_record: the record's run_key disagreeing STOPs",
      stops(OM.arms_from_record, tb, hashlib.sha256(tb).hexdigest(), [r | {"key": "t2/x#e0"} if r["task"] == "t2" and r["world"] == "run1" else r for r in rows]))
check("arms_from_record: two .eval shas for one arm STOP",
      stops(OM.arms_from_record, tb, hashlib.sha256(tb).hexdigest(), [r | {"eval_sha256s": ["a" * 64, "b" * 64]} if r["task"] == "t1" else r for r in rows]))
tw = json.dumps({"what": "something else", "outcome_table": {"outcomes": {}}}).encode()
check("arms_from_record: a record not from tables mode STOPs", stops(OM.arms_from_record, tw, hashlib.sha256(tw).hexdigest(), rows))

# (8) route C A's own selection record (09 RULE item 40 via 2e): the writer is keys only; run_tables requires it to read (A) for this record
sys.path.insert(0, str(HERE.parent / "2026-10-15_fpr_sweep_507_prep"))
import gzip  # noqa: E402

import make_routec_selection as MS  # noqa: E402

real_ms_git = MS.git
J = json.dumps({"rows": {"x": 1}}).encode()
GZ = gzip.compress(J)
FR = {"status": "FROZEN 2026-10-17 22:39 PDT", "test": False, "overrides": {k: {} for k in MS.OVERRIDES_110447}, "ilarms": {"commit": "c"}}


def ms_with(frozen, a_json=J):
    files = {("a" * 40, "r/A.json.gz"): gzip.compress(a_json), ("b" * 40, "c/IF.json"): json.dumps(frozen).encode()}

    def g(*a):
        if a[0] == "rev-parse":
            return (("f" * 40 if a[-1] == "HEAD" else a[-1].split("^")[0]) + "\n").encode()
        c, p = a[-1].split(":", 1)
        return files[(c, p)] if (c, p) in files else real_ms_git(*a)  # the RULE file: read-only from the real repo
    MS.git = g
    fb = files[("b" * 40, "c/IF.json")]
    try:
        return MS.build("r/A.json.gz@" + "a" * 40, hashlib.sha256(files[("a" * 40, "r/A.json.gz")]).hexdigest(), hashlib.sha256(a_json).hexdigest(),
                        "c/IF.json@" + "b" * 40, hashlib.sha256(fb).hexdigest(), "2026-10-18_000000")
    finally:
        MS.git = real_ms_git


sel = ms_with(FR)
check("selection: reads (A), item 40's NOTE verbatim, cites the A record by both shas",
      sel["lead"] == "A" and sel["note"] == "(B) not evaluable at the freeze: IL extra-arm overrides not pinned (build omission)"
      and sel["record"] == {"path": "r/A.json.gz", "commit": "a" * 40, "blob_sha256": hashlib.sha256(GZ).hexdigest(), "json_sha256": hashlib.sha256(J).hexdigest()})
check("selection: an extra (IL) override pinned in FROZEN STOPs (09 words (B))", stops(ms_with, FR | {"overrides": FR["overrides"] | {"il|C": {}}}))
check("selection: a TEST INPUTS_FULL STOPs", stops(ms_with, FR | {"status": "TEST x"}))
MS.git = lambda *a: (_ for _ in ()).throw(SystemExit("no git"))  # noqa: E731
check("selection: an unreadable pin STOPs", stops(MS.build, "r@" + "a" * 40, "0" * 64, "0" * 64, "c@" + "b" * 40, "0" * 64, "s"))
MS.git = real_ms_git
fr_json = json.loads(real_git("show", "1a842a85635bd63a8bfc60071195584834117c11:experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FULL_2026-10-17_233212.json"))
check("selection: route C's FROZEN (K config) pins exactly 110447's twelve overrides (no IL extra-arm override)",
      sorted(fr_json["overrides"]) == MS.OVERRIDES_110447 and fr_json["status"].startswith("FROZEN") and fr_json["test"] is False)
REC = ("r/A.json.gz", "a" * 40, hashlib.sha256(GZ).hexdigest(), hashlib.sha256(J).hexdigest())
check("check_routec_selection: the writer's record passes", not stops(T.check_routec_selection, sel, REC))
check("check_routec_selection: a (B) lead STOPs", stops(T.check_routec_selection, sel | {"lead": "B"}, REC))
check("check_routec_selection: another A record STOPs", stops(T.check_routec_selection, sel, REC[:3] + ("e" * 64,)))
check("route C pins: the selection is a fill token of the route C kind (not the 171-pool RULE pin)",
      committed["selection"]["kind"].startswith(T.ROUTEC_SELECTION_KIND) and "<FILL" in committed["selection"]["sha256"] and "line" not in committed["selection"])

# (9) a5b40fd5's controls (10-18): R1 and the mutants M04, M10, M19 that guard route C's core protections


def ok_run(fn, *a):
    """fn(*a), or None on any exception (a control then reports FAIL instead of crashing; 2e O-a)."""
    try:
        return fn(*a)
    except BaseException:  # noqa: BLE001 -- SystemExit included: the M19 control must FAIL, not end the run
        return None


def clean_stop(fn, *a):
    """True only for a STOP (SystemExit); any other exception (e.g. a parse error before a sha check) is False."""
    try:
        fn(*a)
    except SystemExit:
        return True
    except Exception:  # noqa: BLE001 -- the control: a mutant fails here with a non-STOP error
        return False
    return False


ctext = (HERE / "config" / "lead_pins_pooled_routec.json").read_text()
filled_pins = json.loads(re.sub(r"<FILL:([A-Z0-9_]+)>", lambda m: hashlib.sha256(m.group(1).encode()).hexdigest()[:40 if "COMMIT" in m.group(1) else 64], ctext))
check("R1: the route C pins file passes load_lead once every field token is filled (its note names no token)",
      not stops(lead_of, filled_pins) and "<FILL" not in committed["note"])
check("M04 control: a sidecar that is not JSON and fails its sha STOPs before any parse (compose_label)",
      clean_stop(T.compose_label, b"\x00not json", "0" * 64, A))
meta_ok = {"git": "a" * 40, "script_sha256": "d" * 64, "full_length": {"inputs_full": {"path": "p", "sha256": "1" * 64, "status": "FROZEN x", "test": False}}}


def full_with(rm):
    lb = json.dumps({"meta": meta_ok}).encode()
    p = {"lead_bytes": lb, "lead_path": "r.json", "lead_json_sha256": hashlib.sha256(lb).hexdigest(), "dry": False, "build_commit": "a" * 40,
         "calibration": "pooled", "rule_pin": False, "record_meta": rm, "inputs_full_sha256": "1" * 64, "meta": {"run_label": ""}, "stub": False}
    try:
        T._full(p)
    except SystemExit as e:
        return str(e)
    return "no STOP"


rm_ok = {"git": "a" * 40, "script_sha256": "d" * 64, "inputs_full_sha256": "1" * 64}
check("M10 control: the container STOPs on a record meta that differs from the pins (before anything else reads the record)",
      "record_meta at ['script_sha256']" in full_with(rm_ok | {"script_sha256": "e" * 64}))
check("M10 control, positive: with the pinned meta the run passes the meta check (it then STOPs on the synthetic record's structure)",
      "structure" in full_with(rm_ok) and "record_meta" not in full_with(rm_ok))
check("M19 control: the selection writer succeeds on an A whose bytes are not JSON (it hashes A and never parses it)",
      (ok_run(ms_with, FR, b"not json at all") or {}).get("lead") == "A")

# (7) app names
RX = re.compile(r"probe-shift_aswin-surya_sub-[a-z0-9-]+_[a-z0-9-]+_kill-2026-11-14")
check("app names: <= 63 characters and the route C form", all(len(a) <= 63 and RX.fullmatch(a) for a in (T.APP, OM.APP)))

# (9) launch package (10-18): §7 S7RC's committed copy carries a routec_<TAG>_ prefix the Volume name (the spec's dest) lacks
S7RC_P = "experiments/2026-10-15_fpr_sweep_507_prep/results/routec_records/routec_A_alltasks_x_2026-10-18_000000.json.gz"
check("volume_name: an S7RC routec_A_ copy maps to its Volume record name; a path outside routec_records/ falls back to its own name less .gz",
      T.volume_name(S7RC_P) == "alltasks_x_2026-10-18_000000.json"
      and T.volume_name("experiments/2026-10-15_fpr_sweep_507_prep/results/alltasks_y.json.gz") == "alltasks_y.json")
check("volume_name (af820177 O-b, A only): routec_B_ / routec_R37_ / any other file under routec_records/ STOPs",
      all(stops(T.volume_name, S7RC_P.replace("routec_A_", t)) for t in ("routec_B_", "routec_R37_", "routec_Q_", "")))
check("volume_name feeds the container's record_name (the spec dest check)", '"record_name": volume_name(rp)' in (HERE / "run_tables_v81.py").read_text())
check("volume_name (af820177 M2): the routec_A_ prefix is stripped ONLY under results/routec_records/; a routec_A_ file anywhere else keeps it",
      T.volume_name("experiments/2026-10-15_fpr_sweep_507_prep/results/routec_A_alltasks_x_2026-10-18_000000.json.gz")
      == "routec_A_alltasks_x_2026-10-18_000000.json"
      and T.volume_name("experiments/2026-10-15_fpr_sweep_507_prep/results/other_records/routec_A_alltasks_x_2026-10-18_000000.json.gz")
      == "routec_A_alltasks_x_2026-10-18_000000.json"
      and T.volume_name("experiments/2026-10-16_examples_v81/results/routec_records/routec_A_alltasks_x_2026-10-18_000000.json.gz")
      == "routec_A_alltasks_x_2026-10-18_000000.json")

# (10) af820177 O-a (REQUIRED before release): the sha checks that carry record identity, each with a negative that must STOP with its own
# message (so a mutant that drops the check fails the test) and a positive control that passes the check
import gzip as _gz  # noqa: E402
import pickle  # noqa: E402
import tempfile  # noqa: E402


def stop_msg(fn, *a, **k):
    """The STOP's text, or the type of any other exception, or 'no STOP' (never a record value: synthetic inputs only)."""
    try:
        fn(*a, **k)
    except SystemExit as e:
        return str(e)
    except Exception as e:  # noqa: BLE001  (a mutant that drops a check reaches later code; any other failure is reported by type)
        return f"other: {type(e).__name__}"
    return "no STOP"


lb_ = json.dumps({"meta": meta_ok}).encode()
pj_ = {"lead_bytes": lb_, "lead_path": "r.json", "lead_json_sha256": hashlib.sha256(lb_).hexdigest(), "dry": False, "build_commit": "a" * 40,
       "calibration": "pooled", "rule_pin": False, "record_meta": rm_ok, "inputs_full_sha256": "1" * 64, "meta": {"run_label": ""}, "stub": False}
MSG1 = "the lead record's JSON sha256 differs from the pin"
check("O-a 1: the container STOPs on a lead JSON sha256 that is not the pin (ARS)", stop_msg(T._full, pj_ | {"lead_json_sha256": "0" * 64}) == MSG1)
check("O-a 1: ... and on a .json.gz lead whose decompressed bytes are not the pin",
      stop_msg(T._full, pj_ | {"lead_bytes": _gz.compress(b'{"meta": {}}'), "lead_path": "r.json.gz"}) == MSG1)
check("O-a 1 control: the pinned bytes pass the sha check (the run then STOPs later, on the synthetic structure)",
      stop_msg(T._full, pj_) != MSG1 and stop_msg(T._full, pj_ | {"lead_bytes": _gz.compress(lb_), "lead_path": "r.json.gz"}) != MSG1)

RP_, RC_ = S7RC_P, "c" * 40
blob_ = _gz.compress(lb_)
L_ = {"build_commit": "", "record": (RP_, RC_, hashlib.sha256(blob_).hexdigest(), hashlib.sha256(lb_).hexdigest()), "spec": "xfer/s.pkl",
      "spec_sha256": "7" * 64, "selection": None, "selection_ref": {}}


class _NoRemote:
    def remote(self, *a):
        raise SystemExit("STUB REMOTE (the test never calls Modal)")


def laptop_run(L):
    saved = (T.git, T.load_lead, T.job)
    files = {("status", "--porcelain", "--untracked-files=no"): b"", ("show", f"{RC_}:{RP_}"): blob_, ("rev-parse", "HEAD"): b"f" * 40}
    T.git = lambda *a: files[a]
    T.load_lead, T.job = (lambda cal, pins: (L, False, {"path": "p", "commit": "f" * 40, "sha256": "b" * 64})), _NoRemote()
    try:
        return stop_msg(T.main.info.raw_f, mode="tables", calibration="pooled", pins="p.json", z_from_calib="draw0_z__{nm}")
    finally:
        T.git, T.load_lead, T.job = saved


MSG2 = "the lead record's git blob sha256 differs from the pin"
check("O-a 2: the laptop STOPs on a committed blob whose sha256 is not the pin, before any remote call",
      laptop_run(L_ | {"record": (RP_, RC_, "0" * 64, L_["record"][3])}) == MSG2)
check("O-a 2 control: the pinned blob passes the laptop check (the run then fails later, never reaching Modal)", laptop_run(L_) != MSG2)


def tables_with(dest):
    sp = {"dest": dest, "meta": {"git": "g" * 40, "stamp": "s1"}, "mode": "alltasks", "jobs": []}
    with tempfile.NamedTemporaryFile("wb", suffix=".pkl", delete=False) as fh:
        pickle.dump(sp, fh)
    sb = Path(fh.name).read_bytes()
    saved = (T.load_build, X.find_eval)
    T.load_build = lambda code: None
    try:
        return stop_msg(T._tables, {"record": {"meta": {"git": "g" * 40, "stamp": "s1"}}, "code": None, "spec_path": fh.name,
                                    "spec_sha256": hashlib.sha256(sb).hexdigest(), "calibration": "pooled", "record_name": T.volume_name(S7RC_P)})
    finally:
        T.load_build, X.find_eval = saved
        Path(fh.name).unlink()


MSG3 = "the driver spec is not the one that wrote the lead record (dest, git, stamp, mode): STOP"
check("O-a 3: the container STOPs on a driver spec whose dest names another record",
      tables_with("xfer/fpr507prep/records/results/alltasks_OTHER_2026-10-18_000000.json") == MSG3
      and tables_with("xfer/fpr507prep/records/results/routec_A_alltasks_x_2026-10-18_000000.json") == MSG3)
check("O-a 3 control: the spec whose dest ends with the S7RC copy's Volume name passes the dest check",
      tables_with("xfer/fpr507prep/records/results/alltasks_x_2026-10-18_000000.json") != MSG3)

print(f"{'FAIL' if bad else 'OK'}: {len(bad)} failing")
sys.exit(1 if bad else 0)
