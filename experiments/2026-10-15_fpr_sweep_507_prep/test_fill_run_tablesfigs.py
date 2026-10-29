"""tools_routec/fill_run_tablesfigs.py: the NOTE blocks, the VARS placeholders, the template fills and the run files' syntax (zsh -f -n
and bash -n only; no block is run). Synthetic values only; no marker, record, results/ or READ_LOG is read.

  uv run --no-project --python 3.12 python -I test_fill_run_tablesfigs.py        (from experiments/2026-10-15_fpr_sweep_507_prep)
"""

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location("frt", HERE / "tools_routec" / "fill_run_tablesfigs.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
fails, n = [], 0


def check(name, ok):
    global n
    n += 1
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


def stops(fn, *a):
    try:
        fn(*a)
    except SystemExit as e:
        return str(e).startswith("STOP")
    return False


E = F.E
SYN = {"HB": "1" * 40, "A_LAUNCH": "2" * 40, "A_REC_P": f"{E}/results/a_2026-10-18_000000.json.gz", "A_REC_C": "3" * 40, "A_REC_B": "4" * 64,
       "A_REC_J": "5" * 64, "A_SPEC": "xfer/fpr507prep/drivers/a_2026-10-18_000000.pkl", "A_SPEC_S": "6" * 64,
       "SIDECAR": f"git:{'7' * 40}:{E}/results/routec_A_label_sidecar_2026-10-18_000000.json@{'8' * 64}", "C09": "9" * 40,
       "MP": "experiments/2026-10-14_priority1_spec_09/signoff/S2_2026-10-18_000000.json", "MSHA": "a" * 64,
       "COVER": json.dumps([f"{E}/results/a_2026-10-18_000000.json.gz@{'4' * 64}"], separators=(",", ":")),
       "SEL_P": f"{E}/results/estimator_selection_routec_2026-10-18_000000.json", "SEL_C": "b" * 40, "SEL_S": "c" * 64,
       "STAMP": "2026-10-18_120000", "PFILE": "/tmp/x/filled.json", "PFS": "d" * 64,
       "A_BASE": "a_2026-10-18_000000.json", "A_J": "5" * 64, "A_CP": f"{E}/results/a_2026-10-18_000000.json.gz", "A_CC": "3" * 40, "A_B": "4" * 64,
       "A_SP": f"{E}/results/routec_A_label_sidecar_2026-10-18_000000.json", "A_SC": "7" * 40, "A_SS": "8" * 64,
       "B_BASE": "b_2026-10-18_000000.json", "B_J": "e" * 64, "B_CP": f"{E}/results/b_2026-10-18_000000.json.gz", "B_CC": "f" * 40, "B_B": "0" * 64,
       "B_LAUNCH": "12" * 20, "B_SC_P": f"{E}/results/routec_B_label_sidecar_2026-10-18_000000.json", "B_SC_C": "34" * 20, "B_SC_S": "56" * 32,
       "CVPKG": "tables", "CVP": f"{E}/config/s2_cover/routec_tables_cover_2026-10-18_120000.json", "CVS": "ef" * 32}
for p_, path_ in (("GC", f"{E}/results/group_catch_2026-10-18_000000.json"), ("GX", f"{E}/results/group_excess_2026-10-18_000000_{'ab' * 16}.json"),
                  ("PW", f"{E}/results/pw_shares_routec/pw_shares_pooled_all_origins_a_2026-10-18_000000_routeC_2026-10-18_000001.json")):
    SYN |= {f"{p_}_BASE": Path(path_).name, f"{p_}_P": path_, f"{p_}_C": "78" * 20, f"{p_}_S": "9a" * 32,
            f"{p_}_SP": path_.removesuffix(".json") + "_LABEL.json", f"{p_}_SC": "78" * 20, f"{p_}_SS": "bc" * 32}

check("every placeholder has a regex and a rule, and the synthetic values pass them",
      all(k in F.RX and k in F.RULES and re.fullmatch(F.RX[k], SYN[k]) for k in SYN) and set(SYN) == set(F.RX) == set(F.RULES))
check("a CONTROL_ONLY pw-shares output fails the pw regex",
      not re.fullmatch(F.RX["PW_P"], f"{E}/results/pw_shares_routec/pw_shares_pooled_all_origins_a_CONTROL_ONLY_2026-10-18_000001.json"))
check("a sidecar is not a group catch record", not re.fullmatch(F.RX["GC_P"], SYN["GC_SP"]))

for pkg, (note, vbs, acts, tmpl) in F.PKG.items():
    text = (REPO / note).read_text()
    bl = F.blocks(text, [b for b, _ in vbs] + acts)
    check(f"{pkg}: every block is one BEGIN / END pair", len(bl) == len(vbs) + len(acts))
    filled = ""
    for b, names in vbs:
        have = sorted(m.group(1) for s in bl[b].splitlines() if (m := re.fullmatch(r"([A-Z_0-9]+)='<[^']*>'", s)))
        check(f"{pkg} {b}: its placeholders are exactly the package's", have == sorted(names))
        t_, d_ = F.fill(bl[b], names, SYN)
        check(f"{pkg} {b}: filled, no placeholder left, the diff is the placeholder lines only",
              "='<" not in t_ and len(d_) == 2 * len(names) and all(x[0] in "+-" for x in d_))
        filled += t_
        check(f"{pkg} {b}: a value with a quote STOPs", stops(F.fill, bl[b], names, SYN | {names[0]: SYN[names[0]] + "'"}))
        check(f"{pkg} {b}: a value failing its regex STOPs", stops(F.fill, bl[b], names, SYN | {names[-1]: "x"}))
        check(f"{pkg} {b}: a missing placeholder line STOPs", stops(F.fill, bl[b].replace(f"{names[1]}='<", f"{names[1]}0='<", 1), names, SYN))
    for x in acts:
        body = filled + bl[x]
        with tempfile.NamedTemporaryFile("w", suffix=".zsh", delete=False) as fh:
            fh.write(body)
        rz = subprocess.run(["zsh", "-f", "-n", fh.name], capture_output=True, text=True, check=False)
        rb = subprocess.run(["bash", "-n", fh.name], capture_output=True, text=True, check=False)
        Path(fh.name).unlink()
        check(f"{pkg} {x}: the run file parses (zsh -f -n and bash -n)", rz.returncode == 0 and rb.returncode == 0)
    if tmpl:
        tt = (REPO / tmpl[0]).read_text()
        out = F.fill_template(tt, tmpl[1], SYN)
        check(f"{pkg}: the template fills to valid JSON with no token left", "<FILL" not in out and json.loads(out) is not None)
        tok = next(iter(tmpl[1]))
        check(f"{pkg}: a token count off by one STOPs", stops(F.fill_template, tt.replace(f"<FILL:{tok}>", f"<FILL:{tok}><FILL:{tok}>", 1), tmpl[1], SYN))
        check(f"{pkg}: an unknown token STOPs", stops(F.fill_template, tt.replace("{", '{"x": "<FILL:NOT_A_TOKEN>",', 1), tmpl[1], SYN))

with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "m.txt"
    p.write_text("a b c\n")
    check("fields: one line of 3 fields passes", F.fields(p, 3, "t") == ["a", "b", "c"])
    for bad in ("a b c", "a  b c\n", "a b c\nd\n", "a b\n"):
        p.write_text(bad)
        check(f"fields: {bad!r} STOPs", stops(F.fields, p, 3, "t"))
    p.write_text("x y z\n")
    check("record_commit: a marker outside the TAG's §7 dir STOPs before any git read", stops(F.record_commit, str(p), d, "A", {"rs": "5" * 64}))
    p.write_text("{not json")
    check("jload: a non-JSON input STOPs cleanly (O2)", stops(F.jload, p.read_bytes(), "t"))

# af820177 R1: every placeholder x {newline, tab, CR, NUL, quote, space} (the reviewer's 348 negatives) STOPs in fill()
VARIANTS = (lambda s: s + "\n", lambda s: s[:len(s) // 2] + "\t" + s[len(s) // 2:], lambda s: s + "\r", lambda s: s + "\x00",
            lambda s: s + "'", lambda s: s[:len(s) // 2] + " " + s[len(s) // 2:])
neg = [(k, i) for k in sorted(F.RX) for i, f_ in enumerate(VARIANTS) if not stops(F.fill, f"{k}='<x>'\n", [k], SYN | {k: f_(SYN[k])})]
check(f"R1: all {len(F.RX) * len(VARIANTS)} control-character / quote / space negatives STOP in fill() (missed: {neg[:5]})", not neg)
paths = [k for k in F.RX if k.endswith("_BASE") or any(t in F.RX[k] for t in ("experiments/", "xfer/", "/[^", "git:"))]
leak = sorted({k for k in paths for c in "\n\t\r" if re.fullmatch(F.RX[k], SYN[k][:5] + c + SYN[k][5:])})
check(f"R1: the path and file-name regexes alone refuse a newline, a tab and a CR (leaks: {leak})", not leak and len(paths) >= 20)
check("R1: a template value with a control character STOPs",
      stops(F.fill_template, (REPO / F.PKG["fig4"][3][0]).read_text(), F.TOK_4, SYN | {"B_BASE": "b\n.json"}))

# af820177 O-a (REQUIRED before release): S7RC's record_commit.txt and A's name checks, each negative with its own STOP message (a mutant that
# drops the check fails the test) and a positive control; git is a dict fake, the §7 markers synthetic files (no repo, no record)
import gzip
import hashlib
from types import SimpleNamespace


def stop_msg(fn, *a):
    try:
        out = fn(*a)
    except SystemExit as e:
        return str(e)
    except Exception as e:  # noqa: BLE001  (a mutant reaching later code: reported by type only)
        return f"other: {type(e).__name__}"
    return out


REC = b'{"synthetic": true}'
RS, BASE, C_ = hashlib.sha256(REC).hexdigest(), "alltasks_x_2026-10-18_000000.json", "c" * 40
SCP, SCC = f"{E}/results/routec_A_label_sidecar_2026-10-18_000000.json", "9" * 40


def s7_dir(root, rc_name, rc_sha=RS, where=None):
    d = Path(root) / "S7_A_2026-10-18_000000"
    d.mkdir()
    (d / "ok.txt").write_text(f"xfer/fpr507prep/records/results/{BASE} {RS} xfer/fpr507prep/a.npz {'6' * 64} ap-abcdefghij12\n")
    (d / "launched.txt").write_text(f"ap-abcdefghij12 fc-abc xfer/a.pkl {'7' * 64} {'2' * 40}\n")
    (d / "launch.log").write_text('x\n{"driver_call": "fc-abc", "spec": "xfer/fpr507prep/drivers/a.pkl", "spec_sha256": "' + "7" * 64 + '"}\n')
    (d / "sidecar.txt").write_text(f"{SCP} {hashlib.sha256(b'sc').hexdigest()} {SCC} {'1' * 40}\n")
    path = f"{E}/results/routec_records/{rc_name}"
    m = Path(where) if where else d
    m.mkdir(exist_ok=True)
    (m / "record_commit.txt").write_text(f"{C_} {path} {rc_sha}\n")
    return d, path, m / "record_commit.txt"


def with_git(path, blob, fn, *a):
    saved = (F.git, F.ancestors, F.s2)
    files = {("diff-tree", "--no-commit-id", "--name-only", "-r", C_): path.encode() + b"\n", ("cat-file", "blob", f"{C_}:{path}"): blob,
             ("cat-file", "blob", f"{SCC}:{SCP}"): b"sc"}
    F.git = lambda *k: files[k]
    F.ancestors, F.s2 = (lambda *k: None), (lambda *k: {})
    try:
        return stop_msg(fn, *a)
    finally:
        F.git, F.ancestors, F.s2 = saved


GOOD = gzip.compress(REC)
with tempfile.TemporaryDirectory() as r:
    d, path, mk = s7_dir(r, f"routec_A_{BASE}.gz")
    check("O-a 4 control: S7RC's marker with the right blob gives ARC:ARP@ARS",
          with_git(path, GOOD, F.record_commit, str(mk), str(d), "A", {"rs": RS}) == f"{C_}:{path}@{RS}")
    check("O-a 4: a record_commit blob that decompresses to another sha256 STOPs",
          "the blob does not decompress to its sha256" in with_git(path, gzip.compress(b'{"other": 1}'), F.record_commit, str(mk), str(d), "A", {"rs": RS}))
    bad_z = bytearray(GOOD)
    bad_z[12:16] = b"\xff\xff\xff\xff"
    check("O-c: a blob that is not gzip, or a corrupt deflate stream (zlib.error), STOPs cleanly",
          all("is not a readable gzip blob" in with_git(path, b_, F.record_commit, str(mk), str(d), "A", {"rs": RS}) for b_ in (b"not gzip", bytes(bad_z))))
with tempfile.TemporaryDirectory() as r:
    d, path, mk = s7_dir(r, f"routec_A_{BASE}.gz", where=Path(r) / "elsewhere")
    check("O-a (no directory anchor): a well-formed record_commit.txt outside the TAG's §7 dir STOPs",
          "must be" in with_git(path, GOOD, F.record_commit, str(mk), str(d), "A", {"rs": RS}))
for nm, want in ((f"routec_B_{BASE}.gz", False), (f"routec_R37_{BASE}.gz", False), (f"routec_A_{BASE}.gz", True), (f"{BASE}.gz", True)):
    with tempfile.TemporaryDirectory() as r:
        d, path, mk = s7_dir(r, nm)
        got = with_git(path, GOOD, F.a_values, SimpleNamespace(s7_dir=str(d), fok="", a_final=str(mk)), "1" * 40)
        if want:
            check(f"O-a 5 control: A's committed copy named {nm} is accepted", isinstance(got, dict) and got.get("A_REC_P") == path)
        else:
            check(f"O-a 5: A's committed copy named {nm} (another TAG's) STOPs", isinstance(got, str) and "is not routec_A_" in got)

# 2e (10-18): --gc / --gx / --pw from SIMPROBE's commit_ok.txt, fields 3-5 cross-checked; git is a dict fake, the markers synthetic files
DREC = {"gc": f"{E}/results/group_catch_2026-10-18_000001.json", "gx": f"{E}/results/group_excess_2026-10-18_000002_{'ab' * 16}.json",
        "pw": f"{E}/results/pw_shares_routec/pw_shares_pooled_all_origins_{BASE.removesuffix('.json')}_routeC_2026-10-18_000003.json"}
DBYTES = {r: (f'{{"r": "{r}"}}'.encode(), f'{{"s": "{r}"}}'.encode()) for r in DREC}


def derived_with(edit):
    """edit(role, fields) -> fields for that role's marker (5 strings); runs derived() with a dict-fake git; returns its dict or STOP text."""
    files = {}
    for r, p in DREC.items():
        sp = p.removesuffix(".json") + "_LABEL.json"
        files[("cat-file", "blob", f"{C_}:{p}")], files[("cat-file", "blob", f"{C_}:{sp}")] = DBYTES[r]
    saved = (F.git, F.ancestors)
    with tempfile.TemporaryDirectory() as r:
        args = {}
        for role, p in DREC.items():
            sp = p.removesuffix(".json") + "_LABEL.json"
            fl = edit(role, [C_, p, hashlib.sha256(DBYTES[role][0]).hexdigest(), sp, hashlib.sha256(DBYTES[role][1]).hexdigest()])
            mf = Path(r) / f"{role}_commit_ok.txt"
            mf.write_text(fl if isinstance(fl, str) else " ".join(fl) + "\n")
            args[role] = str(mf)
        F.git, F.ancestors = (lambda *k: files[k]), (lambda *k: None)
        try:
            return stop_msg(F.derived, SimpleNamespace(**args), "1" * 40, BASE)
        finally:
            F.git, F.ancestors = saved


def gx_only(i, val):
    return lambda role, f: [val if (role == "gx" and j == i) else x for j, x in enumerate(f)]


got = derived_with(lambda role, f: f)
check("commit_ok control: all three markers consistent: the GC / GX / PW values come from fields 1:2 and the hashes match fields 3 / 5",
      isinstance(got, dict) and got.get("GX_P") == DREC["gx"] and got.get("GX_C") == C_ and got.get("PW_SP") == DREC["pw"].removesuffix(".json") + "_LABEL.json")
check("commit_ok: a record sha256 (field 3) that is not the record's blob STOPs with its own message",
      "does not hash to commit_ok.txt's record sha256 (field 3)" in str(derived_with(gx_only(2, "0" * 64))))
check("commit_ok: a sidecar path (field 4) that is not <stem>_LABEL.json STOPs with its own message",
      "sidecar path (field 4) is not the record's <stem>_LABEL.json" in str(derived_with(gx_only(3, f"{E}/results/other_LABEL.json"))))
check("commit_ok: a sidecar sha256 (field 5) that is not the sidecar's blob STOPs with its own message",
      "does not hash to commit_ok.txt's sidecar sha256 (field 5)" in str(derived_with(gx_only(4, "0" * 64))))
check("commit_ok: a malformed marker (4 fields, or not one line) STOPs",
      all("single-space-separated fields" in str(derived_with(e)) for e in (lambda role, f: f[:4], lambda role, f: " ".join(f) + "\nx\n")))

# 2e (10-18): the pins / plan templates are pinned by blob; the committed ones pass, any other blob STOPs before the fill
for tp in F.TEMPLATE_BLOB:
    saved = F.git
    F.git = lambda *k, tp=tp, saved=saved: b"0" * 40 + b"\n" if k == ("rev-parse", f"HEAD:{tp}") else saved(*k)
    bad = stop_msg(F.template_text, tp)
    F.git = saved
    check(f"template pin: {Path(tp).name} at HEAD is the pinned blob {F.TEMPLATE_BLOB[tp][:8]}, and another blob STOPs",
          F.template_text(tp) == (REPO / tp).read_text() and "not the pinned" in str(bad))

# 09 via 2e (10-18): Figure 5's E4 citation, checked at fill time by blob (blob_sha faked: no results/ file is read here)
plan156 = (REPO / F.PKG["figs156"][3][0]).read_text()
cited = {(c, p): s for p, c, s in F.E4_CITES}
saved_bs = F.blob_sha
F.blob_sha = lambda c, p: cited[(c, p)]
ok_ = stop_msg(F.check_e4_cites, plan156)
bad_plan = stop_msg(F.check_e4_cites, plan156.replace(F.E4_CITES[1][2], "0" * 64))
F.blob_sha = lambda c, p: "0" * 64
bad_blob = stop_msg(F.check_e4_cites, plan156)
F.blob_sha = saved_bs
check("E4 cites: the committed Figs 1/5/6 plan cites both AUROC files by path@commit and sha256, and matching blobs pass", ok_ is None)
check("E4 cites: a plan whose citation names another sha256 STOPs", "does not cite" in str(bad_plan))
check("E4 cites: a blob that does not hash to its cited sha256 STOPs", "does not hash to its cited sha256" in str(bad_blob))

pins = json.loads(F.fill_template((REPO / F.PKG["tables"][3][0]).read_text(), F.TOK_T, SYN))
check("tables: the filled pins carry A, the selection and record_meta.git = build_commit, by key",
      pins["build_commit"] == SYN["A_LAUNCH"] == pins["record_meta"]["git"] and pins["record"] == {"path": SYN["A_REC_P"], "commit": SYN["A_REC_C"],
      "blob_sha256": SYN["A_REC_B"], "json_sha256": SYN["A_REC_J"]} and pins["selection"]["path"] == SYN["SEL_P"] and pins["spec"] == SYN["A_SPEC"])
plan = json.loads(F.fill_template((REPO / F.PKG["figs156"][3][0]).read_text(), F.TOK_156, SYN))
check("figs156: the filled plan reads the pw-shares record from its commit (git form) with step analyze_pw_shares",
      plan["records"][SYN["PW_BASE"]] == f"git:{SYN['PW_C']}:{SYN['PW_P']}" and plan["records"][SYN["A_BASE"]] == f"sol:xfer/fpr507prep/records/results/{SYN['A_BASE']}")

# GATES Amendment 15 (09 via 2e, 10-18): the per-package COVER set, its committed file, and the S2 marker's set equality (dict-fake git)
CS = {k: F.cover_set(k, SYN) for k in ("tables", "figs156", "fig4")}
check("COVER (09: records + sidecars, .json.gz by decompressed sha): Tables = A + A's sidecar",
      CS["tables"] == sorted([f"{SYN['A_REC_P']}@{SYN['A_REC_J']}", f"{SYN['A_SP']}@{SYN['A_SS']}"]))
check("COVER: Figures 1/5/6 = A, group catch, four-probe excess, pw-shares and their four sidecars (8)",
      len(CS["figs156"]) == 8 and f"{SYN['A_CP']}@{SYN['A_J']}" in CS["figs156"] and all(f"{SYN[p + '_SP']}@{SYN[p + '_SS']}" in CS["figs156"] for p in ("GC", "GX", "PW")))
check("COVER: Figure 4 = B + B's sidecar", CS["fig4"] == sorted([f"{SYN['B_CP']}@{SYN['B_J']}", f"{SYN['B_SC_P']}@{SYN['B_SC_S']}"]))
F.COVER_SIDECARS, F.COVER_GZ_SHA = False, "blob"
flip = {k: F.cover_set(k, SYN) for k in ("tables", "figs156", "fig4")}
F.COVER_SIDECARS, F.COVER_GZ_SHA = True, "json"
check("COVER switches (one line each): records only gives 1 / 4 / 1 entries; 'blob' cites the gz blob's sha256",
      [len(flip[k]) for k in ("tables", "figs156", "fig4")] == [1, 4, 1] and flip["tables"] == [f"{SYN['A_REC_P']}@{SYN['A_REC_B']}"])


def cover_with(cover_bytes, marker, cover_sha=None):
    """committed_cover then s2 for the tables package with a dict-fake git; -> the s2 dict or the STOP text."""
    cvp, cvc, mp = SYN["CVP"], "c" * 40, SYN["MP"]
    files = {("cat-file", "blob", f"{cvc}:{cvp}"): cover_bytes, ("cat-file", "blob", f"{SYN['C09']}:{mp}"): json.dumps(marker).encode()}
    saved = (F.git, F.ancestors)
    F.git, F.ancestors = (lambda *k: files[k]), (lambda *k: None)
    with tempfile.TemporaryDirectory() as r:
        okf = Path(r) / "cover_ok.txt"
        okf.write_text(f"{cvc} {cvp} {cover_sha or hashlib.sha256(cover_bytes).hexdigest()}\n")
        a_ = SimpleNamespace(cover_ok=str(okf), s2=f"{SYN['C09']}:{mp}")
        try:
            return stop_msg(lambda: F.s2(a_, F.committed_cover(a_, "tables", SYN, "1" * 40)))
        finally:
            F.git, F.ancestors = saved


good_b = (json.dumps(CS["tables"], indent=1) + "\n").encode()
mk_ok = {"signoff": "S2", "signed": True, "by": "09", "covered": list(reversed(CS["tables"])), "time": "t", "notes": "n", "for": "tables", "items": []}
got = cover_with(good_b, mk_ok)
check("S2 control: the committed COVER file equals the derived set and the marker's covered (any order) equals it as a set",
      isinstance(got, dict) and json.loads(got["COVER"]) == CS["tables"])
check("S2: a committed COVER file whose bytes are not cover_ok.txt's sha256 STOPs", "does not hash to cover_ok.txt's sha256" in str(cover_with(good_b, mk_ok, "0" * 64)))
check("S2: a committed COVER list that is not exactly the package's derived set STOPs (a subset, a superset)",
      all("not exactly this package's derived set" in str(cover_with((json.dumps(sorted(x), indent=1) + "\n").encode(), mk_ok))
          for x in (CS["tables"][:1], CS["tables"] + [f"{E}/results/x.json@{'1' * 64}"])))
check("S2: a COVER list with a duplicate STOPs", "unique" in str(cover_with((json.dumps(sorted(CS["tables"] * 2), indent=1) + "\n").encode(), mk_ok)))
check("S2: a marker whose covered is a subset, or holds a duplicate, STOPs (true set equality, no duplicates)",
      all("covered set is not exactly" in str(cover_with(good_b, mk_ok | {"covered": c})) for c in (CS["tables"][:1], CS["tables"] + CS["tables"][:1])))
check("S2: a marker not signed S2 by 09 STOPs", all("covered set is not exactly" in str(cover_with(good_b, mk_ok | d))
                                                   for d in ({"signoff": "S1"}, {"signed": False}, {"by": "0d"})))
check("S2 (Amendment 15 path form): a marker path that is not S2_<YYYY-MM-DD_HHMMSS>.json fails the MP regex",
      not re.fullmatch(F.RX["MP"], "experiments/2026-10-14_priority1_spec_09/signoff/S2_A.json") and re.fullmatch(F.RX["MP"], SYN["MP"]) is not None)
check("S2 (af820177 M12): a marker whose covered is a SUPERSET of the COVER list STOPs",
      "covered set is not exactly" in str(cover_with(good_b, mk_ok | {"covered": CS["tables"] + [f"{E}/results/extra.json@{'3' * 64}"]})))

# af820177 R-1: every COVER input (records AND sidecars) is regex-checked before --cover-out writes anything
for pkg_, key_ in (("tables", "A_SP"), ("figs156", "GX_SP"), ("fig4", "B_SC_P")):
    bad_ = [stop_msg(F.cover_set, pkg_, SYN | {key_: SYN[key_].removesuffix(".json") + ext}) for ext in (".json.gz", ".json.GZ", ".JSON.GZ")]
    check(f"R-1: a {pkg_} sidecar ({key_}) named .json.gz / .json.GZ / .JSON.GZ STOPs at --cover-out (cover_set)",
          all(isinstance(b_, str) and "COVER inputs fail their regex" in b_ and key_ in b_ for b_ in bad_))
_msrc = __import__("inspect").getsource(F.main)
check("R-1: --cover-out derives the list with cover_set (which regex-checks every input) before it writes the file",
      0 <= _msrc.find("cov = cover_set(a.pkg, v)") < _msrc.find('with cf.open("xb") as fh:'))
# 185200fb OPTIONAL-2: cover_set's control-character guard (a \x01 inside a sidecar's stem passes the free-form A_SP / B_SC_P regexes)
check("O-2 (185200fb): a sidecar with a control character (\\x01) in its stem STOPs at --cover-out (cover_set's CTRL guard)",
      all("COVER inputs fail their regex" in str(stop_msg(F.cover_set, pkg_, SYN | {key_: SYN[key_].removesuffix(".json") + "\x01.json"}))
          for pkg_, key_ in (("tables", "A_SP"), ("figs156", "GX_SP"), ("fig4", "B_SC_P"))))


# 185200fb OPTIONAL-1: --cover-out driven through main() (behavior, not source text), with a dict-fake git and a scratch toplevel
def cover_out_main(scp):
    """main --pkg tables --cover-out --a-final, sidecar.txt field 1 = scp -> (STOP text or None, COVER files written, --out-dir made, stdout)."""
    import contextlib
    import io

    nc, hb = "e" * 40, "1" * 40
    with tempfile.TemporaryDirectory() as r:
        d, path, mk = s7_dir(r, f"routec_A_{BASE}.gz")
        (d / "sidecar.txt").write_text(f"{scp} {hashlib.sha256(b'sc').hexdigest()} {SCC} {'1' * 40}\n")
        top, out = Path(r) / "top", Path(r) / "out"
        top.mkdir()
        files = {("rev-parse", f"{nc}^{{commit}}"): f"{nc}\n".encode(), ("rev-parse", f"{hb}^{{commit}}"): f"{hb}\n".encode(),
                 ("rev-parse", "HEAD"): f"{hb}\n".encode(), ("status", "--porcelain", "--untracked-files=no"): b"",
                 ("diff-tree", "--no-commit-id", "--name-only", "-r", C_): path.encode() + b"\n", ("cat-file", "blob", f"{C_}:{path}"): GOOD,
                 ("cat-file", "blob", f"{SCC}:{scp}"): b"sc", ("rev-parse", "--show-toplevel"): f"{top}\n".encode(),
                 ("show", f"{nc}:{F.NPK}"): (REPO / F.NPK).read_bytes()}
        saved = (F.git, F.ancestors, sys.argv)
        F.git, F.ancestors = (lambda *k: files[k]), (lambda *k: None)
        sys.argv = ["fill_run_tablesfigs.py", "--pkg", "tables", "--cover-out", "--note-commit", nc, "--head", hb, "--s7-dir", str(d),
                    "--a-final", str(mk), "--out-dir", str(out)]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                got = stop_msg(F.main)
        finally:
            F.git, F.ancestors, sys.argv = saved
        return got, len(list(top.rglob("routec_tables_cover_*.json"))), out.exists(), buf.getvalue()


got_, ncov_, made_, so_ = cover_out_main(SCP)
check("O-1 (185200fb) control: main --cover-out with a good sidecar writes one COVER file and the run-file folder (ROUTEC_FILL_OK)",
      got_ is None and ncov_ == 1 and made_ and "ROUTEC_FILL_OK tables" in so_)
for what_, exts_ in (("named .json.gz / .json.GZ / .JSON.GZ", (".json.gz", ".json.GZ", ".JSON.GZ")), ("with a control character in its stem", ("\x01.json",))):
    res_ = [cover_out_main(SCP.removesuffix(".json") + e_) for e_ in exts_]
    check(f"O-1 (185200fb): main --cover-out with a sidecar {what_} STOPs before any COVER file or --out-dir is written",
          all(isinstance(g_, str) and "COVER inputs fail their regex: ['A_SP']" in g_ and n_ == 0 and not m_ and "ROUTEC_FILL_OK" not in s_
              for g_, n_, m_, s_ in res_))

# af820177 M14: on the --fok path, A's committed .json.gz is checked by its decompressed bytes (gzip -dc) too
for blob_, want_ in ((gzip.compress(b'{"other": 2}'), False), (GOOD, True)):
    with tempfile.TemporaryDirectory() as r:
        d, path, mk = s7_dir(r, f"routec_A_{BASE}.gz")
        fok = Path(r) / "fill_ok.txt"
        fok.write_text(f"{'c' * 40} {'d' * 40} {'e' * 64} {C_}:{path}@{RS} git:{SCC}:{SCP}@{hashlib.sha256(b'sc').hexdigest()} {'f' * 40} {'0' * 40}\n")
        got_ = with_git(path, blob_, F.a_values, SimpleNamespace(s7_dir=str(d), fok=str(fok), a_final=""), "1" * 40)
        if want_:
            check("M14 control: the --fok path with the right .json.gz bytes passes", isinstance(got_, dict) and got_.get("A_REC_J") == RS)
        else:
            check("M14: the --fok path with a .json.gz that decompresses to another sha256 STOPs", "does not decompress to §7's record sha256" in str(got_))
print(f"{n - len(fails)}/{n} checks passed")
sys.exit(1 if fails else 0)
