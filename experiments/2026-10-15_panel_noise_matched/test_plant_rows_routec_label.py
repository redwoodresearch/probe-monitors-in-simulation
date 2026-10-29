"""run_plant_rows.py for route C (09's label ruling, 10-17): the record's pool label is COMPOSED from A's label sidecar, never typed. Lean checks
(synthetic only; the script is read as text, never imported; no record is read):

1. EXACT: this file minus its route C statements (pinned literally) equals run_plant_rows.py @ 4fb7e2063, statement for statement.
2. pool_label: None without a sidecar (the 171-pool configs, unchanged); with one, "<label> | plant_rows (Table 6 verdicts) on route C pooled A
   @ <A sha256>" from 09's registered string (a literal as committed in test_routec_group_steps.py @ 1e9ec0f00 L146-147; READ_LOG @
   8a4e2bf84 L2213; pinned by sha256 and length, check 0). Positive controls: non-JSON bytes with a wrong sha STOP
   on the sha message; a sidecar naming another A, an extra key, a label without "route C", a label without PRELIMINARY, a malformed pointer:
   all STOP.
3. main(): a config with a sidecar must be a non-TEST run with base_record, a git:<40-hex>:<path>.json@<sha256> pointer and a label naming no
   171-pool; the sidecar's bytes are read unparsed and sha-checked; the remote call passes them.
4. The route C configs equal the reviewed 2026-10-17 e2e configs except: record (all four) and base_record / a_label_sidecar / label (FULL),
   all record pins TO_FILL (main STOPs on any), and no "171" or "d8fcb945f" anywhere.

No READ_LOG decode and no live git (2e, 10-18; rule 30): run_plant_rows.py @ 4fb7e2063 (blob 32812bb6) is read from a hash-verified export.

  RUNNER_EXPORTS=<a folder holding the file 32812bb68ffa6c651055ecb038de9136a2801d2a, that blob's bytes> \
  uv run --offline --no-project --python 3.12 python -I experiments/2026-10-15_panel_noise_matched/test_plant_rows_routec_label.py
"""
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = (HERE / "run_plant_rows.py").read_text()
EXPORTS = Path(os.environ.get("RUNNER_EXPORTS") or HERE / "exports")  # 2e / rule 30: pinned blobs from hash-verified exports, no live git


def export(blob_id):
    """A pinned blob read from <RUNNER_EXPORTS>/<blob id> and checked here as that git blob (sha1 of "blob <n>\\0" + bytes)."""
    b = (EXPORTS / blob_id).read_bytes()
    if hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() != blob_id:
        raise SystemExit(f"STOP: the export {blob_id} is not that git blob")
    return b.decode()


OLD = export("32812bb68ffa6c651055ecb038de9136a2801d2a")  # run_plant_rows.py @ 4fb7e2063
# 09's registered route C string (READ_LOG @ 8a4e2bf84 L2213), as committed in the fixture test_routec_group_steps.py @ 1e9ec0f00
# (blob adb58dad, L146-147, LAB09), pinned by sha256 and length (2e, 10-18: no READ_LOG decode)
REG_LABEL = ("route C pooled A (full-length; S_old tau_sim 191 = 149 + 20 pilot + 8 + 14; pilot runs have no 32k cross-check except p2's, unit "
             "identity PASS @ f4d1db66a); PRELIMINARY until S2, the value reproductions and the a1 clean-room reproduction")
REG_PIN = ("6100557b57d8ca57699aa71619231eb6fdcbdf103199faa1298c3474cd4ab15c", 237)
ADDED_TOP = {"POOL_STEP", "SIDECAR_PTR", "pool_label"}
EXPECTED_REMOVED = sorted([
    "plab_ = pool_label(cfg0, side_blob)",
    "side_ = None",
    "if c.get('a_label_sidecar') is not None:\n    import re\n    m_ = re.fullmatch(SIDECAR_PTR, str(c['a_label_sidecar']))\n"
    "    if test_ or not c.get('base_record') or '171-pool' in json.dumps(c.get('label')) or (not m_):\n"
    "        raise SystemExit('STOP: a route C config (a_label_sidecar) is a non-TEST run with base_record, a git:<40-hex>:<path>.json@<sha256> "
    "sidecar pointer, and a label that names no 171-pool')\n    side_ = git('show', f'{m_[1]}:{m_[2]}')\n"
    "    if hashlib.sha256(side_).hexdigest() != m_[3]:\n        raise SystemExit(\"STOP: A's label sidecar's bytes differ from the config's pin\")"])


class Strip(ast.NodeTransformer):
    """The route C statements out (each recorded as written); the extra parameters / arguments / dict entry back to 4fb7e2063's."""

    def __init__(self):
        self.removed = []

    def visit_FunctionDef(self, n):
        a_ = [x.arg for x in n.args.args]
        if n.name == "plant_rows" and a_ == ["cfg", "rec_blob", "base_blob", "ctx", "side_blob"]:
            n.args.args, n.args.defaults = n.args.args[:4], n.args.defaults[:1]
        if n.name == "finalize" and a_ == ["c", "out", "ctx", "plab"]:
            n.args.args, n.args.defaults = n.args.args[:3], []
        self.generic_visit(n)
        return n

    def visit_Call(self, n):
        self.generic_visit(n)
        u_ = ast.unparse(n.func)
        if u_ == "finalize" and len(n.args) == 4 and ast.unparse(n.args[3]) == "plab_":
            n.args = n.args[:3]
        if u_ == "plant_rows.remote" and len(n.args) == 5 and ast.unparse(n.args[4]) == "side_":
            n.args = n.args[:4]
        return n

    def visit_Dict(self, n):
        self.generic_visit(n)
        drop = [i for i, (k, v) in enumerate(zip(n.keys, n.values)) if k is None and ast.unparse(v) == "{'pool_label': plab} if plab is not None else {}"]
        for i in reversed(drop):
            del n.keys[i], n.values[i]
            self.removed.append("**pool_label")
        return n

class Filter(ast.NodeTransformer):
    """Pass 2 (after the trims): the new route C statements out, top-down, each recorded as written (own fields name plab_ / side_, or the
    `if c.get('a_label_sidecar') …` block)."""

    def __init__(self):
        self.removed = []

    def generic_visit(self, n):
        for f in ("body", "orelse", "finalbody"):
            b = getattr(n, f, None)
            if isinstance(b, list) and b and isinstance(b[0], ast.stmt):
                keep = []
                for s in b:
                    own = {x.id for v in (v for f_, v in ast.iter_fields(s) if not (isinstance(v, list) and v and isinstance(v[0], ast.stmt)))
                           for nn in (v if isinstance(v, list) else [v]) if isinstance(nn, ast.AST) for x in ast.walk(nn) if isinstance(x, ast.Name)}
                    if own & {"plab_", "side_"} or (isinstance(s, ast.If) and "a_label_sidecar" in ast.unparse(s.test)):
                        self.removed.append(ast.unparse(s))
                    else:
                        keep.append(s)
                setattr(n, f, keep)
        return super().generic_visit(n)


def stripped(src):
    t = ast.parse(src)
    t.body = [s for s in t.body if not ((isinstance(s, ast.FunctionDef) and s.name in ADDED_TOP)
                                        or (isinstance(s, ast.Assign) and {x.id for x in s.targets if isinstance(x, ast.Name)} & ADDED_TOP))]
    st_, fl_ = Strip(), Filter()
    t = fl_.visit(st_.visit(t))  # pass 1 trims the modified ORIGINAL statements back; pass 2 removes the new ones
    return ast.dump(t), sorted(fl_.removed), st_.removed.count("**pool_label")


nd_, rem_, nplab_ = stripped(NEW)
res = {"1. EXACT: minus its route C statements (pinned) == run_plant_rows.py @ 4fb7e2063, statement for statement": (
    nd_ == stripped(OLD)[0] and rem_ == EXPECTED_REMOVED and nplab_ == 1)}
MUT = ("        out = plant_rows_body(cfg, rec_blob, base_blob)\n", "        out = plant_rows_body(cfg, rec_blob, None)\n")
res["1. positive control: a non-label change (the base blob dropped) is caught"] = NEW.count(MUT[0]) == 1 and stripped(NEW.replace(*MUT))[0] != stripped(OLD)[0]
res["0. pin (no READ_LOG decode): the registered string matches its sha256 and length"] = (
    hashlib.sha256(REG_LABEL.encode()).hexdigest() == REG_PIN[0] and len(REG_LABEL.encode()) == REG_PIN[1])

ns = {"hashlib": hashlib, "json": json}
for n in ast.parse(NEW).body:
    if (isinstance(n, ast.Assign) and {x.id for x in n.targets if isinstance(x, ast.Name)} & {"POOL_STEP", "SIDECAR_PTR"}) or (
            isinstance(n, ast.FunctionDef) and n.name == "pool_label"):
        exec(compile(ast.Module(body=[n], type_ignores=[]), "pr", "exec"), ns)
pool_label = ns["pool_label"]
A_SHA = "a" * 64


def cfg_side(label=REG_LABEL, rsha=A_SHA, extra=None, ptr_sha=None, ptr=None):
    b = json.dumps({"record": "experiments/x/results/alltasks_routeC.json.gz", "record_sha256": rsha, "label": label} | (extra or {})).encode()
    p = ptr or f"git:{'e' * 40}:experiments/x/results/A_label_sidecar.json@{ptr_sha or hashlib.sha256(b).hexdigest()}"
    return {"a_label_sidecar": p, "base_record": {"git": "c:p.json.gz", "sha256": "9" * 64, "decompressed_sha256": A_SHA}}, b


def msg(f, *a):
    try:
        return f"RET {f(*a)}"
    except SystemExit as e:
        return f"STOP {e}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR {type(e).__name__}"


c0, b0 = cfg_side()
res["2. no sidecar (the 171-pool configs): None; route C: the composed label, exactly"] = (
    pool_label({"label": "x"}, None) is None and pool_label(c0, b0) == f"{REG_LABEL} | plant_rows (Table 6 verdicts) on route C pooled A @ {A_SHA}")
cn, _ = cfg_side(ptr_sha="c" * 64)
res["2. M2-type control: non-JSON bytes with a wrong sha STOP on the SHA message (sha before parse)"] = "differ from the pin" in msg(pool_label, cn, b"not json {")
res["2. controls: another A; an extra key; no 'route C' (PRELIMINARY kept); no PRELIMINARY; a malformed pointer: all STOP"] = all(
    msg(pool_label, *cfg_side(**kw)).startswith("STOP") for kw in (
        {"rsha": "b" * 64}, {"extra": {"note": "x"}}, {"label": REG_LABEL.replace("route C", "route X")},
        {"label": REG_LABEL.replace("PRELIMINARY", "FINAL")}, {"ptr": "git:short:experiments/x/A.json@" + "0" * 64}))

BADC = ("abc1234", "g" * 40, "A" * 40, "a" * 39, "a" * 41)  # a63c64c9 R1: the RIGHT sha, ONLY the commit malformed (short, non-hex, upper, 39, 41)


def only_commit_bad(pl, c_):
    cs, bs = cfg_side()
    return msg(pl, cs | {"a_label_sidecar": f"git:{c_}:experiments/x/results/A_label_sidecar.json@{hashlib.sha256(bs).hexdigest()}"}, bs)


res["2. R1 control: the right sha with ONLY the commit malformed (short, non-hex, upper-case, 39, 41 chars): all STOP on the pointer form"] = all(
    only_commit_bad(pool_label, c_).startswith("STOP STOP: A's label sidecar: not git:<40-hex>") for c_ in BADC)  # msg() prefixes "STOP "
MUTP = ('SIDECAR_PTR = r"git:([0-9a-f]{40}):', 'SIDECAR_PTR = r"git:([0-9a-fA-Fg-z]+):')
ns_m = {"hashlib": hashlib, "json": json}
for n in ast.parse(NEW.replace(*MUTP)).body:
    if (isinstance(n, ast.Assign) and {x.id for x in n.targets if isinstance(x, ast.Name)} & {"POOL_STEP", "SIDECAR_PTR"}) or (
            isinstance(n, ast.FunctionDef) and n.name == "pool_label"):
        exec(compile(ast.Module(body=[n], type_ignores=[]), "prm", "exec"), ns_m)
res["2. R1 positive control: with the 40-hex form loosened (mutant), the short-commit pointer is accepted: the form check is load-bearing"] = (
    NEW.count(MUTP[0]) == 1 and only_commit_bad(ns_m["pool_label"], BADC[0]).startswith("RET "))

mainf = next(n for n in ast.parse(NEW).body if isinstance(n, ast.FunctionDef) and n.name == "main")
mu_ = ast.unparse(mainf)
res["3. main: the route C guard (non-TEST, base_record, no 171-pool label, the pointer form), the unparsed sha-checked read, the remote call"] = (
    [ast.unparse(x) for x in ast.walk(mainf) if isinstance(x, ast.If) and "a_label_sidecar" in ast.unparse(x.test)] == [EXPECTED_REMOVED[0]]
    and "plant_rows.remote(c, blobs[0], blobs[1], ctx, side_)" in mu_)

CD = HERE / "config"
ok4 = True
for kind in ("FULL", "TEST_pos", "TEST_neg", "TEST_dd0neg"):
    src_ = json.loads((CD / f"plant_rows_POOLED_{kind}_2026-10-17_e2e.json").read_text())
    rc_txt = (CD / f"plant_rows_POOLED_{kind}_routeC.json").read_text()
    rc_ = json.loads(rc_txt)
    allowed = {"record"} | ({"base_record", "a_label_sidecar", "label"} if kind == "FULL" else set())
    same = {k: v for k, v in src_.items() if k not in allowed} == {k: v for k, v in rc_.items() if k not in allowed}
    pins = rc_["record"] == {"git": "TO_FILL", "sha256": "TO_FILL", "decompressed_sha256": "TO_FILL"} and (
        kind != "FULL" or (rc_["base_record"] == rc_["record"] and rc_["a_label_sidecar"] == "TO_FILL" and "171" not in rc_["label"]
                           and "route c" not in rc_["label"].lower()))  # O3: the route C label only ever comes from the sidecar
    ok4 = ok4 and same and pins and "171" not in rc_txt and "d8fcb945f" not in rc_txt and set(rc_) == set(src_) | ({"a_label_sidecar"} if kind == "FULL" else set())
res["4. the route C configs: the e2e configs except the record pins (TO_FILL), and FULL's sidecar pointer (TO_FILL) and pool-neutral label"] = ok4
for k, v in res.items():
    print(("PASS " if v else "FAIL ") + k)
print(f"PASS {all(res.values())} ({sum(res.values())}/{len(res)})")
sys.exit(0 if all(res.values()) else 1)
