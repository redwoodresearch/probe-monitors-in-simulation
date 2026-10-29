"""Lean tests of run_pilot_32k_crosscheck.py (synthetic records and npz only; no Modal call, no real record is read).

- crosscheck_core (09's ruling: pass = unit identity only): PASS on identical inside-32k units, boundary units excluded from both sides (a
  run of exactly 32,768 tokens keeps its unit at 32766: a63478 O5); the POSITIVE CONTROL (one planted unit-identity mismatch: a unit missing
  on the full-length side) FAILs, as do a changed call_index, an extra key on the full-length side (K5) and zero units compared (N0); a LARGE
  score difference (or NaN on one side) does NOT flip pass but DOES set the flag; a small one sets neither; every pre-condition STOPs:
  the 32k record in its LAUNCH-TIME format (2e's (a) fix: staged a path string, staged_sha, dump_manifest_sha256), one negative per term
  (staged sha, the 41a865a0e list format, the staged file name, the 32k run, the extraction record's run, the dump manifest sha, n_epochs),
  the registered staging's 20 epochs, 19 counted keys, a 32k npz with 19 keys or an extra key (N7), n_tokens != the inventory's full
  length (N6), a Volume file's sha256.
- to_volume (a63478 P2X-R2, option (b)): the record holds the buckets; the reply holds only booleans, counts and the record pointer; the
  record is written once. main()'s text holds no bucket or score field. crosscheck()'s body is exactly to_volume, the commit, and
  json.dumps of to_volume's return (a63478 P2X2-R1: a body returning crosscheck_core's output would put the buckets on the laptop).
- PINS: m32 / keys / inventory are run_simh_scores.PILOT_PINS "source" / "keys" / "inventory" (literal, by syntax tree).
- The unit rule is the B gate's, line for line (run_peraction311.py blob 0306186c8ca4, read with git cat-file; "cu if nocut else " aside).
- Rule 27: the script imports with __file__ = /root/<name>.py (one parent); the unguarded HERE.parents[1] fails there (mutant).

  uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy==2.3.3 python experiments/2026-10-17_p2_pilot_32k_crosscheck/test_pilot_32k_crosscheck.py
  (on an export that is not a git checkout: XCHECK_GIT_DIR=<a checkout holding the B gate blob> for the one git cat-file; without it that
  one check prints NOT RUN, the rest run, and the exit code is 2 when everything that ran passed)
"""

import ast
import hashlib
import io
import json
import os
import subprocess
import tempfile
import types
from pathlib import Path

import numpy as np

import run_pilot_32k_crosscheck as X

HERE = Path(__file__).resolve().parent
SRC = (HERE / "run_pilot_32k_crosscheck.py").read_text()
h = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
SS, ROOT = "5" * 64, "simh_q4b_extract_v1_pilot_c"
TASKS = [f"t{i}_task" for i in range(21, 41)]
RK = {t: f"2026-10-14_140858_simh_pilot/{t}.eval#e0" for t in TASKS}
LONG = set(TASKS[:5])  # runs longer than 32,768 tokens
EDGE = TASKS[5]  # a run of exactly 32,768 tokens: its unit at 32766 is NOT a boundary unit (the rule's "longer than 32,768"; a63478 O5)
FL = {t: 40000 if t in LONG else 32768 if t == EDGE else 20000 for t in TASKS}
TOK = {t: [100, 20000, 32766, 33000, 39000] if t in LONG else [100, 2000, 5000, 10000, 32766 if t == EDGE else 19000] for t in TASKS}  # 32766 on a
# long run: a boundary unit


def npz(**a):
    b = io.BytesIO()
    np.savez(b, **a)
    return b.getvalue()


def files(probe_i, plant=None, fl=None, tok=None):
    """(32k npz, full npz) for one probe. plant: {"score" (+5e-4) | "big" (+5.0) | "drop" | "cidx" | "nan": task} on its unit 0 (drop: unit 1);
    {"cut_drop": task}: the 32k file without that task; {"cut_extra": True} / {"full_extra": True}: a 21st key in that file (full: past 32,768)."""
    plant, fl, tok = plant or {}, fl or FL, tok or TOK
    rng = np.random.default_rng(probe_i)
    cut, full = {f: [] for f in ("key", "unit", "call_index", "score")}, {f: [] for f in ("key", "unit", "call_index", "score", "token", "n_tokens", "run_key")}
    for t in TASKS:
        for u, tk in enumerate(tok[t]):
            s = float(rng.normal())
            if not (plant.get("drop") == t and u == 1):
                full["key"].append(f"simh_pilot|{t}"), full["unit"].append(u), full["call_index"].append(2 * u + (1 if plant.get("cidx") == t and u == 0 else 0))
                full["score"].append(s + ({"score": 5e-4, "big": 5.0}.get(next((k_ for k_, v_ in plant.items() if v_ == t), ""), 0.0) if u == 0 else 0.0)
                                     if plant.get("nan") != t or u else float("nan"))
                full["token"].append(tk), full["n_tokens"].append(fl[t]), full["run_key"].append(RK[t])
            if tk < 32768 and plant.get("cut_drop") != t:  # the 32k render holds every unit before the cut, the boundary unit included
                cut["key"].append(f"simh_pilot|{t}"), cut["unit"].append(u), cut["call_index"].append(2 * u)
                cut["score"].append(s + (5.0 if tk >= 32765 and fl[t] > 32768 else 0.0))  # a boundary unit's 32k score differs
    if plant.get("cut_extra"):
        cut["key"].append("simh_pilot|t99_task"), cut["unit"].append(0), cut["call_index"].append(0), cut["score"].append(0.0)
    if plant.get("full_extra"):  # on a long run, past 32,768: no inside-32k unit changes, only the key set
        t0 = TASKS[0]
        full["key"].append("simh_pilot|t99_task"), full["unit"].append(0), full["call_index"].append(0), full["score"].append(0.0)
        full["token"].append(39500), full["n_tokens"].append(fl[t0]), full["run_key"].append(RK[t0])
    return npz(**{k: np.array(v) for k, v in cut.items()}), npz(**{k: np.array(v) for k, v in full.items()})


def world(plant=None, keys_n=20, staged=None, staged_sha=SS, run=ROOT, reg_run=ROOT, n_ep=20, dmsha=None, epochs=20, tamper_file=False, fl=None,
          tok=None, inv_off=None, verdict="written", mode="simh", out_dir="/simh/pilot/p2", m_world="simh_pilot"):
    """Synthetic records. The 32k record's dump is in the LAUNCH-TIME scorer's format (run_simh_scores.py @ 903ac2905 L322/L325): staged = the
    staged PATH (a string), staged_sha and meta_arm separate, dump_manifest_sha256 = the registered extraction record's own sha256."""
    names = [f"{p}__command.npz" for p in X.PROBES]
    vol, f32, ffl, fl = {}, {}, {}, fl or FL
    for i, n in enumerate(names):
        c, f = files(i, plant, fl, tok)
        vol[f"simh/pilot/p2/{n}"], vol[f"{X.OUT_FULL.strip('/')}/{n}"] = c, f
        f32[n], ffl[n] = h(c), h(f)
    if tamper_file:
        vol[f"simh/pilot/p2/{names[0]}"] += b"x"
    reg_x = json.dumps({"config": {"run": reg_run, "inputs": "simh_q4f_v1_pilot", "inputs_sha256": {"sim_honest@september.jsonl": SS}, "test": False}}).encode()
    blobs = {
        "m32": {"verdict": verdict, "files": f32 | {"fibs__reply_mean.json": "6" * 64},
                "dump": {"mode": mode, "out_dir": out_dir, "world": m_world, "n_epochs": n_ep, "run": run,
                         "staged": staged if staged is not None else f"{ROOT}/inputs/sim_honest@september.jsonl", "staged_sha": staged_sha,
                         "meta_arm": "sim_honest@september", "dump_manifest_sha256": dmsha or h(reg_x)}},
        "mfull": {"verdict": "written", "folders": {X.OUT_FULL: {"files": ffl | {"fibs__reply_mean.npz": "7" * 64}}}},
        "reg": {"simh": {"out": "simh_q4f_v1_pilot", "assembled": {"staged_inputs_sha256": {"sim_honest@september.jsonl": SS}, "epochs": {"sim_honest@september": epochs}}}},
        "keys": {"n": 20, "keys": [{"task": t, "world": "simh_pilot"} for t in TASKS[:keys_n]]},
        "inventory": {"runs": [{"key": RK[t], "full_len": fl[t] + (1 if t == inv_off else 0)} for t in TASKS]},
    }
    return {k: json.dumps(v).encode() for k, v in blobs.items()} | {"reg_x": reg_x}, vol


def run(**kw):
    blobs, vol = world(**kw)
    X.PINS = {n: ("c", "p", h(b)) for n, b in blobs.items()}
    try:
        return X.crosscheck_core(blobs, lambda rel: vol[rel])
    except SystemExit as e:
        return f"STOP {e}"
    except Exception as e:  # noqa: BLE001  (a63478 optional: a crash is a named result, never a STOP)
        return f"ERROR {type(e).__name__}"


REAL_PINS = dict(X.PINS)
n_in = 4 * (15 * 5 + 5 * 2)  # per probe: 15 short runs (the 32,768-token one included) x 5 units + 5 long runs x 2 (their boundary unit excluded)
ok = run()
B0 = {p: "0" for p in X.PROBES}
stop = lambda r: isinstance(r, str) and r.startswith("STOP")  # noqa: E731
P2_STOP = ("STOP STOP: the 32k record did not score the registered 32k pilot staging (its staged file and sha256, the registered run and "
           "extraction record)")  # the second pre-condition's message (run() prefixes "STOP ")
res = {
    "PASS: identical inside-32k units, boundary units excluded from both sides (their 32k scores differ by 5.0); a 32,768-token run's unit at "
    "32766 compared": ok == {"keys_equal": True, "n_units_compared": n_in, "n_mismatches": 0, "rel_diff_bucket": B0, "flag_rel_diff_over_0.1": False, "pass": True},
    "POSITIVE CONTROL: a planted unit-identity mismatch (a unit missing on the full-length side) FAILs": run(plant={"drop": "t30_task"}) == {
        "keys_equal": False, "n_units_compared": n_in - 4, "n_mismatches": 4, "rel_diff_bucket": B0, "flag_rel_diff_over_0.1": False, "pass": False},
    "a changed call_index on a matched unit FAILs (keys equal)": (lambda r: r["keys_equal"] and r["n_mismatches"] == 4 and not r["pass"])(
        run(plant={"cidx": "t30_task"})),
    "K5: an extra key on the full-length side (its unit past 32,768): keys_equal false, FAIL, no unit mismatch": (
        lambda r: not r["keys_equal"] and r["n_mismatches"] == 0 and r["n_units_compared"] == n_in and not r["pass"])(run(plant={"full_extra": True})),
    "N0: zero units compared (every unit in the boundary zone or past it): pass false": run(
        fl={t: 40000 for t in TASKS}, tok={t: [32765, 32766, 32767, 33000, 39000] for t in TASKS}) == {
        "keys_equal": True, "n_units_compared": 0, "n_mismatches": 0, "rel_diff_bucket": {p: "none" for p in X.PROBES}, "flag_rel_diff_over_0.1": False,
        "pass": False},
    "a LARGE score difference (+5.0) does NOT flip pass but DOES set the flag (every probe's bucket >0.1)": run(plant={"big": "t30_task"}) == {
        "keys_equal": True, "n_units_compared": n_in, "n_mismatches": 0, "rel_diff_bucket": {p: ">0.1" for p in X.PROBES}, "flag_rel_diff_over_0.1": True,
        "pass": True},
    "NaN on one side: pass unchanged, bucket nan, flag set": (lambda r: r["pass"] and r["flag_rel_diff_over_0.1"] and set(r["rel_diff_bucket"].values()) == {"nan"})(
        run(plant={"nan": "t30_task"})),
    "a small score difference (+5e-4): pass, no flag, a bucket at most (0.0001,0.001]": (lambda r: r["pass"] and not r["flag_rel_diff_over_0.1"] and set(
        r["rel_diff_bucket"].values()) <= {"(1e-05,0.0001]", "(0.0001,0.001]"})(run(plant={"score": "t30_task"})),
    "STOP (launch-time format): dump.staged_sha is not the registered stage record's sha256": run(staged_sha="4" * 64) == P2_STOP,
    "STOP (launch-time format): the 41a865a0e list format (staged = [{path, sha, meta_arm}])": run(
        staged=[{"path": f"{ROOT}/inputs/sim_honest@september.jsonl", "sha": SS, "meta_arm": "sim_honest@september"}]) == P2_STOP,
    "STOP (launch-time format): the staged file name is not sim_honest@september.jsonl": run(staged=f"{ROOT}/inputs/other.jsonl") == P2_STOP,
    "STOP (launch-time format): dump.run is not the registered root": run(run="other_root") == P2_STOP,
    "STOP (launch-time format): the extraction record's config.run is not the registered root (the 32k run is)": run(reg_run="other_root") == P2_STOP,
    "STOP (launch-time format): dump.dump_manifest_sha256 is not the registered extraction record's sha256": run(dmsha="3" * 64) == P2_STOP,
    "STOP (launch-time format): dump.n_epochs is not 20": run(n_ep=19) == P2_STOP,
    "STOP (a63478 PXA-R1): the 32k record's verdict is not written": run(verdict="gate failed: nothing written") == P2_STOP,
    "STOP (a63478 PXA-R1): dump.mode is not simh": run(mode="relaunch") == P2_STOP,
    "STOP (a63478 PXA-R1): dump.out_dir is not /simh/pilot/p2": run(out_dir="/simh/full/p2") == P2_STOP,
    "STOP (a63478 PXA-R1): dump.world is not simh_pilot": run(m_world="simh_full") == P2_STOP,
    "STOP: the registered staging is not 20 epochs": stop(run(epochs=19)),
    "STOP: 19 counted keys": stop(run(keys_n=19)),
    "N7: STOP: a 32k npz with 19 keys": run(plant={"cut_drop": "t30_task"}) == "STOP STOP: fibs__command.npz: the 32k file does not hold exactly the 20 pilot keys",
    "N7: STOP: a 32k npz with an extra key": run(plant={"cut_extra": True}) == "STOP STOP: fibs__command.npz: the 32k file does not hold exactly the 20 pilot keys",
    "N6: STOP: a run's n_tokens differs from the inventory's full length": run(inv_off="t30_task") == (
        "STOP STOP: fibs__command.npz: n_tokens differs from the pilot inventory's full length (or a run has none)"),
    "STOP: a Volume file differs from its record's sha256": stop(run(tamper_file=True)),
    "the bucket edges": [X.bucket(x) for x in (None, float("nan"), 0.0, 1e-13, 5e-7, 1e-5, 0.5)] == ["none", "nan", "0", "<=1e-12", "(1e-09,1e-06]",
                                                                                                    "(1e-06,1e-05]", ">0.1"],
}
with tempfile.TemporaryDirectory() as td_:  # P2X-R2: the buckets stay in the Volume record; only booleans, counts and its pointer cross back
    full_ = run(plant={"big": "t30_task"})
    rep_ = X.to_volume(full_, "2026-10-17_190000", Path(td_))
    body_ = (Path(td_) / rep_["result"]["path"]).read_bytes()
    try:
        X.to_volume(full_, "2026-10-17_190000", Path(td_))
        twice_ = False
    except FileExistsError:
        twice_ = True
res["P2X-R2: the record holds the buckets; the reply only booleans, counts and the record's pointer; written once"] = (
    list(rep_) == list(X.RELAY_KEYS) and json.loads(body_)["rel_diff_bucket"] == full_["rel_diff_bucket"] and h(body_) == rep_["result"]["sha256"]
    and rep_["result"] == {"volume": X.OUTVOL, "path": "2026-10-17_p2_pilot_32k_crosscheck/xcheck_2026-10-17_190000/result.json", "sha256": h(body_)}
    and all(type(v_) in (bool, int) for k_, v_ in rep_.items() if k_ != "result") and "bucket" not in json.dumps(rep_) and twice_)
main_src = next(ast.get_source_segment(SRC, n) for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "main")
res["P2X-R2 text check: main() names no bucket or score field and parses exactly one reply"] = (
    "bucket" not in main_src.lower() and "score" not in main_src.lower() and main_src.count("json.loads(") == 1 and "rel_diff_bucket" not in X.RELAY_KEYS)
CROSS_BODY = ["out = to_volume(crosscheck_core(blobs, lambda rel: (Path('/cw') / rel).read_bytes()), stamp, Path('/o'))", "o_vol.commit()",
              "return json.dumps(out)"]
CR_LINE = '    out = to_volume(crosscheck_core(blobs, lambda rel: (Path("/cw") / rel).read_bytes()), stamp, Path("/o"))\n'


def cross_body(src):  # P2X2-R1: crosscheck()'s statements after its docstring, unparsed
    f_ = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "crosscheck")
    doc_ = isinstance(f_.body[0], ast.Expr) and isinstance(f_.body[0].value, ast.Constant) and isinstance(f_.body[0].value.value, str)
    return [ast.unparse(s_) for s_ in f_.body[1 if doc_ else 0:]]


res["P2X2-R1: crosscheck() is exactly to_volume(...), o_vol.commit(), return json.dumps(<to_volume's return>); the mutant returning "
    "crosscheck_core's output directly (CR) fails"] = (
    cross_body(SRC) == CROSS_BODY and SRC.count(CR_LINE) == 1
    and cross_body(SRC.replace(CR_LINE, '    out = crosscheck_core(blobs, lambda rel: (Path("/cw") / rel).read_bytes())\n', 1)) != CROSS_BODY)
pp = ast.literal_eval(next(ast.unparse(n.value) for n in ast.parse((HERE.parent / "2026-10-14_p2_simh_scores" / "run_simh_scores.py").read_text()).body
                           if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "PILOT_PINS"))
res["PINS m32 / keys / inventory are the scorer's PILOT_PINS source / keys / inventory"] = (
    (REAL_PINS["m32"], REAL_PINS["keys"], REAL_PINS["inventory"]) == (pp["source"], pp["keys"], pp["inventory"]))
gd_ = subprocess.run(["git", "-C", os.environ.get("XCHECK_GIT_DIR") or str(HERE), "cat-file", "-p", "0306186c8ca4da6bc4451b518549ec312a8b9769"],
                     capture_output=True)  # a63478 O-a: no git dir holding the blob (an export) -> this one check is NOT RUN, named; the rest run
bg = gd_.stdout.decode().splitlines() if gd_.returncode == 0 else None
RULE = ['        key, rk, nt, tok = zu["key"].astype(str), zu["run_key"].astype(str), zu["n_tokens"].astype(np.int64), zu["token"].astype(np.int64)',
        '        rk = np.asarray([x_.removeprefix("runs/") for x_ in rk.tolist()])', "        inside = tok < 32768",
        "        excl = [(full_len.get(r, 0) > 32768) and (32765 <= t_ < 32768) for r, t_ in zip(rk.tolist(), tok.tolist())]",
        "        zkeys = {(k_, str(u)) for k_, u, r, t_ in zip(key.tolist(), zu[uid].astype(str).tolist(), rk.tolist(), tok.tolist())",
        "                 if full_len.get(r, 0) > 32768 and t_ >= 32765}",
        "        cu = Counter((k_, str(u)) for k_, u, i_, x_ in zip(key.tolist(), zu[uid].astype(str).tolist(), inside.tolist(), excl) if i_ and not x_)",
        '        cc = Counter((k_, str(u)) for k_, u in zip(zc["key"].astype(str).tolist(), zc[uid].astype(str).tolist()) if (k_, str(u)) not in zkeys)',
        '    full_len = {r["key"]: r.get("full_len", r.get("tokens_before")) for r in j["inventory"]["runs"]}  # run_gates L1951 (blob 0306186c8)']
B_AT = {0: 1519, 1: 1522, 2: 1581, 3: 1584, 4: 1585, 5: 1586, 6: 1588, 7: 1589, 8: 1951}
res["the unit rule is the B gate's line for line (blob 0306186c8ca4; cc without 'cu if nocut else ', full_len from the pinned inventory)"] = None if bg is None else (
    all(SRC.count(ln + "\n") == 1 for ln in RULE) and all(
        bg[B_AT[i] - 1].replace("cc = cu if nocut else Counter(", "cc = Counter(").replace('inv["runs"]', 'j["inventory"]["runs"]').split("  #")[0].rstrip()
        == ln.split("  #")[0].rstrip() for i, ln in enumerate(RULE)))


def at_root(src):
    m_ = types.ModuleType("rootcheck")
    m_.__file__ = "/root/run_pilot_32k_crosscheck.py"
    try:
        exec(compile(src, m_.__file__, "exec"), m_.__dict__)
        return "ok"
    except Exception as e:  # noqa: BLE001  (the type is the result)
        return type(e).__name__


GUARD = "REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE"
res["rule 27: imports at /root (one parent); the unguarded HERE.parents[1] fails there"] = (
    at_root(SRC) == "ok" and SRC.count(GUARD) == 1 and at_root(SRC.replace(GUARD, "REPO = HERE.parents[1]")) == "IndexError")
res["retries 0, an explicit timeout, 09's Volume read-only; no score tolerance in any spelling (09: scores descriptive only; a63478 O4)"] = (
    'retries=0, volumes={"/cw": cw_vol.read_only(), "/o": o_vol})' in SRC and "timeout=900" in SRC and "tol" not in SRC.lower().replace("tolist", ""))
for k, v in res.items():
    print(("NOT RUN " if v is None else "PASS " if v else "FAIL ") + k)
ran, nr = [v for v in res.values() if v is not None], [k for k, v in res.items() if v is None]
print(f"PASS {all(ran)} ({sum(ran)}/{len(ran)})" + (f"; NOT RUN {len(nr)}: {nr}" if nr else ""))
raise SystemExit(1 if not all(ran) else 2 if nr else 0)  # 2: every check that ran passed, but one or more did not run
