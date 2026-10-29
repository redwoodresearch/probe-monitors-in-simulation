"""Route C numbers verifier (09's pre-stated design, 10-20 PT): every printed number of each released artifact, parsed from the artifact and
recomputed from its committed, S2-signed source record by key through the consumer's own mapping, on Modal. Values never reach the laptop.

The laptop reads the config (keys and pins only), sends each artifact's and source's git bytes UNPARSED with their pinned sha256, and the
consumer code by git blob id. The container checks every sha before it parses, builds one JSONL row per printed number (verify_core.row),
writes the JSONL to this check's Volume, commits it, and returns COUNTS, slot ids, condition names, shas and the 20-row recheck selection
(seeded) only; the reply is whitelisted by type before it leaves. The laptop writes that reply as results/numbers_verify_<stamp>.json.

  uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy==2.3.3 modal run \\
      experiments/2026-10-20_routec_numbers_verify/run_numbers_verify.py --only tables34,rc2
  (from the repo root of the worktree). The v84 pass: --only tables34,rc1a,rc2,rc1b,post (the post runs last, on those artifacts' rows).
  check_payload.py resolves a run's pins to git bytes on the laptop without any Modal call (counts and bytes only).
"""

import gzip
import hashlib
import io
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_core as VC  # noqa: E402

APP = "probe-shift_aswin-surya_sub-routec-numv_check_kill-2026-11-19"  # 61 characters
OUT_VOL = "probe-shift_aswin-surya_sub-routec-numv_rows_kill-2026-11-19"  # 60
SOL_VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"  # A's D1 calibration files and pooled pools files (A's meta pins)
CONFIG = HERE / "config" / "artifacts_routec_2026-10-20.json"
RESULTS = HERE / "results"
SLOT_ID_RX = re.compile(r"^[A-Za-z0-9 _|.,:@+()/'<>=*#\[\]-]{1,200}$")

PURE = ("verify_core.py", "verify_post.py", "predicates.py")  # shipped into the image and sha-checked against the payload in the container
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "matplotlib==3.10.6",  # render_all's image (figures)
                                                                       "zstandard==0.25.0")  # run_examples_v81's .eval reader (C3)
for _f in PURE:
    image = image.add_local_file(HERE / _f, f"/root/{_f}")
app = modal.App(APP, image=image)
sol = modal.Volume.from_name(SOL_VOL, version=2).read_only()
conv = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14").read_only()  # run_tables_v81.VOLS["/conv"]
p2 = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13").read_only()  # run_tables_v81.VOLS["/p2"]
CMD_VOLS = {"/b": "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14", "/q4": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13",
            "/acts": "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14",  # run_examples_v81.VOLS (staged rows, .eval files)
            "/q3": "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13"}  # run_tables_v81.VOLS["/q3"] (A25 sim-honest .eval files)
cmd_vols = {k: modal.Volume.from_name(v).read_only() for k, v in CMD_VOLS.items()}
MOUNT = {"conv": "/conv", "p2": "/p2", "solavg": "/sol"}  # run_tables_v81.MOUNT: a score file's `volume` -> its mount
out_vol = modal.Volume.from_name(OUT_VOL, create_if_missing=True)


# ---------- container ----------

def _bytes_ok(x, role):
    """git bytes (gz decompressed in the container) whose sha256 must be the pin BEFORE any parse."""
    b = x["bytes"]
    if x["path"].endswith(".gz"):
        b = gzip.decompress(b)
    if VC.sha(b) != x["sha256"]:
        raise VC.Stop(f"{role}: sha256 differs from its pin (STOP)")
    return b


def _code(code, name):
    c = code[name]
    if hashlib.sha1(b"blob %d\0" % len(c["bytes"]) + c["bytes"]).hexdigest() != c["blob"]:
        raise VC.Stop(f"{name}: bytes are not git blob {c['blob']} (STOP)")
    return c["bytes"]


def _vol_npz(path, want_sha, what, mount="/sol"):
    import numpy as np

    b = (Path(mount) / path.lstrip("/")).read_bytes()
    if VC.sha(b) != want_sha:
        raise VC.Stop(f"{what}: the Volume file does not hash to its pin (STOP)")
    return np.load(io.BytesIO(b), allow_pickle=False), VC.sha(b)



def _attack(T, a, d1_npz, unit_z, argmax_command, lookup_cell, e12_cell, sp_alias, pins):
    """09's ruling (b) for each drawn attack side of Table 4: FROZEN's pin for its score file (by the record's path), the file read from its
    Volume and sha-checked before parsing, the run's units (attack_units); their file keys must be the drawn cell (lookup_cell == e12_cell(c))
    and this side's arm by the consumer's alias rule (world == arm, arm == world + "_e2", or (world, arm) in rp's SP_WORLD_ALIAS) (O2);
    unit_z with the outcome label's D1 draw-0 moments, argmax. A failure is kept by class only (R5, O1); the side then STOPs."""
    out = {}
    for oc, d in T["outcome_table"]["outcomes"].items():
        if d.get("drawn") in (None, "none"):
            continue
        c = d["drawn"]["cell"].split("|", 1)[1]
        for side, nm in VC.SIDES:
            v = d["drawn"][side]
            if v == "no command":
                continue
            arm = "run1" if nm == "B" else "sim"
            ent = {"frozen_ok": False, "score_ref": "-", "result": None, "why": "", "file_cells_ok": False, "file_arm_ok": False}
            try:
                sf = v["score_file"]
                shas = pins.get(sf["path"].lstrip("/"), set())
                ent["score_ref"] = f"vol:{sf['volume']}:{sf['path']}@{next(iter(shas)) if len(shas) == 1 else '?'}"
                ent["frozen_ok"] = len(shas) == 1 and shas == {sf["sha256"]}
                if len(shas) != 1:
                    raise VC.Stop("not exactly one FROZEN pin for the score file")
                npz, _ = _vol_npz(sf["path"], next(iter(shas)), "attack score file", MOUNT[sf["volume"]])
                mom, fm, folds = VC.d1_moments(d1_npz[VC.OUTCOME_LABEL], nm)
                units = VC.attack_units(npz, v["run_key"], 5 if fm else 1)
                ent["file_cells_ok"] = all(lookup_cell(k_) == e12_cell(c) for k_ in units["file_keys"])
                ent["file_arm_ok"] = all((w == arm or arm == w + "_e2" or (w, arm) in sp_alias) for w in (k_.split("|", 1)[0] for k_ in units["file_keys"]))
                ent["result"] = VC.attack_argmax(units, mom, fm, folds, unit_z, argmax_command)
            except (Exception, SystemExit) as e:  # noqa: BLE001  O1: kept by class only; the side's rows STOP as unmatched / orphan
                ent["why"] = f"attack recompute failed: {type(e).__name__}"
            out[f"{oc}|{side}"] = ent
    return out


def _fa_units(T, d1_npz, pins):
    """RQ2: each false alarm's run in its FROZEN-pinned score file (the draw's score_file path looked up in FROZEN's chain, sha-checked
    before parsing), by run_key, read as FileIndex reads it; tables34_rows then requires its scores to equal the pool's for the alias."""
    out = {}
    for i, d in enumerate(T["false_alarms"]["draws"], 1):
        ent = {"frozen_ok": False, "score_ref": "-", "units": None, "why": ""}
        try:
            sf = d["score_file"]
            shas = pins.get(sf["path"].lstrip("/"), set())
            ent["score_ref"] = f"vol:{sf['volume']}:{sf['path']}@{next(iter(shas)) if len(shas) == 1 else '?'}"
            ent["frozen_ok"] = len(shas) == 1 and shas == {sf["sha256"]}
            if len(shas) != 1:
                raise VC.Stop("not exactly one FROZEN pin for the score file")
            npz, _ = _vol_npz(sf["path"], next(iter(shas)), "false-alarm score file", MOUNT[sf["volume"]])
            _, fm, _ = VC.d1_moments(d1_npz[d["probe"]], VC.POOL_NM[d["pool"]])
            ent["units"] = VC.attack_units(npz, d["run_key"], 5 if fm else 1)
        except (Exception, SystemExit) as e:  # noqa: BLE001  kept by class only; the draw's token row STOPs
            ent["why"] = f"false-alarm score rows failed: {type(e).__name__}"
        out[i] = ent
    return out


def _tables34(a, signed):
    import ast as ast_
    import types

    T = json.loads(_bytes_ok(a["artifact"], "tables record"))
    A = json.loads(_bytes_ok(a["sources"]["A"], "A"))
    rp = _code(a["code"], "run_peraction311.py")
    rt = _code(a["code"], "run_tables_v81.py")
    stops0 = []
    if VC.sha(rp) != A["meta"].get("script_sha256") or VC.sha(rt) != T.get("script_sha256"):  # O4: the producers' own script shas
        stops0.append("tables34: the shipped runner or tables script is not the one that wrote A or the tables record")
    wthr = VC.extract_function(rp, "wthr")
    e12_cell = VC.extract_function(rp, "e12_cell")
    sp = [n for n in ast_.parse(rp).body if isinstance(n, ast_.Assign) and any(getattr(t, "id", "") == "SP_WORLD_ALIAS" for t in n.targets)]
    if len(sp) != 1:
        raise VC.Stop("rp: not exactly one SP_WORLD_ALIAS (STOP)")
    sp_alias = ast_.literal_eval(sp[0].value)
    unit_z = VC.extract_function(rt, "unit_z")
    lookup_cell = VC.extract_function(rt, "lookup_cell", {"_R": types.SimpleNamespace(e12_cell=e12_cell)})
    argmax_command = VC.extract_function(_code(a["code"], "run_examples_v81.py"), "argmax_command")
    pw = A["meta"]["pooled_world"]
    d1, d1_npz, pools, d1_ref, d1_sha = {}, {}, {}, {}, {}
    for lb in VC.FA_PROBES:
        pin = pw["calibrations"]["by_label_combo"][f"{lb} | {VC.LEAD}"]
        z, s_ = _vol_npz(pin["path"], pin["sha256"], f"D1 {lb}")
        d1_npz[lb] = z
        d1[lb] = {"sha256_read": s_, "path": pin["path"], "z": {nm: z[f"draw0_z__{nm}"] for nm in ("B", "S")}}
        d1_ref[lb], d1_sha[lb] = f"vol:sol:{pin['path']}@{pin['sha256']}", pin["sha256"]
        pp = pw["manifest_per_label"][f"{lb} | {VC.LEAD}"]
        p_, _ = _vol_npz(pp["path"], pp["sha256"], f"pools {lb}")
        pools[lb] = {nm: p_[f"{nm}_run"] for nm in ("B", "S")} | {f"{nm}_{x}": p_[f"{nm}_{x}"] for nm in ("B", "S") for x in ("s", "sbm")}
    frozen = json.loads(_bytes_ok(a["sources"]["FROZEN"], "FROZEN"))
    pins = VC.score_pins(frozen, a["overrides"])
    attack = _attack(T, a, d1_npz, unit_z, argmax_command, lookup_cell, e12_cell, sp_alias, pins)
    fa_units = _fa_units(T, d1_npz, pins)
    refs = {"A": f"{a['sources']['A']['commit']}:{a['sources']['A']['path']}", "A_sha": a["sources"]["A"]["sha256"], "d1": d1_ref,
            "d1_sha": d1_sha, "frozen_sha": VC.sha(a["sources"]["FROZEN"]["bytes"]), "not_printed_approved": a.get("not_printed_approved")}
    rows, plan, stops, dg = VC.tables34_rows(T, A, d1, pools, wthr, refs, signed | set(d1_sha.values()), attack, fa_units)
    SHARED["tables34"] = {"kind": "tables34", "sha": a["artifact"]["sha256"], "T": T, "not_printed": dg["not_printed"]["paths"]}  # for the post
    return rows, plan, stops0 + stops, {
        "d1_conditions_failed": {k: v for k, v in dg["conditions"].items() if v}, "d1_labels_pools_checked": len(dg["conditions"]),
        "attack_sides": len(attack), "attack_sides_failed": sorted(k for k, e in attack.items() if e["result"] is None),
        "attack_sides_failed_class": sorted({e["why"] for e in attack.values() if e["why"]}),
        "attack_sides_not_frozen_pinned": sorted(k for k, e in attack.items() if not e["frozen_ok"]),
        "false_alarm_score_rows_failed": sorted(i for i, e in fa_units.items() if e["units"] is None),
        "not_printed": dg["not_printed"], "numeric_leaves": dg["numeric_leaves"], "scope": dg["scope"],
        "printed_numbers_verified": dg["printed_numbers_verified"]}


def _slots(name, a, signed):
    R = json.loads(_bytes_ok(a["artifact"], f"{name} readout"))
    if R.get("code", {}).get("blob", {}).get("slot_table.py") != a["code"]["slot_table.py"]["blob"]:  # the producer's own code pin
        raise VC.Stop("the readout was not produced by the configured slot_table blob (STOP)")
    recs, shas, role_ref = {}, {}, {}
    for role, x in a["sources"].items():
        b = _bytes_ok(x, role)
        if R["inputs"].get(role, {}).get("sha256") != x["sha256"]:
            raise VC.Stop("a readout input sha is not the committed source's (STOP)")
        recs[role], shas[role], role_ref[role] = json.loads(b), x["sha256"], f"{x['commit']}:{x['path']}"
    if set(R["inputs"]) != set(recs):
        raise VC.Stop("the readout's inputs are not the configured sources (STOP)")
    ST = VC.instrumented_slot_table(_code(a["code"], "slot_table.py"))
    paths = {r: R["inputs"][r]["path"] for r in recs}  # the readout's own pointer paths (they feed only the label rule's name check)
    ST._NV_ABS.update({id(recs[r]): (r,) for r in recs})
    rebuilt = ST.build(recs, shas, paths, control=False)
    rows, plans, stops = VC.slots_rows(name, R["slots"], rebuilt, ST._NV_LOG, recs, role_ref, shas, signed)
    raw = {}  # the post's predicates read a slot's unrounded point: the first number of its "full" string, only where it equals the re-run's
    for k_, f_ in (R.get("full") or {}).items():
        n_ = VC.NUM_RX.findall(f_) if isinstance(f_, str) and f_ == rebuilt.get("full", {}).get(k_) else []
        raw[k_] = float(n_[0].replace(VC.MINUS, "-").replace(",", "")) if n_ else None
    SHARED[name] = {"kind": "slots", "sha": a["artifact"]["sha256"], "slots": R["slots"], "raw": raw}
    kinds = {}
    for r in rows:  # how each slot was checked: summ / ci / leaf = string AND raw numbers by key; derived / text = string only
        if r["unit"] == "slot":
            k_ = r["rule"].split("/")[-1]
            kinds[k_] = kinds.get(k_, 0) + 1
    return rows, plans, stops, {"rebuilt_missing": len(rebuilt["missing"]), "rebuilt_not_carried": sorted(rebuilt["not_carried"]),
                                "slots_by_check_kind": kinds}


def _figrender(name, a, signed):
    """A figure group re-rendered by its own committed scripts (the render log's plan and per-figure args, the render driver's env, the
    drawing commit's blobs, the plan's records from their committed copies), in a fresh folder as render_all._draw does; then each
    figure's committed _source.json and _text.md are compared number by number with the re-render, and every plotted value with the
    record read here by its logged key (verify_core.fig_rows)."""
    import os
    import subprocess
    import tempfile

    log = json.loads(_bytes_ok(a["render_log"], f"{name} render log"))
    pl = log["plan"]
    recs_b = {base: _bytes_ok(x, f"{name} record {base}") for base, x in a["records"].items()}
    if set(pl["records"]) != set(recs_b):
        raise VC.Stop(f"{name}: the plan's records are not the configured ones (STOP)")
    root = Path(tempfile.mkdtemp()) / "exp"
    (root / "results").mkdir(parents=True)
    for n_ in a["code"]:
        (root / n_).write_bytes(_code(a["code"], n_))
    for base, b in recs_b.items():
        (root / "results" / base).write_bytes(b)
    env = {"FIG_COMPARATOR_OVERRIDES": json.dumps(pl.get("comparator_overrides") or {}), "FIG_DRAW_EXTRA": json.dumps(pl.get("draw_extra") or []),
           "FIG_UNION_NOTE": pl.get("union_note") or "", "FIG_TITLE_BANNER": pl.get("title_banner") or "", "FIG_BUILD": json.dumps(pl.get("build") or {}),
           "FIG_KEEP_GOING": "1" if pl.get("keep_going") else "", "FIG_PARALLEL": "1" if pl.get("parallel") else "",
           "FIG_COMPARATOR": pl.get("comparator") or "",
           "FIG_ESTIMATOR": (json.dumps(pl["estimator"]) if isinstance(pl.get("estimator"), dict) else pl.get("estimator") or "")}  # render_all L181-188
    before = {p.name for p in (root / "results").iterdir()}
    rcs = []
    for i, fg in enumerate(log["logs"]):
        r = subprocess.run([sys.executable, fg["script"], *fg["args"], "--git", a["scripts_commit"]], cwd=root, capture_output=True, text=True,
                           env=os.environ | env)
        rcs.append(r.returncode)
        if OUT_DIR[0] is not None:  # the scripts' own output (it may carry values): the Volume only, never the reply
            (OUT_DIR[0] / f"render_{name}_{i}.log").write_text(f"rc={r.returncode}\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr}\n")
    outs = {p.name: p.read_bytes() for p in (root / "results").iterdir() if p.name not in before}
    stops = [f"{name}: re-render exit codes {rcs}"] if any(rcs) else []
    want = {f"{s}{x}" for s in a["figures"] for x in (".png", "_source.json", "_text.md")}
    if log.get("script_commit") != a["scripts_commit"] or set(log.get("outputs", {})) != want:  # O3: the render log is this figure set's
        stops.append(f"{name}: the render log's script commit or output files are not the configured ones")
    rec_parsed = {base: json.loads(b) for base, b in recs_b.items()}
    rec_sha = {base: a["records"][base]["sha256"] for base in recs_b}
    rows, plans, diag = [], {}, {"render_exit_codes": rcs}
    for stem in a["figures"]:
        ps = json.loads(_bytes_ok(a["outputs"][f"{stem}_source.json"], f"{stem} source"))
        if ps.get("git") != a["scripts_commit"]:  # the producer's own pin: the committed figure was drawn by the scripts re-run here
            stops.append(f"{name}:{stem}: the committed source JSON was not drawn at the configured scripts commit")
        pt = _bytes_ok(a["outputs"][f"{stem}_text.md"], f"{stem} text").decode()
        rs_b, rt_b = outs.get(f"{stem}_source.json", b"{}"), outs.get(f"{stem}_text.md", b"")
        r_, p_, s_ = VC.fig_rows(f"{name}:{stem}", ps, json.loads(rs_b), pt, rt_b.decode(), rec_parsed, rec_sha, signed)
        rows += r_
        stops += s_
        plans[f"{name}:{stem}"] = p_
        # RQ1 (09): the PNG's printed numbers are checked by byte identity with the container's re-render from verified values
        ps_png, pd_png = VC.png_gate(f"{name}:{stem}", _bytes_ok(a["outputs"][f"{stem}.png"], f"{stem} png"), outs.get(f"{stem}.png", b""))
        stops += ps_png
        diag[stem] = {"source_bytes_identical": VC.sha(rs_b) == a["outputs"][f"{stem}_source.json"]["sha256"],
                      "text_bytes_identical": VC.sha(rt_b) == a["outputs"][f"{stem}_text.md"]["sha256"]} | pd_png
    return rows, plans, stops, diag


def _consumer_x(code):
    """run_examples_v81 (blob-checked) imported from its bytes, with run_tables_v81's find_eval_v2 (its EVAL_ROOTS, incl. the A25 sim-honest
    root) as the locator's .eval search, as run_tables_v81._tables sets it."""
    import ast as ast_
    import importlib.util

    d = Path("/tmp/nv_consumer")
    d.mkdir(exist_ok=True)
    (d / "run_examples_v81.py").write_bytes(_code(code, "run_examples_v81.py"))
    spec = importlib.util.spec_from_file_location("run_examples_v81", d / "run_examples_v81.py")
    X = importlib.util.module_from_spec(spec)
    sys.modules["run_examples_v81"] = X
    spec.loader.exec_module(X)
    rt = _code(code, "run_tables_v81.py")
    er = [n for n in ast_.parse(rt).body if isinstance(n, ast_.Assign) and any(getattr(t, "id", "") == "EVAL_ROOTS" for t in n.targets)]
    if len(er) != 1:
        raise VC.Stop("run_tables_v81: not exactly one EVAL_ROOTS (STOP)")
    X.find_eval = VC.extract_function(rt, "find_eval_v2", {"EVAL_ROOTS": ast_.literal_eval(er[0].value), "VOLS": dict(X.VOLS) | CMD_VOLS,
                                                           "X": X, "Path": Path, "hashlib": hashlib})
    return X


def _post(name, a, signed):
    """The v84 pass (09's design, 10-20): the frozen post's every number token, classed and checked by verify_post.post_rows against this
    run's verified artifact rows (tables34 and the readouts must run first, in the same run), the signed records, the configured
    not-route-C records, writing's 09-approved maps, 09's pinned patterns and header labels, and the swap re-run (positions tracked).
    Every input is sha-checked before it is parsed."""
    import tempfile

    import verify_post as VP

    text_b = _bytes_ok(a["artifact"], "post")
    text = text_b.decode("utf-8")
    inp = {k: _bytes_ok(x, f"post input {k}") for k, x in a["inputs"].items()}
    missing = [n for n in a["requires"] if n not in SHARED]
    if missing:
        raise VC.Stop(f"post: the artifacts it reads were not verified in this run: {','.join(sorted(missing))} (STOP)")
    jl = lambda b: [json.loads(x) for x in b.decode("utf-8").splitlines() if x.strip()]  # noqa: E731
    prov, dfr, cons, csrc = jl(inp["provenance"]), jl(inp["deferred"]), jl(inp["constants"]), jl(inp["constants_sources"])
    recs, rec_ok = {}, set()
    for x in a["records"]:  # every cited record: bytes hashed to the pin before parsing
        b = _bytes_ok(x, "post record")
        rec_ok.add(x["sha256"])
        recs[x["sha256"]] = json.loads(b)
    stops = []
    acc = json.loads(inp["accepted_reasons"])
    if not (isinstance(acc, list) and all(isinstance(x, str) for x in acc)):  # O6
        raise VC.Stop("post: 09's accepted reasons are not a list of strings (STOP)")
    pats = VP.compile_patterns(json.loads(inp["patterns"]))
    hl = json.loads(inp["header_labels"])
    if (hl.get("post") or {}).get("sha256") != VC.sha(text_b):
        stops.append("post: the header-label list was not built from this post")
    # 09's approved tolerance table must be this provenance's: each row's op and non-expected arguments, row by row
    tol = json.loads(inp["tolerances"])
    # value_string (op 23) is masked in 09's table too; 09 (10-20, table v15): component_string (op 23) and expected (op 25) as well,
    # never size or level (design constants 09 reads); the same set as tools/build_tolerance_table.sh
    mask = {"value", "values", "target", "min", "max", "value_string", "component_string", "expected"}
    want = [{"row": p.get("row"), "op": (p.get("predicate") or {}).get("op"),
             "args": {k: ("<expected value: checked in the container>" if k in mask else v) for k, v in (p.get("predicate") or {}).items()
                      if k not in ("op", "note", "report")}} for p in prov]
    got = [{k: r.get(k) for k in ("row", "op", "args")} for r in tol.get("rows") or []]
    if want != got or len(tol.get("closed_set") or []) != len(VP.PR.OPS) or set(tol["closed_set"]) != set(VP.PR.OPS):
        stops.append("post: 09's tolerance table is not this provenance's (op, arguments, row order) or not the code's closed set")
    # 2e / 09 (10-20), O1: writing's later label-only drafts (0240, 0256) must equal this post with hidden comments stripped (STOP on False);
    # the pass itself, and the header-label positions, stay on this post
    label_same = {}
    for k_ in sorted(x for x in inp if x.startswith("label_update")):
        label_same[k_] = VP.HIDDEN.sub("", inp[k_].decode("utf-8")) == VP.HIDDEN.sub("", text)
        if not label_same[k_]:
            stops.append(f"post: {k_}: the later draft's visible text differs from this post's")
    # the swap: Run A reproduces the swap-only output, the hand-edit diff gives the frozen post, Run B on the post reproduces it
    w = Path(tempfile.mkdtemp())

    def kept(fn, *x):  # a failure here is a STOP with its class only, and the pass goes on (its numbers then fall to other classes)
        try:
            return fn(*x), ""
        except (Exception, SystemExit) as e:  # noqa: BLE001
            return None, type(e).__name__

    ra, err_a = kept(VP.rerun_apply, inp["apply"], inp["rules"], inp["draft_in"], inp["merged_slots"], w / "A")
    out_a, log_a, spans_a, info_a = ra if ra else (b"", [], [], {"unattributable": 0, "spans_lost": 0, "outside_changes": 0,
                                                                 "output_is_final_text": None})
    run_a_ok = not err_a and VC.sha(out_a) == a["inputs"]["swaponly"]["sha256"]
    if not run_a_ok:
        stops.append(f"post: Run A (apply_v84 on the draft and the merged slots) does not reproduce the committed swap-only output {err_a}")
    if info_a["unattributable"]:
        stops.append(f"post: Run A made {info_a['unattributable']} changes the observer cannot attribute (no rule positions kept)")
    if ra and not info_a["output_is_final_text"]:
        stops.append("post: Run A's written output is not its final text (no rule positions kept)")
    swap_txt = inp["swaponly"].decode("utf-8")
    dm, err_d = kept(VP.apply_diff_map, swap_txt, inp["hand_diff"].decode("utf-8"))
    tgt, keep, added, bad_h = dm if dm else ("", {}, [], 0)
    if err_d or tgt.encode("utf-8") != text_b:
        stops.append(f"post: the hand-edit diff does not turn the swap-only output into the frozen post {err_d}")
    if bad_h:
        stops.append(f"post: {bad_h} hand-edit diff hunk headers disagree with their bodies")
    rb, err_b = kept(VP.rerun_apply, inp["apply"], inp["rules"], text_b, inp["drop1b_slots"], w / "B")
    out_b, log_b = (rb[0], rb[1]) if rb else (b"", [])
    if err_b or out_b != text_b:
        stops.append(f"post: Run B (apply_v84 on the frozen post and the drop-1b slots) does not reproduce it {err_b}")
    # R3: the positions Run A wrote, in the swap-only text, mapped through the diff's kept lines into the post
    claims = []
    if run_a_ok and dm and not err_d:
        for s, e, x in VP.map_spans(swap_txt, keep, tgt, spans_a):
            claims.append({"rule": log_a[x]["rule"], "keys": log_a[x]["keys"], "inserted": log_a[x]["inserted"], "start": s, "end": e})
    merged = json.loads(inp["merged_slots"])["slots"]
    rd = {n: s for n, s in SHARED.items() if s["kind"] == "slots"}
    bad_m = [k for k, v in merged.items() if not any(rd[n]["slots"].get(k) == v for n in ("rc1a", "rc2") if n in rd)]
    d1b = json.loads(inp["drop1b_slots"])["slots"]  # first-run finding: 18 of its slots are rc1a / rc2 slots, equal to the verified strings
    bad_d = [k for k, v in d1b.items() if (rd.get("rc1b", {}).get("slots", {}).get(k, v) != v)  # a key rc1b has must be rc1b's own string
             or not any(rd[n]["slots"].get(k) == v for n in ("rc1b", "rc1a", "rc2") if n in rd)]
    d1b_from_rc1a_rc2 = sum(1 for k in d1b if "rc1b" not in rd or k not in rd["rc1b"]["slots"])
    if bad_m or bad_d:
        stops.append(f"post: {len(bad_m)} merged slots are not rc1a / rc2's and {len(bad_d)} drop-1b slots are not a verified readout's (rc1b, rc1a, rc2)")
    # deferred rows: record_ok only for a non-null record whose every entry was read here by sha, and a code-map region that resolves
    cm = json.loads(inp["code_map"])
    cm_ref = f"{a['inputs']['code_map']['path']}@{a['inputs']['code_map']['commit']}"
    d_ok, cohort = {}, {"cited": 0, "resolved": 0}
    for j, r in enumerate(dfr):
        rr = r.get("record")
        ok = isinstance(rr, list) and bool(rr) and all(isinstance(e, dict) and e.get("sha256") in rec_ok for e in rr)
        ok = ok and isinstance(r.get("code_map"), str) and r["code_map"].startswith(cm_ref) and r.get("code_map_region") in cm
        if r.get("medians_cohort_block"):  # the App G rows: the cited cohort block must resolve by key in a cited record
            cohort["cited"] += 1
            m = re.match(r"\.results\['([^']+)'\]\.pooled", r["medians_cohort_block"])
            hit = bool(m) and any(isinstance(((recs.get(e.get("sha256")) or {}).get("results") or {}).get(m.group(1)), dict)
                                  and "pooled" in recs[e["sha256"]]["results"][m.group(1)] for e in rr or [])
            cohort["resolved"] += hit
            ok = ok and hit
        d_ok[j] = ok
    # constants: the approved sources file joined by row (its literal sha binds each row); each row at its own anchor (R4)
    src_by = {s["row"]: s for s in csrc}
    add_by = {x["constants_row"]: x for x in jl(inp["constants_sources_additions"])} if "constants_sources_additions" in inp else {}
    if set(add_by) & set(src_by):
        stops.append("post: a constants row has both a final sources row and an additions row")
    constants = []
    for c in cons:
        s_ = src_by.get(c["row"]) or {}
        bound = s_.get("text_sha256") == VC.sha(c["text"].encode())
        sourced = s_.get("class") != "constant" or any(x.get("contains_literal") or x.get("contains_value") for x in s_.get("sources") or [])
        cls_ = s_.get("class")
        if c["row"] in add_by:  # 09 (10-20): rows the approved additions file sources (its literal must be the row's; a primary source cited)
            x_ = add_by[c["row"]]
            s_, bound, sourced = x_, x_.get("text") == c["text"], isinstance(x_.get("primary"), dict) and bool(x_["primary"])
            cls_ = c.get("kind") if c.get("kind") in ("constant", "definitional", "citation") else None
        constants.append({"row": c["row"], "text": c["text"], "class": cls_, "anchor": c.get("anchor"),
                          "anchor_sentence": c.get("anchor_sentence"), "occurrence_of_literal": c.get("occurrence_of_literal"),
                          "checks_ok": bool(s_) and bound and sourced})
    PINS["accepted_reasons"] = acc
    PINS["nonresult_approved"] = a.get("nonresult_approved")
    rows_so_far = RUN["rows"]
    slot_rows = {}
    for r in rows_so_far:
        if r["artifact"] in rd and "#" in r["slot"]:
            slot_rows.setdefault((r["artifact"], r["slot"].rsplit("#", 1)[0]), []).append(r)
    counter, X_ = {}, []

    def command_check(side, form, n):
        if not X_:
            X_.append(_consumer_x(a["code"]))
        return VP.command_equal(side, form, n, X_[0], counter)

    ctx = {"text_sha": VC.sha(text_b), "provenance": prov, "deferred": dfr, "deferred_record_ok": d_ok, "constants": constants,
           "hyp_exprs": json.loads(inp["hyp_exprs"])["rows"], "rule_claims": claims, "released_all": merged,
           "readout_of_slot": {k: [n for n in rd if k in rd[n]["slots"]] for n0 in rd for k in rd[n0]["slots"]},
           "verified": {(r["artifact"], r["slot"]): r for r in rows_so_far}, "slot_rows": slot_rows,
           "by_jpath": {(r["artifact"], r["jpath"]): r for r in rows_so_far if r.get("jpath")},
           "artifact_of_sha": {s["sha"]: n for n, s in SHARED.items()}, "kind_of": {n: s["kind"] for n, s in SHARED.items()},
           "released": {n: s["slots"] for n, s in rd.items()}, "slot_raw": {n: s["raw"] for n, s in rd.items()},
           "records": recs, "signed": signed, "not_route_C": set(a["not_route_C"]), "T": SHARED["tables34"]["T"],
           "not_printed": SHARED["tables34"]["not_printed"], "not_printed_v84_approved": a.get("not_printed_v84_approved"),
           "command_check": command_check, "added_spans": added, "orphan_notes": a.get("orphan_notes") or [], "patterns": pats,
           "header_labels": hl.get("entries") or [], "accepted_reasons": acc,
           "upstream_sources": {x for r in rows_so_far if not r["orphan"] for x in r["sources"]},
           "literals": VP.compile_literals(json.loads(inp["nonresult_literals"])) if "nonresult_literals" in inp else None,
           "preliminary": set(a.get("preliminary") or []), "row_scope_notes": a.get("row_scope_notes") or {},
           "preliminary_labels": a.get("preliminary_labels") or {},
           "deferred_record_checks": a.get("deferred_record_checks") or []}
    if "numv5_result" in inp:  # 09 (10-20): numv5's number-word POSITIONS only (its sentences are never read), on this same post text
        nv5, x5 = json.loads(inp["numv5_result"]), a["inputs"]["numv5_result"]
        if nv5.get("pins", {}).get("post", {}).get("artifact", {}).get("sha256") != a["artifact"]["sha256"]:
            raise VC.Stop("post: the pinned numv5 result was not run on this post text (STOP)")
        ctx["numv5_positions"] = [int(x["position"]) for x in nv5["diagnostics"]["post"]["number_words"]["list"]]
        ctx["numv5_source"] = f"{x5['path']}@{x5['commit'][:9]} (sha256 {x5['sha256'][:8]})"
    rows, plans, st, dg = VP.post_rows(name, text, ctx)
    dg |= {"command_error_phases": counter,
           "swap": {"run_a_reproduces": run_a_ok, "run_b_reproduces": out_b == text_b, "run_a_rules_changed": sum(x["changed"] for x in log_a),
                    "run_b_rules_changed": sum(x["changed"] for x in log_b), "run_a_positions": len(spans_a), "positions_mapped": len(claims),
                    "positions_lost": info_a["spans_lost"], "outside_sub_changes": info_a["outside_changes"], "diff_added_lines": len(added), "diff_header_problems": bad_h,
                    "merged_not_verified": len(bad_m), "drop1b_not_verified": len(bad_d), "drop1b_slots_from_rc1a_rc2": d1b_from_rc1a_rc2},
           "deferred_rows_record_ok": sum(d_ok.values()), "deferred_rows": len(d_ok), "deferred_cohort_blocks": cohort,
           "label_update_visible_text_identical": label_same, "tolerance_table_matches": "post: 09's tolerance table" not in " ".join(stops),
           "constants_rows_checks_ok": sum(c["checks_ok"] for c in constants)}
    return rows, plans, stops + st, dg


KINDS = {"tables34": _tables34, "slots": _slots, "figrender": _figrender, "post": _post}
OUT_DIR = [None]  # this run's Volume folder (set in verify): logs and tracebacks go there, never into the reply
SHARED, RUN, PINS = {}, {"rows": []}, {}  # verified artifacts' objects for the post; the rows so far; 09's pins for summarize (C1)


@app.function(image=image, cpu=2, memory=(8192, 32768), timeout=1800, retries=0,
              volumes={"/sol": sol, "/conv": conv, "/p2": p2, "/out": out_vol} | cmd_vols)
def verify(payload: dict) -> dict:
    """R5: every phase runs under VC.guarded: an exception's message and traceback go to the Volume only (errors.txt); the reply carries
    its class and phase, never its text (consumer STOP messages can hold record-derived counts). The post (kind "post") runs last, on
    the rows and objects of the artifacts verified before it; summarize takes 09's pins from it (C1)."""
    d = Path("/out") / payload["stamp"]
    d.mkdir(parents=True, exist_ok=False)
    OUT_DIR[0] = d
    errors, tb = [], []

    def run_all():
        for f_, want in payload["pure_sha256"].items():
            if VC.sha(Path(f"/root/{f_}").read_bytes()) != want:
                raise VC.Stop(f"{f_} in the container is not the pinned copy (STOP)")
        signed = set(payload["signed"])
        rows, plans, stops, diag = [], {}, [], {}
        RUN["rows"] = rows
        order = sorted(payload["artifacts"], key=lambda n: payload["artifacts"][n]["kind"] == "post")
        for name in order:
            a = payload["artifacts"][name]
            fn = KINDS[a["kind"]]
            res, err, t_ = VC.guarded(fn, name, *((a, signed) if a["kind"] == "tables34" else (name, a, signed)))
            if err:
                errors.append(err)
                tb.append(f"== {name}\n{t_}")
                stops.append(f"{name}: {err['class']} (traceback on the Volume)")
                plans[name] = None
                diag[name] = {"error": err}
                continue
            r_, p_, s_, d_ = res
            rows += r_
            plans |= p_ if isinstance(p_, dict) else {name: p_}
            diag[name] = d_
            stops += s_
        summ = VC.summarize(rows, plans, stops, pins=PINS)
        for n_, d_ in diag.items():  # 09 (10-20): the post's residual STOP items by position (no text), with 09's notes from config
            if payload["artifacts"][n_]["kind"] == "post" and isinstance(d_, dict):
                import verify_post as VP_

                d_["residual_items"] = VP_.residual_items(summ["stop"], n_, payload["artifacts"][n_].get("residual_notes") or {})
        pw = next((d_.get("number_words") for n_, d_ in diag.items() if payload["artifacts"][n_]["kind"] == "post" and isinstance(d_, dict)
                   and d_.get("number_words")), None)
        if pw:  # 09 (10-20): the verdict line names the number words the pass does not gate
            pl_ = next((d_.get("preliminary_labels_used") or [] for n_, d_ in diag.items()
                        if payload["artifacts"][n_]["kind"] == "post" and isinstance(d_, dict)), [])
            summ["verdict"] = VC.verdict_line(summ["verdict_base"], summ["n_not_route_C_checked"], pw, n_pre=summ.get("n_preliminary_checked", 0),
                                              pre_labels=pl_)  # R7-1: the PRELIMINARY clause names the per-record labels
        jb = VC.jsonl_bytes(rows)
        pick = VC.select_recheck(rows, payload["seed"], payload["k"])
        sel = json.dumps({"seed": payload["seed"], "k": payload["k"], "rows": pick, "jsonl_sha256": VC.sha(jb)}, sort_keys=True).encode()
        (d / "rows.jsonl").write_bytes(jb)
        (d / "recheck_selection.json").write_bytes(sel)
        return reply(summ, diag, {"volume": OUT_VOL, "path": f"{payload['stamp']}/rows.jsonl", "sha256": VC.sha(jb), "n_rows": len(rows)},
                     {"seed": payload["seed"], "k": payload["k"], "path": f"{payload['stamp']}/recheck_selection.json", "sha256": VC.sha(sel),
                      "rows": pick})

    out, err, t_ = VC.guarded(run_all, "verify")
    if err:
        errors.append(err)
        tb.append(f"== verify\n{t_}")
        out = {"summary": {"verdict": "STOP"}, "errors": errors}
    else:
        out["errors"] = errors
    if tb:
        (d / "errors.txt").write_text("\n".join(tb))
    out_vol.commit()
    return out


def reply(summ, diag, jsonl, recheck):
    """The only thing that leaves the container: ints, slot ids, condition names, paths and shas (type-checked; any float STOPs)."""
    out = {"summary": summ, "diagnostics": diag, "jsonl": jsonl, "recheck": recheck}

    def chk(x, where):
        if isinstance(x, bool) or x is None or isinstance(x, int):
            return
        if isinstance(x, str):
            if len(x) > 300:
                raise VC.Stop(f"reply: a long string at {where} (STOP)")
            return
        if isinstance(x, list):
            for i, y in enumerate(x):
                chk(y, f"{where}[{i}]")
            return
        if isinstance(x, dict):
            for k, y in x.items():
                chk(k, where)
                chk(y, f"{where}.{k}")
            return
        raise VC.Stop(f"reply: a {type(x).__name__} at {where} (STOP)")

    chk(out, "reply")
    for k in ("unmatched", "orphan", "duplicate_slots"):
        for s in summ["stop"][k]:
            if not SLOT_ID_RX.fullmatch(s):
                raise VC.Stop("reply: a stop entry is not a slot id (STOP)")
    return out


# ---------- laptop ----------

def git(*a, cwd=None) -> bytes:
    return subprocess.run(["git", *a], check=True, capture_output=True, cwd=cwd).stdout


def git_bytes(commit, path):
    return git("show", f"{commit}:{path}")


def pointer(x, role):
    """{commit, path, sha256} -> the git bytes, unparsed (the container checks the sha)."""
    if not (re.fullmatch(r"[0-9a-f]{7,40}", x["commit"]) and re.fullmatch(r"[0-9a-f]{64}", x["sha256"])):
        sys.exit(f"STOP: {role}: malformed pin")
    return x | {"bytes": git_bytes(x["commit"], x["path"])}


@app.local_entrypoint()
def main(only: str = "tables34,rc2", config: str = str(CONFIG), seed: int = VC.SEED, k: int = 20, dry_run: bool = False):
    cfg = json.loads(Path(config).read_text())
    names = [n for n in only.split(",") if n]
    if not names or any(n not in cfg["artifacts"] for n in names):
        sys.exit(f"STOP: --only must name configured artifacts: {sorted(cfg['artifacts'])}")
    if git("symbolic-ref", "-q", "HEAD").strip() != b"refs/heads/aswin/routec-numbers-verify":
        sys.exit("STOP: run from the attached branch aswin/routec-numbers-verify")
    if git("status", "--porcelain", "--", str(HERE)).strip():
        sys.exit("STOP: the verifier folder has uncommitted changes; the run must cite committed code")
    head = git("rev-parse", "HEAD").decode().strip()
    arts = build_arts(cfg, names)
    core_sha = VC.sha((HERE / "verify_core.py").read_bytes())
    pure = {f_: VC.sha((HERE / f_).read_bytes()) for f_ in PURE}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    payload = {"artifacts": arts, "signed": sorted(s["sha256"] for s in cfg["signed"]), "seed": seed, "k": k, "stamp": stamp,
               "pure_sha256": pure}
    run_payload(payload, cfg, names, head, core_sha, pure, stamp, config, dry_run)


def build_arts(cfg, names):
    """Every selected artifact's pins resolved to git bytes, unparsed (the container checks each sha before it parses)."""
    arts = {}
    for n in names:
        a = cfg["artifacts"][n]
        if a["kind"] == "figrender":  # bytes only; the render log, outputs and records are parsed in the container
            arts[n] = {"kind": a["kind"], "render_log": pointer(a["render_log"], f"{n}.render_log"), "scripts_commit": a["scripts_commit"],
                       "figures": a["figures"], "outputs": {o: pointer(x, f"{n}.{o}") for o, x in a["outputs"].items()},
                       "records": {r: pointer(x, f"{n}.{r}") for r, x in a["records"].items()},
                       "code": {c: {"blob": b, "bytes": git("cat-file", "blob", b)} for c, b in a["code"].items()}}
            continue
        if a["kind"] == "post":  # bytes only; writing's files are parsed in the container (never on the laptop)
            if any(r not in names for r in a["requires"]):
                sys.exit(f"STOP: post needs {a['requires']} in the same run (--only)")
            arts[n] = {k_: a.get(k_) for k_ in ("kind", "requires", "not_route_C", "nonresult_approved", "not_printed_v84_approved", "orphan_notes",
                                                "preliminary", "row_scope_notes", "deferred_record_checks", "preliminary_labels")} | {
                "artifact": pointer(a["artifact"], n), "inputs": {k_: pointer(x, f"{n}.{k_}") for k_, x in a["inputs"].items()},
                "records": [pointer(x, f"{n}.record") for x in a["records"]],
                "code": {c: {"blob": b, "bytes": git("cat-file", "blob", b)} for c, b in a["code"].items()}}
            continue
        arts[n] = {"kind": a["kind"], "artifact": pointer(a["artifact"], n), "not_printed_approved": a.get("not_printed_approved"),
                   "sources": {r: pointer(x, f"{n}.{r}") for r, x in a["sources"].items()},
                   "code": {c: {"blob": b, "bytes": git("cat-file", "blob", b)} for c, b in a["code"].items()}}
        if "FROZEN" in a["sources"]:  # FROZEN INPUTS_FULL is an inputs config (keys and pins, no result value): its C overrides' pointers are
            fz = json.loads(arts[n]["sources"]["FROZEN"]["bytes"])  # read here only to ship their bytes; the container re-checks every sha
            arts[n]["overrides"] = {k_: {x_: v_[x_] for x_ in ("commit", "path", "sha256")} | {"bytes": git_bytes(v_["commit"], v_["path"])}
                                    for k_, v_ in fz["overrides"].items() if k_.endswith("|C")}
    return arts


def bytes_sent(arts):
    return {n: sum(len(x["bytes"]) for k_ in ("artifact", "render_log") if k_ in a for x in [a[k_]])
            + sum(len(x["bytes"]) for k_ in ("sources", "outputs", "records", "overrides", "inputs")
                  for x in (lambda v: v if isinstance(v, list) else v.values())(a.get(k_, {})))
            for n, a in arts.items()}


def run_payload(payload, cfg, names, head, core_sha, pure, stamp, config, dry_run):
    arts = payload["artifacts"]
    print(json.dumps({"stamp": stamp, "head": head, "artifacts": names, "pure_sha256": pure,
                      "bytes_sent": bytes_sent(arts)}))
    if dry_run:
        print("dry run: pins and git bytes resolved; nothing sent")
        return
    res = verify.remote(payload)
    rec = {"what": "route C numbers verifier (09's design): counts, slot ids, shas only", "stamp": stamp, "git": head,
           "script_sha256": VC.sha(Path(__file__).read_bytes()), "core_sha256": core_sha, "pure_sha256": pure, "config": str(Path(config).resolve().relative_to(HERE.parents[1])),
           "config_sha256": VC.sha(Path(config).read_bytes()), "app": APP,
           "pins": {n: {k_: v for k_, v in cfg["artifacts"][n].items() if k_ != "note"} for n in names}} | res
    run_err = next((e for e in res.get("errors") or [] if e.get("where") == "verify"), None)  # a whole-run error labels every artifact
    rec["scope"], rec["printed_numbers_verified"] = VC.result_scope({n: cfg["artifacts"][n]["kind"] for n in names}, res.get("diagnostics"),
                                                                    res.get("errors"), run_err, res.get("summary"))  # top level, every artifact (09 / RT1, 10-20)
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"numbers_verify_{stamp}.json"
    with p.open("x") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    s = res["summary"]
    pn = next((n for n in names if cfg["artifacts"][n]["kind"] == "post"), None)
    pd = (res.get("diagnostics") or {}).get(pn) if pn else None
    if isinstance(pd, dict) and "nonresult_entries" in pd:  # 09 (10-20): the non-result list as committed output, its sha by construction
        nl = s.get("nonresult_list") or {}
        try:
            text, srt = VC.emit_nonresult_list(pd["nonresult_entries"], nl.get("sha256"), nl.get("count"))
        except VC.Stop as e:
            print(json.dumps({"file": str(p.relative_to(HERE.parents[1])), "nonresult_list_emit": str(e)}))
            sys.exit(2)
        tp, jp_ = RESULTS / f"nonresult_list_{stamp}.txt", RESULTS / f"nonresult_list_{stamp}.json"
        with tp.open("xb") as fh:
            fh.write(text)
        with jp_.open("x") as fh:
            json.dump({"what": "the v84 pass's non-result list: one entry per non-result token (class, the token as printed, the matched post "
                               "text, position, classifier); the .txt beside it is the list 09 approves by {sha256, count}; no record values",
                       "result": str(p.relative_to(HERE.parents[1])), "sha256": VC.sha(text), "count": len(srt),
                       "entries": [{k_: e[k_] for k_ in ("class", "token", "matched", "pos", "by")} for e in srt],
                       "header_labels_taken_by_other_classes": pd.get("header_labels_taken_by_other_classes")}, fh, indent=1, ensure_ascii=False)
        print(json.dumps({"nonresult_list": str(tp.relative_to(HERE.parents[1])), "sha256": VC.sha(text), "count": len(srt)}))
        prev = cfg["artifacts"][pn].get("nonresult_previous")
        if prev:  # 09 (10-20): the new list's by-key diff against the approved one (entries added / removed, by class); 09 approves the diff
            pb = git_bytes(prev["commit"], prev["path"])
            if VC.sha(pb) != prev["sha256"]:
                print(json.dumps({"nonresult_previous": "the approved list file does not hash to its pin (STOP)"}))
                sys.exit(2)
            diff = VC.nonresult_diff(pb.decode().split("\n") if pb else [], text.decode().split("\n") if text else [])
            dp = RESULTS / f"nonresult_list_diff_{stamp}.json"
            with dp.open("x") as fh:
                json.dump({"what": "the non-result list's by-key diff against the approved list (post text only, no record values)",
                           "previous": prev, "new": {"path": str(tp.relative_to(HERE.parents[1])), "sha256": VC.sha(text), "count": len(srt)}} | diff,
                          fh, indent=1, ensure_ascii=False)
            print(json.dumps({"nonresult_list_diff": str(dp.relative_to(HERE.parents[1])), "added_by_class": diff["added_by_class"],
                              "removed_by_class": diff["removed_by_class"]}))
    print(json.dumps({"file": str(p.relative_to(HERE.parents[1])), "verdict": s.get("verdict"),
                      **{k_: s[k_] for k_ in ("n_printed", "n_equal", "n_unmatched", "n_orphan", "n_slot_rows") if k_ in s},
                      "per_artifact": s.get("per_artifact"), "jsonl_sha256": (res.get("jsonl") or {}).get("sha256"),
                      "stop_counts": {k_: len(v) for k_, v in (s.get("stop") or {}).items()}, "errors": res.get("errors")}))
