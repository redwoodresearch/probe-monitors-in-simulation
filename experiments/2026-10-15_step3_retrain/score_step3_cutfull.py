"""Step-3 recipes (r1only, r2only, r12; L48; R and C) on the full-length (cutfull, uncut) roots of the (a'') scope: +311 run1, run2, sim,
honest_a, honest_b and the +311 sim-honest batches (RULE Addendum F.7; 4f's option A). ONE scorer for every (a'') arm; the sim-honest
scorer's cutfull path is retired. Input: messages from build_cutfull_message.py over d9's cutfull routing records (spec v2), each equal
by sha256 to a committed adapter record with passing gates. Same staged-input checks, fixed-token skips, step-3 models and row code as
pass 2 (byte-identical copies). Routing by main with runs36 over the +311 pin: a mapped main uses its held-out model, an unmapped main
fold -1 (score NaN, all five kept). Every epoch must consume its full render: meta n_tokens == the registered tokens_before, cut "full".
Output (SEALED, create-only, line "cutfull"): convaudit_work:/cutfull/<batch>/step3/<cutfull folder>/step3_<recipe>_L48__<R|C>.npz,
per-row only (2e's driver fields: key, run_key, unit, call_index, irid, token, n_tokens, score, fold, score_by_model), and
manifest_step3.json last; <batch> = the routing record's batch (09's layout ruling, option (c)). Each key is reserved once per batch.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach <this file> --runs-file <json>
(--runs-file: [{"message": <path>, "source_record": "<commit>:<adapter record path>"}, ...])
"""

import hashlib
import importlib.util
import inspect
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
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
FAM = REPO / "experiments/2026-10-14_p2_family_expansion"
APP = "probe-shift_aswin-surya_sub-step3_cutfull_kill-2026-10-22"
SPEC = "2026-10-14_163027"
SPEC_SHA = "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61"
FIT3 = {"stamp": "2026-10-15_114402", "run": "fit_2026-10-15_114402", "volume_record_sha256": "0aa4ad7db54aea16597588eda368c43053684456ef463efd59c3761eccfa2bed"}
VARIANTS = [f"step3_{r}_L48" for r in ("r1only", "r2only", "r12")]
CUTFULL_RE = re.compile(r"^([a-z0-9_]+?)_cutfull_v1(?:_try[0-9]+)?__([A-Za-z0-9_@.]+)$")  # 09's RULE Append 41 am. 2 @ bfc8742a6, verbatim
CUTFULL_VOLUME = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"  # d9's write-once cutfull roots; mounted read-only
CUTFULL_KEYS = "probe-shift_aswin-surya_sub-step3_cutfull-keys_kill-2026-11-14"  # Modal Dict: <batch>|<world|cell> -> the one cutfull folder scoring it
INVENTORY = ("b8a39f329", "experiments/2026-10-15_cut_sensitivity/results/inventory_v3_2026-10-15.json", "19306ffa7c42ab35a62715c1a18822a43c6273492e5ede9b2f62cd9e428ad9a3")
ROUTING_BRANCH = "origin/aswin/q3-cut-option"
LABEL = "uncut: all tokens; POST HOC (Append 41); step-3 recipes fit on rows cut at 32,768 tokens, so rows past the cut are out of their training range"
NEWMAINS = ("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json", "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e")
POOL = {"S_new": "q2", "simh_311": "q2", "S_newmains": "newmains", "simh_nm360": "newmains", "S_old": "t196"}  # stratum -> the pinned cell list
Q2 = ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json", "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b")
T196 = ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")
R36 = ("c24106ca0", "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json", "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96")

if modal.is_local():
    sys.path.insert(0, str(FAM))
    import run_famexp as rf
    import run_famexp_pass2 as p2

    BUILD = rf.BUILD
else:
    BUILD = Path("/tmp/famexp_build")
B_ASIS = {"R": "unused/{L}", "C": "unused"}  # never reached: empty layer pairs, L != 63


def _x2e():
    sys.path.insert(0, "/root/x2e")
    import run_correctedb as rc

    return rc, rc.rat


def _models(spec, fit, L, readout):
    """(variant, fold) -> weights at layer L and readout: pass 1's fit record (own_a, adapted; certified only) or 2e's b_asis file."""
    pair, out = spec["layer_choice_a"], {}
    for x in fit:
        if x["layer"] == L and x["readout"] == readout and x["certified"]:
            out[(x["variant"], x["fold"])] = ("p1", x["sha256"]["weights"], f"{x['variant']}__{readout}__L{L}__f{x['fold']}.safetensors")
    if readout == "C" and L == 63:
        out |= {("b_asis", f): ("b_asis", None, B_ASIS["C"]) for f in range(5)}
    if readout == "R":
        out |= {("b_asis", f): ("b_asis", None, B_ASIS["R"].format(L=L)) for f in range(5) if L in pair[f]}
    return out


def _score_rows(own, L, task_fold, spec, fit, fit_run, base, readout, cache):
    """Rows of `own` at L, scored by every (variant, fold) model there: {variant: (names, units, call_index, folds, S [N x 5])}, weights."""
    import numpy as np
    from safetensors.numpy import load_file

    rc, rat = _x2e()
    models = _models(spec, fit, L, readout)
    if not models:
        return {}, {}
    try:
        X, names, folds = (rat._b_rows if readout == "R" else rat._b_rows_cmd)(own, L, task_fold)
    except ValueError as e:  # 2e's loader cannot concatenate zero arrays: no staged epoch has a row in this readout
        if "need at least one array to concatenate" not in str(e):
            raise
        return {}, {}  # every run of the message is then listed as runs_without_rows (orchestrate2), never imputed
    ci, _ = rc._row_calls(own, cmd=(readout == "C"))
    X32, X = X.astype(np.float32), X.astype(np.float64)
    out, wsha = {}, {}
    for v in sorted({k[0] for k in models}):
        S = np.full((len(names), 5), np.nan)
        for f in range(5):
            if (v, f) not in models:
                continue
            kind, want, path = models[(v, f)]
            p = (base / f"run_{fit_run}" / "weights" / path) if kind == "p1" else Path("/sol", path)
            if p not in cache:
                b_ = p.read_bytes()
                cache[p] = (hashlib.sha256(b_).hexdigest(), load_file(str(p)) if kind == "p1" else dict(np.load(io.BytesIO(b_))))
            h, Z = cache[p]
            if want and h != want:
                raise SystemExit(f"{v} f{f} L{L}: pass-1 weights sha256 differs from the fit record")
            pre = "" if kind == "p1" else next((q for q in ("b_asis__", "b_asis_tight__") if f"{q}f{f}__W" in Z), None)
            if pre is None:
                raise SystemExit(f"b_asis weights: no b_asis(_tight)__f{f}__W key in {path}")
            w = {k: Z[f"{pre}f{f}__{k}" if pre else k] for k in ("W", "b", "mean", "std")}
            Xs = ((X - w["mean"]) / w["std"]) if kind == "p1" else ((X32 - w["mean"].astype(np.float32)) / w["std"].astype(np.float32)).astype(np.float64)
            S[:, f] = Xs @ np.asarray(w["W"], np.float64).reshape(-1) + float(np.asarray(w["b"]).reshape(-1)[0])
            wsha[f"{v}|{readout}|L{L}|f{f}"] = h
        out[v] = (names, np.asarray(rc._units(names), np.int64), np.asarray(ci, np.int64), np.asarray(folds, np.int64), S)
    return out, wsha


def _own_spec(epochs, readout, skip):
    return {"layout": "b", "kind": "output_end_mean" if readout == "R" else "call_end", "map": {e["key"]: [e["world"], e["cell"]] for e in epochs},
            "arm_of": {e["key"]: e["arm"] for e in epochs}, "run_of": {e["key"]: e["run"] for e in epochs}} | ({"skip": skip} if readout == "C" else {})


if modal.is_local():
    for fn in (_models, _score_rows, _own_spec):
        if inspect.getsource(fn) != inspect.getsource(getattr(p2, fn.__name__)):
            raise SystemExit(f"{fn.__name__} differs from run_famexp_pass2.py's")

app = modal.App(APP)
OUTV = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2)
W3 = modal.Volume.from_name("probe-shift_aswin-surya_sub-step3_work_kill-2026-11-13", version=2).read_only()
CFV = modal.Volume.from_name(CUTFULL_VOLUME, version=2).read_only()
img = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "safetensors==0.8.0").add_local_dir(BUILD, "/root/x2e")


def _row_meta(own: dict, cmd: bool):
    """Per row, in _row_calls order (same map order, same filters): the staged epoch key, irid, readout token and the epoch's n_tokens."""
    rc, _ = _x2e()
    rk, ir, tk, nt = [], [], [], []
    for key in own["map"]:
        m = json.loads((Path("/b") / own["run_of"][key] / "meta" / own["arm_of"][key] / f"{rc.stem(key)}.json").read_text())
        if cmd:
            skip = {tuple(x) for x in own.get("skip", {}).get(key, [])}
            cs = [c for c in m["calls"] if c["kind"] == own["kind"] and c.get("row") is not None and (c["irid"], c["call_index"]) not in skip]
        else:
            cs = [c for c in m["calls"] if c["kind"] == "output_end" and c.get("row_mean") is not None]
        if any(not isinstance(c.get("readout"), int) or not 0 <= c["readout"] < m["n_tokens"] for c in cs):
            raise SystemExit(f"{key}: a unit without a readout position inside the epoch")
        rk += [key] * len(cs)
        ir += [str(c["irid"]) for c in cs]
        tk += [c["readout"] for c in cs]
        nt += [m["n_tokens"]] * len(cs)
    return rk, ir, tk, nt


@app.function(image=img, cpu=(8, 8), memory=(32768, 32768), timeout=900, retries=0, scaledown_window=2, volumes={"/out": OUTV, "/w3": W3, "/b": CFV})
def score(msg: dict, task_fold: dict, head: dict) -> dict:
    import numpy as np

    rc, _ = _x2e()
    base3 = Path("/w3/step3") / SPEC
    if hashlib.sha256((base3 / f"fit_{FIT3['stamp']}.json").read_bytes()).hexdigest() != FIT3["volume_record_sha256"]:
        raise SystemExit("the step-3 fit record on the Volume differs from its committed copy")
    if hashlib.sha256((base3 / f"spec_{SPEC}.json").read_bytes()).hexdigest() != SPEC_SHA:
        raise SystemExit("the spec copy differs from famexp's prep sha256 (fixed_ids drive the command skips)")
    spec = json.loads((base3 / f"spec_{SPEC}.json").read_text())
    spec_np = spec | {"layer_choice_a": [[] for _ in spec["layer_choice_a"]]}
    fit3 = [x for g in json.loads((base3 / f"fit_{FIT3['stamp']}.json").read_text())["groups"] for x in g["fits"] if x["layer"] == 48]
    if head["pool"] == "t196":  # S_old: every cell must use the fold model that held its task out (the fit's own fot)
        bad_f = [e["cell"] for e in msg["epochs"] if task_fold[e["cell"]] != spec["fot"].get(head["base_task"].get(e["cell"], e["cell"]), -9)]
        if bad_f:
            raise SystemExit(f"S_old routing differs from the fit's held-out folds: {bad_f[:3]}")
    eps, fixed, bad, full = {}, set(spec["fixed_ids"]), [], []
    for path, h in sorted({(e["staged"], e["staged_sha256"]) for e in msg["epochs"]}):
        raw = Path("/b", path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != h:
            raise SystemExit(f"staged inputs {path}: sha256 differs from the message")
        eps |= {(path, x["key"]): x for x in (json.loads(y) for y in raw.decode().splitlines() if y)}
    for e in msg["epochs"]:
        st = eps.get((e["staged"], e["key"]))
        mt = json.loads((Path("/b") / e["run"] / "meta" / e["arm"] / f"{rc.stem(e['key'])}.json").read_text()) if st else None
        if not st or mt["key"] != e["key"] or mt.get("inputs_sha256") != e["staged_sha256"] or [(c["irid"], c["call_index"], c["kind"]) for c in st["calls"]] != \
                [(c["irid"], c["call_index"], c["kind"]) for c in mt["calls"]]:
            bad.append(e["key"])
        elif mt.get("cut") != "full" or mt.get("truncated") is not False or mt.get("n_tokens") != e["tokens_before"] or mt.get("tokens_total", mt["n_tokens"]) != mt["n_tokens"]:
            full.append(e["key"])  # 09 item 2: full consumption vs the REGISTERED tokens_before; never counted as uncut
    if bad or full:
        raise SystemExit(f"{len(bad)} epochs missing from staging or inconsistent with the meta, {len(full)} not fully consumed, e.g. {(bad + full)[:2]}")
    skip = {e["key"]: [[r["irid"], r["call_index"]] for r in eps[(e["staged"], e["key"])]["calls"]
                       if r["kind"] == "call_end" and r["readout"] is not None and eps[(e["staged"], e["key"])]["ids"][r["readout"]] in fixed] for e in msg["epochs"]}
    excl = {}
    for p_ in msg.get("provenance", []):
        for x in p_.get("excluded_by_staging", []):
            excl.setdefault(x["folder"], []).append(x)
    folders = sorted({e["folder"] for e in msg["epochs"]} | set(excl))
    tree = Path("/out/cutfull") / head["batch"] / "step3"  # 09 item 4 + layout ruling (c): never a registered dir
    keys_msg = {e["world"] + "|" + e["cell"] for e in msg["epochs"]}
    for mp in sorted(tree.glob("*/manifest_step3.json")):
        m_ = json.loads(mp.read_text())
        if m_.get("line") != "cutfull":
            raise SystemExit(f"ERROR: {mp} is a pack of the other line")
        if set(m_["keys"]) & keys_msg:
            raise SystemExit(f"{mp}: a key of this message is already scored in this batch's cutfull tree (each key once per batch)")
    rsv = modal.Dict.from_name(CUTFULL_KEYS, create_if_missing=True)  # atomic: concurrent launches cannot both pass the scan
    me = f"{modal.current_function_call_id()}|{head['message_sha256']}"  # one invocation owns each key (Codex cutfull r1 item 2): a duplicate launch,
    for e in msg["epochs"]:  # even of the same folder, is refused; a retry after a failure needs its stale reservation cleared by hand
        k_ = f"{head['batch']}|{e['world']}|{e['cell']}"
        if not rsv.put(k_, {"folder": e["folder"], "owner": me}, skip_if_exists=True):
            raise SystemExit(f"{k_} is already reserved ({rsv.get(k_)}): each key once per batch, one invocation per key")
    ods = {fo: tree / fo for fo in folders}
    for od in ods.values():
        if od.exists() and any(od.iterdir()):
            raise SystemExit(f"{od} already holds files: never overwritten")
    keys_of = {fo: {e["world"] + "|" + e["cell"] for e in msg["epochs"] if e["folder"] == fo} for fo in folders}
    fo_of = {e["world"] + "|" + e["cell"]: e["folder"] for e in msg["epochs"]}
    blobs, files, counts, missing = ({fo: {} for fo in folders} for _ in range(4))
    wsha = {}
    for readout in ("R", "C"):
        own = _own_spec(msg["epochs"], readout, skip)
        res, ws = _score_rows(own, 48, task_fold, spec_np, fit3, FIT3["run"], base3, readout, {})
        wsha |= ws
        meta = _row_meta(own, readout == "C") if res else None
        for v in VARIANTS:
            if v not in res:
                for fo in folders:
                    missing[fo][f"{v}|{readout}"] = sorted(keys_of[fo])
                continue
            names, units, ci, folds, S = res[v]
            names = names.astype(str)
            rk, ir, tk, nt = (np.asarray(x) for x in meta)
            ci_ref, _ = rc._row_calls(own, cmd=(readout == "C"))
            if len(rk) != len(names) or list(ci_ref) != ci.tolist() or [str(own["map"][k][0]) + "|" + str(own["map"][k][1]) for k in rk.tolist()] != names.tolist():
                raise SystemExit(f"{v} {readout}: the per-row meta does not line up with the scored rows")
            if not np.isfinite(S).all() or any(folds[i] != task_fold[k.split("|", 1)[1]] for i, k in enumerate(names.tolist())):
                raise SystemExit(f"{v} {readout}: a non-finite score or a row not routed by its main")
            for fo in folders:
                sel = np.array([fo_of[k] == fo for k in names.tolist()])
                if not sel.any():
                    missing[fo][f"{v}|{readout}"] = sorted(keys_of[fo])
                    continue
                f_, S_ = folds[sel], S[sel]
                sc = np.where(f_ >= 0, S_[np.arange(len(f_)), np.clip(f_, 0, 4)], np.nan)
                buf = io.BytesIO()
                np.savez(buf, key=names[sel], run_key=rk[sel], unit=units[sel], call_index=ci[sel], irid=ir[sel], token=tk[sel].astype(np.int64),
                         n_tokens=nt[sel].astype(np.int64), score=sc, fold=f_, score_by_model=S_)
                blobs[fo][f"{v}__{readout}.npz"] = buf.getvalue()
                files[fo][f"{v}__{readout}.npz"] = hashlib.sha256(buf.getvalue()).hexdigest()
                counts[fo][f"{v}__{readout}"] = {"n_runs": len(set(rk[sel].tolist())), "n_units": int(sel.sum()), "n_units_fold_minus1": int((f_ < 0).sum()),
                                                 "n_units_past_32768": int((tk[sel] >= 32768).sum())}
                gone = sorted(keys_of[fo] - set(names[sel].tolist()))
                if gone:
                    missing[fo][f"{v}|{readout}"] = gone
    out = {}
    orig = {e["folder"]: e["original_folder"] for e in msg["epochs"]} | {x["folder"]: x["original_folder"] for xs in excl.values() for x in xs}
    for fo, od in ods.items():
        od.mkdir(parents=True, exist_ok=True)
        for fn, b_ in sorted(blobs[fo].items()):
            (od / fn).open("xb").write(b_)
        shas = {e["staged_sha256"] for e in msg["epochs"] if e["folder"] == fo}
        if len(shas) != 1:
            raise SystemExit(f"{fo}: not exactly one staged sha256 for the folder")
        man = head | {"folder": fo, "original_folder": orig[fo], "staged_sha256": next(iter(shas)), "excluded_by_staging": excl.get(fo, []), "files": files[fo], "counts": counts[fo],
                      "runs_without_rows": missing[fo], "keys": sorted(keys_of[fo]),
                      "routing": {k: {"fold": task_fold[k.split("|", 1)[1]], "basis": "mapped: held-out step-3 model" if task_fold[k.split("|", 1)[1]] >= 0 else "unmapped: fold -1, all five models"}
                                  for k in sorted(keys_of[fo])}, "weights_sha256": wsha, "complete": not missing[fo] and len(files[fo]) == 2 * len(VARIANTS)}
        mb = json.dumps(man, indent=1, default=str).encode()
        (od / "manifest_step3.json").open("xb").write(mb)
        out[fo] = {"manifest": str(od / "manifest_step3.json"), "manifest_sha256": hashlib.sha256(mb).hexdigest(), "counts": counts[fo], "complete": man["complete"]}
    OUTV.commit()
    return out


@app.local_entrypoint()
def main(runs_file: str):
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout  # noqa: E731
    on_origin = lambda c: any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", c).decode().splitlines())  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip() or git("rev-parse", "HEAD") != git("rev-parse", "origin/aswin/step3-retrain") \
            or git("ls-files", "--error-unmatch", str(Path(__file__).relative_to(REPO))).strip() == b"":
        raise SystemExit("uncommitted changes, HEAD not pushed, or this scorer untracked")
    sp = importlib.util.spec_from_file_location("build_cutfull_message", HERE / "build_cutfull_message.py")  # one S_old grammar / world map, shared
    A = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(A)
    pins = {}
    for nm, (c, p, h) in {"q2": Q2, "newmains": NEWMAINS, "t196": T196, "r36": R36, "inventory": INVENTORY}.items():
        b = git("show", f"{c}:{p}")
        if not on_origin(c) or hashlib.sha256(b).hexdigest() != h:
            raise SystemExit(f"{nm}: not on origin, or its sha256 differs from the pin")
        pins[nm] = json.loads(b)
    invs = {"a2": {r["key"]: r for r in pins["inventory"]["runs"]}}
    if A.INVENTORY_B is not None:
        b = git("show", f"{A.INVENTORY_B[0]}:{A.INVENTORY_B[1]}")
        if not on_origin(A.INVENTORY_B[0]) or hashlib.sha256(b).hexdigest() != A.INVENTORY_B[2]:
            raise SystemExit("inventory_b: not on origin, or its sha256 differs from the pin")
        invs["b"] = {r["key"]: r for r in json.loads(b)["runs"]}
        orphans = {f"{x['record']}#e0" for x in json.loads(git("show", f"{A.S4_ORPHANS[0]}:{A.S4_ORPHANS[1]}"))["outside_source_map"]}
    lists = {nm: {c["key"]: c["main"] for c in pins[nm]["cells"]} for nm in ("q2", "newmains", "t196")}
    t196_mains = set(lists["t196"].values())
    fom = {}
    for r in pins["r36"]:
        if fom.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("r36: a main in two folds")
    strip_try = lambda f: re.sub(r"_try[0-9]+__", "__", f)  # noqa: E731
    jobs = []
    for job in json.loads(Path(runs_file).read_text()):
        raw = Path(job["message"]).read_bytes()
        rc_, rp_ = job["source_record"].split(":", 1)
        if not on_origin(rc_):
            raise SystemExit(f"{rp_}: the adapter record's commit is not on origin")
        rb_ = git("show", f"{rc_}:{rp_}")
        rec = json.loads(rb_)
        if hashlib.sha256(raw).hexdigest() != rec.get("message_sha256") or not rec.get("gate") or not all(g.get("pass") is True for g in rec["gate"].values()):
            raise SystemExit(f"{rp_}: the message sha256 differs from the committed adapter record's, or its gates did not pass")
        msg = json.loads(raw)
        rr = rec["pins"]["routing_record"]["path"]
        stem = re.sub(r"(_r[0-9]+)?\.json$", "", Path(rr).name)
        ks = [int(m_.group(1) or 1) for x in git("ls-tree", "--name-only", f"{ROUTING_BRANCH}:{Path(rr).parent}").decode().split()
              if (m_ := re.fullmatch(re.escape(stem) + r"(?:_r([0-9]+))?\.json", x))]
        if max(ks) != rec["revision"]:
            raise SystemExit(f"{rr}: a later routing revision exists on {ROUTING_BRANCH} (highest k is authoritative): rebuild the message")
        scope = msg.get("scope", "a2")
        if scope not in invs:
            raise SystemExit(f"{rp_}: scope {scope}: its inventory is not pinned")
        inv = invs[scope]
        tb = lambda k: inv[k].get("tokens_before", inv[k].get("full_len"))  # noqa: E731  ((b): full_len, equal by 4f)
        up = lambda k: inv[k]["uncut_planned"] if isinstance(inv[k]["uncut_planned"], dict) else {}  # noqa: E731
        derived = lambda f: {strip_try(f"{CUTFULL_RE.fullmatch(f).group(1)}{s_}__{CUTFULL_RE.fullmatch(f).group(2)}") for s_ in ("_v1", "")}  # noqa: E731  (addendum 2)
        rows = msg["epochs"] + [y for p_ in msg.get("provenance", []) for y in p_.get("excluded_by_staging", [])]
        bad = [x["folder"] for x in rows if "_cutfull_gate_" in x["folder"] or not CUTFULL_RE.fullmatch(x["folder"]) or strip_try(x["original_folder"]) not in derived(x["folder"])]
        bad_i = [e["key"] for e in msg["epochs"] if e["key"] not in inv or up(e["key"]).get("folder") != e["folder"] or inv[e["key"]]["original_folder"] != e["original_folder"]
                 or inv[e["key"]]["arm"] != e["arm"] or tb(e["key"]) != e["tokens_before"] or inv[e["key"]].get("original_folder_registered") is not True]
        fos = {e["folder"] for e in msg["epochs"]}
        out_k = set(msg.get("out_by_key") or {})
        if out_k and (scope != "b" or not out_k <= orphans or out_k & set(inv) or out_k & {e["key"] for e in msg["epochs"]}):
            raise SystemExit(f"{rp_}: keys held OUT that are not (b) S4 orphans outside the inventory (09)")
        if bad or bad_i or {k for k in inv if up(k).get("folder") in fos} != {e["key"] for e in msg["epochs"]}:
            raise SystemExit(f"{rp_}: folder grammar (09 (a)-(c)) or inventory ({scope}) mismatch: {(bad + bad_i)[:3]}")
        if msg["volume"] != CUTFULL_VOLUME or not re.fullmatch(r"[a-z0-9_-]+", msg.get("batch") or "") or len({e["stratum"] for e in msg["epochs"]}) != 1:
            raise SystemExit(f"{rp_}: not the cutfull Volume, no chain batch, or mixed strata")
        pool = POOL.get(msg["epochs"][0]["stratum"])
        if pool is None or (pool == "t196") != (scope == "b" and msg["epochs"][0]["stratum"] == "S_old"):
            raise SystemExit(f"{rp_}: a stratum outside the pinned pools, or S_old outside the (b) scope")
        # the relaunch pool repeats T196 tasks: its cells are "<T196 key>_r<k>"; main and fold come from the T196 key (09 world label "relaunch")
        base = {e["cell"]: (re.sub(r"_r[0-9]+$", "", e["cell"]) if e["world"] == "relaunch" else e["cell"]) for e in msg["epochs"]}
        main_of = {c: lists[pool][b] for c, b in base.items() if b in lists[pool]}
        if any(e["cell"] not in main_of for e in msg["epochs"]) or (pool != "t196" and any(main_of[e["cell"]] in t196_mains for e in msg["epochs"])):
            raise SystemExit(f"{rp_}: a cell outside its pool's pinned list, or a T196 main outside S_old (the step-3 models trained on T196 mains)")
        if pool == "t196" and any(A.sold_world(e["original_folder"]) != e["world"] for e in msg["epochs"]):
            raise SystemExit(f"{rp_}: an S_old world differs from 09's per-folder map (never the arm token)")
        task_fold = {e["cell"]: fom.get(main_of[e["cell"]], -1) for e in msg["epochs"]}
        base_task = {c: b for c, b in base.items() if c != b}
        if pool == "t196" and any(f < 0 for f in task_fold.values()):
            raise SystemExit(f"{rp_}: an S_old cell without a runs36 fold")
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        head = {"label": f"step-3 recipes on a full-length root, scope {scope} (RULE Addenda F.7 / F.10): SEALED; hashes, counts and routing only", "row_label": LABEL,
                "scope": scope, "pool": pool, "format_check": msg.get("format_check"), "out_by_key": msg.get("out_by_key") or {}, "base_task": base_task,
                "world_labels": {"relaunch": A.RELAUNCH_LABEL} if any(e["world"] == "relaunch" for e in msg["epochs"]) else {},
                "status": "POST HOC (uncut)", "line": "cutfull", "batch": msg["batch"], "scorer_git": git("rev-parse", "HEAD").decode().strip(), "stamp (PT)": stamp,
                "message_sha256": rec["message_sha256"], "source_record": {"commit": rc_, "path": rp_, "sha256": hashlib.sha256(rb_).hexdigest()},
                "routing_record": rec["pins"]["routing_record"], "revision": rec["revision"], "extraction_root": msg["provenance"][0]["run"],
                "extraction_records": msg["provenance"][0]["extraction_records"], "staged_set": msg["provenance"][0]["staged_set"],
                "inventory": dict(zip(("commit", "path", "sha256"), INVENTORY if scope == "a2" else A.INVENTORY_B)), "step3_fit": FIT3,
                "schema": "per-row only: key (world|cell), run_key (staged epoch key), unit, call_index, irid, token (readout position), n_tokens, "
                          "score (designated model; NaN on fold -1), fold, score_by_model (n x 5)"}
        jobs.append((msg, task_fold, head, sorted(c for c, f in task_fold.items() if f < 0)))
    calls = [(m, h, u, score.with_options(memory=(65536, 65536) if len(m["epochs"]) > 500 else (32768, 32768)).spawn(m, t, h)) for m, t, h, u in jobs]  # no map (F.8)
    (HERE / "results").mkdir(exist_ok=True)
    failed = []
    for msg, head, unmapped, fc in calls:
        p = HERE / "results" / f"cutfull_{msg['d9_arm']}_r{head['revision']}_{head['stamp (PT)']}.json"
        try:
            out = fc.get()
        except (Exception, SystemExit) as ex:  # a remote gate's SystemExit too (Codex cutfull-b r1)  # recorded; the other messages' records are still written
            failed.append(f"{msg['d9_arm']}: {type(ex).__name__}: {ex}")
            p.with_name(p.stem + "_FAILED.json").open("x").write(json.dumps(head | {"error": failed[-1]}, indent=1))
            continue
        p.open("x").write(json.dumps(head | {"function_call": fc.object_id, "result": out, "unmapped_cells": unmapped}, indent=1))
        print(json.dumps({fo: {k: v for k, v in r.items() if k != "counts"} for fo, r in out.items()}, indent=1))
    if failed:
        raise SystemExit(f"{len(failed)} of {len(jobs)} messages FAILED: {failed}")
