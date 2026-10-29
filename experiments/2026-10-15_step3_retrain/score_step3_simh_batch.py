"""PRE-STAGED (not run): step-3 recipes on a later sim-honest batch (simh311_q1..q4, simh_nm360_<g>, simh_t196fill), RULE.md Addendum A + 09's Append 25h.
Input: ONE message in famexp pass 2's schema for that batch ({"d9_arm", "volume", "epochs": [{key, cell, world, stratum, run, arm,
staged, staged_sha256, folder}]}, world simh_311 or simh_nm360), whose sha256 must equal a committed famexp record's message_sha256
(or, if famexp builds none, a committed adapter output over d9's routing record). Same staged-input checks and fixed-token skips as
pass 2's score_layer; step3_{r1only,r2only,r12}_L48 x R/C through pass 2's _score_rows (byte-identical copies). Routing by main with
the step-3 fold map (r36): a mapped main uses the held-out model; an unmapped main gets fold -1 (score and run_score NaN;
score_by_model and run_score_by_model carry all five). Output, SEALED and create-only, on convaudit_work:
/simh/<batch>/step3/<d9 routed folder>/step3_<recipe>_L48__<R|C>.npz (pass 1's schema + run_score_by_model) and manifest_step3.json
last; a full-length (cutfull, --cutfull) message instead under /cutfull/<batch>/step3/<folder>/ with line "cutfull" (RULE F.5).
No threshold, rate, catch or comparison. Launch only on 4f's relay of 09's 25h sha and d9's routing.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach <this file> --message <json> --source-record <commit>:<path> --append-25h <sha>
"""

import hashlib
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
APP = "probe-shift_aswin-surya_sub-step3_simhb_kill-2026-10-22"
SPEC = "2026-10-14_163027"
SPEC_SHA = "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61"
FIT3 = {"stamp": "2026-10-15_114402", "run": "fit_2026-10-15_114402", "volume_record_sha256": "0aa4ad7db54aea16597588eda368c43053684456ef463efd59c3761eccfa2bed"}
VARIANTS = [f"step3_{r}_L48" for r in ("r1only", "r2only", "r12")]
WORLDS = ("simh_311", "simh_nm360", "t196fill")
APPEND_25H, ATTEST_0D = "fd163c4c788f3bdc6809e3376d11bd8d69035655", "5aa9f32bfc0b7e78f60e79899f070449203e1e4f"  # 09's Append 25h and 0d's section-10 attestation (pre-scoring gate met)
_FOLDER_RE_BASE = re.compile(r"^(simh311_q[1-4]|simh_nm360_(?:s1|s2|s3|s4d|s4r|s5))_sim(?:_s[0-9]+|_rlr(?:[2-9]|[1-9][0-9]+)?)?_v1(?:_try[0-9]+)?__sim_honest$")  # 09 @ 91485ce80 + 502c1b05e
# A25 fill-in (09 @ 286b2f518, ACK of d9's root regex relayed by 4f 10-15 19:4x): T196 sim-honest, its own batch / world "t196fill", stratum
# S_old, every row labelled; never in the registered primary_v3 / S_old pool (/simh/primary_v3/). SEALED until the second refresh's input freeze.
_T196FILL_RE = re.compile(r"^simh_t196fill(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__sim_honest$")  # 09's ACK regex, verbatim
A25_FILL_LABEL = "A25 fill-in (POST HOC, generated after A25 results were read); registered A25 rows unchanged"  # 09 @ 286b2f518 section 4, verbatim


def _registered(f):
    """The accepted registered grammar; group(1) is the batch."""
    return _FOLDER_RE_BASE.fullmatch(f) or (re.match(r"(simh_t196fill)", f) if _T196FILL_RE.fullmatch(f) else None)


# CUT lift: 09's RULE Append 41 amendment 2 @ bfc8742a6 (tested), adopted verbatim. A folder name N is a full-length set iff (a) N has no
# "_cutfull_gate_" (gate sets are refused, never scored or pooled); (b) N fully matches CUTFULL["regex"] with groups (target, arm); (c) the
# derived original "<target>_v1__<arm>" fully matches this scorer's own accepted regex; (d) the cutfull manifest names the authoritative
# original folder by key (checked in main() against each epoch's "original_folder", carried from the manifest by the adapter).
# Admission still needs --cutfull at launch (flag OFF by default).
CUTFULL_VOLUME = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"  # d9's write-once cutfull roots; mounted read-only here
# (a'') run and folder list: inventory v3 (09's OK relayed by 4f 10-15 ~21:3x); cutfull rows must be exactly its runs for their folders
INVENTORY = ("b8a39f329", "experiments/2026-10-15_cut_sensitivity/results/inventory_v3_2026-10-15.json", "19306ffa7c42ab35a62715c1a18822a43c6273492e5ede9b2f62cd9e428ad9a3")
CUTFULL_KEYS = "probe-shift_aswin-surya_sub-step3_cutfull-keys_kill-2026-11-14"  # Modal Dict: <batch>|<world|cell> -> the one cutfull folder scoring it
CUTFULL = {"enabled": False, "regex": r"^([a-z0-9_]+?)_cutfull_v1(?:_try[0-9]+)?__([A-Za-z0-9_@.]+)$"}


def orig_name(f):
    """The derived original name of a cutfull folder ((c) above); any other name unchanged."""
    m = re.fullmatch(CUTFULL["regex"], f) if "_cutfull_" in f else None
    return f"{m.group(1)}_v1__{m.group(2)}" if m else f


class _Grammar:
    """This scorer's folder grammar plus 09's cutfull rule: a cutfull name is admitted only with the flag ON, and is matched through its
    derived original, so batch / pool groups and every rule keyed on them are unchanged; output paths keep the real folder name."""

    def fullmatch(self, f):
        if "_cutfull_gate_" in f:
            return None
        if "_cutfull_" in f:  # arm list v2 (a'') has no A25 fill-in root: cutfull names derive only into the base grammar (Codex t196fill r1 item 1)
            if not CUTFULL["enabled"] or not re.fullmatch(CUTFULL["regex"], f):
                return None
            return _FOLDER_RE_BASE.fullmatch(orig_name(f))
        return _registered(f)


FOLDER_RE = _Grammar()
RLR_RE = re.compile(r"_sim_rlr(?:[2-9]|[1-9][0-9]+)?_v1(?:_try[0-9]+)?__sim_honest$")  # a replacement root: k = 1 is _rlr, k >= 2 is _rlr<k> (_rlr0/_rlr1 invalid)
is_rlr = lambda f: bool(FOLDER_RE.fullmatch(f)) and bool(RLR_RE.search(orig_name(f)))  # noqa: E731  (cutfull: on the derived original)
base_root = lambda f: re.sub(r"_try[0-9]+__sim_honest$", "__sim_honest", orig_name(f))  # noqa: E731
# 09's RL / terminal-error replacement rulings: an _rlr root admits ONLY these (parent batch, cell) keys (@ 91485ce80 section 2)
RLR_RULINGS = {"13be06982": "13be06982", "c5420f2f5": "c5420f2f5", "c10a14948": "c10a14948", "91485ce80": "91485ce80", "dd93a41cd": "dd93a41cd", "81f75b763": "81f75b763",
               "502c1b05e": "502c1b05e", "322282c09": "322282c09"}
# 09 @ 502c1b05e section 2: these ruled keys moved to their parent's SECOND replacement launch; every other ruled key sits in the k = 1 root
RLR_ROOT_MOVED = {("simh311_q2", "t293_media_processing"): "simh311_q2_sim_rlr2_v1__sim_honest",
                  ("simh_nm360_s4d", "nm107_media_processing"): "simh_nm360_s4d_sim_rlr2_v1__sim_honest",
                  # 09 @ 322282c09 (RULE l.3698-3699): Q3's first _rlr launch has no rows (EC2 stop), so t298 and t310 move to Q3 rlr2
                  ("simh311_q3", "t298_user_tracking"): "simh311_q3_sim_rlr2_v1__sim_honest", ("simh311_q3", "t310_file_indexing"): "simh311_q3_sim_rlr2_v1__sim_honest"}
READ_LOG = "experiments/2026-10-14_priority1_spec_09/READ_LOG.md"  # 09's; standing-rule admissions (@ dd93a41cd section 2)
SR_LINE = re.compile(r"^- STANDING_RULE_ADD key=(\S+)\|(\S+) class=(?:unknown|infrastructure) m_r=(\d+) evidence=(\S+) rootcheck=(\S+) root=(\S+)\s*$")  # RL: class=infrastructure
RLR_KEYS = {("simh_nm360_s3", "nm103_file_indexing"): "13be06982", ("simh311_q3", "t298_user_tracking"): "13be06982",
            ("simh311_q3", "t310_file_indexing"): "c5420f2f5", ("simh311_q2", "t271_slack_clone"): "c5420f2f5", ("simh311_q2", "t410_file_indexing"): "c5420f2f5",
            ("simh311_q1", "t462_file_indexing"): "c5420f2f5", ("simh311_q1", "t242_model_registry"): "c5420f2f5",
            ("simh311_q1", "t237_media_processing"): "c10a14948", ("simh311_q1", "t300_user_tracking"): "c10a14948", ("simh311_q1", "t389_model_registry"): "c10a14948",
            ("simh311_q4", "t361_file_indexing"): "c10a14948", ("simh311_q4", "t411_media_processing"): "c10a14948", ("simh311_q4", "t413_media_processing"): "c10a14948",
            ("simh_nm360_s4d", "nm107_media_processing"): "c10a14948", ("simh_nm360_s4d", "nm258_tiktok"): "c10a14948",
            ("simh_nm360_s3", "nm134_media_processing"): "c10a14948", ("simh311_q2", "t293_media_processing"): "dd93a41cd",
            ("simh311_q2", "t240_model_registry"): "81f75b763", ("simh311_q2", "t275_shared_dotfiles"): "81f75b763", ("simh311_q2", "t457_media_processing"): "81f75b763"}
# every _rlr key needs d9's committed by-key root check (RL keys: @ 98ffa9e40 / c5420f2f5 section 4; terminal-error keys: c10a14948 section 2)
Q2 = ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json", "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b")
NEWMAINS = ("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json", "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e")
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
img = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "safetensors==0.8.0").add_local_dir(BUILD, "/root/x2e")


def _runs(key, fold, sbm, readout):
    """09 (3): per model on raw logits, R = mean of the run's rows, C = max; designated run score NaN on fold -1."""
    import numpy as np
    keys, inv = np.unique(key, return_inverse=True)
    rf = np.full(len(keys), -2, np.int64)
    rf[inv] = fold
    if any(len(set(fold[inv == i].tolist())) != 1 for i in range(len(keys))):
        raise SystemExit("a run has rows from two folds")
    if readout == "R":
        rsm = np.stack([np.bincount(inv, weights=sbm[:, j]) / np.bincount(inv) for j in range(5)], axis=1)
    else:
        rsm = np.full((len(keys), 5), -np.inf)
        for j in range(5):
            np.maximum.at(rsm[:, j], inv, sbm[:, j])
    rs = np.where(rf >= 0, rsm[np.arange(len(keys)), np.clip(rf, 0, 4)], np.nan)
    return keys, rf, rs, rsm


@app.function(image=img, cpu=(8, 8), memory=(32768, 32768), timeout=1200, retries=0, scaledown_window=2, volumes={"/out": OUTV, "/w3": W3})
def score(msg: dict, task_fold: dict, head: dict) -> dict:
    import numpy as np
    CUTFULL["enabled"] = bool(head.get("cutfull"))  # the launch's flag, in-container
    rc, _ = _x2e()
    base3 = Path("/w3/step3") / SPEC
    if hashlib.sha256((base3 / f"fit_{FIT3['stamp']}.json").read_bytes()).hexdigest() != FIT3["volume_record_sha256"]:
        raise SystemExit("the step-3 fit record on the Volume differs from its committed copy")
    if hashlib.sha256((base3 / f"spec_{SPEC}.json").read_bytes()).hexdigest() != SPEC_SHA:
        raise SystemExit("the spec copy differs from famexp's prep sha256 (fixed_ids drive the command skips)")
    spec = json.loads((base3 / f"spec_{SPEC}.json").read_text())
    spec_np = spec | {"layer_choice_a": [[] for _ in spec["layer_choice_a"]]}
    fit3 = [x for g in json.loads((base3 / f"fit_{FIT3['stamp']}.json").read_text())["groups"] for x in g["fits"] if x["layer"] == 48]
    eps, fixed, bad, cfgs = {}, set(spec["fixed_ids"]), [], {}
    for e in msg["epochs"]:
        if e["staged_sha256"] == "from_config":
            if e["run"] not in cfgs:
                cfgs[e["run"]] = json.loads((Path("/b") / e["run"] / "config.json").read_text())["inputs_sha256"]
            e["staged_sha256"] = cfgs[e["run"]][f"{e['arm']}.jsonl"]
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
    if bad:
        raise SystemExit(f"{len(bad)} epochs missing from staging or inconsistent with the extraction meta, e.g. {bad[:2]}")
    if head["batch"] == "simh_t196fill":  # T196 cells: the designated model must be the one that held the task out (spec fot)
        bad_f = [e["cell"] for e in msg["epochs"] if task_fold[e["cell"]] != spec["fot"].get(e["cell"], -9)]
        if bad_f:
            raise SystemExit(f"T196 fill-in routing differs from the fit's held-out folds: {bad_f[:3]}")
    skip = {e["key"]: [[r["irid"], r["call_index"]] for r in eps[(e["staged"], e["key"])]["calls"]
                       if r["kind"] == "call_end" and r["readout"] is not None and eps[(e["staged"], e["key"])]["ids"][r["readout"]] in fixed] for e in msg["epochs"]}
    excl = {}
    for p_ in msg.get("provenance", []):  # pass 2's staging exclusions: kept per folder, never dropped (Codex simhb r1 item 4); same grammar and batch
        if any(not FOLDER_RE.fullmatch(x["folder"]) or FOLDER_RE.fullmatch(x["folder"]).group(1) != head["batch"] for x in p_.get("excluded_by_staging", [])):
            raise SystemExit("a staging exclusion outside this batch's folder grammar")
        for x in p_.get("excluded_by_staging", []):
            excl.setdefault(x["folder"], []).append(x)
    folders = sorted({e["folder"] for e in msg["epochs"]} | set(excl))
    # 09 @ 77a770eec (Append 41 am. 2 item 4): separate output roots. Cutfull packs go to /cutfull/<batch>/step3/<folder>/ with line "cutfull",
    # never into a registered batch dir; each reader stays in its own tree, and a pack of the other line there is an ERROR
    tree = Path("/out") / ("cutfull" if head.get("line") == "cutfull" else "simh")
    for mp in sorted((tree / head["batch"] / "step3").glob("*/manifest_step3.json")):
        m_ = json.loads(mp.read_text())
        if (m_.get("line") == "cutfull") != (head.get("line") == "cutfull"):
            raise SystemExit(f"ERROR: {mp} is a pack of the other line")
        if head.get("line") == "cutfull" and set(m_["keys"]) & {e["world"] + "|" + e["cell"] for e in msg["epochs"]}:
            raise SystemExit(f"{mp}: a key of this message is already scored in this batch's cutfull tree (each key once per batch)")
    if head.get("line") == "cutfull":  # an atomic per-key reservation, so concurrent launches cannot both pass the scan (Codex cutfull r2 item 2)
        rsv = modal.Dict.from_name(CUTFULL_KEYS, create_if_missing=True)
        for e in msg["epochs"]:
            k_ = f"{head['batch']}|{e['world']}|{e['cell']}"
            if not rsv.put(k_, e["folder"], skip_if_exists=True) and rsv.get(k_) != e["folder"]:
                raise SystemExit(f"{k_} is reserved by another cutfull folder ({rsv.get(k_)}): each key once per batch")
    ods = {fo: tree / FOLDER_RE.fullmatch(fo).group(1) / "step3" / fo for fo in folders}
    for od in ods.values():
        if od.exists() and any(od.iterdir()):
            raise SystemExit(f"{od} already holds files: never overwritten")
    keys_of = {fo: {e["world"] + "|" + e["cell"] for e in msg["epochs"] if e["folder"] == fo} for fo in folders}
    blobs, files, counts, missing = ({fo: {} for fo in folders} for _ in range(4))
    wsha = {}
    fo_of = {e["world"] + "|" + e["cell"]: e["folder"] for e in msg["epochs"]}
    for readout in ("R", "C"):
        res, ws = _score_rows(_own_spec(msg["epochs"], readout, skip), 48, task_fold, spec_np, fit3, FIT3["run"], base3, readout, {})
        wsha |= ws
        for v in VARIANTS:
            if v not in res:
                for fo in folders:
                    missing[fo][f"{v}|{readout}"] = sorted(keys_of[fo])
                continue
            names, units, ci, folds, S = res[v]
            names = names.astype(str)
            if not np.isfinite(S).all() or any(folds[i] != task_fold[k.split("|", 1)[1]] for i, k in enumerate(names.tolist())):
                raise SystemExit(f"{v} {readout}: a non-finite score or a row not routed by its main")
            for fo in folders:
                sel = np.array([fo_of[k] == fo for k in names.tolist()])
                if not sel.any():
                    missing[fo][f"{v}|{readout}"] = sorted(keys_of[fo])
                    continue
                k_, u_, c_, S_, f_ = names[sel], units[sel], ci[sel], S[sel], folds[sel]
                score = np.where(f_ >= 0, S_[np.arange(len(f_)), np.clip(f_, 0, 4)], np.nan)
                rk, rfo, rs, rsm = _runs(k_, f_, S_, readout)
                buf = io.BytesIO()
                np.savez(buf, key=k_, unit=u_, call_index=c_, score=score, score_by_model=S_, fold=f_, run_key=rk, run_fold=rfo, run_score=rs, run_score_by_model=rsm)
                blobs[fo][f"{v}__{readout}.npz"] = buf.getvalue()
                files[fo][f"{v}__{readout}.npz"] = hashlib.sha256(buf.getvalue()).hexdigest()
                counts[fo][f"{v}__{readout}"] = {"n_runs": len(rk), "n_units": len(k_), "n_runs_fold_minus1": int((rfo < 0).sum())}
                gone = sorted(keys_of[fo] - set(rk.tolist()))
                if gone:
                    missing[fo][f"{v}|{readout}"] = gone
    out = {}
    for fo, od in ods.items():
        od.mkdir(parents=True, exist_ok=True)
        for fn, b_ in sorted(blobs[fo].items()):
            (od / fn).open("xb").write(b_)
        man = head | {"folder": fo, "excluded_by_staging": excl.get(fo, []), "files": files[fo], "counts": counts[fo], "runs_without_rows": missing[fo], "keys": sorted(keys_of[fo]),
                      "routing": {k: {"fold": task_fold[k.split("|", 1)[1]], "basis": "mapped: held-out step-3 model" if task_fold[k.split("|", 1)[1]] >= 0 else "unmapped: fold -1, all five models"}
                                  for k in sorted(keys_of[fo])}, "weights_sha256": wsha, "complete": not missing[fo] and len(files[fo]) == 2 * len(VARIANTS)}
        mb = json.dumps(man, indent=1, default=str).encode()
        (od / "manifest_step3.json").open("xb").write(mb)
        out[fo] = {"manifest": str(od / "manifest_step3.json"), "manifest_sha256": hashlib.sha256(mb).hexdigest(), "counts": counts[fo], "complete": man["complete"]}
    OUTV.commit()
    return out


@app.local_entrypoint()
def main(message: str, source_record: str, append_25h: str, rl_root_check: str = "", standing_rule_log: str = "", cutfull: bool = False, profile: str = "cut32k"):
    if profile != "cut32k":  # the full-length refit profile (RULE Addendum F) is not pinned yet: refused
        raise SystemExit(f"profile {profile!r} is not pinned (only cut32k: the fit on cut rows)")
    if cutfull:  # retired (4f, option A): every (a'') full-length root is scored by score_step3_cutfull.py, one code path
        raise SystemExit("--cutfull is retired here: full-length roots go through build_cutfull_message.py and score_step3_cutfull.py")
    CUTFULL["enabled"] = bool(cutfull)
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted changes: commit and push first")
    if append_25h not in (APPEND_25H, APPEND_25H[:9]):
        raise SystemExit("--append-25h must be 09's Append 25h commit pinned in this file")
    for c_ in ("51a612b29", APPEND_25H, ATTEST_0D):
        if not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", c_).decode().splitlines()):
            raise SystemExit(f"09's commit {c_} is not on origin")
    raw = Path(message).read_bytes()
    rc_, rp_ = source_record.split(":", 1)  # a COMMITTED source record on origin (famexp record or this folder's adapter record; Codex simhb r1 item 2)
    if not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", rc_).decode().splitlines()):
        raise SystemExit("the famexp record's commit is not on origin")
    rb_ = git("show", f"{rc_}:{rp_}")
    rec = json.loads(rb_)
    res_ = rec.get("result") or {}
    passed = res_.get("gate_pass") is True or (isinstance(rec.get("gate"), dict) and rec["gate"] and all(g.get("pass") is True for g in rec["gate"].values()))
    if hashlib.sha256(raw).hexdigest() != rec.get("message_sha256") or not passed or res_.get("error"):
        raise SystemExit("the message sha256 differs from the committed famexp record's, or that record's gates did not pass")
    if git("rev-parse", "HEAD") != git("rev-parse", "origin/aswin/step3-retrain") or git("ls-files", "--error-unmatch", str(Path(__file__).relative_to(REPO))).strip() == b"":
        raise SystemExit("this scorer must be tracked and HEAD pushed")
    msg = json.loads(raw)
    keys = [e["key"] for e in msg["epochs"]]
    batches = {FOLDER_RE.fullmatch(e["folder"]).group(1) if FOLDER_RE.fullmatch(e["folder"]) else None for e in msg["epochs"]}
    if any(e["world"] not in WORLDS or not FOLDER_RE.fullmatch(e["folder"]) for e in msg["epochs"]) or len({(e["world"], e["cell"]) for e in msg["epochs"]}) != len(msg["epochs"]) \
            or len(set(keys)) != len(keys) or len(batches) != 1:
        raise SystemExit("a world or folder outside the sim-honest grammar, a duplicate world|cell or key, or more than one batch")
    (batch,) = batches
    if (msg["volume"] == CUTFULL_VOLUME) != bool(cutfull):
        raise SystemExit(f"--cutfull messages read only {CUTFULL_VOLUME} (read-only), and registered messages never do")

    # 09 Append 41 am. 2 (d): a cutfull message names, per key, the authoritative original folder (from d9's cutfull manifest, via the adapter)
    all_rows = msg["epochs"] + [y for p_ in msg.get("provenance", []) for y in p_.get("excluded_by_staging", [])]
    cut_rows = [x for x in all_rows if "_cutfull_" in x["folder"]]
    if cut_rows != (all_rows if cutfull else []):  # --cutfull: every row is cutfull (Codex cutfull r2 item 1); otherwise none is
        raise SystemExit("--cutfull needs every row in a cutfull folder, and cutfull folders need --cutfull")
    if cut_rows:
        strip_try = lambda f: re.sub(r"_try[0-9]+__", "__", f)  # noqa: E731
        bad_o = [x.get("folder") for x in cut_rows if not x.get("original_folder") or not _FOLDER_RE_BASE.fullmatch(x["original_folder"])
                 or strip_try(x["original_folder"]) != strip_try(orig_name(x["folder"]))]
        if bad_o:
            raise SystemExit(f"09 (d): a cutfull row without a manifest-named original folder that matches its derived original: {bad_o[:3]}")
        # ... and the named original is authoritative: a folder this scorer scored (a tracked record here) whose committed message holds that key
        tracked_ = set(git("ls-files", "--", str((HERE / "results").relative_to(REPO))).decode().split())
        msgs_ = {hashlib.sha256(q.read_bytes()).hexdigest(): q for q in (HERE / "results" / "simh_msgs").glob("msg_*.json") if str(q.relative_to(REPO)) in tracked_}
        okeys = {}
        for rp2 in sorted((HERE / "results").glob("simh_batch_*.json")):
            r_ = json.loads(rp2.read_text()) if str(rp2.relative_to(REPO)) in tracked_ else {}
            if r_.get("message_sha256") in msgs_:
                m2 = json.loads(msgs_[r_["message_sha256"]].read_text())
                for fo in r_.get("result") or {}:
                    okeys.setdefault(fo, set()).update(x["key"] for x in m2["epochs"] + [y for q_ in m2.get("provenance", []) for y in q_.get("excluded_by_staging", [])]
                                                       if x["folder"] == fo)
        bad_k = sorted({x["original_folder"] for x in cut_rows if x["key"] not in okeys.get(x["original_folder"], set())})
        if bad_k:
            raise SystemExit(f"09 (d): no tracked scoring record here holds these keys in the named original folder: {bad_k[:3]}")
        # inventory v3 pin: each epoch is an inventory run with the same cutfull folder and original folder, and the epochs of these folders
        # are exactly the inventory's runs for them (staging exclusions are not (a'') runs)
        if not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", INVENTORY[0]).decode().splitlines()):
            raise SystemExit("the inventory's commit is not on origin")
        ib = git("show", f"{INVENTORY[0]}:{INVENTORY[1]}")
        if hashlib.sha256(ib).hexdigest() != INVENTORY[2]:
            raise SystemExit("inventory v3 sha256 differs from its pin")
        inv = {r["key"]: r for r in json.loads(ib)["runs"]}
        bad_i = [e["key"] for e in msg["epochs"] if e["key"] not in inv or inv[e["key"]]["uncut_planned"]["folder"] != e["folder"]
                 or inv[e["key"]]["original_folder"] != e["original_folder"]]
        fos = {e["folder"] for e in msg["epochs"]}
        if bad_i or {k for k, r in inv.items() if r["uncut_planned"]["folder"] in fos} != {e["key"] for e in msg["epochs"]}:
            raise SystemExit(f"cutfull epochs differ from inventory v3 for their folders (folder, original folder or key set): {bad_i[:3]}")
    rlr = {}
    if any(is_rlr(x["folder"]) for p_ in msg.get("provenance", []) for x in p_.get("excluded_by_staging", [])) and \
            not any(is_rlr(e["folder"]) for e in msg["epochs"]):
        raise SystemExit("an _rlr folder appears only among staging exclusions: an _rlr root must be routed as its own message")
    if any(is_rlr(e["folder"]) for e in msg["epochs"]):  # 09 @ 91485ce80: the regex admits nothing by itself
        for c_ in RLR_RULINGS:
            if not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", c_).decode().splitlines()):
                raise SystemExit(f"RL ruling {c_} is not on origin")
        xs = [x for p_ in msg.get("provenance", []) for x in p_.get("excluded_by_staging", [])]
        keys_ok = dict(RLR_KEYS)
        sr = {}
        if standing_rule_log:  # 09 @ dd93a41cd section 2: a STANDING_RULE_ADD line in 09's READ_LOG admits exactly that key in its parent's _rlr root
            if not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", standing_rule_log).decode().splitlines()):
                raise SystemExit("the READ_LOG commit is not on origin")
            rl = git("show", f"{standing_rule_log}:{READ_LOG}")
            for ln in rl.decode().splitlines():
                m_ = SR_LINE.match(ln.strip())
                rf_ = FOLDER_RE.fullmatch(m_.group(6)) if m_ else None  # the line's root binds the batch; its key may name the pool (e.g. simh_nm360)
                pool = "simh_nm360" if batch.startswith("simh_nm360_") else "simh311"
                if m_ and rf_ and rf_.group(1) == batch and is_rlr(m_.group(6)) and m_.group(1) in (batch, pool):
                    keys_ok[(batch, m_.group(2))] = f"standing rule (READ_LOG @ {standing_rule_log})"
                    sr[m_.group(2)] = {"line": ln.strip(), "m_r": int(m_.group(3)), "evidence": m_.group(4), "rootcheck": m_.group(5), "root": m_.group(6)}
            rlr["standing_rule_log"] = {"commit": standing_rule_log, "path": READ_LOG, "sha256": hashlib.sha256(rl).hexdigest(), "admitted": sr}
        bad_k = [e["cell"] for e in msg["epochs"] + xs if (batch, e["cell"]) not in keys_ok]
        bad_k += [e["cell"] for e in msg["epochs"] + xs if e["cell"] in sr and base_root(e["folder"]) != sr[e["cell"]]["root"]]  # a standing-rule key only in its named root
        # each ruled key only in its assigned root (k = 1, or the second launch for the moved keys); standing-rule keys in their line's root
        bad_k += [e["cell"] for e in msg["epochs"] + xs if e["cell"] not in sr and (batch, e["cell"]) in keys_ok
                  and base_root(e["folder"]) != RLR_ROOT_MOVED.get((batch, e["cell"]), f"{batch}_sim_rlr_v1__sim_honest")]
        if bad_k or not all(is_rlr(e["folder"]) for e in msg["epochs"] + xs) or len({e["folder"] for e in msg["epochs"] + xs}) != 1:
            raise SystemExit(f"an _rlr root holds keys (scored or staged out) outside 09's committed RL rulings, or mixes roots: {bad_k}")
        need = sorted({e["cell"] for e in msg["epochs"]})
        if not rl_root_check:
            raise SystemExit(f"09: {need} need d9's committed by-key root check (--rl-root-check commit:path[,commit:path])")
        rlr["root_check"], txt = [], ""
        for rc in rl_root_check.split(","):
            kc, kp = rc.split(":", 1)
            if not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", kc).decode().splitlines()):
                raise SystemExit("a root-check record is not on origin")
            rb2 = git("show", f"{kc}:{kp}")
            txt += rb2.decode()
            rlr["root_check"].append({"commit": kc, "path": kp, "sha256": hashlib.sha256(rb2).hexdigest()})
        if any(c not in txt for c in need):
            raise SystemExit(f"the root-check record(s) do not name every replaced key: {[c for c in need if c not in txt]}")
        full_c = {git("rev-parse", "--verify", f"{x['commit']}^{{commit}}").decode().strip() for x in rlr["root_check"]}
        full_f = {x["sha256"] for x in rlr["root_check"]}
        present = {e["cell"] for e in msg["epochs"] + xs}
        for c, v in sr.items():  # every standing-rule key present (scored or staged out): its rootcheck must be a pinned root-check record
            if c not in present:
                continue
            rcv = v["rootcheck"]  # (was rc_, which overwrote the source-record commit: routing 59's record; see its correction sidecar)
            if len(rcv) == 64:
                ok_ = rcv in full_f
            elif len(rcv) >= 7:
                r_ = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", "--quiet", f"{rcv}^{{commit}}"], capture_output=True, text=True)
                ok_ = r_.returncode == 0 and r_.stdout.strip() in full_c
            else:
                ok_ = False
            if not ok_:
                raise SystemExit(f"{c}: READ_LOG rootcheck {rcv} does not resolve to a pinned root-check record (full commit or file sha256)")
        own = {}  # this scorer's own COMMITTED evidence: each original was staged out of a root that a committed scoring record covers
        tracked = set(git("ls-files", "--", str((HERE / "results").relative_to(REPO))).decode().split())
        for rp2 in sorted((HERE / "results").glob("simh_batch_*.json")):
            if str(rp2.relative_to(REPO)) not in tracked:
                continue
            r_ = json.loads(rp2.read_text())
            src_ = r_.get("source_record") or {}
            if not src_.get("commit") or not all(m_.get("complete") is True for m_ in (r_.get("result") or {}).values()):
                continue
            sh_ = subprocess.run(["git", "-C", str(REPO), "show", f"{src_['commit']}:{src_['path']}"], capture_output=True)
            ab_ = sh_.stdout if sh_.returncode == 0 else git("show", f"HEAD:{src_['path']}")  # identity is the sha256 below, not the commit field
            if hashlib.sha256(ab_).hexdigest() != src_.get("sha256"):
                raise SystemExit(f"{rp2.name}: its source adapter record does not match the recorded sha256")
            a_ = json.loads(ab_)  # the committed adapter record the scoring used
            if a_.get("message_sha256") != r_.get("message_sha256"):
                continue
            fm_ = FOLDER_RE.fullmatch(a_["folder"])
            if fm_ and fm_.group(1) == batch and not is_rlr(a_["folder"]) and a_["folder"] in (r_.get("result") or {}):
                for x in a_["excluded_by_staging"]:
                    own.setdefault(x["cell"], []).append(a_["folder"])
        missing_own = [c for c in need if c not in own]
        if missing_own:
            raise SystemExit(f"no step-3 record shows the original of {missing_own} staged out in a root of {batch}: stop")
        rlr["originals_staged_out_in"] = {c: own[c] for c in need}
        rlr["keys"] = {e["cell"]: {"ruling": keys_ok[(batch, e["cell"])], "grammar": "91485ce80",
                                   "label": "RL replacement (Append 35 infrastructure)" if keys_ok[(batch, e["cell"])] in ("13be06982", "c5420f2f5", "81f75b763")
                                   or ("standing rule" in keys_ok[(batch, e["cell"])] and "class=infrastructure" in sr.get(e["cell"], {}).get("line", ""))
                                   else "terminal-error replacement (09 @ c10a14948 / dd93a41cd)"}
                       for e in msg["epochs"]}
    want = ("simh_311", "S_new") if batch.startswith("simh311_") else ("t196fill", "S_old") if batch == "simh_t196fill" else ("simh_nm360", "S_newmains")
    if batch == "simh_t196fill" and any(e.get("label") != A25_FILL_LABEL for e in msg["epochs"]):
        raise SystemExit("an A25 fill-in row without 09's label")
    if any((e["world"], e["stratum"]) != want for e in msg["epochs"]):
        raise SystemExit(f"{batch}: every epoch must be world/stratum {want}")
    prov = {}
    for p_ in msg.get("provenance", []):  # extraction evidence per contributing root (Codex simhb r1 item 1): H200 on every shard, staging record
        dr, sr = p_.get("device_record") or {}, p_.get("staging_record") or {}
        if not dr.get("commit") or not sr.get("commit"):
            raise SystemExit(f"{p_.get('run')}: provenance lacks a device or staging record")
        drec = json.loads(git("show", f"{dr['commit']}:{dr['path']}"))
        shards = drec.get("shards") if isinstance(drec, dict) else None
        if not isinstance(shards, list) or not shards or any((sh.get("gpu") or sh.get("gpu_name")) != "NVIDIA H200" for sh in shards):
            raise SystemExit(f"{p_.get('run')}: the device record's shards do not each show NVIDIA H200 (a missing GPU field fails)")
        gpus = []

        def scan(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k in ("gpu", "gpu_name") and isinstance(v, str):
                        gpus.append(v)
                    scan(v)
            elif isinstance(o, list):
                for v in o:
                    scan(v)
        scan(drec)
        if not gpus or set(gpus) != {"NVIDIA H200"}:
            raise SystemExit(f"{p_.get('run')}: the device record does not show NVIDIA H200 on every shard")
        prov[p_["run"]] = {"device_record": dr | {"sha256": hashlib.sha256(git("show", f"{dr['commit']}:{dr['path']}")).hexdigest(), "gpu_entries": len(gpus)},
                           "staging_record": sr | {"sha256": hashlib.sha256(git("show", f"{sr['commit']}:{sr['path']}")).hexdigest()}}
    if {e["run"] for e in msg["epochs"]} - set(prov):
        raise SystemExit("an extraction root without provenance in the message")
    pins = {}
    for nm, (c, p, h) in {"q2": Q2, "newmains": NEWMAINS, "t196": T196, "r36": R36}.items():
        b = git("show", f"{c}:{p}")
        if hashlib.sha256(b).hexdigest() != h:
            raise SystemExit(f"{nm} sha256 differs from its pin")
        pins[nm] = json.loads(b)
    main_of = {c["key"]: c["main"] for nm in ("q2", "newmains", "t196") for c in pins[nm]["cells"]}
    lists = {"S_new": {c["key"] for c in pins["q2"]["cells"]}, "S_newmains": {c["key"] for c in pins["newmains"]["cells"]}, "S_old": {c["key"] for c in pins["t196"]["cells"]}}
    if any(e["cell"] not in lists[e["stratum"]] for e in msg["epochs"]):
        raise SystemExit("a cell outside its stratum's pinned list")
    t196_mains = {c["main"] for c in pins["t196"]["cells"]}
    fom = {}
    for r in pins["r36"]:
        if fom.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("r36: a main in two folds")
    if batch != "simh_t196fill" and any(main_of[e["cell"]] in t196_mains for e in msg["epochs"]):
        raise SystemExit("a sim-honest main is a T196 main: the step-3 models may have trained on it")
    task_fold = {e["cell"]: fom.get(main_of[e["cell"]], -1) for e in msg["epochs"]}
    if batch == "simh_t196fill" and any(f < 0 for f in task_fold.values()):  # T196 cells: the fold model that held the task out (checked in-job vs fot)
        raise SystemExit("a T196 fill-in cell without a runs36 fold")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    head = {"label": "step-3 recipes on a sim-honest batch (RULE Addendum A; 09 @ 51a612b29 and Append 25h): SEALED; hashes, counts and routing only",
            "status": "POST HOC end-to-end inputs", "scorer_git": git("rev-parse", "HEAD").decode().strip(), "stamp (PT)": stamp, "append_25h": append_25h,
            "message_sha256": rec["message_sha256"], "source_record": {"commit": rc_, "path": rp_, "sha256": hashlib.sha256(rb_).hexdigest()}, "step3_fit": FIT3,
            "batch": batch, "provenance_checked": prov, "rl_replacement": rlr or None,
            "cutfull": bool(cutfull), "profile": profile, **({"line": "cutfull"} if cutfull else {}),
            **({"pool": "A25 + fill-in (labelled; NOT the registered primary_v3 / S_old pool)", "row_label": A25_FILL_LABEL,
                "sealed_until": "the second refresh's input freeze (09 @ 286b2f518 section 4)"} if batch == "simh_t196fill" else {}),
            "run_transform": {"R": "per model, mean of reply rows", "C": "per model, max of content-only command rows"},
            "schema": "pass 1's + run_score_by_model (N_runs x 5); score / run_score NaN on fold -1"}
    out = score.with_options(volumes={"/out": OUTV, "/w3": W3, "/b": modal.Volume.from_name(msg["volume"], version=2).read_only()}).remote(msg, task_fold, head)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"simh_batch_{msg['d9_arm']}_{stamp}.json").open("x").write(json.dumps(head | {"result": out,
        "unmapped_cells": sorted(c for c, f in task_fold.items() if f < 0)}, indent=1))
    print(json.dumps({fo: {k: v for k, v in r.items() if k != "counts"} for fo, r in out.items()}, indent=1))
