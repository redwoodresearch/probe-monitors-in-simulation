"""RULE.md "Pass 2" (09's scale-up schema): score ONE d9 message (several worlds) with pass 1's frozen fold models; no refit, no
re-selection. Per (variant, readout) npz of RAW logits from all five fold models (`score_by_model`, N x 5), plus `fold` (the
main's Sol #36 fold, -1 outside every fold); no run-level score (09's scale-up X_err builds R mean / C max, z-scores and applies
P: Append 21b @ b42fc829). Output: convaudit_work:/scaleup/<d9_arm>/famexp/[sim/]; manifest_famexp.json written LAST.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach <this file> --spec <prep stamp> \\
      --fit-run <pass-1 fit stamp> --message <json> --message-sha <sha256> --r36 <runs36.json> --open-manifest </p2/...> --open-sha <sha>
      [--elig-commit <c> --elig-path <p> --elig-sha256 <h>]
Message (d9's, or a committed adapter's output): {"d9_arm", "volume", "epochs": [{"key", "cell", "world", "stratum", "run", "arm",
"staged", "staged_sha256"}]} in extract_q4b.py's layout (meta/<arm>/<stem>.json, by_layer/L<L>/<chunk>.npy).
"""

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
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-p2probes_famexp2_kill-2026-11-13"
BUILD = Path("/tmp/famexp_build")
BUNDLE = ("avg_v1.json", "corrected_map.json", "merge.json", "run_avg_transfer.py", "run_correctedb.py", "t196.json", "torch_probe_pinned.py")
LAYERS = (16, 32, 48, 34, 46, 47, 49, 63)
B_ASIS = {"R": "xfer/asis_2026-10-14_135701/outer_tight_L{L}.npz", "C": "xfer/asis_2026-10-14_135701_cmd/outer_tight_L63.npz"}
WORLDS = ("run1", "run2", "sim", "honest_a", "honest_b", "honest_c", "run3", "sim2", "sim_ug")  # run3 / sim2: the interleaved (IL) launch (09's Append 32); sim_ug: the ungrounded arm (Append 34)
STRATA = ("S_old", "S_new", "S_newmains")  # 09's ORIGIN_STRATUM: T196, q2's +311 list, the launcher's 360 new-main cells (Append 21c)
ROSTER = [f"own_frac{p}_L{L}" for p in (10, 25, 100) for L in (16, 32, 48)] + ["ad_k5", "ad_k10", "ad_k20", "b_asis"]  # x readouts R, C = 26
FOLDER_RE = re.compile(r"^(su311|nm360)_(q[0-9]+r?|s[0-9]+)_(main|run2|sim)(_s[0-9]+)?_v1(_try[0-9]+)?__(honest_a|honest_b|attack|sim_attack)$")  # 09's ONE canonical regex, corrected (via 4f, 10-14 ~19:4x)
FOLDER_OK = ("il311_q2r_sim2_v1__sim_attack", "ilold_oldr_run3_v1__attack", "ilnm_s4dr_sim2_s1_v1__sim_attack", "il311_q1_run3_v1__attack", "ilnm_s4d_sim2_s1_v1__sim_attack", "ilold_old_run3_v1_try2__attack", "su311_q2r_sim_v1__sim_attack", "nm360_s3_main_s1_v1__honest_a", "nm360_s3_run2_s1_v1_try2__attack", "nm360_s3_sim_s2_v1__sim_attack", "su311_q1_main_v1__attack",
             "su311_q4_sim_v1_try3__sim_attack", "su311_q2_run2_s4_v1__honest_b")
FOLDER_BAD = ("il311_q2rr_sim2_v1__sim_attack", "il311_q2_rlr_sim2_v1__sim_attack", "il311_rlr_q2_sim2_v1__sim_attack", "il311_q1_sim2_v1__attack", "il311_q5_run3_v1__attack", "su311_q2r_main_v1__attack", "su311_q2r_run2_v1__attack", "nm360_s3_main_s1_extract_v1__honest_a", "su311_q1_main_v1__sim_honest", "../su311_q1_main_v1__attack", "su311_q1_main_v1__attack/x",
              "su311_q1_main_v1__attack\n", "su311_q1t_main_TEST_v1_pilot3__attack", "xx_q1_main_v1__attack", "su311_q1_main_v2__attack")
FOLDER_RE_IL = re.compile(r"^(?:il311_q[1-4]r?|ilnm_s(?:[1-5]|4d)r?|ilold_oldr?)_(?:run3(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__attack|sim2(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__sim_attack)$")  # Append 32 @ 9fd14891a (r = rate-limit rerun)
FOLDER_RE_NM360 = re.compile(r"^nm360_(?:(?:s4d_sim(?:_s[0-9]+)?|(?:s5r|s4r)_sim)_v1(?:_try[0-9]+)?__sim_attack|s4r_(?:main_v1(?:_try[0-9]+)?__(?:attack|honest_a|honest_b)|run2_v1(?:_try[0-9]+)?__attack))$")  # 09 §1 "nm360 follow-up and relaunch names" @ bffe22e3e, verbatim (s5r/s4r/s4d follow-ups, s4r relaunch)
FOLDER_RE_ILC = re.compile(r"^(?:il311_q[1-4](?:r|r2|r3|c)?|ilnm_s(?:[1-5]|4d)(?:r|r2|r3)?|ilold_old(?:r|r3|c)?)_(?:run3(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__attack|sim2(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__sim_attack)$")  # 09 Append 32 Addendum 1 item 4 @ 7bd9d5e89, verbatim: tokens r2, r3, c
IL_TOKEN_RE = re.compile(r"^(?:il311_q[1-4](r|r2|r3|c)|ilnm_s(?:[1-5]|4d)(r|r2|r3)|ilold_old(r|r3|c))_")  # an IL rerun (r, r2, r3) or completion (c) folder
TOKEN_PINS = {"r": ("supersession", "freeze_addendum"),  # IL rule @ c1c770694 item 6: d9's supersession map + the freeze addendum, per message (fail closed)
              "c": ("registration", "supersession"), "r2": ("registration", "supersession"),  # Addendum 1 item 4: a token needs its committed registration + supersession record
              "r3": ("registration", "supersession", "rlr_final", "rlr3_selection", "rlr3_validation")}  # item 2: rlr3 also needs the final, selection and validation records
FOLDER_OK_NM360 = ("nm360_s5r_sim_v1__sim_attack", "nm360_s4d_sim_s1_v1__sim_attack", "nm360_s4r_sim_v1__sim_attack", "nm360_s4r_main_v1__honest_a", "nm360_s4r_run2_v1_try2__attack")  # 09 §1 forms
FOLDER_BAD_NM360 = ("nm360_s4r_run2_v1__honest_a", "nm360_s5r_main_v1__attack", "nm360_s5r_sim_s1_v1__sim_attack")  # 09 §1 rejects (snapshots only on s4d)
FOLDER_OK_ILC = ("il311_q1c_run3_v1__attack", "ilold_oldc_sim2_v1__sim_attack", "ilnm_s1r2_sim2_v1__sim_attack", "il311_q4r2_run3_v1_try1__attack", "ilnm_s4dr3_run3_v1__attack", "ilold_oldr3_sim2_v1__sim_attack")  # Addendum 1 forms
FOLDER_BAD_ILC = ("ilnm_s1c_run3_v1__attack", "ilold_oldr2_sim2_v1__sim_attack", "il311_q1c_sim2_v1__attack", "il311_q1cr_run3_v1__attack", "il311_q5r2_run3_v1__attack", "ilnm_s1r4_sim2_v1__sim_attack")
FOLDER_RE_UG = re.compile(r"^ug_sim(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__sim_attack_ug$")  # 09's Append 32 format note §2 (ug, Append 34), verbatim
FOLDER_OK_UG, FOLDER_BAD_UG = ("ug_sim_v1__sim_attack_ug", "ug_sim_v1_try2__sim_attack_ug", "ug_sim_s1_v1__sim_attack_ug"), ("ug_sim_v1_try2__sim_attack", "ug_sim_v1__sim_attack_ug_x", "ug_main_v1__sim_attack_ug")
FOLDER_RE_HA25F = re.compile(r"^su311_q[124]_main_ha25f_v1(_try[0-9]+)?__honest_a$")  # 09's 25f H_A folder-name note @ ef5654c0e, extended @ 1bc718e98, verbatim
HA25F = {"su311_q1_main_ha25f_v1": ("t389_model_registry", "d160ffdc956ffc89180043b3575b002d4b0f1cc3f7600082f38decc81f28dae0"),  # 09's 25f H_A note @ 46dda27f7 (+ t309 @ 1bc718e98):
         "su311_q2_main_ha25f_v1": ("t309_slack_clone", "69c964700620933d0fbfa771fc83068d208a3e20c48dfad47d443a2ec88f5028"),  # each dedicated folder holds exactly its cell's
         "su311_q4_main_ha25f_v1": ("t325_model_registry", "e1cdf8d1594bac78b6e64377557bf63d008f2af0248912f21c7bd60c523b7247")}  # selected honest_a run, by source .eval sha256
FOLDER_OK_HA25F, FOLDER_BAD_HA25F = ("su311_q4_main_ha25f_v1__honest_a", "su311_q1_main_ha25f_v1_try2__honest_a"), ("su311_q3_main_ha25f_v1__honest_a", "su311_q4_main_ha25f_v1__honest_b", "su311_q4_main_ha25f_v2__honest_a")
FOLDER_RE_EP2 = re.compile(r"^ep2_(t196|311_l1|311_l2|nm)_v1(_try[0-9]+)?__honest_(a|b)$")  # 09's epoch-2 real-honest folders (Append 38 §4/§5), verbatim
EP2_ORIGIN = {"t196": "S_old", "311_l1": "S_new", "311_l2": "S_new", "nm": "S_newmains"}
EP2_SEALED = "SEALED (09 Append 38 §4/§5): per-row scores only, nothing derived; unread until 09 verifies the epoch-2 pool-key artifact"
FOLDER_OK_EP2, FOLDER_BAD_EP2 = ("ep2_311_l2_v1__honest_a", "ep2_t196_v1_try2__honest_b", "ep2_nm_v1__honest_a"), ("ep2_311_l3_v1__honest_a", "ep2_t196_v1__honest_c", "ep2_311_l2_v1__attack")
FOLDER_RE_OLDSEL = re.compile(r"^ilold_oldsel_(?:run3_v1(?:_try[0-9]+)?__attack|sim2_v1(?:_try[0-9]+)?__sim_attack)$")  # 09's sel97 add-on note (Append 32 @ 2026-10-15 11:51), verbatim
OLDSEL_LABEL = "descriptive add-on; not a registered IL launch or replacement; cannot amend the registered IL result"  # 09: marked on each oldsel manifest
FOLDER_OK_OLDSEL, FOLDER_BAD_OLDSEL = ("ilold_oldsel_run3_v1__attack", "ilold_oldsel_sim2_v1_try2__sim_attack"), ("ilold_oldsel_run3_v1__sim_attack", "ilold_oldsel_sim2_s1_v1__sim_attack", "ilold_oldselr_run3_v1__attack")
_folder_ok = lambda fo: (bool(FOLDER_RE.fullmatch(fo)) and (not re.match(r"^[a-z0-9]+_q[0-9]+r_", fo) or "_sim" in fo.split("__")[0])) or bool(FOLDER_RE_IL.fullmatch(fo)) or bool(FOLDER_RE_NM360.fullmatch(fo)) or bool(FOLDER_RE_ILC.fullmatch(fo)) \
    or bool(FOLDER_RE_UG.fullmatch(fo)) or bool(FOLDER_RE_HA25F.fullmatch(fo)) or bool(FOLDER_RE_OLDSEL.fullmatch(fo)) or bool(FOLDER_RE_EP2.fullmatch(fo))  # noqa: E731  (r: sim arms only)
if not all(_folder_ok(x) for x in FOLDER_OK + FOLDER_OK_NM360 + FOLDER_OK_ILC + FOLDER_OK_UG + FOLDER_OK_HA25F + FOLDER_OK_OLDSEL + FOLDER_OK_EP2) \
        or any(_folder_ok(x) for x in FOLDER_BAD + FOLDER_BAD_NM360 + FOLDER_BAD_ILC + FOLDER_BAD_UG + FOLDER_BAD_HA25F + FOLDER_BAD_OLDSEL + FOLDER_BAD_EP2) \
        or any(IL_TOKEN_RE.match(x) for x in FOLDER_OK_OLDSEL) \
        or not all(IL_TOKEN_RE.match(x) for x in FOLDER_OK_ILC) or any(IL_TOKEN_RE.match(x) for x in ("il311_q2_sim2_v1__sim_attack", "ilnm_s1_sim2_v1__sim_attack", "su311_q2r_sim_v1__sim_attack")) \
        or [IL_TOKEN_RE.match(x).group(0) for x in ("il311_q2r_sim2_v1__sim_attack", "ilold_oldr_run3_v1__attack")] != ["il311_q2r_", "ilold_oldr_"] \
        or FOLDER_RE_NM360.fullmatch("nm360_s5_sim_v1__sim_attack"):  # 09's accept / reject examples
    raise SystemExit("FOLDER_RE fails its accept / reject examples")
FOLDER_RE_TEST = re.compile(r"[A-Za-z0-9_-]+__[A-Za-z0-9_]+")
NEWMAINS = ("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json", "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e")  # membership
FOLDER_STRATUM = {"su311_": "S_new", "nm360_": "S_newmains", "il311_": "S_new", "ilnm_": "S_newmains", "ilold_": "S_old"}  # 09: an su311_ folder holds ONLY S_new rows, an nm360_ folder ONLY S_newmains


def _stratum_folder_fails(epochs) -> list:  # a ug_ folder holds ONLY world sim_ug, whose stratum is the cell's origin (S_old / S_new; 09 Q4 @ 48363962a: registry label sim_ug)
    def ep2(e):  # an ep2_ folder holds ONLY honest_a / honest_b rows of registry stratum ep2, whose stratum is the folder label's origin
        m_ = FOLDER_RE_EP2.fullmatch(str(e["folder"]))
        return bool(m_) and e.get("registry_stratum") == "ep2" and e["world"] == f"honest_{m_[3]}" and e["stratum"] == EP2_ORIGIN[m_[1]]
    ug = lambda e: str(e["folder"]).startswith("ug_") or e["world"] == "sim_ug" or e.get("registry_stratum") == "sim_ug"  # noqa: E731
    isep2 = lambda e: str(e["folder"]).startswith("ep2_") or e.get("registry_stratum") == "ep2"  # noqa: E731
    return [f"{e['world']}|{e['cell']}" for e in epochs if (isep2(e) and not ep2(e))
            or (not isep2(e) and not ug(e) and e["stratum"] != next((v for k, v in FOLDER_STRATUM.items() if str(e["folder"]).startswith(k)), None))
            or (not isep2(e) and ug(e) and not (str(e["folder"]).startswith("ug_") and e["world"] == "sim_ug" and e.get("registry_stratum") == "sim_ug" and e["stratum"] in ("S_old", "S_new")))]


def _token_unpinned(msg) -> list:
    """Addendum 1 item 4 (fail closed): a c / r2 / r3 folder whose runs do not all have message parts citing every pin its token needs (commit, path, sha256)."""
    bad = []
    for fo in sorted({e["folder"] for e in msg["epochs"]} | {x["folder"] for p_ in msg.get("provenance", []) for x in p_.get("excluded_by_staging", [])}):
        m_ = IL_TOKEN_RE.match(str(fo))
        if not m_:
            continue
        runs = {e["run"] for e in msg["epochs"] if e["folder"] == fo}
        parts = [p_ for p_ in msg.get("provenance", []) if p_.get("run") in runs or any(x.get("folder") == fo for x in p_.get("excluded_by_staging", []))]
        if not parts or not runs <= {p_.get("run") for p_ in parts} or not all(
                all(((p_.get("citations") or {}).get(k) or {}).get(f) for k in TOKEN_PINS[next(g for g in m_.groups() if g)] for f in ("commit", "path", "sha256")) for p_ in parts):
            bad.append(fo)
    return bad


SIMH = ("573054a4d", "experiments/2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717/runs.json",
        "f420ed2401dcc776cbbcfd414bbe9d6d1132a3bbd012b529a33aba7fa5ecd068")  # b_asis's SH source (2e's run_asis_fits.py)
OUT_SUB = "famexp_r2"  # 09's ruling (a): this scorer routes PER PROBE; it writes famexp_r2/ and marks an older famexp/ SUPERSEDED
NM_BUILT = ("6feeee4b7", "experiments/2026-10-14_q3_new_mains_launch/config/nm_built.json", "93d9b538710b6bd930e7df01b0f5296d417550a0f0a35bcb6eb696166cf39a04")  # d9's built key -> main
Q2 = ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json",
      "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b")
T196 = ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")
B_T196 = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
TOL_GATE = {"p1": 1e-9, "b_asis": 1e-4}  # CPU float64 vs pass 1's GPU float64 (dot-product order); b_asis vs 2e's stored rows
TIMEOUT_LAYER, TIMEOUT_GATE, TIMEOUT_ORCH = 600, 900, 2400
if modal.is_local():  # run_famexp.py's import materializes the pinned 2e row code into BUILD and refuses any byte difference
    sys.path.insert(0, str(HERE))
    import run_famexp  # noqa: F401

app = modal.App(APP)
OUT = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2)
SOL = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2).read_only()
WORK = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2)
img = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "safetensors==0.8.0").add_local_dir(BUILD, "/root/x2e"))


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


@app.function(image=img, cpu=(8, 8), memory=(32768, 32768), timeout=TIMEOUT_LAYER, retries=0, scaledown_window=2, volumes={"/out": OUT, "/sol": SOL})
def score_layer(spec_stamp: str, fit_run: str, msg: dict, task_fold: dict, L: int, stamp: str, fp: str = "") -> dict:
    _check_fp(fp)
    import numpy as np

    rc, _ = _x2e()
    base = Path("/out/famexp") / spec_stamp
    spec = json.loads((base / f"spec_{spec_stamp}.json").read_text())
    fit = [x for g in json.loads((base / f"fit_{fit_run}.json").read_text())["groups"] for x in g["fits"]]
    eps, fixed, bad = {}, set(spec["fixed_ids"]), []
    cfgs = {}
    for e in msg["epochs"]:  # d9's chain: the staged sha256 comes from the extraction run's own config.json (make_pass2_message.py)
        if e["staged_sha256"] == "from_config":
            if e["run"] not in cfgs:
                cfgs[e["run"]] = json.loads((Path("/b") / e["run"] / "config.json").read_text())["inputs_sha256"]
            e["staged_sha256"] = cfgs[e["run"]][f"{e['arm']}.jsonl"]
    for path, h in sorted({(e["staged"], e["staged_sha256"]) for e in msg["epochs"]}):
        raw = Path("/b", path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != h:
            raise SystemExit(f"staged inputs {path}: sha256 differs from the message")
        eps |= {(path, x["key"]): x for x in (json.loads(y) for y in raw.decode().splitlines() if y)}
    for e in msg["epochs"]:  # staged coverage and extraction consistency (Codex round 11)
        st = eps.get((e["staged"], e["key"]))
        mt = json.loads((Path("/b") / e["run"] / "meta" / e["arm"] / f"{rc.stem(e['key'])}.json").read_text()) if st else None
        if not st or mt["key"] != e["key"] or mt.get("inputs_sha256") != e["staged_sha256"] or [(c["irid"], c["call_index"], c["kind"]) for c in st["calls"]] != \
                [(c["irid"], c["call_index"], c["kind"]) for c in mt["calls"]]:
            bad.append(e["key"])
    if bad:
        raise SystemExit(f"{len(bad)} epochs missing from staging or inconsistent with the extraction meta, e.g. {bad[:2]}")
    skip = {e["key"]: [[r["irid"], r["call_index"]] for r in eps[(e["staged"], e["key"])]["calls"]
                       if r["kind"] == "call_end" and r["readout"] is not None and eps[(e["staged"], e["key"])]["ids"][r["readout"]] in fixed] for e in msg["epochs"]}
    od, rec, cache = base / f"pass2_{stamp}" / "layers", {"layer": L, "files": {}, "weights": {}}, {}
    od.mkdir(parents=True, exist_ok=True)
    for readout in ("R", "C"):
        res, wsha = _score_rows(_own_spec(msg["epochs"], readout, skip), L, task_fold, spec, fit, fit_run, base, readout, cache)
        rec["weights"] |= wsha
        for v, (names, units, ci, folds, S) in res.items():
            buf = io.BytesIO()
            np.savez(buf, key=np.asarray(names), unit=units, call_index=ci, fold=folds, score_by_model=S)
            fp = od / f"{v}__{readout}__L{L}.npz"
            fp.open("xb").write(buf.getvalue())
            rec["files"][fp.name] = hashlib.sha256(buf.getvalue()).hexdigest()
    OUT.commit()
    return rec


def _needs(spec, v, readout):
    """The layers a variant x readout's designated score needs (as _designated combines them)."""
    if v.startswith("own_"):
        return {int(v.rsplit("_L", 1)[1])}
    return {L for pair in spec["layer_choice_a"] for L in pair} if readout == "R" else {63}


def _fingerprint() -> str:
    """sha256 over this module's bytes and every file of the pinned helper bundle (local BUILD, or /root/x2e in a container):
    the code a launch expects, checked inside each worker before any work (Codex round 37)."""
    root = BUILD if modal.is_local() else Path("/root/x2e")
    h = hashlib.sha256(Path(__file__).read_bytes())
    for name in BUNDLE:  # run_famexp.PINNED's files: a fixed list, so other writers of the shared BUILD dir cannot change it
        h.update(name.encode() + b"\0" + (root / name).read_bytes())
    return h.hexdigest()


def _check_fp(fp: str):
    if fp and fp != _fingerprint():
        raise SystemExit("in-job: this deployment's code fingerprint differs from the launch's (stale deploy): stop before any work")


def _designated(spec, per_layer, v, readout):
    """Combine a variant x readout's layer outputs into score_by_model: own_a = its layer; adapted / b_asis R = for fold model f, the
    mean of its two avg_top2 layers' logits (as pass 1); C = L63 (or own_a's layer)."""
    import numpy as np

    Ls = sorted(per_layer)
    names, units, ci, folds, _ = per_layer[Ls[0]]
    for L in Ls[1:]:
        if not (np.array_equal(per_layer[L][0], names) and np.array_equal(per_layer[L][1], units)):
            raise SystemExit(f"{v} {readout}: layers hold different rows")
    if (v.startswith("ad_") or v == "b_asis") and readout == "R":
        S = np.full((len(names), 5), np.nan)
        for f, (a, b) in enumerate(spec["layer_choice_a"]):
            S[:, f] = (per_layer[a][4][:, f] + per_layer[b][4][:, f]) / 2
    else:
        (S,) = [per_layer[L][4] for L in Ls]
    return names, units, ci, folds, S


@app.function(image=img, cpu=(8, 8), memory=(32768, 32768), timeout=TIMEOUT_GATE, retries=0, scaledown_window=2, volumes={"/out": OUT, "/sol": SOL})
def repro_gate(spec_stamp: str, fit_run: str, open_ref: dict, elig_rec: dict | None = None, fp: str = "") -> dict:
    """09's honest_b gate: re-score T196 honest_b with the same five models; the designated column must equal pass 1's open pack,
    and no honest_b run's hit may FLIP at any alpha (thresholds: v7's phase-1 record if given, else pass 1's own per-fold z).
    Honest runs only: no sim key is ever read here (the open pack holds none; the re-score reads honest_b epochs only)."""
    _check_fp(fp)
    import numpy as np

    base = Path("/out/famexp") / spec_stamp
    spec = json.loads((base / f"spec_{spec_stamp}.json").read_text())
    fit = [x for g in json.loads((base / f"fit_{fit_run}.json").read_text())["groups"] for x in g["fits"]]
    mp = Path("/out", open_ref["path"].removeprefix("/p2/"))
    if hashlib.sha256(mp.read_bytes()).hexdigest() != open_ref["sha256"]:
        raise SystemExit("pass-1 open manifest sha256 differs")
    om = json.loads(mp.read_text())
    ep = [{"key": k, "world": w, "cell": t, "run": spec["run_of"][k], "arm": spec["arm_of"][k]} for k, (w, t) in spec["map"].items() if w == "honest_b"]
    per, cache = {}, {}
    for L in LAYERS:
        for readout in ("R", "C"):
            res, _ = _score_rows(_own_spec(ep, readout, spec["skip"]), L, spec["fot"], spec, fit, fit_run, base, readout, cache)
            for v, r in res.items():
                per.setdefault((v, readout), {})[L] = r
    out = {}
    dec = ((elig_rec or {}).get("result", elig_rec or {}) or {}).get("phase1_eligibility_record", (elig_rec or {}).get("result", elig_rec or {}))
    dec = (dec or {}).get("decisions") or {}
    pools = {"B": ("honest_b",), "BuC": ("honest_b", "honest_c")}

    def runs(keys, units_, vals, readout):  # run level: reply mean, command max, summed in row order exactly as v7's pack_runs
        tot, cnt, mx = {}, {}, {}
        for k, x in zip(keys, vals, strict=True):
            tot[k] = tot.get(k, 0.0) + float(x)
            cnt[k] = cnt.get(k, 0) + 1
            mx[k] = max(mx.get(k, -np.inf), float(x))
        return {k: (tot[k] / cnt[k] if readout == "R" else float(mx[k])) for k in tot}

    def flips(r1, r2, readout, key, tol):
        """Count honest_b runs whose hit differs between pass-1 (r1) and pass-2 (r2) run scores, per pool source and alpha.
        v7's threshold is an order statistic (a pool score itself; hit = score > T), so the run AT the threshold flips on any
        re-score difference. A change with |z1 - T| within the accepted score tolerance (tol / sd, in z) is a tie: listed with
        its margins, not counted as a flip. Only flips outside that band fail the gate."""
        fold = lambda k: spec["fot"][k.split("|", 1)[1]]  # noqa: E731
        d_ = dec.get(key) or {}
        if d_.get("moments") and d_.get("pool"):
            srcs = {f"v7:{d_['pool']}": (d_["moments"], {a: d_.get(a, {}).get("threshold") for a in ("0.01", "0.05")})}
        else:
            srcs = {}
            for pk, ws in pools.items():
                pool = {k: x for k, x in r1.items() if k.split("|", 1)[0] in ws}
                mom = {"mu": {}, "sd": {}}
                for f in range(5):
                    v_ = np.array([x for k, x in pool.items() if fold(k) == f])
                    if len(v_) < 2 or not v_.std() > 0:
                        mom = None
                        break
                    mom["mu"][str(f)], mom["sd"][str(f)] = float(v_.mean()), float(v_.std())
                if mom is None:
                    srcs[f"pass1:{pk}"] = (None, {})
                    continue
                zp = np.sort([(x - mom["mu"][str(fold(k))]) / mom["sd"][str(fold(k))] for k, x in pool.items()])[::-1]
                srcs[f"pass1:{pk}"] = (mom, {a: float(zp[min(int(np.floor(float(a) * len(zp) + 1e-9)), len(zp) - 1)]) for a in ("0.01", "0.05")})
        res = {}
        for src, (mom, T) in srcs.items():
            if mom is None or any(t is None for t in T.values()):
                res[src] = {"usable": False}
                continue
            z = lambda k, x: (x - mom["mu"][str(fold(k))]) / mom["sd"][str(fold(k))]  # noqa: E731
            hb = sorted(k for k in r2 if k in r1 and k.startswith("honest_b|"))
            res[src] = {"usable": True, "n_honest_b": len(hb)}
            for a, t in T.items():
                ch = [k for k in hb if (z(k, r1[k]) > t) != (z(k, r2[k]) > t)]
                band = {k: tol / mom["sd"][str(fold(k))] for k in ch}
                ties = [k for k in ch if abs(z(k, r1[k]) - t) <= band[k]]
                res[src][f"flips@{a}"] = len(ch) - len(ties)
                res[src][f"ties@{a}"] = [{"key": k, "z1_minus_T": z(k, r1[k]) - t, "z2_minus_T": z(k, r2[k]) - t, "band": band[k],
                                          "is_threshold_run": z(k, r1[k]) - t == 0.0} for k in ties]  # 09: the order-statistic run itself
                n_thr = sum(x["is_threshold_run"] for x in res[src][f"ties@{a}"])
                res[src][f"excused@{a}"] = {"threshold_run": n_thr, "not_threshold_run": len(ties) - n_thr}
                res[src][f"review_flag@{a}"] = len(ties) > 1 or n_thr < len(ties)  # 09: reported for review, not a fail
        return res

    for (v, readout), pl in sorted(per.items()):
        names, units, _, folds, S = _designated(spec, pl, v, readout)
        mine = {(k, int(u)): float(S[i, folds[i]]) for i, (k, u) in enumerate(zip(names.tolist(), units.tolist(), strict=True))}
        z = np.load(io.BytesIO(Path("/out", om["variants"][f"{v}|{readout}"]["npz"]["path"]).read_bytes()))
        ref = {(k, int(u)): float(s) for k, u, s in zip(z["key"].tolist(), z["unit"].tolist(), z["score"].tolist(), strict=True) if k.startswith("honest_b|")}
        same = set(mine) == set(ref)
        d = max((abs(mine[k] - ref[k]) for k in ref), default=float("inf")) if same else float("inf")
        tol = TOL_GATE["b_asis" if v == "b_asis" else "p1"]
        zz = np.load(io.BytesIO(Path("/out", om["variants"][f"{v}|{readout}"]["npz"]["path"]).read_bytes()))
        hk = [k.startswith("honest_") for k in zz["key"].tolist()]  # firewall: honest keys only enter thresholds and hits
        r1 = runs([k for k, h in zip(zz["key"].tolist(), hk, strict=True) if h], None, [x for x, h in zip(zz["score"].tolist(), hk, strict=True) if h], readout)
        r2 = runs(names.tolist(), None, [float(S[i, folds[i]]) for i in range(len(names))], readout)
        if any(not k.startswith("honest_") for k in list(r1) + list(r2)):
            raise SystemExit("internal: a non-honest key reached the flip gate")
        fl = flips(r1, r2, readout, f"{v}|{readout}", tol)
        no_flip = bool(fl) and all(x.get("usable") and all(x[k] == 0 for k in x if k.startswith("flips@")) for x in fl.values())
        out[f"{v}|{readout}"] = {"rows": len(ref), "same_rows": same, "max_abs_diff": d, "byte_identical": same and d == 0.0, "tol": tol,
                                 "flip_gate": fl, "threshold_source": "v7 phase-1 record" if dec.get(f"{v}|{readout}", {}).get("moments") else "pass 1's own per-fold z (B and BuC)",
                                 "pass": same and d <= tol and no_flip}
    flags = sorted(f"{vr} {src} alpha={k.split('@', 1)[1]}" for vr, x in out.items() for src, g in x["flip_gate"].items() for k, on in g.items()
                   if k.startswith("review_flag@") and on)
    return {"open_manifest": open_ref, "variants": out, "review_flags": flags, "pass": bool(out) and all(x["pass"] for x in out.values())}


@app.function(image=img, cpu=(2, 2), memory=(8192, 8192), timeout=TIMEOUT_ORCH, retries=0, scaledown_window=2, volumes={"/out": OUT, "/work": WORK})
def orchestrate2(spec_stamp: str, fit_run: str, msg: dict, task_fold: dict, stamp: str, head: dict, task_fold_b: dict | None = None) -> dict:
    """The gate and one container per layer in parallel (a detached launch keeps every call); one retry per failed layer; combine;
    write the per-(variant, readout) npz files to /work, then manifest_famexp.json LAST."""
    import numpy as np

    base = Path("/out/famexp") / spec_stamp
    fit = json.loads((base / f"fit_{fit_run}.json").read_text())
    if fit["failed_groups"] or fit["uncertified"]:
        raise SystemExit("pass 1 has failed groups or uncertified fits: pass 2 scores only a complete pass 1")
    spec = json.loads((base / f"spec_{spec_stamp}.json").read_text())
    _check_fp(head.get("code_fingerprint", "missing"))  # the deployment (or this run's upload) is exactly the launch's code
    nm_maps = head.pop("_nm_maps", None)
    if task_fold_b is None or set(task_fold_b) != set(task_fold):  # per-probe routing (09's revised 21b): b_asis has its own map
        raise SystemExit("in-job: b_asis's per-probe fold map is missing or covers other cells")
    if not nm_maps or nm_maps[0] != nm_maps[1]:  # 09: in-job, the two new-main pins must agree on keys and mains
        raise SystemExit("in-job: the new-main membership list and d9's built map disagree: stop")
    if _token_unpinned(msg):  # IL rule item 6 (r) and Addendum 1 item 4 (r2 / r3 / c), in-job before any worker starts
        raise SystemExit(f"in-job: an r / r2 / r3 / c folder without its cited pins: {_token_unpinned(msg)}")
    if _stratum_folder_fails(msg["epochs"]):  # 09's rule, checked in-job before any container starts (again before writing)
        raise SystemExit("in-job: a row's stratum does not fit its folder (su311_ = S_new only, nm360_ = S_newmains only)")
    gate = repro_gate.with_options(volumes={"/out": OUT, "/sol": SOL, "/b": modal.Volume.from_name(B_T196, version=2).read_only()}).spawn(
        spec_stamp, fit_run, head["open_manifest"], head.pop("_elig_content", None), head["code_fingerprint"])
    sc = score_layer.with_options(volumes={"/out": OUT, "/sol": SOL, "/b": modal.Volume.from_name(msg["volume"], version=2).read_only()})
    recs, fails, todo, where = {}, [], list(LAYERS), {}
    for attempt in (1, 2):
        st = stamp if attempt == 1 else f"{stamp}_a2"
        calls = {L: sc.spawn(spec_stamp, fit_run, msg, task_fold, L, st, head["code_fingerprint"]) for L in todo}
        todo = []
        for L, c in calls.items():
            try:
                recs[L], where[L] = c.get(), st
            except BaseException as e:  # noqa: BLE001 - retried once into its own folder, then recorded
                (todo.append(L) if attempt == 1 else fails.append({"layer": L, "error": type(e).__name__}))
        if not todo:
            break
    try:
        g = gate.get()
    except BaseException as e:  # noqa: BLE001 - recorded; 09's job treats a missing gate as a failed gate
        g = {"pass": False, "error": type(e).__name__}
    OUT.reload()  # the workers committed their layer files after this container mounted the Volume (Codex round 13)
    per = {}
    for L, st in where.items():
        for fn in recs[L]["files"]:
            v, readout, _ = fn.removesuffix(".npz").split("__")
            z = np.load(base / f"pass2_{st}" / "layers" / fn)
            per.setdefault((v, readout), {})[L] = (z["key"], z["unit"], z["call_index"], z["fold"], z["score_by_model"])
    fold_of_run = {e["world"] + "|" + e["cell"]: e["folder"] for e in msg["epochs"]}  # 09: one output folder per (set, meta arm)
    folders = sorted(set(fold_of_run.values()) | {x["folder"] for p_ in msg.get("provenance", []) for x in p_.get("excluded_by_staging", [])})  # round 18
    if not re.fullmatch(r"scaleup|scaleup_TEST(_[0-9]+)?", head["out_prefix"]) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(fo)) or ".." in str(fo) for fo in folders):
        raise SystemExit("an output prefix or folder name is not a single plain path component (Codex round 22)")
    if any(not (FOLDER_RE_TEST.fullmatch(str(fo)) if head["test"] else _folder_ok(str(fo))) for fo in folders):
        raise SystemExit("a folder name outside 09's canonical grammar")
    bad_s = _stratum_folder_fails(msg["epochs"])
    if bad_s:  # 09's consistency rule, in-job: su311_ folders hold only S_new rows, nm360_ folders only S_newmains
        raise SystemExit(f"in-job: {len(bad_s)} rows whose stratum does not fit their folder (su311_ = S_new, nm360_ = S_newmains), e.g. {bad_s[:3]}")
    files, strat = {fo: {} for fo in folders}, {e["world"] + "|" + e["cell"]: e["stratum"] for e in msg["epochs"]}
    missing = []
    for (v, readout), pl in sorted(per.items()):
        try:
            names, units, ci, folds, S = _designated(spec, pl, v, readout)
            if v == "b_asis":  # its OWN map (SA + SH): a main in it uses its designated held-out model, else fold -1
                folds = np.array([task_fold_b[k.split("|", 1)[1]] for k in names.tolist()], dtype=np.int64)
        except KeyError as e:  # a layer this combination needs failed twice: listed with every affected run, never substituted (7b item 5)
            missing.append({"variant_readout": f"{v}|{readout}", "missing_layer": str(e), "layers_present": sorted(pl),
                            "runs": sorted(strat)})  # 09: run keys "<world>|<cell_key>" so double-filling can be checked
            continue
        gone = sorted(set(strat) - set(names.tolist()))  # message runs with no usable rows here: listed (7b item 5), never imputed
        if gone:
            missing.append({"variant_readout": f"{v}|{readout}", "runs_without_rows": gone})
        dest = np.array([fold_of_run.get(k, "") for k in names.tolist()])
        for fo in folders:
            sel = dest == fo
            if not sel.any():
                continue
            rk, inv = np.unique(names[sel], return_inverse=True)
            rf = np.zeros(len(rk), np.int64)
            rf[inv] = folds[sel]
            buf = io.BytesIO()
            np.savez(buf, key=names[sel], unit=units[sel], call_index=ci[sel], score_by_model=S[sel], fold=folds[sel], run_key=rk, run_fold=rf,
                     run_stratum=np.array([strat[k] for k in rk.tolist()]))
            fp = Path("/work") / head["out_prefix"] / fo / OUT_SUB / f"{v}__{readout}.npz"
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.open("xb").write(buf.getvalue())
            files[fo][str(fp)] = hashlib.sha256(buf.getvalue()).hexdigest()
    have_ml = {m_["variant_readout"] for m_ in missing if "missing_layer" in m_}
    failed_l = {x["layer"] for x in fails}
    for v in ROSTER:  # a variant x readout with no output and no failed layer: no run of the message has a row in that readout
        for r in ("R", "C"):
            if (v, r) not in per and not (_needs(spec, v, r) & failed_l):
                missing.append({"variant_readout": f"{v}|{r}", "runs_without_rows": sorted(strat), "cause": "no row in this readout for any run of the message"})
    for x in fails:  # Codex round 28: a variant x readout whose ONLY layer failed has no entry in `per`; list it here too, in the field 09 reads
        for vr in (f"{v}|{r}" for v in ROSTER for r in ("R", "C") if x["layer"] in _needs(spec, v, r)):
            if vr not in have_ml:
                have_ml.add(vr)
                missing.append({"variant_readout": vr, "missing_layer": str(x["layer"]), "layers_present": [], "runs": sorted(strat),
                                "cause": "the layer failed twice; no output for this variant x readout"})
    out_m = {}
    for fo in folders:  # one manifest per folder, written LAST, its folder name the single key of the arm -> world map
        ep_ = [e for e in msg["epochs"] if e["folder"] == fo]
        ks = {e["world"] + "|" + e["cell"] for e in ep_}
        miss_fo = [m_ | {f_: [k for k in m_[f_] if k in ks] for f_ in ("runs_without_rows", "runs") if f_ in m_} for m_ in missing
                   if any(k in ks for f_ in ("runs_without_rows", "runs") for k in m_.get(f_, []))]
        # 09: a layer that failed twice affects every run of this folder for every variant x readout that needs it (own_a: its
        # layer, both readouts; adapted and b_asis: R = Sol (a)'s avg_top2 layers, C = L63); run keys "<world>|<cell_key>"
        fails_fo = [x | {"runs": sorted(ks), "variant_readouts": [f"{v}|{r}" for v in ROSTER for r in ("R", "C") if x["layer"] in _needs(spec, v, r)]} for x in fails]
        rmap = {}  # per variant: the map source and, per run, its fold and routing basis (09)
        for v in ROSTER:
            tf = task_fold_b if v == "b_asis" else task_fold
            rmap[v] = {"map_sources": head["fold_maps"]["b_asis" if v == "b_asis" else "own_a_adapted"],
                       "runs": {k: {"fold": tf[k.split("|", 1)[1]], "basis": "mapped" if tf[k.split("|", 1)[1]] >= 0 else "unseen: excluded by all five"}
                                for k in sorted(ks)}}
        runs_fo = {e["run"] for e in ep_} | {p_["run"] for p_ in msg.get("provenance", []) if any(x.get("folder") == fo for x in p_.get("excluded_by_staging", []))}
        roots = [{"root": p_["run"], "version": (re.search(r"_extract_(v[0-9]+)", p_["run"]) or [None, None])[1], "device_record": p_.get("device_record"),
                  "staging_record": p_.get("staging_record"), "manifest": p_.get("manifest"), "setup_failures": p_.get("setup_failures"), "unusable_honest": p_.get("unusable_honest"), "citations": p_.get("citations"),
                  "ended_by_header_verdict": [x for x in p_.get("ended_by_header_verdict", []) if x.get("folder") == fo]}
                 for p_ in msg.get("provenance", []) if p_.get("run") in runs_fo]
        # ended_by_header_verdict: staged runs relayed as 'stopped' whose .eval header says success (d9's v2 record); ENDED, scored, read
        row_meta = {e["world"] + "|" + e["cell"]: e["row_meta"] for e in ep_ if e.get("row_meta")}  # IL: stage, il_order, first_start, pause_gate_at_start, ...
        man = head | {"d9_arm": fo, "routing_per_variant": rmap, "extraction_roots": roots, "ended_by_header_verdict": [x for r_ in roots for x in r_["ended_by_header_verdict"]],
                      "row_meta": row_meta, "volume": msg["volume"], "arm_to_world": {fo: {w: sum(e["world"] == w for e in ep_) for w in sorted({e["world"] for e in ep_})}},
                      "strata": {s_: sum(strat[k] == s_ for k in ks) for s_ in sorted({strat[k] for k in ks})}, "files": files[fo],
                      "weights_sha256": {k: v for r in recs.values() for k, v in r["weights"].items()}, "failed_layers": fails_fo, "honest_b_gate": g,
                      "incomplete_variant_readouts": miss_fo, "excluded_by_staging": [x for p_ in msg.get("provenance", []) for x in p_.get("excluded_by_staging", []) if x.get("folder") == fo],
                      "complete": not fails and not miss_fo and g.get("pass") is True and len(files[fo]) == 26, "variant_readouts": len(files[fo])}
        mp = Path("/work") / head["out_prefix"] / fo / OUT_SUB / "manifest_famexp.json"
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.open("x").write(json.dumps(man, indent=1, default=float))
        out_m[fo] = {"manifest": str(mp), "sha256": hashlib.sha256(mp.read_bytes()).hexdigest(), "complete": man["complete"], "files": len(files[fo])}
        old = Path("/work") / head["out_prefix"] / fo / "famexp" / "manifest_famexp.json"
        if old.exists():  # 09 (a): mark the older per-folder output superseded; never modify or delete its files
            sp = old.parent / "SUPERSEDED.json"
            sp.open("x").write(json.dumps({"superseded_by": str(mp), "superseded_by_sha256": out_m[fo]["sha256"], "old_manifest_sha256": hashlib.sha256(old.read_bytes()).hexdigest(),
                                           "reason": "per-probe fold routing (09's revised Append 21b): b_asis's map includes the simh-mapped mains, which the older scorer wrote with fold -1",
                                           "scorer_git": head["scorer_git"], "stamp (PT)": head["stamp (PT)"]}, indent=1))
            out_m[fo]["superseded"] = {"path": str(sp), "sha256": hashlib.sha256(sp.read_bytes()).hexdigest()}
    WORK.commit()
    return {"manifests": out_m, "failed_layers": fails, "gate_pass": g.get("pass")}


@app.function(image=img, cpu=(1, 1), memory=(1024, 1024), timeout=60, retries=0, scaledown_window=2)
def code_version() -> dict:
    """For the deployed path (launch_pass2_deployed.py): the sha256 of the code this deployment runs."""
    return {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "code_fingerprint": _fingerprint()}


@app.local_entrypoint()
def main(spec: str, fit_run: str, message: str, message_sha: str, r36: str, open_manifest: str, open_sha: str,
         elig_commit: str = "", elig_path: str = "", elig_sha256: str = "", out_prefix: str = "scaleup"):
    _launch(spec, fit_run, message, message_sha, r36, open_manifest, open_sha, elig_commit, elig_path, elig_sha256, out_prefix)


def _launch(spec: str, fit_run: str, message: str, message_sha: str, r36: str, open_manifest: str, open_sha: str,
            elig_commit: str = "", elig_path: str = "", elig_sha256: str = "", out_prefix: str = "scaleup", call=None, extra_head: dict | None = None):
    """Every launcher check, then orchestrate2: this app's (modal run) or, with `call`, the deployed one's .remote."""
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    show = lambda c, p: subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
    me = [str((HERE / f).relative_to(REPO)) for f in ("run_famexp_pass2.py", "RULE.md")]
    if git("status", "--porcelain", "--untracked-files=no") or git("status", "--porcelain", "--untracked-files=all", "--", *me):
        raise SystemExit("uncommitted changes, or this launcher / RULE.md is untracked: commit and push first")
    raw = Path(message).read_bytes()
    if hashlib.sha256(raw).hexdigest() != message_sha:
        raise SystemExit("message sha256 differs from --message-sha")
    msg = json.loads(raw)
    need = ("key", "cell", "world", "stratum", "run", "arm", "staged", "staged_sha256", "folder")
    if not {"d9_arm", "volume", "epochs"} <= set(msg) or not msg["epochs"] or any(not set(need) <= set(e) or e["world"] not in WORLDS
                                                                                 or e["stratum"] not in STRATA for e in msg["epochs"]):
        raise SystemExit("message schema (write a committed adapter for d9's format)")
    q2, t196, rb = show(*Q2[:2]), show(*T196[:2]), Path(r36).read_bytes()
    if hashlib.sha256(q2).hexdigest() != Q2[2] or hashlib.sha256(t196).hexdigest() != T196[2] or not hashlib.sha256(rb).hexdigest().startswith("76b11486cc353ab1eec3"):
        raise SystemExit("q2 list, T196 or r36 sha256 differs from its pin")
    nm, nb = show(*NEWMAINS[:2]), show(*NM_BUILT[:2])
    if hashlib.sha256(nm).hexdigest() != NEWMAINS[2] or hashlib.sha256(nb).hexdigest() != NM_BUILT[2]:
        raise SystemExit("the new-main membership list or d9's built map differs from its pin")
    cells2 = [json.loads(x)["cells"] for x in (nm, nb)]
    if any(len({c["key"] for c in cs}) != len(cs) for cs in cells2):  # Codex round 29: duplicates are refused before any map collapses them
        raise SystemExit("a duplicate key in the new-main membership list or d9's built map: stop")
    nm_maps = [{c["key"]: c["main"] for c in cs} for cs in cells2]
    if nm_maps[0] != nm_maps[1]:
        raise SystemExit("09: the new-main membership list and d9's built map disagree on keys or mains: stop")
    src = {"S_old": json.loads(t196)["cells"], "S_new": json.loads(q2)["cells"], "S_newmains": json.loads(nm)["cells"]}
    stratum_of = {c["key"]: s_ for s_, cs in src.items() for c in cs}
    main_of = {c["key"]: c["main"] for cs in src.values() for c in cs}
    if len(stratum_of) != sum(map(len, src.values())):
        raise SystemExit("a cell key appears in two of T196, q2's +311 list and the new-main list")
    if [e["cell"] for e in msg["epochs"] if e["cell"] not in main_of] or len({(e["world"], e["cell"]) for e in msg["epochs"]}) != len(msg["epochs"]):
        raise SystemExit("a cell outside T196, q2's +311 list and the new-main list, or a duplicate world|cell")
    if any(e["stratum"] != stratum_of[e["cell"]] for e in msg["epochs"]):
        raise SystemExit("an epoch's stratum differs from its cell's list (S_old = T196, S_new = +311, S_newmains = new mains)")
    if _stratum_folder_fails(msg["epochs"]):
        raise SystemExit("a row's stratum does not fit its folder (su311_ = S_new only, nm360_ = S_newmains only)")
    folders_ = {e["folder"] for e in msg["epochs"]} | {x["folder"] for p_ in msg.get("provenance", []) for x in p_.get("excluded_by_staging", [])}
    if _token_unpinned(msg):
        raise SystemExit(f"an r / r2 / r3 / c folder without its cited pins (IL rule item 6; Addendum 1 item 4): {_token_unpinned(msg)}")
    pins_ = sorted({k for fo in folders_ for m_ in [IL_TOKEN_RE.match(str(fo))] if m_ for k in TOKEN_PINS[next(g for g in m_.groups() if g)]})
    for p_ in msg.get("provenance", []):  # every token pin is a committed record on origin, byte-identical to its cited sha256
        for k in pins_:
            c_ = (p_.get("citations") or {}).get(k) or {}
            if c_ and (not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", c_["commit"]).splitlines())
                       or hashlib.sha256(show(c_["commit"], c_["path"])).hexdigest() != c_["sha256"]):
                raise SystemExit(f"Addendum 1: the {k} pin {c_['path']} @ {c_['commit']} is not on origin or its sha256 differs")
    if any(not (FOLDER_RE_TEST.fullmatch(str(fo)) if msg.get("test") else _folder_ok(str(fo))) for fo in folders_):
        raise SystemExit("a folder name outside 09's grammar")
    sel_ = {bool(FOLDER_RE_OLDSEL.fullmatch(str(x))) for x in folders_}
    if len(sel_) > 1:  # 09: every oldsel manifest is marked as the descriptive add-on, so an oldsel message holds only oldsel folders
        raise SystemExit("a message mixing oldsel (sel97 add-on) folders with any other folder")
    e2_ = {bool(FOLDER_RE_EP2.fullmatch(str(x))) for x in folders_}
    if len(e2_) > 1:  # every epoch-2 manifest is marked SEALED, so an epoch-2 message holds only epoch-2 folders
        raise SystemExit("a message mixing epoch-2 (ep2_) folders with any other folder")
    for fo in (x for x in folders_ if FOLDER_RE_HA25F.fullmatch(str(x))):  # 09's folder-name note: exactly its cell, one run, the selected source .eval
        ep_ = [e for e in msg["epochs"] if e["folder"] == fo]
        cell_, sha_ = HA25F[re.sub(r"(_try[0-9]+)?__honest_a$", "", fo)]
        if len(ep_) != 1 or ep_[0]["world"] != "honest_a" or ep_[0]["cell"] != cell_ or ep_[0].get("eval_sha256") != sha_:
            raise SystemExit(f"{fo}: must hold exactly honest_a|{cell_} with source .eval sha256 {sha_[:8]}… (09's 25f H_A notes)")
    fom = {r["main_task"]: r["fold"] for r in json.loads(rb)}  # Sol #36's folds; a main outside them is a no-fold main (21b)
    task_fold = {e["cell"]: fom.get(main_of[e["cell"]], -1) for e in msg["epochs"]}
    sh_raw = show(*SIMH[:2])  # b_asis = (SA, SH): its own map is runs36's sim-attack folds plus the simh file's sim-honest folds
    if hashlib.sha256(sh_raw).hexdigest() != SIMH[2]:
        raise SystemExit("the simh runs file differs from its pin")
    sa_rows, sh_rows = [r for r in json.loads(rb) if r["arm"] == "sim-attack"], [r for r in json.loads(sh_raw) if r["arm"] == "sim-honest"]
    fb = {}
    for r in sa_rows + sh_rows:
        fb.setdefault(r["main_task"], set()).add(r["fold"])
    if any(len(v_) != 1 for v_ in fb.values()):
        raise SystemExit("STOP (09): b_asis's SA and SH folds disagree for a main")
    fom_b = {m: next(iter(v_)) for m, v_ in fb.items()}
    task_fold_b = {e["cell"]: fom_b.get(main_of[e["cell"]], -1) for e in msg["epochs"]}
    elig, elig_content = None, None
    if elig_commit:
        if git("cat-file", "-t", elig_commit) != "commit" or not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", elig_commit).splitlines()) \
                or hashlib.sha256(show(elig_commit, elig_path)).hexdigest() != elig_sha256:
            raise SystemExit("the eligibility record is not on origin or its sha256 differs")
        elig = {"commit": elig_commit, "path": elig_path, "sha256": elig_sha256}
        elig_content = json.loads(show(elig_commit, elig_path))
    if not re.fullmatch(r"scaleup|scaleup_TEST(_[0-9]+)?", out_prefix) or (out_prefix == "scaleup") == bool(msg.get("test")):
        raise SystemExit("--out-prefix: scaleup for real messages, scaleup_TEST[_<n>] for a TEST message (never mixed)")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    nofold = sorted({main_of[c] for c, f in task_fold.items() if f < 0})
    t196_main = {c["key"]: c["main"] for c in json.loads(t196)["cells"]}
    t196_fold = {c: fom[m] for c, m in t196_main.items()}
    train_mains = [sorted({t196_main[c] for c, g in t196_fold.items() if g != f} | {r["main_task"] for r in json.loads(rb) if r["fold"] != f}) for f in range(5)]
    tm_sha = [hashlib.sha256(json.dumps(x).encode()).hexdigest() for x in train_mains]  # fold model f's training mains (T196 + Sol #36), all variants
    if any(m in set(x) for m in nofold for x in train_mains):
        raise SystemExit("a no-fold main appears in a fold model's training mains")
    tm_b = [sorted({r["main_task"] for r in sa_rows + sh_rows if r["fold"] != f}) for f in range(5)]  # b_asis's fold models (SA + SH, folds != f)
    tm_b_sha = [hashlib.sha256(json.dumps(x).encode()).hexdigest() for x in tm_b]
    for c in task_fold:  # every routed cell: never scored by a model that trained on its main (per probe)
        m, d, db = main_of[c], task_fold[c], task_fold_b[c]
        if (d >= 0 and m in set(train_mains[d])) or (db >= 0 and m in set(tm_b[db])) or (db < 0 and any(m in set(x) for x in tm_b)):
            raise SystemExit(f"STOP: cell {c} (main {m}) would be scored by a fold model trained on its main")
    nofold_b = sorted({main_of[c] for c, f in task_fold_b.items() if f < 0})
    head = {"scorer_git": git("rev-parse", "HEAD"), "stamp (PT)": stamp, "app": APP, "out_prefix": out_prefix, "test": bool(msg.get("test")), "spec": spec, "fit_run": fit_run, "message_sha256": message_sha,
            "open_manifest": {"path": open_manifest, "sha256": open_sha}, "eligibility_record": elig or {"status": "pending: phase 1 not yet committed at pass-2 time"},
            "q2_list": {"commit": Q2[0], "path": Q2[1], "sha256": Q2[2]}, "new_mains_list": {"commit": NEWMAINS[0], "path": NEWMAINS[1], "sha256": NEWMAINS[2]},
            "new_mains_built": {"commit": NM_BUILT[0], "path": NM_BUILT[1], "sha256": NM_BUILT[2]},
            "append_21b": "b42fc8290d06063f0688d24f12efeb59ff6f5727",
            "routing": {m: {"excluded_by_all_five": True, "training_manifest_sha256": tm_sha} for m in nofold},
            "routing_b_asis": {"mapped_here": {main_of[c]: task_fold_b[c] for c in sorted(task_fold_b) if task_fold_b[c] >= 0 and task_fold[c] < 0},
                               "unseen": {m: {"excluded_by_all_five": True, "training_manifest_sha256": tm_b_sha} for m in nofold_b}},
            "fold_maps": {"own_a_adapted": [{"file": "runs36.json (Sol #36)", "sha256": hashlib.sha256(rb).hexdigest(), "field": "main_task -> fold (all arms)"}],
                          "b_asis": [{"file": "runs36.json (Sol #36)", "sha256": hashlib.sha256(rb).hexdigest(), "field": "sim-attack main_task -> fold (SA)"},
                                     {"commit": SIMH[0], "path": SIMH[1], "sha256": SIMH[2], "field": "sim-honest main_task -> fold (SH)"}]},
            "output_subdir": OUT_SUB,
            "training_manifests": "per fold model f: sha256 of json.dumps(sorted training mains) = T196 cells' mains with Sol #36 fold != f, plus Sol #36 mains with fold != f (union over the roster's training sources)",
            "label": "pass 2 scoring only; frozen roster, weights and eligibility; RAW logits of five fold models; PRELIMINARY"}
    if sel_ == {True}:
        head["descriptive_addon"] = OLDSEL_LABEL
    if e2_ == {True}:
        head["sealed_epoch2"] = EP2_SEALED
    head |= (extra_head or {}) | {"code_fingerprint": _fingerprint()}
    out = (call or orchestrate2.remote)(spec, fit_run, msg, task_fold, stamp, head | {"_elig_content": elig_content, "_nm_maps": nm_maps}, task_fold_b)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"pass2_{msg['d9_arm']}_{stamp}.json").open("x").write(json.dumps(head | {"result": out}, indent=1, default=float))
    print(json.dumps(head | {"result": out}, indent=1, default=float)[:3000])
