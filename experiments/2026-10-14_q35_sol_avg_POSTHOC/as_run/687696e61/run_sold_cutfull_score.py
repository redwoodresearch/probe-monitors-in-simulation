"""S_old full-length ("cutfull") Sol scoring (RULE.md Amendment 51; design config/sold_cutfull_design.json @ eac0152da; 2e 10-16 02:2x-02:5x).
The (b) S_old folders (q4f_main x4, q4f_run2, q4f_c, rh1e4_stage, A25 simh b1/b2) have no chain routing record of the Amendment 46 kind: their authority is
d9's S_old attestation (make_sold_attestation.py; stager provenance 9527f70d2) and their registered Sol files are Amendment 23 / the relaunch / A25, so
run_scaleup_score.py's cutfull line does not admit them. This script does, with its own gates, and writes the Amendment 46 per-row format for EVERY probe of
a weights spec, at the reply readout (avg_top2) and the command readout (cmd_A, content-only call_end at L63, dropping the committed rebuilt skip list).

Admission from git only, before any spend (--phase dry runs it alone): the folder's epochs are inventory_b_v2's S_old rows with that uncut folder, bound
BY FULL EPOCH KEY to the highest-revision attestation (failing {}), its extraction attestation, the cutfull stage record and extraction records (H200), the
registered cut root per key (the corrected-B manifest for q4f_main / q4f_run2, else the set's one root), the registered Sol file per key, the r36 fold map
(T196 cell -> main) and, when given, d9's routing record for the folder (--routing; equality by key, fail closed). Per epoch on Modal, before any row is used:
full consumption, unit positions, and the binding of the reply and command units to the registered cut metas (score_r_cf's rule; withheld epochs fail the
folder). Weights (--weights <spec>@<commit>, default = the deployed asis_2026-10-14_135701 tight outer models): every file sha256-checked first.

  (env) SOLD_WEIGHTS_VOL=<the spec's Volume> PYTHONPATH=src <modal> run --detach experiments/2026-10-14_q35_sol_avg_POSTHOC/run_sold_cutfull_score.py
      --phase dry|skip|score --folder <cutfull folder> --r36 <runs.json> [--weights <spec path>@<commit>] [--routing <record path>@<commit>]
      [--skip-list <results/sold_skiplist_*.json>@<commit>]   (score: the folder's routing record and the committed skip list are required)
Outputs: probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14:/cutfull/sold/<out_name>/<folder>/ (create-only, manifest_sol.json last); local
record results/sold_cutfull_<folder>[_<label>]_<stamp>.json. Firewall: shas, counts, keys and gates only; no score is printed or returned.
"""

import hashlib
import io
import itertools
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
Q_REL = "experiments/2026-10-14_q35_sol_avg_POSTHOC"
APP = "probe-shift_aswin-surya_sub-q35solavg_soldcf_kill-2026-11-14"
SOLVOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"  # the deployed asis weights (default spec)
OUTVOL = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"  # 09's work Volume; this job writes only /cutfull/sold/<out_name>/
CFVOL = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"  # cutfull staged sets and extraction roots
BVOL = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"  # registered q4b / q4f staging and extraction (corrected B, batch C)
RHVOL = "probe-shift_aswin-surya_sub-q3realhonest_data_kill-2026-11-14"  # registered rh1e4 staging and extraction
SIMHVOL = "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13"  # registered A25 sim-honest staging and extraction
HFVOL = "probe-shift_aswin-surya_hf-cache"
WVOL = os.environ.get("SOLD_WEIGHTS_VOL", SOLVOL)  # the weights spec's Volume, mounted at /w (checked against the spec at launch)
MOUNT = {"b": "/b", "rh": "/rh", "simh": "/simh"}

CF_BRANCH = "origin/aswin/q3-cut-option"
CHAIN = "experiments/2026-10-14_q3_overnight_chain/results"
ADIR, RDIR = f"{CHAIN}/cutfull/attest", f"{CHAIN}/cutfull/routing"
INVB = ("experiments/2026-10-15_cut_sensitivity/results/inventory_b_v2_2026-10-16_014820.json", "a0bf963a7",
        "fc0e84ea5260193f7e5f249f022b2f4b04de544794e06a7137d777c222d5bbf8")
PROV = ("experiments/2026-10-15_cut_sensitivity/results/stager_provenance_b_sold_2026-10-15.json", "9527f70d2",
        "aeba2ce0cb76f97d39845acfb486101817f6e8b25fd353a4cfeb311f66a119a4")  # the AUTHORITY of the S_old attestations
CF_GATE = (f"{CHAIN}/cutfull_step0_compare_2026-10-15_214206.json", "e6896f33b", "efc7421c7942d0a74b8ba0d1468d3563f51030b5a894ccd02a41fb26892b2925")
CF_KEYS = ("experiments/2026-10-15_cut_sensitivity/results/step0_keys_v2_2026-10-15.json", "f3bfe9208", "cf5c4f5e3e0e066414f193ea5df16ac11c7bc995ac8de48f8b7f876521dc0a02")
CORRB = (f"{Q_REL}/results/correctedb_manifest_v1.json", "46ca25fc4", "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc")  # per-key cut root (A23)
Q4F_STAGE = ("experiments/2026-10-13_q4_why_weaker_POSTHOC/results/q4f_stage_2026-10-14_105839.json", "948ecd8e8", None)  # changed_keys (sha recorded)
T196 = ("experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "debef0af2", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")
R36_PIN = "76b11486cc353ab1eec3"  # the #36 run table (run_scaleup_score.PINS["r36"])
Q4F_C_EXTRACT = ("experiments/2026-10-13_q4_why_weaker_POSTHOC/results/extract_q4b_q4b_extract_c_v1_a1_2026-10-14_121640.json", "34ed9fcd9",
                 "d362ab2c23f0d05105901955aaa335eca98e70cb8422ce3285ae863116ac43f1")  # batch C's registered extraction record (Amendment 23 inputs.c)
RH_COUNTED = ("experiments/2026-10-14_q3_q35_realhonest_1e4/config/manifest_rh1e4_eval_upload_v1.json", "7bbc0e3df",
              "9785118d230c4ec4e67bb73ea6821547c672705689c8bc3d519bef56b49068ac")  # d9's counted rows: per-key world honest_r<i>
SOL_REC = {  # the registered Sol score files of the original folders (committed records; Volume packs bound on Modal by manifest sha256)
    "asis": (f"{Q_REL}/results/asis_scores_2026-10-14_142228.json", "74992a373", "59c051a4ff9cc40811745a119251224f031cdaf479b8d3fb909c1059ab94da31"),
    "rh": (f"{Q_REL}/results/scaleup507_rh1e4_v1__honest_b_2026-10-14_235427.json", "f874fb9c1", "afc52f7911536610f6a055a95b9b1303455b12e2be92e2627e7c6ef45f7f8eed"),
    "simh_b1": (f"{Q_REL}/results/simh_score_2026-10-14_230251.json", "c108cef81", "2fd5f5c8877f585659a4c1781d313399248e51a31f9a07b4c838c874097bc1a9"),
    "simh_b2": (f"{Q_REL}/results/simh_score_2026-10-14_232103.json", "d13079e99", "aae5e67b73436c3c6a63924e23279ed9a988271a78397f68cfda79325890f983")}
# original target -> the registered cut side (Volume key, root and staged set, or the corrected-B manifest per key), key world, registered Sol file
SETS = {"q4f_main": {"vol": "b", "cut": "corrected_b", "sol": "asis"}, "q4f_run2": {"vol": "b", "cut": "corrected_b", "sol": "asis"},
        "q4f_c": {"vol": "b", "cut": ("q4b_extract_c_v1", "q4f_c_v1"), "world": "honest_c", "sol": "asis"},
        "rh1e4_stage": {"vol": "rh", "cut": ("rh1e4_extract_v1", "rh1e4_stage_v1"), "world": "honest_r", "sol": "rh"},
        "simh_q4f_v1_full_b1": {"vol": "simh", "cut": ("simh_full_b1", "simh_q4f_v1_full_b1"), "world": "simh_full", "sol": "simh_b1"},
        "simh_q4f_v1_full_b2": {"vol": "simh", "cut": ("simh_full_b2", "simh_q4f_v1_full_b2"), "world": "simh_full", "sol": "simh_b2"}}
TAG = "asis_2026-10-14_135701"
CHOICE_REC = (f"{Q_REL}/results/avg_transfer_avg_v1_2026-10-14_011347.json", "aa41408f6", "cc90ed914bdee4119e9648f7bc85f0e835f9fcd61e2ade3a6185b448b95fd4f4")
FIT_REC = (f"{Q_REL}/results/asis_fit_2026-10-14_135701.json", "04f4e4acb", "9c5479ed4e93b4df83cfc3bf602065fd1e4ea695e579594f88e6a0f74bae9102")
FIT_NPZ_SHA256 = {f"xfer/{TAG}/outer_tight_L34.npz": "82f57ecfd2478be0daacc9e0d60220fdd3c16f8e26f9967fbc1de58bb32f9a93",  # run_scaleup_score.FIT_NPZ_SHA256 (A25 @ c108cef81)
                  f"xfer/{TAG}/outer_tight_L46.npz": "ed561f100a573a5c0940b6ca2b3b06324de1958fb7ad42776a2cbb19acc672a1",
                  f"xfer/{TAG}/outer_tight_L47.npz": "1488630dda37c8b1ca9ef83ce19fcada88d36afc2fb0f10d54ea1434be87cd3c",
                  f"xfer/{TAG}/outer_tight_L49.npz": "5d98db68f81df37b3890780a49c338b7ce5672cce5a52149b20aa1680f9ac5bc",
                  f"xfer/{TAG}_cmd/outer_tight_L63.npz": "a76a03cced4122804c0eb2630aa879d355493d49eb6ce071e26d68faad25990b"}
FIT_MODEL_SHA256 = {"a": ["4e3afc3272e9fe930d34f5e3f059760d53a970b8819053eb0ee1bf5eb612c441", "ac25b2003489dcbbe4bb9949c527a9e00065c487ef38870ead8be908aee46bdb",
                          "e8c33690ad6b08a57deb58f76816ee52f2811373e360f39080cbab6f2654a652", "aff4824ccbf58ab7aca12c152c88bc27015bf03aacb5dc60ca91da8b34aad4de",
                          "3828be20fc84202c04b1c906ba1a4a499a81f16eff9832ba9bfa7265cab20595"],  # run_scaleup_score.FIT_MODEL_SHA256 (example manifest @ 7a343c6f7)
                    "c_asis": ["7001e33c8dea684faae82f6c209ac488ca8653d6d4d71b7b92d84c90d05a5bf1", "56420843beba427d1c98243c0a3a31e6f0fa5c4353006c241d4e76278b9a0194",
                               "7873f4cc093e90b79d05548b95c742d6848594ffa9c8dc1eac8b1d29504fee95", "0cfae59e9cbe1a28b2126fd72ef9f17cf06b73e74cd3de674d7542d13b6e03ed",
                               "0654a3b30ec5c89abf3b75c720b2f0878bd31af979635825816bd919a7a71225"]}
CMD_L = 63
CUTFULL_SKIP = ("src/probe_shift/cutfull_skip.py", "7f15d2c0b", "0b448ef424e3d5c5f26f6c9102e98ee917e2ad7810ffb63acf36178743d18fcf")  # SOLA-UNCUT's shared rebuild
# (origin/aswin/sol-a-uncut-score-2e; 2e 03:0x: one implementation), copied byte-identically into this branch; its sha256 is asserted locally and in every container
LABEL = "uncut: all tokens; POST HOC (Append 41); OOD caveat per probe from the committed training-length table (Append 41 §5 amendment); S_old path (Amendment 51)"
CMD_RULE = "content-only call_end at L63: rows on a fixed call token ('0', 'n'; fixed_call_tokens('q4b_main_v1')) dropped by the committed rebuilt skip list"

app = modal.App(APP)
cpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
gate_image = cpu_image.add_local_python_source("probe_shift")
tok_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "numpy==2.3.3")
             .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))
RO = {"/cf": CFVOL, "/b": BVOL, "/rh": RHVOL, "/simh": SIMHVOL}
ro_vols = {m: modal.Volume.from_name(v, version=2).read_only() for m, v in RO.items()}
w_vol = modal.Volume.from_name(WVOL, version=2).read_only()
out_vol = modal.Volume.from_name(OUTVOL, version=2)


# ---- git (records are read at pinned commits only)
def git(*a, text=True):
    r = subprocess.run(["git", *a], capture_output=True, text=text, cwd=HERE, check=False)
    if r.returncode:
        raise SystemExit(f"git {' '.join(a)[:200]}: failed (fetch {CF_BRANCH} first?)")
    return r.stdout


def show(commit: str, path: str) -> bytes:
    return git("show", f"{commit}:{path}", text=False)


def full(commit: str) -> str:
    f = git("rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}").strip()
    if not re.fullmatch(r"[0-9a-f]{9,40}", str(commit)) or not re.fullmatch(r"[0-9a-f]{40}", f) or not f.startswith(commit):
        raise SystemExit(f"commit {commit!r} does not resolve to one full sha")
    return f


def on_branch(commit: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", full(commit), CF_BRANCH], capture_output=True, cwd=HERE, check=False).returncode == 0


def pinned(spec: tuple) -> tuple[bytes, dict]:
    """(path, commit, sha256 or None) -> bytes at commit:path and a citation; a different sha256 stops the job."""
    b = show(spec[1], spec[0])
    h = hashlib.sha256(b).hexdigest()
    if spec[2] is not None and h != spec[2]:
        raise SystemExit(f"{spec[0]} @ {spec[1]}: sha256 {h[:16]} differs from the pin {spec[2][:16]}")
    return b, {"path": spec[0], "commit": spec[1], "sha256": h}


def cite(s: str) -> tuple[str, str]:
    """'<path> @ <commit>' -> (path, full commit)."""
    p, _, c = str(s).rpartition(" @ ")
    if not p or not c:
        raise SystemExit(f"not a '<path> @ <commit>' citation: {s!r}")
    return p, full(c)


def ls(ref: str, d: str) -> list:
    return git("ls-tree", "--full-tree", "--name-only", f"{ref}:{d}").split()


def highest(d: str, base: str) -> tuple[int, str]:
    """The highest _r<k> revision of <base>[_r<k>].json in d on CF_BRANCH -> (k, path)."""
    revs = {int(m.group(1) or 1): f"{d}/{n}" for n in ls(CF_BRANCH, d) if (m := re.fullmatch(rf"{re.escape(base)}(?:_r([2-9]|[1-9][0-9]+))?\.json", n))}
    if not revs:
        raise SystemExit(f"{d}/{base}[_r<k>].json: none on {CF_BRANCH}")
    return max(revs), revs[max(revs)]


# ---- the weights spec (--weights): per probe, the reply files per layer, the command file, layer choice and training arms; every file sha256-pinned
def default_spec() -> dict:
    choice = json.loads(pinned(CHOICE_REC)[0])["analysis"]["layer_choice"]["a"]
    R = {str(L): {"path": f"xfer/{TAG}/outer_tight_L{L}.npz", "sha256": FIT_NPZ_SHA256[f"xfer/{TAG}/outer_tight_L{L}.npz"]} for L in sorted({L for c in choice for L in c})}
    C = {"path": f"xfer/{TAG}_cmd/outer_tight_L{CMD_L}.npz", "sha256": FIT_NPZ_SHA256[f"xfer/{TAG}_cmd/outer_tight_L{CMD_L}.npz"]}
    return {"label": f"deployed {TAG} tight outer models (Amendment 23; registered weights, no refit)", "volume": SOLVOL, "out_name": "sol",
            "fit_record": dict(zip(("path", "commit", "sha256"), FIT_REC)),
            "probes": {f"{p}_tight": {"npz_prefix": p, "reply_files": R, "cmd_file": C, "layer_choice": choice,
                                      "layer_choice_source": f"{CHOICE_REC[0]} @ {CHOICE_REC[1]} analysis.layer_choice.a (layers inherited from (a))",
                                      "training_arms": arms, "fold_model_sha256": FIT_MODEL_SHA256[p],
                                      "fold_model_hash": "per fold f: W then b as float64 at sorted(layer_choice[f]) reply files, then the cmd file"}
                       for p, arms in (("a", ["real-attack-e1", "real-honest"]), ("c_asis", ["sim-attack", "real-honest"]))}}


def load_spec(arg: str) -> tuple[dict, dict]:
    if not arg:
        return default_spec(), {"default": True, "constants": "FIT_NPZ_SHA256 / FIT_MODEL_SHA256 / CHOICE_REC of this file (= run_scaleup_score.py's pins)"}
    p, _, c = arg.rpartition("@")
    b, ct = pinned((p, full(c), None))
    s = json.loads(b)
    ok = isinstance(s.get("probes"), dict) and s["probes"] and re.fullmatch(r"[a-z0-9_]+", str(s.get("out_name"))) and isinstance(s.get("volume"), str)
    for n, pr in (s.get("probes") or {}).items():
        files = [*(pr.get("reply_files") or {}).values(), pr.get("cmd_file") or {}]
        ok = ok and re.fullmatch(r"[a-z0-9_]+", n) and re.fullmatch(r"[a-z0-9_]+", str(pr.get("npz_prefix"))) and all(
            isinstance(f, dict) and re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256"))) and isinstance(f.get("path"), str) and ".." not in f["path"] for f in files) \
            and isinstance(pr.get("layer_choice"), list) and len(pr["layer_choice"]) == 5 and all(len(x) == 2 and len(set(x)) == 2 and all(isinstance(L, int) and str(L) in pr["reply_files"] for L in x) for x in pr["layer_choice"]) \
            and isinstance(pr.get("training_arms"), list) and pr["training_arms"]
    if not ok:
        raise SystemExit(f"--weights {arg}: a spec needs volume, out_name, and per probe npz_prefix, reply_files {{layer: {{path, sha256}}}}, cmd_file, "
                         "layer_choice (5 x 2 layers of reply_files) and training_arms")
    return s, ct


def weight_gate(root: Path, spec: dict) -> dict:
    """BEFORE any row: every spec file's bytes hash to its pin; every probe x fold x layer has {prefix}__f{f}__{W,b,mean,std}; fold-model shas where pinned."""
    import numpy as np

    files, bad, models = {}, [], {}
    for n, pr in spec["probes"].items():
        for f_ in [*pr["reply_files"].values(), pr["cmd_file"]]:
            h = hashlib.sha256((root / f_["path"]).read_bytes()).hexdigest()
            files[f_["path"]] = h
            if h != f_["sha256"]:
                bad.append(f_["path"])
    miss = []
    if not bad:
        for n, pr in spec["probes"].items():
            for f in range(5):
                h = hashlib.sha256()
                for f_ in [pr["reply_files"][str(L)] for L in sorted(pr["layer_choice"][f])] + [pr["cmd_file"]]:
                    Z = np.load(root / f_["path"])
                    ks = [f"{pr['npz_prefix']}__f{f}__{q}" for q in ("W", "b", "mean", "std")]
                    miss += [f"{f_['path']}:{k}" for k in ks if k not in Z.files]
                    if ks[0] in Z.files and ks[1] in Z.files:
                        h.update(np.asarray(Z[ks[0]], np.float64).tobytes())
                        h.update(np.asarray(Z[ks[1]], np.float64).tobytes())
                models.setdefault(n, []).append(h.hexdigest())
    bad_m = [f"{n}/f{f}" for n, pr in spec["probes"].items() if pr.get("fold_model_sha256") for f in range(5) if (models.get(n) or [None] * 5)[f] != pr["fold_model_sha256"][f]]
    return {"files_sha256": files, "files_differing": bad, "keys_missing": miss, "fold_models_sha256": models, "fold_models_differing": bad_m,
            "passes": not bad and not miss and not bad_m}


# ---- admission from git (no Modal): one S_old cutfull folder -> its epochs, bound by full epoch key to every committed record
def admit(folder: str, r36: str, routing: str = "") -> dict:
    m = re.fullmatch(r"([a-z0-9_]+?)_cutfull_v1__([a-z_]+)", folder)
    if not m or m.group(1) not in SETS:
        raise SystemExit(f"{folder}: not <S_old target>_cutfull_v1__<arm token> (addendum 2: no _try), targets {sorted(SETS)}")
    target, tok, st = m.group(1), m.group(2), SETS[m.group(1)]
    inv_b, inv_c = pinned(INVB)
    rows = [r for r in json.loads(inv_b)["runs"] if (r.get("uncut_planned") or {}).get("folder") == folder]
    keys = [r["key"] for r in rows]
    if not rows or len(set(keys)) != len(keys) or len({r["original_folder"] for r in rows}) != 1 or len({r["arm"] for r in rows}) != 1:
        raise SystemExit(f"inventory: {folder} has no rows, a key twice, or more than one original folder / arm")
    orig, arm = rows[0]["original_folder"], rows[0]["arm"]
    # the attestation: highest revision on the branch, the one each inventory row names, failing {}
    k, a_path = highest(ADIR, f"attestations_sold__{target}_cutfull")
    a_raw = show(CF_BRANCH, a_path)
    att = json.loads(a_raw)
    up = {json.dumps({x: r["uncut_planned"].get(x) for x in ("staged_set", "extraction_root", "volume", "staged_sha256", "routing", "revision", "staged_inputs_file")}, sort_keys=True)
          for r in rows}
    if len(up) != 1:
        raise SystemExit(f"inventory: {folder}'s rows disagree on the uncut plan")
    u = json.loads(next(iter(up)))
    rp, rc = cite(u["routing"])
    prov_b, prov_c = pinned(PROV)
    gate_c = f"{CF_GATE[0]} @ "
    if rp != a_path or show(rc, a_path) != a_raw or not on_branch(rc) or u["revision"] != k or att.get("revision") != k or att.get("failing") != {} \
            or (att.get("cutfull") or {}).get("folders", {}).get(tok) != folder or att["cutfull"].get("staged_set") != u["staged_set"] \
            or att["cutfull"].get("extraction_root") != u["extraction_root"] or att["cutfull"].get("volume") != CFVOL or u["volume"] != CFVOL \
            or (att.get("original_folders") or {}).get(tok) != orig or att.get("original_target") != target or u["staged_inputs_file"] != f"{arm}.jsonl" \
            or (att.get("authority") or {}).get("provenance") != f"{PROV[1]}:{PROV[0]}" or att["authority"].get("provenance_sha256") != PROV[2] \
            or not str(att.get("repro_gate", "")).startswith(gate_c) or full(str(att["repro_gate"])[len(gate_c):].split()[0]) != full(CF_GATE[1]):
        raise SystemExit(f"{folder}: the inventory's attestation ({u['routing']}, revision {u['revision']}) is not the highest revision {a_path} (r{k}) at the branch tip, "
                         "or its failing / folder / staged set / root / Volume / original folder / authority / step-0 gate differ")
    tc = (att.get("template_class") or {}).get(tok)
    if not isinstance(tc, dict) or set(tc) != set(keys) or (att.get("units") or {}).get(tok, {}).get("epochs") != len(keys) \
            or not set(tc.values()) <= {"fixed, render unchanged", "render changed with the cut"} or not (att.get("format_check") or {}).get(tok):
        raise SystemExit(f"{folder}: the attestation's template classes / unit epochs / format check are not the inventory's {len(keys)} keys")
    # the extraction attestation (by key) and the cutfull stage record (staged sha)
    xa = [r for ref in att.get("extraction_attestation") or [] for r in json.loads(show(*cite(ref)[::-1]))["roots"]
          if r.get("root") == u["extraction_root"] and r.get("uncut_set") == u["staged_set"]]
    xs = (xa[0]["arms"].get(tok) or {}) if len(xa) == 1 else {}
    if len(xa) != 1 or xa[0].get("failing") != {} or xs.get("keys") is None or sorted(xs["keys"]) != sorted(keys) or xs.get("staged_sha256") != u["staged_sha256"] \
            or xs.get("meta_arm") != arm or xs.get("epochs") != len(keys):
        raise SystemExit(f"{folder}: the extraction attestation's {tok} arm is not these keys / staged sha / meta arm (by key)")
    sp, sc = cite(att["cutfull"]["stage_record"])
    srec = [v for v in json.loads(show(sc, sp)).values() if isinstance(v, dict) and v.get("out") == u["staged_set"]]
    if len(srec) != 1 or (srec[0].get("assembled") or {}).get("staged_inputs_sha256", {}).get(f"{arm}.jsonl") != u["staged_sha256"]:
        raise SystemExit(f"{folder}: the cutfull stage record {sp} does not stage {u['staged_set']}/{arm}.jsonl with sha256 {u['staged_sha256'][:16]}")
    # the cutfull extraction records on the branch: attempt-ordered, this root / staged set / sha / cut full, every shard H200
    xr, chunks, gpus, last, chunk_rows = [], set(), {}, 0, {}
    for n in sorted(ls(CF_BRANCH, CHAIN), key=lambda n: (int(mm.group(1)) if (mm := re.search(r"_a([0-9]+)_", n)) else 0, n)):
        if not re.fullmatch(rf"extract_chain_{re.escape(u['extraction_root'])}_a[0-9]+_\d{{4}}-\d\d-\d\d_\d{{6}}\.json", n):
            continue
        b = show(CF_BRANCH, f"{CHAIN}/{n}")
        x = json.loads(b)
        cfg, an = x.get("config") or {}, x.get("attempt")
        if not isinstance(an, int) or an <= last or cfg.get("attempt") != an or cfg.get("run") != u["extraction_root"] or cfg.get("inputs") != u["staged_set"] \
                or cfg.get("cut") != "full" or cfg.get("test") is not False or (cfg.get("inputs_sha256") or {}).get(f"{arm}.jsonl") != u["staged_sha256"] or x.get("volume") != CFVOL:
            raise SystemExit(f"extraction record {n}: not attempt-ordered, or not this root / staged set / sha / cut full / Volume / non-test")
        last = an
        for s_ in x.get("shards") or []:
            if "error" in s_ or "H200" not in str(s_.get("gpu_name")):
                raise SystemExit(f"{n}: shard {s_.get('shard')} errored or ran on {s_.get('gpu_name')!r}, not H200")
            gpus[f"a{an}/shard{s_['shard']}"] = s_["gpu_name"]
            for c in s_.get("chunks") or []:
                if re.fullmatch(rf"shard0*{s_['shard']}_a{an}_chunk[0-9]+", c["chunk"]):
                    if not isinstance(c.get("rows"), int) or c["rows"] < 0 or c["chunk"] in chunk_rows:
                        raise SystemExit(f"{n}: chunk {c['chunk']} has no row count, or appears twice")
                    chunk_rows[c["chunk"]] = c["rows"]
                    chunks.add(c["chunk"])
        xr.append({"path": f"{CHAIN}/{n}", "sha256": hashlib.sha256(b).hexdigest(), "attempt": an})
    if not xr or not chunks:
        raise SystemExit(f"{folder}: no committed extraction record with H200 shards and chunks for {u['extraction_root']}")
    # the step-0 gate (as run_scaleup_score.cf_load)
    g, ks0 = json.loads(pinned(CF_GATE)[0]), json.loads(pinned(CF_KEYS)[0]).get("keys") or []
    gr = g.get("runs") or []
    if g.get("passes") is not True or not gr or g.get("n_pass") != len(gr) or any(r_.get("passes") is not True for r_ in gr) or sorted(r_.get("key") for r_ in gr) != sorted(ks0):
        raise SystemExit("the step-0 gate record is not a full pass bound to the step-0 key list")
    # the registered cut side and key per epoch (committed records only)
    t196_b, t196_c = pinned(T196)
    main_of = {c["key"]: c["main"] for c in json.loads(t196_b)["cells"]}
    eps, cite_cut, reg_sha = [], {}, None
    if st["cut"] == "corrected_b":
        cb_b, cb_c = pinned(CORRB)
        cb = {e["key"]: e for e in json.loads(cb_b)["epochs"]}
        q4_b, q4_c = pinned(Q4F_STAGE)
        changed = set(json.loads(q4_b)[{"q4f_main": "main", "q4f_run2": "run2"}[target]]["compare"]["changed_keys"][arm])
        bad = [k_ for k_, r in zip(keys, rows) if k_ not in cb or (cb[k_]["world"], cb[k_]["task"], cb[k_]["arm"]) != (r["world"], r["task"], arm)
               or cb[k_]["meta_file"] != f"{cb[k_]['extraction_run']}/meta/{arm}/{hashlib.sha1(k_.encode()).hexdigest()[:20]}.json"
               or cb[k_]["extraction_run"].endswith("_rendercut_v1") != (k_ in changed) or not isinstance(r["cut"].get("extraction_root"), str)
               or cb[k_]["extraction_run"] not in re.findall(r"[a-z0-9_]+_extract_[a-z0-9_]+", r["cut"]["extraction_root"])]
        if bad:
            raise SystemExit(f"{folder}: {len(bad)} keys whose corrected-B entry (world, task, arm, meta path, rendercut root iff a changed key, root named by the inventory) "
                             f"disagrees: {bad[:3]}")
        cite_cut = {"corrected_b_manifest": cb_c, "q4f_stage_changed_keys": q4_c | {"arm": arm, "n": len(changed)}}
        for k_, r in zip(keys, rows):
            e = cb[k_]
            eps.append({"run_key": k_, "world": e["world"], "task": e["task"], "cell": e["task"], "cut_root": e["extraction_run"], "cut_meta": f"/b/{e['meta_file']}",
                        "cut_staged": {"file": e["staged_inputs_file"], "sha256": e["staged_inputs_sha256"], "mount": "/b"}})
    else:
        root, sset = st["cut"]
        if rows[0]["cut"].get("extraction_root") != root or rows[0]["cut"].get("staged_set") != sset or any(r["cut"] != rows[0]["cut"] for r in rows):
            raise SystemExit(f"{folder}: the inventory's cut side is not {sset} / {root}")
        reg_sha = (att["authority"].get("registered_staged_sha256") or {}).get(f"{arm}.jsonl")
        if not re.fullmatch(r"[0-9a-f]{64}", str(reg_sha)) or f"{arm}.jsonl {reg_sha}" not in prov_b.decode():
            raise SystemExit(f"{folder}: the registered staged sha256 is not the provenance's")
        world_of = {}
        if st["world"] == "honest_r":
            rc_b, rc_c = pinned(RH_COUNTED)
            for w, rr in json.loads(rc_b)["worlds"].items():
                for x in rr:
                    if not re.fullmatch(r"honest_r[1-5]", w) or not x["task"].endswith("_r" + w[-1]) or world_of.setdefault(f"{x['path']}#e0", w) != w:
                        raise SystemExit(f"{RH_COUNTED[0]}: a counted row outside honest_r1..5 / <cell>_r<i>, or a path in two worlds")
            cite_cut["rh_counted_rows"] = rc_c
        for k_, r in zip(keys, rows):
            w = world_of.get(k_) if st["world"] == "honest_r" else st["world"]
            if w is None:
                raise SystemExit(f"{folder}: run {k_} has no counted honest_r<i> row")
            task = r["task"].removesuffix("_r" + w[-1]) if st["world"] == "honest_r" else r["task"]
            eps.append({"run_key": k_, "world": w, "task": task, "cell": task, "cut_root": root,
                        "cut_meta": f"{MOUNT[st['vol']]}/{root}/meta/{arm}/{hashlib.sha1(k_.encode()).hexdigest()[:20]}.json",
                        "cut_staged": {"file": f"{sset}/inputs/{arm}.jsonl", "sha256": reg_sha, "mount": MOUNT[st["vol"]]}})
    by_key = {e["run_key"]: e for e in eps}
    for k_, r in zip(keys, rows):
        if not isinstance(r.get("tokens_before"), int) or r["tokens_before"] != r.get("full_len"):
            raise SystemExit(f"inventory row {k_}: tokens_before differs from full_len")
        by_key[k_] |= {"key": f"{by_key[k_]['world']}|{by_key[k_]['task']}", "arm": arm, "cf_root": u["extraction_root"], "tokens_before": r["tokens_before"],
                       "template_class": tc[k_]}
    if len({e["key"] for e in eps}) != len(eps) or any(e["cell"] not in main_of for e in eps):
        raise SystemExit(f"{folder}: a scoring key twice, or cells outside the T196 list: {sorted({e['cell'] for e in eps} - set(main_of))[:5]}")
    # registered Sol file per key: committed run keys for Amendment 23; Volume packs (rh, A25) bound on Modal by their manifest sha256
    srb, src = pinned(SOL_REC[st["sol"]])
    sr = json.loads(srb)
    # the registered Sol record scored THIS cut side (committed fields): corrected-B manifest / batch C stage + extraction records / rh staged sha / A25 dump
    if st["cut"] == "corrected_b":
        bound = (sr.get("inputs") or {}).get("manifest", {}).get("sha256") == CORRB[2]
    elif st["sol"] == "asis":
        cshas = {v.get("sha256") for v in ((sr.get("inputs") or {}).get("c") or {}).values()}
        rr_c, rr_p = att["authority"]["registered_record"].split(":", 1)
        xb_, _ = pinned(Q4F_C_EXTRACT)
        bound = hashlib.sha256(show(rr_c, rr_p)).hexdigest() in cshas and hashlib.sha256(xb_).hexdigest() in cshas and json.loads(xb_)["config"]["run"] == st["cut"][0]
    elif st["sol"] == "rh":
        bound = (sr.get("provenance") or {}).get("staged_inputs_sha256") == reg_sha
    else:
        bound = (sr.get("dump") or {}).get("extract_run") == st["cut"][0] and sr["dump"].get("staged_run") == st["cut"][1]
    if not bound:
        raise SystemExit(f"{folder}: the registered Sol record {SOL_REC[st['sol']][0]} did not score this registered cut side")
    if st["sol"] == "asis":
        nr = sr.get("runs_without_rows") or {}  # run keys (launch/rel#e0)

        def st_(v, nk):
            hv = set(sr["run_scores"][v]["a_tight"]) & set(sr["run_scores"][v]["c_asis_tight"])
            return lambda e: "registered run score" if e["key"] in hv else "registered: no rows (runs_without_rows)" if e["run_key"] in set(nr.get(nk) or []) else "none"
        r_, c_ = st_("avg_top2", "avg"), st_("cmd_A", "cmd_A")
        reg = {"record": src, "per_key": {e["key"]: f"avg_top2: {r_(e)}; cmd_A: {c_(e)}" for e in eps}, "pack": None}
    else:
        o = sr.get("outputs") or {}
        mm = re.fullmatch(rf"{re.escape(OUTVOL)}:/((?:ref/relaunch|simh/full_b[12])/sol)/", str(o.get("path")))
        if not mm or not re.fullmatch(r"[0-9a-f]{64}", str(o.get("manifest_sha256"))) or sr.get("stage") not in ("done", "scoring and publication"):
            raise SystemExit(f"{SOL_REC[st['sol']][0]}: not a published registered Sol pack with a manifest sha256")
        reg = {"record": src, "per_key": "checked on Modal against the pack's manifest folds", "pack": {"dir": mm.group(1), "manifest_sha256": o["manifest_sha256"],
               "extract_run": st["cut"][0], "staged_run": st["cut"][1], "staged_sha256": reg_sha}}
    # fold routing: r36 by main (T196 cell -> main); training lists per probe written beside the outputs
    rb = Path(r36).read_bytes()
    if not hashlib.sha256(rb).hexdigest().startswith(R36_PIN):
        raise SystemExit("--r36 sha256 mismatch")
    base = json.loads(rb)
    fold_of_main = {}
    for r in base:
        if fold_of_main.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("r36: a main task in two folds")
    for e in eps:
        e["main"] = main_of[e["cell"]]
        e["fold"] = fold_of_main.get(e["main"])
    rt = {"supplied": False}
    if routing:
        rt = routing_equal(routing, folder, target, tok, orig, arm, u, k, a_path, hashlib.sha256(a_raw).hexdigest(), keys, eps)
    return {"folder": folder, "target": target, "token": tok, "arm": arm, "original_folder": orig, "set": st, "simh": st["sol"].startswith("simh"),
            "uncut": u, "epochs": eps, "chunks": sorted(chunks), "chunk_rows": chunk_rows, "gpus": gpus, "r36_rows": base, "routing": rt,
            "cite": {"inventory": inv_c, "attestation": {"path": a_path, "revision": k, "commit_cited": rc, "sha256": hashlib.sha256(a_raw).hexdigest(), "failing": {}},
                     "provenance": prov_c, "extraction_attestation": att.get("extraction_attestation"), "stage_record": {"path": sp, "commit": sc},
                     "extraction_records": xr, "step0_gate": f"{CF_GATE[0]} @ {CF_GATE[1]} (sha256 {CF_GATE[2][:16]}...)", "t196": t196_c,
                     "r36": {"path": r36, "sha256": hashlib.sha256(rb).hexdigest()}, "registered_cut": cite_cut, "registered_sol": reg,
                     "template_class_counts": {v: sum(x == v for x in tc.values()) for v in sorted(set(tc.values()))}}}


def routing_equal(spec: str, folder: str, target: str, tok: str, orig: str, arm: str, u: dict, k: int, a_path: str, a_sha: str, keys: list, eps: list) -> dict:
    """09 @ 2d61af835 ("S_old (b) routing records (record-only)", name corrected by 09 / d9 via 2e 03:0x), 0d's one source of truth: d9's routing record for
    this ROOT, {RDIR}/sold__<target>_cutfull[_r<k>].json, the HIGHEST _r<k> of exactly that base name (as cf_load; never a prefix glob), byte-identical at the branch tip, must
    cite this attestation by path + sha256 and its revision, and equal it by key: staged set, root, Volume, cut full, folder and original folder of the
    token, the arm's staged sha and epochs, and every epoch key it lists. The root must appear in exactly ONE active routing file of the routing dir. Any
    difference stops the job (fail closed); with no record, --phase score refuses."""
    path, _, c = spec.rpartition("@")
    base = f"sold__{target}_cutfull"
    if not re.fullmatch(rf"{re.escape(RDIR)}/{re.escape(base)}(?:_r([2-9]|[1-9][0-9]+))?\.json", path):
        raise SystemExit(f"--routing: <path>@<commit> of {RDIR}/{base}[_r<k>].json (batch sold, one record per root; the highest revision)")
    kk, hp = highest(RDIR, base)
    raw = show(full(c), path)
    if hp != path or not on_branch(c) or show(CF_BRANCH, path) != raw:
        raise SystemExit(f"--routing {path}: not the highest revision on {CF_BRANCH} ({hp}), or not on it byte-identically")
    rec = json.loads(raw)
    diffs = []

    def need(ok, what):
        if not ok:
            diffs.append(what)
    rows_ = [a_ for a_ in rec.get("arms") or [] if str(a_.get("meta_arm")) == arm]
    a_ = rows_[0] if len(rows_) == 1 else {}
    att_ = rec.get("attestation") if isinstance(rec.get("attestation"), dict) else {}
    need(rec.get("batch") == "sold", "batch sold")
    need(rec.get("target") == f"{target}_cutfull", "target")
    need(len(rows_) == 1, f"one arms[] row with meta_arm {arm}")
    need(rec.get("staged_set") == u["staged_set"], "staged_set")
    need(rec.get("extraction_root") == u["extraction_root"], "extraction_root")
    need(rec.get("volume") == CFVOL, "volume")
    need(rec.get("cut") == "full", "cut full")
    need((rec.get("original_folders") or {}).get(tok) == orig, "original_folders[token]")
    need((rec.get("folders") or {}).get(tok) == folder, "folders[token]")
    need(a_.get("staged_sha") == u["staged_sha256"], "arm staged_sha")
    need(a_.get("epochs") == len(keys), "arm epochs")
    need(a_.get("staged_inputs_file") == f"{u['staged_set']}/inputs/{arm}.jsonl", "arm staged_inputs_file")
    need(str(att_.get("path", "")).split(" @ ")[0] == a_path and att_.get("sha256") == a_sha, f"attestation {{path, sha256}} = {a_path}, {a_sha[:16]}...")
    need(att_.get("revision") == k, f"attestation revision {k}")
    lk = a_.get("keys")
    need(isinstance(lk, list) and len(lk) == len(set(lk)) and sorted(lk) == sorted(keys), "arm keys: every epoch key, equal by key to the attestation's")
    wl = a_.get("world")
    if wl is not None:
        ws = {e["world"] for e in eps}
        need(all(w in wl or (w.startswith("honest_r") and "honest_r" in wl) for w in ws), f"key worlds {sorted(ws)} within the record's {wl}")
    act = active_roots()
    need(act.get(u["extraction_root"]) == [path], f"the root {u['extraction_root']} in exactly one active routing file (found {act.get(u['extraction_root'])})")
    if diffs:
        raise SystemExit(f"--routing {path}: differs from the attestation path: {diffs} (fail closed; 0d's one-cohort rule)")
    return {"supplied": True, "path": path, "commit": full(c), "sha256": hashlib.sha256(raw).hexdigest(), "revision": kk, "passes": True,
            "one_active_file_per_root": True, "rule": "09 @ 2d61af835 (S_old (b) routing records, record-only)",
            "keys_compared": f"{len(lk)} keys listed by the record, equal to the attestation's and the inventory's"}


def active_roots() -> dict:
    """Every ACTIVE routing file of the routing dir on the branch (the highest revision of each <batch>__<target>_cutfull name; _attest pointers aside)
    -> {extraction root: [files naming it as their extraction_root, or as any string value equal to a root name]}."""
    names = ls(CF_BRANCH, RDIR)
    best = {}
    for n in names:
        mm = re.fullmatch(r"(.+?_cutfull)(?:_r([2-9]|[1-9][0-9]+))?\.json", n)
        if mm and not n.endswith("_attest.json"):
            kv = int(mm.group(2) or 1)
            if kv >= best.get(mm.group(1), (0, ""))[0]:
                best[mm.group(1)] = (kv, f"{RDIR}/{n}")
    paths = sorted(p for _, p in best.values())
    blob = subprocess.run(["git", "cat-file", "--batch"], input="".join(f"{CF_BRANCH}:{p}\n" for p in paths).encode(), capture_output=True, cwd=HERE, check=True).stdout
    out, pos = {}, 0
    for p in paths:
        hdr_end = blob.index(b"\n", pos)
        size = int(blob[pos:hdr_end].split()[2])
        body = blob[hdr_end + 1:hdr_end + 1 + size]
        pos = hdr_end + 1 + size + 1
        rec = json.loads(body)
        roots = {rec.get("extraction_root")} | {v for v in _strings(rec) if re.fullmatch(r"[a-z0-9_]+_cutfull_extract_v1(?:_try[0-9]+)?", v)}
        for r_ in roots - {None}:
            out.setdefault(r_, []).append(p)
    return out


def _strings(o):
    return [o] if isinstance(o, str) else [z for v in (o.values() if isinstance(o, dict) else o if isinstance(o, list) else []) for z in _strings(v)]


# ---- scoring (Modal side; pure arithmetic, testable with local roots)
def check_epoch(e: dict, m: dict, cm: dict, chunk_rows: dict | None = None) -> tuple[list, dict]:
    """Per epoch BEFORE any row (score_r_cf's rule, extended to the command units): full consumption, unit positions, spans, and the binding of the reply
    (output_end, readout + span) and command (call_end, readout) units below L - 3 (L = min(n, 32768); all when n <= 32768) to the registered cut meta."""
    n, why = m.get("n_tokens"), []
    if m.get("key") != e["run_key"] or cm.get("key") != e["run_key"] or m.get("arm") != e["arm"]:
        raise ValueError("meta key or arm mismatch")
    if not (isinstance(n, int) and not isinstance(n, bool) and n == m.get("tokens_total") == e["tokens_before"] and m.get("cut") == "full" and m.get("truncated") is False):
        why.append("full consumption")
        n = -1
    cs = [c for c in m["calls"] if c["kind"] == "output_end" and c.get("row_mean") is not None]
    ks = [c for c in m["calls"] if c["kind"] == "call_end" and c.get("row") is not None]
    allu = [c["readout"] for c in m["calls"] if c.get("row") is not None or c.get("row_mean") is not None]
    st = {"zone": 0, "past": 0, "zone_c": 0, "past_c": 0}
    ra = [c[f_] for c in m["calls"] for f_ in ("row", "row_mean") if c.get(f_) is not None]
    rs_ = m.get("row_start")
    if ra and (not isinstance(rs_, int) or isinstance(rs_, bool) or rs_ < 0 or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in ra) or len(set(ra)) != len(ra)
               or rs_ + max(ra) + 1 > (chunk_rows or {}).get(m.get("chunk"), -1)):
        why.append("row addresses")
    st["rows"] = (rs_, rs_ + max(ra) + 1) if ra and not why else None
    if any(not isinstance(r, int) or isinstance(r, bool) or not 0 <= r < n for r in allu) or m.get("max_position") != (max(allu) if allu else None):
        why.append("unit positions")
    elif any(not isinstance(c.get("span"), list) or len(c["span"]) != 2 or c["span"][1] - 1 != c["readout"] or not 0 <= c["span"][0] < c["span"][1] for c in cs):
        why.append("span")
    else:
        L = min(n, 32768)
        lo = L - 3 if n > L else L
        ident = lambda c: (c["irid"], str(c["src_irid"]), int(c["call_index"]))  # noqa: E731
        for kind, mine, rk, sp in (("output_end", cs, "row_mean", True), ("call_end", ks, "row", False)):
            uu = {ident(c): (c["readout"], tuple(c["span"]) if sp else None) for c in mine}
            cc = [c for c in cm["calls"] if c["kind"] == kind and c.get(rk) is not None]
            cu = {ident(c): (c["readout"], (tuple(c["span"]) if c.get("span") else None) if sp else None) for c in cc}
            if len(uu) != len(mine) or len(cu) != len(cc):
                why.append(f"duplicate {kind} unit")
            elif {q: v for q, v in cu.items() if v[0] < lo} != {q: v for q, v in uu.items() if v[0] < lo} or any(v[0] >= L for v in cu.values()):
                why.append(f"binding to the registered cut {kind} units")
            else:
                sfx = "" if sp else "_c"
                st["zone" + sfx] += sum(lo <= v[0] < L for v in uu.values())
                st["past" + sfx] += sum(v[0] >= 32768 for v in uu.values())
    return why, {"cs": cs, "ks": ks, "n": n} | st


def score_cf(epochs: list, spec: dict, wroot: str, skip: dict | None, chunk_rows: dict, all_models: bool, cf_root: str = "/cf") -> dict:
    """Every probe of the spec at avg_top2 (score_r_cf's arithmetic, verbatim) and, when skip is given, at cmd_A (score_all's: the fold model's L63 logit on the
    content-only call_end rows). A failing epoch is WITHHELD by key and nothing is scored."""
    import numpy as np

    withheld, kept, stats = {}, [], {"zone": 0, "past": 0, "zone_c": 0, "past_c": 0}
    for e in epochs:
        h = hashlib.sha1(e["run_key"].encode()).hexdigest()[:20]
        m = json.loads((Path(cf_root) / e["cf_root"] / "meta" / e["arm"] / f"{h}.json").read_text())
        cm = json.loads(Path(e["cut_meta"]).read_text())
        if (m.get("row_start") is not None) and m.get("chunk") not in chunk_rows:
            raise ValueError(f"chunk {m.get('chunk')} has no extraction-record evidence")
        why, u = check_epoch(e, m, cm, chunk_rows)
        if why:
            withheld[e["key"]] = {"run_key": e["run_key"], "reasons": why}
            continue
        for q in stats:
            stats[q] += u[q]
        kept.append((e, m, u))
    spans_ = sorted((m["chunk"], *u["rows"], e["key"], e["run_key"]) for e, m, u in kept if u["rows"])
    for a_, b_ in itertools.pairwise(spans_):  # no two epochs of the folder share activation rows of one chunk
        if a_[0] == b_[0] and b_[1] < a_[2]:
            withheld.setdefault(b_[3], {"run_key": b_[4], "reasons": []})["reasons"].append(f"row addresses overlap {a_[3]}")
    base = {"withheld": withheld, "units": stats}
    if withheld:
        return base | {"out": None}
    if skip is not None and set(skip) != {e["run_key"] for e in epochs}:
        raise ValueError("the skip list's run keys are not the folder's")
    cache = {}

    def npz(p):
        if p not in cache:
            cache[p] = np.load(Path(wroot) / p)
        return cache[p]
    fr_e = {e["key"]: e["fold"] for e in epochs}
    out, no_rows = {}, {}
    for v in ("avg_top2",) + (("cmd_A",) if skip is not None else ()):
        per_ep = [(e, m, u["cs"] if v == "avg_top2" else [c for c in u["ks"] if (c["irid"], c["call_index"]) not in {tuple(x) for x in skip[e["run_key"]]}]) for e, m, u in kept]
        units = [(e, m, c) for e, m, sel in per_ep for c in sel]
        no_rows[v] = sorted(e["key"] for e, _, sel in per_ep if not sel)
        keys = np.array([e["key"] for e, _, _ in units], dtype=str)
        fr = [fr_e[k] for k in keys.tolist()]
        nof = np.array([f is None for f in fr], dtype=bool)
        order, seen_ = [], {}
        for k in keys.tolist():
            order.append(seen_.get(k, 0))
            seen_[k] = order[-1] + 1
        extra = {"run_key": np.array([e["run_key"] for e, _, _ in units], dtype=str), "token": np.array([c["readout"] for _, _, c in units], np.int64),
                 "n_tokens": np.array([m["n_tokens"] for _, m, _ in units], np.int64), "irid": np.array([str(c["irid"]) for _, _, c in units], dtype=str),
                 "src_irid": np.array([str(c["src_irid"]) for _, _, c in units], dtype=str)}
        if v == "avg_top2":
            extra |= {"span_start": np.array([c["span"][0] for _, _, c in units], np.int64), "span_end": np.array([c["span"][1] for _, _, c in units], np.int64)}
        out[v] = {}
        rk = "row_mean" if v == "avg_top2" else "row"
        for name, pr in spec["probes"].items():
            files = {L: pr["reply_files"][str(L)]["path"] for L in sorted({L for c in pr["layer_choice"] for L in c})} if v == "avg_top2" else {CMD_L: pr["cmd_file"]["path"]}
            logit = {}
            for L, p in files.items() if units else []:  # score_r_cf's row assembly and logits, verbatim
                Z = npz(p)
                X = np.concatenate([np.asarray(np.load(Path(cf_root) / e["cf_root"] / "by_layer" / f"L{L}" / f"{m['chunk']}.npy", mmap_mode="r")[
                    [m["row_start"] + c[rk] for c in sel]]).astype(np.float32) for e, m, sel in per_ep if sel])
                if not np.isfinite(X).all():
                    raise ValueError(f"{v} L{L}: non-finite rows")
                for f in range(5):
                    W, b, mu, sd = (Z[f"{pr['npz_prefix']}__f{f}__{q}"] for q in ("W", "b", "mean", "std"))
                    logit[(f, L)] = ((X - mu) / sd) @ W + float(b.reshape(-1)[0])
            pick = ((lambda f, sel, lg=logit, ch=pr["layer_choice"]: np.mean([lg[(f, L)][sel] for L in ch[f]], axis=0)) if v == "avg_top2"
                    else (lambda f, sel, lg=logit: lg[(f, CMD_L)][sel]))
            des, sbm = np.full(len(units), np.nan), None
            for f in range(5):
                mk = np.array([x == f for x in fr], dtype=bool)
                if mk.any():
                    des[mk] = pick(f, mk)
            if not np.isfinite(des[~nof]).all():
                raise ValueError(f"{v} {name}: non-finite designated scores")
            if all_models and not units:
                sbm = np.zeros((0, 5), np.float64)
            if units and (all_models or nof.any()):
                sbm = np.full((len(units), 5), np.nan)
                sel = slice(None) if all_models else nof
                for f in range(5):
                    sbm[sel, f] = pick(f, sel)
                if not np.isfinite(sbm if all_models else sbm[nof]).all():
                    raise ValueError(f"{v} {name}: non-finite by-model scores")
                fi = np.array([-1 if x is None else x for x in fr], np.int64)
                if all_models and des[fi >= 0].tobytes() != sbm[fi >= 0, fi[fi >= 0]].tobytes():
                    raise ValueError(f"{v} {name}: a designated score differs from its fold model's score_by_model column (raw bytes)")
            runs, by_model = {}, {}
            for k in dict.fromkeys(keys.tolist()):
                mk = keys == k
                runs[k] = float(des[mk].mean()) if not nof[mk][0] else float("nan")
                if nof[mk][0]:
                    by_model[k] = [float(x) for x in sbm[mk].mean(axis=0)]
            rows = {"key": keys, "score": des, "unit": np.array(order, np.int64), "call_index": np.array([int(c["call_index"]) for _, _, c in units], np.int64),
                    "fold": np.array([-1 if f is None else f for f in fr], np.int64)} | extra | ({"score_by_model": sbm} if sbm is not None else {})
            out[v][name] = {"rows": rows, "runs": runs, "by_model": by_model}
    return base | {"out": out, "no_rows": no_rows, "chunks": sorted({m["chunk"] for _, m, _ in kept})}


def target_check(work: Path, out_name: str, folder: str, keys: list) -> dict:
    """Create-only under /cutfull/sold/<out_name>/; every folder of /cutfull/*/<out_name>/ holds a cutfull manifest; each key at most once across that tree."""
    if (work / "cutfull" / "sold" / out_name / folder).exists():
        raise ValueError(f"cutfull/sold/{out_name}/{folder}/ already exists: create-only (operator decision)")
    others, bad = {}, {}
    for d in sorted((work / "cutfull").glob(f"*/{out_name}/*")) if (work / "cutfull").is_dir() else []:
        mf = d / "manifest_sol.json"
        if not mf.is_file():
            raise ValueError(f"{d.relative_to(work)}: no manifest_sol.json (an interrupted publication?): operator decision")
        man = json.loads(mf.read_bytes())
        if man.get("line") != "cutfull" or not isinstance(man.get("folds"), dict) or not man["folds"]:
            raise ValueError(f"{d.relative_to(work)}: a pack of another line under /cutfull/, or a manifest without its key map (folds)")
        ov = sorted(set(man.get("folds") or {}) & set(keys))
        if ov:
            bad[str(d.relative_to(work))] = ov[:5]
        others[str(d.relative_to(work))] = len(man.get("folds") or {})
    if bad:
        raise ValueError(f"keys already held by other cutfull folders of this tree: {bad}")
    return {"target": f"cutfull/sold/{out_name}/{folder}/", "target_absent": True, "other_folders": len(others), "keys_overlap": 0, "passes": True}


def reg_pack_check(work: Path, pack: dict | None, keys: list) -> dict:
    """The registered Sol pack (rh / A25): its manifest hashes to the committed record's, no SUPERSEDED.json, its dump root and staged set are the set's,
    and its scored keys are the folder's keys (both directions; listed by key)."""
    if pack is None:
        return {"passes": True, "note": "Amendment 23: run keys read from the committed record by key (no Volume pack)"}
    d = work / pack["dir"]
    b = (d / "manifest_sol.json").read_bytes() if (d / "manifest_sol.json").is_file() else b""
    man = json.loads(b) if b else {}
    dump, got = man.get("dump") or {}, set(man.get("folds") or man.get("run_folds") or {})
    sha_ok = bool(re.fullmatch(r"[0-9a-f]{64}", str(pack.get("staged_sha256")))) and dump.get("sha256") == pack["staged_sha256"]
    ok = bool(b) and hashlib.sha256(b).hexdigest() == pack["manifest_sha256"] and not (d / "SUPERSEDED.json").exists() and got == set(keys) \
        and dump.get("extract_run") == pack["extract_run"] and (dump.get("staged_run") == pack["staged_run"] or sha_ok)
    return {"dir": f"/{pack['dir']}/", "manifest_sha256_equal": bool(b) and hashlib.sha256(b).hexdigest() == pack["manifest_sha256"],
            "dump": {"extract_run": dump.get("extract_run"), "staged_run": dump.get("staged_run")}, "keys_not_in_pack": sorted(set(keys) - got),
            "pack_keys_not_here": sorted(got - set(keys)), "passes": ok}


@app.function(image=cpu_image, cpu=(16, 16), memory=(65536, 65536), timeout=3600, retries=0,
              volumes={**ro_vols, "/w": w_vol, "/work": out_vol})
def score_folder(plan: dict, spec: dict, skip_doc: dict, stamp: str, prov: dict, train_manifests: dict) -> dict:
    """One folder x one weights spec: weight gate, staged-file binding by key, create-only and registered-pack checks, then score_cf; files written to
    /work/cutfull/sold/<out_name>/<folder>/ ONLY when every gate passed and no epoch is withheld (each created exclusively, manifest_sol.json last)."""
    import numpy as np

    gates = {"weights": weight_gate(Path("/w"), spec)}
    if not gates["weights"]["passes"]:
        return {"published": False, "gates": gates}
    eps = plan["epochs"]
    b = (Path("/cf") / plan["uncut"]["staged_set"] / "inputs" / f"{plan['arm']}.jsonl").read_bytes()
    staged = [json.loads(x) for x in b.decode().splitlines()]
    n_ids = {s["key"]: len(s["ids"]) for s in staged}
    gates["staged_inputs"] = {"sha256": hashlib.sha256(b).hexdigest(), "keys_equal": sorted(n_ids) == sorted(e["run_key"] for e in eps) and len(n_ids) == len(staged),
                              "len_ids_not_tokens_before": sorted(e["key"] for e in eps if n_ids.get(e["run_key"]) != e["tokens_before"])}
    gates["staged_inputs"]["passes"] = gates["staged_inputs"]["sha256"] == plan["uncut"]["staged_sha256"] and gates["staged_inputs"]["keys_equal"] \
        and not gates["staged_inputs"]["len_ids_not_tokens_before"]
    out_vol.reload()
    keys = [e["key"] for e in eps]
    gates["registered_sol_pack"] = reg_pack_check(Path("/work"), plan["cite"]["registered_sol"]["pack"], keys)
    gates["create_only_each_key_once"] = target_check(Path("/work"), spec["out_name"], plan["folder"], keys)
    if not all(g["passes"] for g in gates.values()):
        return {"published": False, "gates": gates}
    skip = None
    if skip_doc:
        fd = (skip_doc.get("folders") or {}).get(plan["folder"]) or {}
        if fd.get("staged_sha256") != plan["uncut"]["staged_sha256"] or fd.get("gate", {}).get("passes") is not True:
            raise ValueError("the committed skip list is not for this staged file, or its registered-equality gate did not pass")
        skip = {k: [list(x) for x in v] for k, v in fd["skip"].items()}
    res = score_cf(eps, spec, "/w", skip, plan["chunk_rows"], plan["simh"])
    gates["full_consumption_positions_binding"] = {"withheld": res["withheld"], "passes": not res["withheld"],
                                                   "rule": "routing v2 (09 item 2): a withheld epoch fails the folder; withholding never lets the remaining keys route alone"}
    if res["withheld"]:
        return {"published": False, "gates": gates, "withheld": res["withheld"]}
    ks_all = set(keys)
    cov = {}
    for v, x in res["out"].items():
        got = [set(x[n]["runs"]) for n in x]
        cov[v] = {"scored": len(got[0]), "without_rows": res["no_rows"][v], "passes": all(gg == got[0] for gg in got) and got[0] | set(res["no_rows"][v]) == ks_all
                  and not got[0] & set(res["no_rows"][v])}
    gates["scored_coverage"] = cov | {"passes": all(c["passes"] for c in cov.values())}
    rm = {}
    for v, x in res["out"].items():
        for name, y in x.items():
            dmax, bad = 0.0, []
            for k, r in y["runs"].items():
                mk = y["rows"]["key"] == k
                if np.isnan(r):
                    cm = y["rows"]["score_by_model"][mk].mean(axis=0)
                    d = float(np.max(np.abs(cm - np.array(y["by_model"][k])))) if np.isfinite(cm).all() else float("inf")
                else:
                    d = abs(float(np.mean(y["rows"]["score"][mk])) - r)
                dmax = max(dmax, d)
                if not d <= 1e-9:
                    bad.append(k)
            rm[f"{v}|{name}"] = {"max_abs_diff": dmax, "runs_over_1e-9": bad, "passes": not bad}
    gates["row_mean_equals_run_score"] = rm | {"passes": all(g["passes"] for g in rm.values())}
    if not all(g["passes"] for g in gates.values()):
        return {"published": False, "gates": gates}
    blobs = {fn: s.encode() for fn, s in train_manifests.items()}
    for v, x in res["out"].items():
        for name, y in x.items():
            js = {"run_scores": y["runs"], f"runs_without_{'reply' if v == 'avg_top2' else 'cmd'}_rows": res["no_rows"][v]} | ({"by_model": y["by_model"]} if y["by_model"] else {})
            blobs[f"{name}__{v}.json"] = json.dumps(js, indent=1).encode()
            buf = io.BytesIO()
            np.savez(buf, **y["rows"])
            blobs[f"{name}__{v}.npz"] = buf.getvalue()
    files = {fn: hashlib.sha256(bb).hexdigest() for fn, bb in blobs.items()}
    if sorted(prov.get("folds") or {}) != sorted(keys):
        raise ValueError("the manifest's folds are not the folder's keys")
    man = prov | {"folds": {e["key"]: (-1 if e["fold"] is None else e["fold"]) for e in eps}, "line": "cutfull", "label": LABEL, "folder": plan["folder"], "batch": "sold", "stamp": stamp, "files": files, "gates": gates,
                  "units": res["units"], "chunks_scored": res["chunks"], "runs_without_rows": res["no_rows"],
                  "readouts": {"avg_top2": "reply (R): the mean of the fold model's two chosen layers of the output_end span-mean rows",
                               "cmd_A": (CMD_RULE + f"; skip list {skip_doc.get('cite')}") if skip is not None else "not scored (no committed skip list given)"}}
    mb = json.dumps(man, indent=1).encode()
    od = Path("/work") / "cutfull" / "sold" / spec["out_name"] / plan["folder"]
    od.mkdir(parents=True, exist_ok=False)
    (od / ".publishing").mkdir(exist_ok=False)
    for fn, bb in list(blobs.items()) + [("manifest_sol.json", mb)]:
        with (od / fn).open("xb") as fh:
            fh.write(bb)
    (od / ".publishing").rmdir()
    out_vol.commit()
    return {"published": True, "files": files, "manifest_sha256": hashlib.sha256(mb).hexdigest(), "gates": gates, "units": res["units"],
            "runs_without_rows": {v: len(x) for v, x in res["no_rows"].items()}, "path": f"{OUTVOL}:/cutfull/sold/{spec['out_name']}/{plan['folder']}/"}


def shared_skip():
    """The shared rebuild module, its file sha256 asserted (byte-identical to SOLA-UNCUT's @ 7f15d2c0b)."""
    from probe_shift import cutfull_skip as CS

    if hashlib.sha256(Path(CS.__file__).read_bytes()).hexdigest() != CUTFULL_SKIP[2]:
        raise ValueError(f"probe_shift/cutfull_skip.py is not the pinned shared rebuild ({CUTFULL_SKIP[1]}, sha256 {CUTFULL_SKIP[2][:16]}...)")
    return CS


def read_staged(path: Path, sha: str, keys: set) -> list:
    b = path.read_bytes()
    if hashlib.sha256(b).hexdigest() != sha:
        raise ValueError(f"{path}: sha256 differs from the routed / registered sha")
    return [ep for ep in (json.loads(x) for x in b.decode().splitlines()) if ep["key"] in keys]


@app.function(image=tok_image, cpu=(8, 8), memory=(32768, 32768), timeout=3600, retries=0, volumes={**ro_vols, "/hf": modal.Volume.from_name(HFVOL).read_only()})
def skip_rebuild(plan: dict) -> dict:
    """The C skip-list rebuild for one folder (09's conditions; the shared function): the registered rule, unchanged, on the cutfull staged epochs and on
    the registered 32k staged epochs of the same keys (each file sha256-checked). Returns per key the decisions of both sides and the rebuilt lists
    ([[irid, call_index], ...]); no activation and no score is read."""
    from transformers import AutoTokenizer

    from probe_shift.models import QWEN35_27B
    from probe_shift.q1_eval import fixed_call_tokens

    CS = shared_skip()
    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"], local_files_only=True)
    fixed = fixed_call_tokens("q4b_main_v1")
    fixed_ids = set(tok.convert_tokens_to_ids(fixed))
    keys = {e["run_key"] for e in plan["epochs"]}
    rebuilt = read_staged(Path("/cf") / plan["uncut"]["staged_set"] / "inputs" / f"{plan['arm']}.jsonl", plan["uncut"]["staged_sha256"], keys)
    files = sorted({(e["cut_staged"]["mount"], e["cut_staged"]["file"], e["cut_staged"]["sha256"]) for e in plan["epochs"]})
    want = {(e["cut_staged"]["mount"], e["cut_staged"]["file"]): set() for e in plan["epochs"]}
    for e in plan["epochs"]:
        want[(e["cut_staged"]["mount"], e["cut_staged"]["file"])].add(e["run_key"])
    registered = []
    for mnt, f_, sha in files:
        got = read_staged(Path(mnt) / f_, sha, want[(mnt, f_)])
        if sorted(ep["key"] for ep in got) != sorted(want[(mnt, f_)]):
            raise ValueError(f"{mnt}/{f_}: does not hold each of its {len(want[(mnt, f_)])} keys exactly once")
        registered += got
    if sorted(ep["key"] for ep in rebuilt) != sorted(keys):
        raise ValueError("the cutfull staged file does not hold each of the folder's keys exactly once")
    r = CS.rebuild_skip(rebuilt, registered, fixed_ids)
    return {"decisions": r["decisions"], "lists": r["lists"]["rebuilt"], "fixed_tokens": fixed, "fixed_ids": sorted(fixed_ids),
            "tokenizer": {"name": QWEN35_27B["name"], "revision": QWEN35_27B["revision"]}, "registered_files": [{"mount": m_, "file": f_, "sha256": h_} for m_, f_, h_ in files]}


@app.function(image=gate_image, cpu=(2, 2), memory=(8192, 8192), timeout=1800, retries=0)
def skip_gate(registered: dict, rebuilt: dict, registered_count) -> dict:
    """09's rebuild gate in its OWN call (the shared gate_report): 0 mismatches both directions on every registered unit, the registered dropped count
    equal to the registered pack's when it has one, a perturbed-unit negative test that must fail, and the units past 32,768."""
    return shared_skip().gate_report(registered, rebuilt, registered_count)


def training_manifests(spec: dict, base: list, eps: list) -> tuple[dict, dict, dict]:
    """The five training-main lists per probe (r36 rows of the probe's training arms, folds != f), the fold-routing gate and fold -1 routing evidence."""
    tm_files, tsha, tsets = {}, {}, {}
    for name, pr in spec["probes"].items():
        for f in range(5):
            tm = sorted({r["main_task"] for r in base if r["arm"] in pr["training_arms"] and r["fold"] != f})
            b = json.dumps({"probe": name, "fold_model": f, "training_mains": tm, "source": "#36 run table (r36), arms " + ", ".join(pr["training_arms"]) + f", folds != {f}"}, indent=1)
            tm_files[f"training_mains__{name}__f{f}.json"] = b
            tsha.setdefault(name, []).append(hashlib.sha256(b.encode()).hexdigest())
            tsets.setdefault(name, []).append(set(tm))
    fold_of = {e["main"]: e["fold"] for e in eps}
    bad = {n: sorted(mn for mn, f in fold_of.items() if (f is None and any(mn in t for t in tsets[n])) or (f is not None and any((mn in tsets[n][g]) != (g != f) for g in range(5))))
           for n in tsets}
    routing = {mn: {n: {"excluded_by_all_five": True, "training_manifest_sha256": tsha[n]} for n in tsets} for mn, f in fold_of.items() if f is None}
    return tm_files, {"mains_mapped": sum(f is not None for f in fold_of.values()), "mains_fold_minus1": sorted(mn for mn, f in fold_of.items() if f is None),
                      "failures": bad, "passes": not any(bad.values()), "rule": "r36 fold map by main; a mapped main is absent from its fold model's training list and in the "
                      "other four; a fold -1 main is in none"}, routing


FOLDERS = ("q4f_main_cutfull_v1__attack", "q4f_main_cutfull_v1__honest_a", "q4f_main_cutfull_v1__honest_b", "q4f_main_cutfull_v1__sim_attack",
           "q4f_run2_cutfull_v1__attack", "q4f_c_cutfull_v1__honest_b", "rh1e4_stage_cutfull_v1__honest_b",
           "simh_q4f_v1_full_b1_cutfull_v1__sim_honest", "simh_q4f_v1_full_b2_cutfull_v1__sim_honest")


def route_for(routing: str, folder: str) -> str:
    """--routing is one <path>@<commit> or a comma list (one record per root): the one whose name is sold__<this folder's target>_cutfull[_r<k>].json."""
    t = re.fullmatch(r"([a-z0-9_]+?)_cutfull_v1__[a-z_]+", folder).group(1)
    hits = [x for x in routing.split(",") if x and re.fullmatch(rf"{re.escape(RDIR)}/sold__{re.escape(t)}_cutfull(?:_r[0-9]+)?\.json@[0-9a-f]{{9,40}}", x)]
    if len(hits) > 1:
        raise SystemExit(f"--routing names two records for {t}")
    return hits[0] if hits else ""


def registered_count(plan: dict):
    """Amendment 23's registered dropped count for this folder's keys (skip_dropped_rows of the files they were staged in, staged shas equal), else None."""
    if plan["set"]["sol"] != "asis":
        return None
    sr = json.loads(pinned(SOL_REC["asis"])[0])
    files = {e["cut_staged"]["file"]: e["cut_staged"]["sha256"] for e in plan["epochs"]}
    if any(sr["staged_sha256"].get(f_) != h_ for f_, h_ in files.items()) or any(f_ not in sr["skip_dropped_rows"] for f_ in files):
        raise SystemExit(f"{plan['folder']}: Amendment 23's staged sha256 / dropped rows do not cover the registered files {sorted(files)}")
    return sum(sr["skip_dropped_rows"][f_] for f_ in files)


@app.local_entrypoint()
def main(phase: str = "dry", folder: str = "", r36: str = "", weights: str = "", routing: str = "", skip_list: str = ""):
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if phase not in ("dry", "skip", "score") or not folder or not r36 or (folder == "ALL" and phase == "score"):
        raise SystemExit("--phase dry|skip|score --folder <S_old cutfull folder> (ALL for dry / skip) --r36 <runs.json>")
    if phase != "dry" and git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    if hashlib.sha256((REPO / CUTFULL_SKIP[0]).read_bytes()).hexdigest() != CUTFULL_SKIP[2] or show(CUTFULL_SKIP[1], CUTFULL_SKIP[0]) != (REPO / CUTFULL_SKIP[0]).read_bytes():
        raise SystemExit(f"{CUTFULL_SKIP[0]} is not byte-identical to the shared rebuild @ {CUTFULL_SKIP[1]}")
    if phase == "skip":  # the C skip list (one container per folder for the rebuild, then 09's gate in its own call per folder)
        plans = [admit(f, r36, route_for(routing, f)) for f in (FOLDERS if folder == "ALL" else (folder,))]
        for pl in plans:
            pl.pop("r36_rows")
        rb = list(skip_rebuild.map(plans))
        gates = list(skip_gate.starmap([(r_["decisions"]["registered"], r_["decisions"]["rebuilt"], registered_count(pl)) for pl, r_ in zip(plans, rb)]))
        ok = all(g_["passes"] for g_ in gates)
        doc = {"label": "S_old C skip lists, rebuilt from the cutfull staged inputs (Amendment 51; 09's conditions; keys and [irid, call_index] only)", "stamp": stamp,
               "git": git("rev-parse", "HEAD").strip(), "rule": CMD_RULE, "shared_function": dict(zip(("path", "commit", "sha256"), CUTFULL_SKIP)), "passes": ok,
               "folders": {pl["folder"]: {"staged_sha256": pl["uncut"]["staged_sha256"], "staged_file": f"{pl['uncut']['staged_set']}/inputs/{pl['arm']}.jsonl",
                                          "registered_files": r_["registered_files"], "fixed_tokens": r_["fixed_tokens"], "fixed_ids": r_["fixed_ids"], "tokenizer": r_["tokenizer"],
                                          "routing_record": pl["routing"], "attestation": pl["cite"]["attestation"], "gate": g_, "skip": r_["lists"]}
                           for pl, r_, g_ in zip(plans, rb, gates)}}
        dest = HERE / "results" / f"sold_skiplist_{stamp}{'' if ok else '_GATE_FAILED'}.json"
        with dest.open("x") as fh:
            fh.write(json.dumps(doc, indent=1))
        for pl, g_ in zip(plans, gates):
            print(pl["folder"], "gate", g_["passes"], "units", g_["units_compared"], "mismatches", g_["mismatches"], "registered dropped", g_["registered_dropped"],
                  "count given", g_["registered_count_given"], "negative test fails as required", (g_["negative_test"] or {}).get("fails_as_required"), "past 32768", g_["past_32768"])
        print(dest.name, hashlib.sha256(dest.read_bytes()).hexdigest(), "PASS" if ok else "GATE FAILED")
        if not ok:
            raise SystemExit("the skip-list gate failed")
        return
    plan = admit(folder, r36, route_for(routing, folder))
    spec, spec_c = load_spec(weights)
    tm_files, fr_gate, fr_routing = training_manifests(spec, plan.pop("r36_rows"), plan["epochs"])
    if not fr_gate["passes"]:
        raise SystemExit(f"fold routing disagrees with the training lists: {fr_gate['failures']}")
    rec = {"label": "S_old cutfull Sol scoring (Amendment 51): shas, counts, keys and gates only", "phase": phase, "folder": folder, "stamp": stamp,
           "git": git("rev-parse", "HEAD").strip(), "weights": {"spec": spec_c, "label": spec["label"], "volume": spec["volume"], "out_name": spec["out_name"],
           "probes": sorted(spec["probes"])}, "admission": plan["cite"], "routing_record": plan["routing"], "fold_routing": fr_gate,
           "epochs": len(plan["epochs"]), "folds": {str(f): sum(e["fold"] == f for e in plan["epochs"]) for f in (None, 0, 1, 2, 3, 4)}}
    print(folder, "epochs", rec["epochs"], "attestation r", plan["cite"]["attestation"]["revision"], "template", plan["cite"]["template_class_counts"],
          "cut roots", sorted({e["cut_root"] for e in plan["epochs"]}), "routing", plan["routing"].get("passes", "not supplied"))
    if phase == "dry":
        return
    if spec["volume"] != WVOL:
        raise SystemExit(f"SOLD_WEIGHTS_VOL ({WVOL}) is not the spec's Volume ({spec['volume']})")
    if not plan["routing"].get("passes"):
        raise SystemExit("--phase score needs d9's routing record for this folder (--routing <path>@<commit>; 0d's one-cohort rule)")
    if not skip_list:
        raise SystemExit("--phase score needs the committed rebuilt skip list (--skip-list <path>@<commit>): the headline leads with C (2e 02:5x)")
    skip_doc = {}
    if skip_list:
        p, _, c = skip_list.rpartition("@")
        b, sc = pinned((p, full(c), None))
        skip_doc = json.loads(b) | {"cite": f"{p} @ {c} (sha256 {sc['sha256']})"}
        if not re.fullmatch(rf"{re.escape(Q_REL)}/results/sold_skiplist_\d{{4}}-\d\d-\d\d_\d{{6}}\.json", p) or skip_doc.get("passes") is not True \
                or skip_doc.get("shared_function") != dict(zip(("path", "commit", "sha256"), CUTFULL_SKIP)) or folder not in (skip_doc.get("folders") or {}):
            raise SystemExit("--skip-list: not a committed passing results/sold_skiplist_<stamp>.json of the pinned shared rebuild holding this folder")
    prov = {"scorer_git": rec["git"], "weights": rec["weights"] | {"spec_doc": spec}, "admission": plan["cite"], "routing_record": plan["routing"],
            "fold_routing": fr_gate, "fold_minus1_routing": fr_routing, "folds": {e["key"]: (-1 if e["fold"] is None else e["fold"]) for e in plan["epochs"]},
            "per_key": {e["key"]: {"run_key": e["run_key"], "fold": e["fold"], "main": e["main"], "template_class": e["template_class"], "cut_root": e["cut_root"],
                                   "cut_meta": e["cut_meta"], "tokens_before": e["tokens_before"]} for e in plan["epochs"]},
            "gpu_per_shard": plan["gpus"], "original_folder": plan["original_folder"], "uncut": plan["uncut"],
            "formats": {"json": "<probe>__<readout>.json: run_scores (designated raw run score = row mean), runs_without_<reply|cmd>_rows, by_model (fold -1 runs)",
                        "npz": "<probe>__<readout>.npz per row: key (world|task), run_key, score (designated; NaN for fold -1), unit, call_index, fold, token, n_tokens, "
                               "irid, src_irid[, span_start, span_end (avg_top2)][, score_by_model (rows x 5: every row for sim-honest, else fold -1 rows)]"},
            "not_done_here": "no z-scoring, ensembles, thresholds, false-alarm rates or comparisons"}
    rec["stage"] = "scoring"
    dest = HERE / "results" / f"sold_cutfull_{folder}{'' if spec['out_name'] == 'sol' else '_' + spec['out_name']}_{stamp}.json"
    r = score_folder.remote(plan, spec, skip_doc, stamp, prov, tm_files)
    rec |= {"stage": "done" if r.get("published") else "not published", "outputs": r}
    with dest.open("x") as fh:
        fh.write(json.dumps(rec, indent=1))
    print(dest.name, "published" if r.get("published") else "NOT published", {k: g.get("passes") for k, g in r.get("gates", {}).items()})
    if not r.get("published"):
        raise SystemExit("not published")
