"""sol_a_uncut on the full-length (cutfull) sets (POST HOC). 09's RULE (experiments/2026-10-14_priority1_spec_09/RULE.md, aswin/convergence-audit-09):
uncut "(c) GO" names sol_a_uncut; full-length analysis pre-registered @ 07ad588be §2; lead ruling @ 129b7c4f1; the Sol (a) C skip-list rebuild
OK'd by 09 via 2e (10-16 ~02:15; its RULE sha is cited in the records once it lands). Raw decision values, counts, shas and gates only.

The job is BOUND to the registered cutfull Sol packs (run_scaleup_score.py's cutfull line, Amendment 46): per folder of a committed index
(cutfull_sol_index_*.json), the registered pack's manifest_sol.json and a_tight__avg_top2.npz are read by sha256. Every admission gate (routing
revision, attestation, full consumption, binding of the reply units to the registered cut units, the r36 fold routing) passed there and is inherited.

  --phase skip (CPU + tokenizer): the C skip list REBUILT from each folder's cutfull staged inputs with the registered rule, unchanged
    (run_scaleup_score.skip_rows: a call_end call is dropped when its readout token is one of fixed_call_tokens("q4b_main_v1") = '0', 'n'); the same rule on
    the registered 32k staged inputs gives the registered list. Gate, in its own call (skip_gate): on every call_end unit of the 32k staged inputs, the
    rebuilt keep/skip decision equals the registered one by key, in both directions, 0 mismatches; a negative test (one perturbed unit) must fail; the
    registered list's dropped count equals the registered pack's skip_dropped_rows. Counts past token 32,768 per folder. Writes
    results/cutfull/sola_uncut_skiplist_<stamp>.json: commit it (sha256) BEFORE any C score.
  --phase score (CPU): R (avg_top2) is score_r_cf's arithmetic verbatim (output_end span-mean rows; per fold model the mean of its two chosen layers);
    positive control first: the REGISTERED weights reproduce the registered pack's a_tight rows (unit identity arrays exactly; scores within 1e-9, bytes
    reported); then sol_a_uncut's weights, pinned by file sha256 and fold-model sha256 from its PASSED fit record. C (cmd_A) is score_all's cmd_A
    arithmetic (L63 call_end rows minus the committed rebuilt skip list, the designated fold model; fold -1 and sim-honest rows carry all five models);
    binding: the C units below the 32k boundary zone (readout < L - 3, L = min(n, 32768)) equal the registered cut C units by (irid, src_irid,
    call_index) with readout. Output, create-only, manifest last: sol_a_uncut__avg_top2.{npz,json},
    sol_a_uncut__cmd_A.{npz,json}, under /cutfull/<batch>/sola_uncut_<variant>/<cutfull folder>/ with manifest_sola_uncut_<variant>.json (cites the inventory's
    sha256 from the registered pack). --regc (once per folder): the REGISTERED a_tight and c_asis_tight command models on the rebuilt skip list, under
    /cutfull/<batch>/sol_c/<cutfull folder>/ (a_tight__cmd_A.*, c_asis_tight__cmd_A.*, manifest_sol_c.json). Both destinations are checked absent first.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run experiments/2026-10-14_q35_sol_avg_POSTHOC/run_sola_uncut_score.py \\
      --phase skip --index <repo-relative index json>@<commit>
  ... --phase score --index <..>@<commit> --fit-record <repo-relative sola_uncut fit record>@<commit> --skip-list <..>@<commit> [--readouts R]
  (each @<commit> file is read as committed, from any branch of this repository)
"""

import hashlib
import io
import json
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-q35solavg_solauncut_kill-2026-11-14"
SOLVOL, OUTVOL, CFVOL = ("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14",
                         "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14")
DUMPVOL, HFVOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", "probe-shift_aswin-surya_hf-cache"
MOUNT = {CFVOL: "/cf", DUMPVOL: "/d"}  # activation Volumes by name (a pack naming any other Volume stops the job)
REG_TAG = "asis_2026-10-14_135701"
REG_CHOICE = [[49, 46], [46, 47], [46, 47], [46, 47], [34, 46]]  # (a)'s as-run layer choice (avg_transfer_avg_v1_2026-10-14_011347.json)
# run_scaleup_score.py's pins of the registered tight files and (a)'s fold models (its fit_sha256 method), verbatim
REG_NPZ = {"asis_2026-10-14_135701/outer_tight_L34.npz": "82f57ecfd2478be0daacc9e0d60220fdd3c16f8e26f9967fbc1de58bb32f9a93",
           "asis_2026-10-14_135701/outer_tight_L46.npz": "ed561f100a573a5c0940b6ca2b3b06324de1958fb7ad42776a2cbb19acc672a1",
           "asis_2026-10-14_135701/outer_tight_L47.npz": "1488630dda37c8b1ca9ef83ce19fcada88d36afc2fb0f10d54ea1434be87cd3c",
           "asis_2026-10-14_135701/outer_tight_L49.npz": "5d98db68f81df37b3890780a49c338b7ce5672cce5a52149b20aa1680f9ac5bc",
           "asis_2026-10-14_135701_cmd/outer_tight_L63.npz": "a76a03cced4122804c0eb2630aa879d355493d49eb6ce071e26d68faad25990b"}
REG_MODEL = {"0": "4e3afc3272e9fe930d34f5e3f059760d53a970b8819053eb0ee1bf5eb612c441", "1": "ac25b2003489dcbbe4bb9949c527a9e00065c487ef38870ead8be908aee46bdb",
             "2": "e8c33690ad6b08a57deb58f76816ee52f2811373e360f39080cbab6f2654a652", "3": "aff4824ccbf58ab7aca12c152c88bc27015bf03aacb5dc60ca91da8b34aad4de",
             "4": "3828be20fc84202c04b1c906ba1a4a499a81f16eff9832ba9bfa7265cab20595"}
REG_MODEL_C = {"0": "7001e33c8dea684faae82f6c209ac488ca8653d6d4d71b7b92d84c90d05a5bf1", "1": "56420843beba427d1c98243c0a3a31e6f0fa5c4353006c241d4e76278b9a0194",
               "2": "7873f4cc093e90b79d05548b95c742d6848594ffa9c8dc1eac8b1d29504fee95", "3": "0cfae59e9cbe1a28b2126fd72ef9f17cf06b73e74cd3de674d7542d13b6e03ed",
               "4": "0654a3b30ec5c89abf3b75c720b2f0878bd31af979635825816bd919a7a71225"}  # run_scaleup_score.FIT_MODEL_SHA256["c_asis"], verbatim
NAME = "sol_a_uncut"
PROBES = {"sol_a_uncut": {"rc": "a", "prefix": "sola_uncut", "variants": ("reselect", "pinned")},  # probe -> npz recipe key, scorer-folder prefix, variants
          "c_asis_uncut": {"rc": "c_asis", "prefix": "casis_uncut", "variants": ("inherit", "pinned")}}  # c_asis_uncut: 09 RULE @ c0afdf2a6 item 22
REGC_SCORER = "sol_c"  # the REGISTERED a_tight and c_asis_tight command models on the rebuilt skip list (the Sol launcher's packs are reply-only)
LABEL = "new probe trained on full transcripts (OOD class (i)); POST HOC; not the registered probe"
CUT, POS_TOL = 32768, 1e-9
UNIT_FIELDS = ("key", "run_key", "token", "n_tokens", "irid", "src_irid", "span_start", "span_end", "unit", "call_index", "fold")

app = modal.App(APP)
sol_vol = modal.Volume.from_name(SOLVOL, version=2).read_only()
out_vol = modal.Volume.from_name(OUTVOL, version=2)
cf_vol = modal.Volume.from_name(CFVOL, version=2).read_only()
d_vol = modal.Volume.from_name(DUMPVOL, version=2).read_only()
cpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3").add_local_python_source("probe_shift")
tok_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "numpy==2.3.3")
             .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))
VOLS = {"/sol": sol_vol, "/work": out_vol, "/cf": cf_vol, "/d": d_vol}


def vpath(volume: str, rel: str) -> Path:
    if volume not in MOUNT:
        raise ValueError(f"Volume {volume} is not mounted by this job")
    return Path(MOUNT[volume]) / rel


def h20(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:20]


def read_pack(ent: dict) -> dict:
    """The registered cutfull pack of one index entry, every file by sha256; its epochs (key, world, task, fold) rebuilt from its staged inputs with
    the registered scorer's world rule and checked equal to the pack's folds by key; the registered cut pack it binds to."""
    m_ = re.fullmatch(rf"{re.escape(OUTVOL)}:/((cutfull|cutfull_ep2)/([^/]+)/sol/([^/]+))/", ent["volume_path"])  # cutfull_ep2: the Sol launcher's epoch-2 tree
    if not m_ or m_.group(4) != ent["folder"]:
        raise ValueError(f"{ent['folder']}: volume_path is not {OUTVOL}:/cutfull[_ep2]/<batch>/sol/<folder>/")
    rel, tree, batch = m_.group(1), m_.group(2), m_.group(3)
    d = Path("/work") / rel
    mb = (d / "manifest_sol.json").read_bytes()
    if hashlib.sha256(mb).hexdigest() != ent["manifest_sha256"]:
        raise ValueError(f"{rel}: manifest sha256 differs from the index")
    man = json.loads(mb)
    blobs = {}
    for fn in ("a_tight__avg_top2.npz", "a_tight__avg_top2.json"):
        blobs[fn] = (d / fn).read_bytes()
        if hashlib.sha256(blobs[fn]).hexdigest() != ent["files_sha256"][fn] or man["files"][fn] != ent["files_sha256"][fn]:
            raise ValueError(f"{rel}/{fn}: sha256 differs from the index or the manifest")
    if man.get("line") != "cutfull" or man.get("folder") != ent["folder"] or man.get("batch") != batch or man.get("layer_choice_a") != REG_CHOICE:
        raise ValueError(f"{rel}: not a cutfull pack of this folder and batch, or not (a)'s layer choice")
    dump, cf = man["dump"], man["cutfull"]
    sb = vpath(dump["volume"], f"{dump['staged_run']}/inputs/{dump['meta_arm']}.jsonl").read_bytes()
    if hashlib.sha256(sb).hexdigest() != dump["sha256"]:
        raise ValueError(f"{rel}: cutfull staged inputs sha256 differs from the pack's")
    (world,) = man["arm_to_world"].values()
    r2 = man.get("r2_suffix_world")
    eps = []
    for line in sb.decode().splitlines():
        ep = json.loads(line)
        t = ep["task"]
        w, t = ("run2", t[:-3]) if r2 == "run2" and t.endswith("_r2") else (world, t)
        in_pack = f"{w}|{t}" in man["folds"]  # not in the pack: an orphan, filtered below
        f_ = man["folds"].get(f"{w}|{t}")
        if in_pack and not (f_ in (-1, None) or (isinstance(f_, int) and not isinstance(f_, bool) and 0 <= f_ < 5)):
            raise ValueError(f"{rel}: {w}|{t} has an invalid fold in the pack")
        eps.append({"key": ep["key"], "world": w, "task": t, "fold": None if f_ in (-1, None) else f_})  # -1 / None = main absent from r36 (fold -1)
    names = [f"{e['world']}|{e['task']}" for e in eps]
    if len(set(names)) != len(names):
        raise ValueError(f"{rel}: two staged epochs share a world and task")
    orphans = sorted(e["key"] for e in eps if f"{e['world']}|{e['task']}" not in man["folds"])  # staged but not in the registered pack (inventory keys only, 2e)
    eps = [e for e in eps if f"{e['world']}|{e['task']}" in man["folds"]]
    if {f"{e['world']}|{e['task']}" for e in eps} != set(man["folds"]):
        raise ValueError(f"{rel}: a key of the registered pack (folds) has no staged epoch")
    reg_dir = ent["cut_pack"]["dir"]  # the registered cut pack, pinned by its committed landing record's manifest sha256 (client side)
    cs_ = (man.get("gates") or {}).get("cut_side_registered_score_files")
    if cs_ and (cs_.get("registered_manifest_sha256") != ent["cut_pack"]["manifest_sha256"] or cs_.get("registered_dir", "").strip("/") != reg_dir):
        raise ValueError(f"{rel}: the pack's own cut-side binding names another registered pack than the committed landing record")
    cmb = (Path("/work") / reg_dir / "manifest_sol.json").read_bytes()
    if hashlib.sha256(cmb).hexdigest() != ent["cut_pack"]["manifest_sha256"]:
        raise ValueError(f"{rel}: the registered cut pack {reg_dir} manifest sha256 differs from its committed landing record's")
    cman = json.loads(cmb)
    cd = cman["dump"]
    if cd["staged_run"] != cf["original_staged_set"] or cd["meta_arm"] != dump["meta_arm"] or cd["sha256"] != cf["original_staged_sha256"]:
        raise ValueError(f"{rel}: the registered cut pack {reg_dir} is not the original staged set / arm / staged sha of this folder")
    return {"rel": rel, "tree": tree, "batch": batch, "man": man, "manifest_sha256": ent["manifest_sha256"], "blobs": blobs, "epochs": eps, "staged_sha256": dump["sha256"], "orphans": orphans,
            "cut": {"dir": reg_dir, "manifest_sha256": hashlib.sha256(cmb).hexdigest(), "volume": cd["volume"], "extract_run": cd["extract_run"],
                    "staged_run": cd["staged_run"], "meta_arm": cd["meta_arm"], "staged_sha256": cd["sha256"],
                    "skip_dropped_rows": cman.get("skip_dropped_rows"), "fixed_tokens": cman.get("fixed_tokens"), "files": cman.get("files") or {}}}


# ---- phase skip -------------------------------------------------------------------------------------------------------------------------------

@app.function(image=tok_image, cpu=(4, 4), memory=(16384, 16384), timeout=1800, retries=0,
              volumes=VOLS | {"/hf": modal.Volume.from_name(HFVOL).read_only()})
def skip_build(ent: dict) -> dict:
    """One folder: per epoch the registered (32k staged inputs) and rebuilt (cutfull staged inputs) decisions, by probe_shift.cutfull_skip's rule
    (no comparison here: the gate is its own call)."""
    from transformers import AutoTokenizer

    from probe_shift.cutfull_skip import rebuild_skip
    from probe_shift.models import QWEN35_27B
    from probe_shift.q1_eval import fixed_call_tokens

    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"], local_files_only=True)
    fixed = fixed_call_tokens("q4b_main_v1")
    fixed_ids = set(tok.convert_tokens_to_ids(fixed))
    p = read_pack(ent)
    c = p["cut"]
    if c["fixed_tokens"] != fixed:
        raise ValueError(f"{p['rel']}: the registered pack's fixed tokens {c['fixed_tokens']} are not {fixed}")
    files = {"rebuilt": (p["man"]["dump"]["volume"], f"{p['man']['dump']['staged_run']}/inputs/{p['man']['dump']['meta_arm']}.jsonl", p["staged_sha256"]),
             "registered": (c["volume"], f"{c['staged_run']}/inputs/{c['meta_arm']}.jsonl", c["staged_sha256"])}
    from probe_shift.cutfull_skip import skip_decisions

    eps = {}
    for side, (vol, rel, want) in files.items():
        b = vpath(vol, rel).read_bytes()
        if hashlib.sha256(b).hexdigest() != want:
            raise ValueError(f"{rel}: sha256 differs from the pack's")
        eps[side] = [json.loads(line) for line in b.decode().splitlines()]
        if len({e["key"] for e in eps[side]}) != len(eps[side]):
            raise ValueError(f"{rel}: a key twice")
    # the gate compares EVERY epoch present in both staged files (09: every unit of the registered 32k inputs that the rebuild holds); registered-only
    # and cutfull-only epochs are listed by key; the registered count is over the WHOLE registered file (as skip_dropped_rows); every pack epoch must be common
    keys = {side: {e["key"] for e in v} for side, v in eps.items()}
    common = keys["rebuilt"] & keys["registered"]
    pack = {e["key"] for e in p["epochs"]}
    if not pack <= common:
        raise ValueError(f"{p['rel']}: {len(pack - common)} registered-pack epochs are not in both staged files")
    rb = rebuild_skip([e for e in eps["rebuilt"] if e["key"] in common], [e for e in eps["registered"] if e["key"] in common], fixed_ids)  # the one implementation
    reg_full = sum(s_ for e in eps["registered"] for s_, _ in skip_decisions(e["ids"], e["calls"], fixed_ids).values())
    dec = {side: {k: [[u[0], u[1], int(s_), int(r_)] for u, (s_, r_) in d.items()] for k, d in dd.items()} for side, dd in rb["decisions"].items()}
    return {"folder": ent["folder"], "rel": p["rel"], "fixed_tokens": fixed, "fixed_ids": sorted(fixed_ids), "decisions": dec, "orphans_unscored": p["orphans"],
            "pack_epochs": sorted(pack), "registered_only_epochs": sorted(keys["registered"] - common), "cutfull_only_epochs": sorted(keys["rebuilt"] - common),
            "registered_full_file_dropped": reg_full, "registered_pack_skip_dropped_rows": (c["skip_dropped_rows"] or {}).get(files["registered"][1]),
            "staged": {k: {"volume": v[0], "path": v[1], "sha256": v[2]} for k, v in files.items()}, "cut_pack": {k: c[k] for k in ("dir", "manifest_sha256")}}


def as_dec(rows: list) -> dict:
    """[[irid, call_index, dropped, readout], ...] -> {(irid, call_index): (dropped, readout)} (JSON types kept)."""
    return {(r[0], r[1]): (bool(r[2]), r[3]) for r in rows}


@app.function(image=cpu_image, cpu=(2, 2), memory=(8192, 8192), timeout=900, retries=0)
def skip_gate(builds: list) -> dict:
    """09's gate (RULE @ 36b9b47df §2), in its own call: probe_shift.cutfull_skip.gate_report per folder (0 mismatches both directions, the
    registered dropped count equal to the registered pack's skip_dropped_rows, a negative test that must fail, counts past 32,768)."""
    from probe_shift.cutfull_skip import gate_report

    out = {b["folder"]: gate_report({k: as_dec(v) for k, v in b["decisions"]["registered"].items()}, {k: as_dec(v) for k, v in b["decisions"]["rebuilt"].items()})
           for b in builds}  # over the epochs common to both staged files
    for b in builds:  # the count check: the registered rule over the WHOLE registered staged file == the registered pack's skip_dropped_rows (required)
        o = out[b["folder"]]
        o |= {"registered_full_file_dropped": b["registered_full_file_dropped"], "registered_pack_skip_dropped_rows": b["registered_pack_skip_dropped_rows"],
              "registered_full_count_matches_pack": b["registered_full_file_dropped"] == b["registered_pack_skip_dropped_rows"],
              "registered_only_epochs": b["registered_only_epochs"], "cutfull_only_epochs": b["cutfull_only_epochs"], "pack_epochs": len(b["pack_epochs"])}
        o["passes"] = o["passes"] and b["registered_pack_skip_dropped_rows"] is not None and o["registered_full_count_matches_pack"]
    return {"by_folder": out, "passes": len(out) == len(builds) and all(v["passes"] for v in out.values())}


# ---- phase score ------------------------------------------------------------------------------------------------------------------------------

def weight_gate(root: Path, tag: str, files_pin: dict, models_pin: dict, choice: list = REG_CHOICE, rc: str = "a") -> dict:
    """Before scoring: the tight files used, by sha256, and (a)'s five fold models by run_scaleup_score's fit_sha256 method (W then b as float64 at the
    fold's chosen layers in sorted order, then the command L63)."""
    import numpy as np

    used = sorted({f"{tag}/outer_tight_L{L}.npz" for c in choice for L in c} | {f"{tag}_cmd/outer_tight_L63.npz"})
    files = {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in used}
    models = {}
    for f in range(5):
        h = hashlib.sha256()
        for tg, L in [(tag, L) for L in sorted(choice[f])] + [(f"{tag}_cmd", 63)]:
            Z = np.load(root / tg / f"outer_tight_L{L}.npz")
            h.update(np.asarray(Z[f"{rc}__f{f}__W"], np.float64).tobytes())
            h.update(np.asarray(Z[f"{rc}__f{f}__b"], np.float64).tobytes())
        models[str(f)] = h.hexdigest()
    return {"tag": tag, "files": files, "fold_models": models, "passes": files == files_pin and models == models_pin}


def rows_r(p: dict):
    """score_r_cf's unit selection, verbatim: per epoch (staged order) the output_end calls with row_mean, in meta order."""
    dump = p["man"]["dump"]
    kept, empty = [], []
    for e in p["epochs"]:
        m = json.loads(vpath(dump["volume"], f"{dump['extract_run']}/meta/{dump['meta_arm']}/{h20(e['key'])}.json").read_text())
        if m.get("key") != e["key"] or m.get("arm") != dump["meta_arm"]:
            raise ValueError("meta key or arm mismatch")
        if not (m.get("cut") == "full" and m.get("truncated") is False and m.get("n_tokens") == m.get("tokens_total")):
            raise ValueError(f"{e['key']}: not a full-consumption meta")
        cs = [c for c in m["calls"] if c["kind"] == "output_end" and c.get("row_mean") is not None]
        if cs:
            kept.append((e, m, cs))
        else:
            empty.append(f"{e['world']}|{e['task']}")
    return kept, empty


def rows_c(p: dict, skip: dict):
    """score_all's cmd_A unit selection on the cutfull metas: call_end calls with a row, minus the rebuilt skip list; binding of the units below the
    32k boundary zone to the registered cut units (same selection on the cut metas with the registered skip list)."""
    dump, c = p["man"]["dump"], p["cut"]
    kept, empty, bad, zone, past, cut_kept = [], [], {}, 0, 0, []
    ident = lambda c_: (c_["irid"], str(c_["src_irid"]), int(c_["call_index"]))  # noqa: E731  score_r_cf's identity (irid's JSON type kept)
    for e in p["epochs"]:
        h = h20(e["key"])
        m = json.loads(vpath(dump["volume"], f"{dump['extract_run']}/meta/{dump['meta_arm']}/{h}.json").read_text())
        cm = json.loads(vpath(c["volume"], f"{c['extract_run']}/meta/{c['meta_arm']}/{h}.json").read_text())
        if m.get("key") != e["key"] or cm.get("key") != e["key"]:
            raise ValueError("meta key mismatch")
        sk_n = {tuple(x) for x in skip["rebuilt"][e["key"]]}  # the registered skip format, JSON types kept (score_all's comparison)
        sk_c = {tuple(x) for x in skip["registered"][e["key"]]}
        cs = [x for x in m["calls"] if x["kind"] == "call_end" and x.get("row") is not None and (x["irid"], x["call_index"]) not in sk_n]
        cc = [x for x in cm["calls"] if x["kind"] == "call_end" and x.get("row") is not None and (x["irid"], x["call_index"]) not in sk_c]
        n = m["n_tokens"]
        L = min(n, CUT)
        lo = L - 3 if n > L else L
        uu, cu = {ident(x): x["readout"] for x in cs}, {ident(x): x["readout"] for x in cc}
        if len(uu) != len(cs) or len(cu) != len(cc):
            bad[e["key"]] = "duplicate unit"
        elif {u: r for u, r in uu.items() if r < lo} != {u: r for u, r in cu.items() if r < lo} or any(r >= L for r in cu.values()):
            bad[e["key"]] = "binding to the registered cut units"
        zone += sum(lo <= r < L for r in uu.values())
        past += sum(r >= CUT for r in uu.values())
        if cs:
            kept.append((e, m, cs))
        else:
            empty.append(f"{e['world']}|{e['task']}")
        if cc:
            cut_kept.append((e, cm, cc))
    return kept, empty, bad, zone, past, cut_kept


def score_units(kept: list, field: str, tag: str, layers_of_fold: list, all_models: bool, dump: dict, rc: str = "a") -> dict:
    """score_r_cf's / score_all's arithmetic: per layer, rows as float32 in epoch order; per fold model ((X - mean) / std) @ W + b; a row's designated
    score is its fold model's mean over the fold's layers (avg_top2) or L63 (cmd_A); fold -1 rows NaN with the five models' scores in score_by_model
    (every row for a sim-honest folder)."""
    import numpy as np

    units = [(f"{e['world']}|{e['task']}", e["key"], c, m["n_tokens"], e["fold"]) for e, m, cs in kept for c in cs]
    logit = {}
    for L in sorted({L for ls in layers_of_fold for L in ls}) if units else []:
        Z = np.load(Path("/sol") / tag / f"outer_tight_L{L}.npz")
        X = np.concatenate([np.asarray(np.load(vpath(dump["volume"], f"{dump['extract_run']}/by_layer/L{L}/{m['chunk']}.npy"), mmap_mode="r")[
            [m["row_start"] + c[field] for c in cs]]).astype(np.float32) for e, m, cs in kept])
        if not np.isfinite(X).all():
            raise ValueError(f"L{L}: non-finite rows")
        for f in range(5):
            W, b, mu, sd = (Z[f"{rc}__f{f}__{p}"] for p in ("W", "b", "mean", "std"))
            logit[(f, L)] = ((X - mu) / sd) @ W + float(b.reshape(-1)[0])
    fr = [u[4] for u in units]
    nof = np.array([f is None for f in fr], dtype=bool)
    des = np.full(len(units), np.nan)
    for f in range(5):
        mk = np.array([x == f for x in fr], dtype=bool)
        if mk.any():
            des[mk] = np.mean([logit[(f, L)][mk] for L in layers_of_fold[f]], axis=0)
    if not np.isfinite(des[~nof]).all():
        raise ValueError("non-finite designated scores")
    sbm = None
    if all_models and not units:
        sbm = np.zeros((0, 5), np.float64)
    if units and (all_models or nof.any()):
        sbm = np.full((len(units), 5), np.nan)
        sel = slice(None) if all_models else nof
        for f in range(5):
            sbm[sel, f] = np.mean([logit[(f, L)][sel] for L in layers_of_fold[f]], axis=0)
        if not np.isfinite(sbm if all_models else sbm[nof]).all():
            raise ValueError("non-finite by-model scores")
        fi = np.array([-1 if x is None else x for x in fr], np.int64)
        if all_models and des[fi >= 0].tobytes() != sbm[fi >= 0, fi[fi >= 0]].tobytes():
            raise ValueError("a designated score differs from its fold model's score_by_model column (raw bytes)")
    keys = np.array([u[0] for u in units], dtype=str)
    order, seen = [], {}
    for u in units:
        order.append(seen.get(u[0], 0))
        seen[u[0]] = order[-1] + 1
    runs, by_model = {}, {}
    for k in dict.fromkeys(keys.tolist()):
        mk = keys == k
        runs[k] = float(des[mk].mean()) if not nof[mk][0] else float("nan")
        if nof[mk][0]:
            by_model[k] = [float(x) for x in sbm[mk].mean(axis=0)]
    rows = {"key": keys, "score": des, "unit": np.array(order, np.int64), "call_index": np.array([int(u[2]["call_index"]) for u in units], np.int64),
            "fold": np.array([-1 if f is None else f for f in fr], np.int64), "run_key": np.array([u[1] for u in units], dtype=str),
            "token": np.array([u[2]["readout"] for u in units], np.int64), "n_tokens": np.array([u[3] for u in units], np.int64),
            "irid": np.array([str(u[2]["irid"]) for u in units], dtype=str), "src_irid": np.array([str(u[2]["src_irid"]) for u in units], dtype=str)}
    if field == "row_mean":
        rows |= {"span_start": np.array([u[2]["span"][0] for u in units], np.int64), "span_end": np.array([u[2]["span"][1] for u in units], np.int64)}
    if sbm is not None:
        rows["score_by_model"] = sbm
    return {"rows": rows, "runs": runs, "by_model": by_model}


def npz_bytes(arrs: dict) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    np.savez(buf, **arrs)
    return buf.getvalue()


def rows_equal(a: dict, b, fields) -> dict:
    """Per field: arrays equal (shape, values); score / score_by_model: presence, shape and NaN pattern equal and finite values within POS_TOL."""
    import numpy as np

    out = {f: bool(f in b and np.array_equal(a[f], b[f])) for f in fields}
    for f in ("score", "score_by_model"):
        if (f in a) != (f in b):
            out[f] = {"present_in_both": False, "passes": False}
            continue
        if f not in a:
            continue
        x, y = np.asarray(a[f]), np.asarray(b[f])
        same = x.shape == y.shape and bool(np.array_equal(np.isnan(x), np.isnan(y)))
        d = float(np.max(np.abs(x[~np.isnan(x)] - y[~np.isnan(y)]), initial=0.0)) if same else float("inf")
        out[f] = {"shape_and_nan_pattern_equal": same, "max_abs_diff": d, "bytes_identical": same and x.tobytes() == y.tobytes(), "passes": same and d <= POS_TOL}
    out["passes"] = all(v if isinstance(v, bool) else v["passes"] for k, v in out.items() if k != "passes")
    return out


def publish(rel: str, blobs: dict, mfest: dict, name: str) -> dict:
    """Create-only folder, files created exclusively, the manifest last; returns shas and sizes."""
    mb = json.dumps(mfest, indent=1).encode()
    od = Path("/work") / rel
    od.mkdir(parents=True, exist_ok=False)
    for fn, b in list(blobs.items()) + [(name, mb)]:
        with (od / fn).open("xb") as fh:
            fh.write(b)
    return {"path": f"{OUTVOL}:/{rel}/", "files": {fn: hashlib.sha256(b).hexdigest() for fn, b in blobs.items()}, "file_bytes": {fn: len(b) for fn, b in blobs.items()},
            "manifest": {"name": name, "sha256": hashlib.sha256(mb).hexdigest(), "bytes": len(mb)}}


def run_blobs(prefix: str, v: str, o: dict) -> dict:
    return {f"{prefix}__{v}.npz": npz_bytes(o["rows"]),
            f"{prefix}__{v}.json": json.dumps({"run_scores": o["runs"], "runs_without_rows": sorted(o["no_rows"])} | ({"by_model": o["by_model"]} if o["by_model"] else {}),
                                              indent=1).encode()}


def coverage(out: dict, keys_all: set) -> dict:
    import numpy as np

    g = {}
    for v, o in out.items():
        got, none_ = set(o["runs"]), set(o["no_rows"])
        dmax = 0.0
        for k, r in o["runs"].items():  # the run score is the row mean (designated) or the by-model row mean (fold -1)
            mk = o["rows"]["key"] == k
            d_ = (abs(float(np.mean(o["rows"]["score"][mk])) - r) if not np.isnan(r)
                  else float(np.max(np.abs(o["rows"]["score_by_model"][mk].mean(axis=0) - np.array(o["by_model"][k])))))
            dmax = max(dmax, d_)
        g[v] = {"scored": len(got), "without_rows": sorted(none_), "of": len(keys_all), "row_mean_max_abs_diff": dmax,
                "passes": got | none_ == keys_all and not got & none_ and dmax <= 1e-9}
    return g | {"passes": all(x["passes"] for x in g.values())}


@app.function(image=cpu_image, cpu=(4, 4), memory=(32768, 32768), timeout=3600, retries=0, volumes=VOLS)
def score_folder(ent: dict, sola: dict, skip: dict | None, cites: dict, stamp: str, regc_on: bool = False) -> dict:
    """One folder: weight gates; the R positive control on the registered pack; sol_a_uncut at R (and C, with the C positive control on the registered
    CUT pack and the C binding); with C and --regc, also the REGISTERED a_tight and c_asis_tight command models on the rebuilt skip list (scorer sol_c). Each
    output folder is written create-only, only when every gate passed; shas, counts and gates only are returned (no score leaves the container)."""
    import numpy as np

    p = read_pack(ent)
    man, simh = p["man"], "simh" in p["man"]
    dump = man["dump"]
    gates = {"registered_weights_a": weight_gate(Path("/sol/xfer"), REG_TAG, REG_NPZ, REG_MODEL),
             "registered_weights_c_asis": weight_gate(Path("/sol/xfer"), REG_TAG, REG_NPZ, REG_MODEL_C, rc="c_asis"),
             "new_probe_weights": weight_gate(Path("/sol/cutfull/xfer"), sola["tag"], sola["files"], sola["fold_models"], sola["choice"], rc=sola["rc"]) | {"probe": sola["name"]}}
    if not all(g["passes"] for g in gates.values()):
        raise ValueError(f"{ent['folder']}: a weight gate failed: nothing scored")
    kept, empty = rows_r(p)
    reg = np.load(io.BytesIO(p["blobs"]["a_tight__avg_top2.npz"]))
    regj = json.loads(p["blobs"]["a_tight__avg_top2.json"])
    chunks = sorted({m["chunk"] for _, m, _ in kept})
    pos = score_units(kept, "row_mean", f"xfer/{REG_TAG}", REG_CHOICE, simh, dump)
    pc = rows_equal(pos["rows"], reg, UNIT_FIELDS)
    gates["r_positive_control_registered_pack"] = pc | {
        "runs_without_reply_rows_equal": sorted(empty) == sorted(regj["runs_without_reply_rows"]), "chunks_equal": chunks == man["chunks_scored"], "tolerance": POS_TOL,
        "passes": pc["passes"] and sorted(empty) == sorted(regj["runs_without_reply_rows"]) and chunks == man["chunks_scored"]}
    if not gates["r_positive_control_registered_pack"]["passes"]:
        return {"folder": ent["folder"], "published": [], "gates": gates}
    out = {"avg_top2": score_units(kept, "row_mean", f"cutfull/xfer/{sola['tag']}", sola["choice"], simh, dump, rc=sola["rc"]) | {"no_rows": empty}}
    regc, cnote = None, "not scored in this pass"
    if skip is not None:
        if set(skip["rebuilt"]) != {e["key"] for e in p["epochs"]} or skip["staged"]["rebuilt"]["sha256"] != p["staged_sha256"]:
            raise ValueError(f"{ent['folder']}: the committed skip list does not cover this folder's staged epochs / staged sha")
        kc, ec, badc, zone, past, cut_kept = rows_c(p, skip)
        gates["c_binding_to_registered_cut_units"] = {"epochs_failing": badc, "units_in_boundary_zone": zone, "units_past_32768": past, "passes": not badc}
        if badc:
            return {"folder": ent["folder"], "published": [], "gates": gates}
        cb = (Path("/work") / p["cut"]["dir"] / "a_tight__cmd_A.npz").read_bytes()  # C positive control on the CUT side
        if hashlib.sha256(cb).hexdigest() != p["cut"]["files"].get("a_tight__cmd_A.npz"):
            raise ValueError(f"{ent['folder']}: the registered cut pack's a_tight__cmd_A.npz differs from its (pinned) manifest")
        cpos = score_units(cut_kept, "row", f"xfer/{REG_TAG}_cmd", [[63]] * 5, simh, {"volume": p["cut"]["volume"], "extract_run": p["cut"]["extract_run"]})
        creg = np.load(io.BytesIO(cb))
        pk = sorted({f"{e['world']}|{e['task']}" for e in p["epochs"]})
        inpk = np.isin(creg["key"], pk)  # the registered 32k pack may hold epochs outside the cutfull pack (IL routed-not-admitted): compare the pack's rows
        creg_f = {f: creg[f][inpk] for f in creg.files}
        sbm_dropped = False
        if "score_by_model" in creg_f and not simh and bool((creg_f["fold"] != -1).all()) and bool(np.isnan(creg_f["score_by_model"]).all()):
            del creg_f["score_by_model"]  # the excluded rows held every fold -1 row: score_all writes score_by_model only with fold -1 rows (Codex r11 optional)
            sbm_dropped = True
        gates["c_positive_control_registered_cut_pack"] = rows_equal(cpos["rows"], creg_f, ("key", "unit", "call_index", "fold")) | {
            "registered_cut_file_sha256": hashlib.sha256(cb).hexdigest(), "tolerance": POS_TOL,
            "registered_cut_rows_outside_the_pack": {"rows": int((~inpk).sum()), "keys": sorted(set(creg["key"][~inpk].tolist()))},
            "score_by_model_all_nan_after_filter_dropped": sbm_dropped}
        if not gates["c_positive_control_registered_cut_pack"]["passes"]:
            return {"folder": ent["folder"], "published": [], "gates": gates}
        out["cmd_A"] = score_units(kc, "row", f"cutfull/xfer/{sola['tag']}_cmd", [[63]] * 5, simh, dump, rc=sola["rc"]) | {"no_rows": ec}
        if regc_on:  # once per folder (--regc): the registered command models on the rebuilt list, its own scorer folder sol_c
            regc = {rc_: score_units(kc, "row", f"xfer/{REG_TAG}_cmd", [[63]] * 5, simh, dump, rc=rc_) | {"no_rows": ec} for rc_ in ("a", "c_asis")}
        cnote = "scored: L63 call_end minus the rebuilt skip list (09 RULE @ 36b9b47df §2)"
    keys_all = {f"{e['world']}|{e['task']}" for e in p["epochs"]}
    gates["scored_coverage"] = coverage(out, keys_all)
    if regc is not None:
        gates["scored_coverage_registered_c"] = coverage({f"{k}|cmd_A": v for k, v in regc.items()}, keys_all)
    dests = [f"{p['tree']}/{p['batch']}/{sola['scorer']}/{ent['folder']}"] + ([f"{p['tree']}/{p['batch']}/{REGC_SCORER}/{ent['folder']}"] if regc is not None else [])
    gates["destinations_absent"] = {"dests": dests, "existing": [d_ for d_ in dests if (Path("/work") / d_).exists()]}
    gates["destinations_absent"]["passes"] = not gates["destinations_absent"]["existing"]  # both resolved before writing either (create-only)
    if not all(g["passes"] for g in gates.values()):
        return {"folder": ent["folder"], "published": [], "gates": gates}
    common = {"line": "cutfull", "label": LABEL, "folder": ent["folder"], "batch": p["batch"], "stamp": stamp, "cites": cites,
              "registered_pack": {"path": f"{OUTVOL}:/{p['rel']}/", "manifest_sha256": p["manifest_sha256"], "files_sha256": ent["files_sha256"], "record": ent.get("record")},
              "orphans_unscored": {"keys": p["orphans"], "rule": "staged epochs not in the registered pack (inventory keys only; 2e 10-16): not scored"},
             "registered_pack_blocks": {k: man[k] for k in ("sealed_epoch2", "epoch2", "sealed_sp493", "sp493", "sealed_t196fill", "simh", "il", "ha25f") if k in man},
             "role": ("thresholds only: epoch-2 honest runs (never training, never a cohort arm); rows identified by physical run key (npz run_key) and the "
                      ".eval sha in registered_pack_blocks.epoch2.keys") if (p["tree"] == "cutfull_ep2" or "epoch2" in man) else None,
             "registered_cut_pack": {k: v for k, v in p["cut"].items() if k != "files"}, "inventory": (man.get("cutfull") or {}).get("inventory"),
              "dump": {k: dump[k] for k in ("volume", "extract_run", "staged_run", "meta_arm", "sha256")}, "folds": man["folds"], "arm_to_world": man["arm_to_world"],
              "r2_suffix_world": man.get("r2_suffix_world"), "gates": gates,
              "formats": "<probe>__<readout>.npz per row: key (world|task), score (designated; NaN on fold -1), unit, call_index, fold (-1 = main absent from r36), "
                         "run_key, token (readout position), n_tokens, irid, src_irid[, span_start, span_end for avg_top2][, score_by_model rows x 5]; "
                         "<probe>__<readout>.json: run_scores (row means), runs_without_rows, by_model (fold -1 runs)",
              "not_done_here": "no z-scoring, thresholds, false-alarm rates or comparisons"}
    blobs = {k: v for ro, o in out.items() for k, v in run_blobs(sola["name"], ro, o).items()}
    pubs = [publish(dests[0], blobs, common | {
        "scorer": sola["scorer"], "probe": sola["name"], "variant": sola["variant"], "label": sola["label"], "layer_choice_a": sola["choice"],
        "units": {v: int(len(o["rows"]["key"])) for v, o in out.items()},
        "readouts": {"avg_top2": "reply (R): output_end span-mean rows; per fold model the mean of its two chosen layers (score_r_cf)", "cmd_A": cnote},
        "weights": {"volume": SOLVOL, "tag": sola["tag"], "files_sha256": sola["files"], "fold_models_sha256": sola["fold_models"]},
        "files": {fn: hashlib.sha256(b).hexdigest() for fn, b in blobs.items()}}, f"manifest_{sola['scorer']}.json")]
    if regc is not None:
        rb = {k: v for rc_, o in regc.items() for k, v in run_blobs(f"{rc_}_tight", "cmd_A", o).items()}
        pubs.append(publish(dests[1], rb, common | {
            "scorer": REGC_SCORER, "probes": {"a_tight": "Sol (a), registered weights", "c_asis_tight": "c_asis, registered weights"},
            "label": "registered probe scored on full transcripts (OOD past 32k); command readout (C) on the rebuilt skip list",
            "units": {rc_: int(len(o["rows"]["key"])) for rc_, o in regc.items()}, "readouts": {"cmd_A": cnote},
            "weights": {"volume": SOLVOL, "tag": f"{REG_TAG}_cmd", "files_sha256": REG_NPZ, "fold_models_sha256": {"a": REG_MODEL, "c_asis": REG_MODEL_C}},
            "files": {fn: hashlib.sha256(b).hexdigest() for fn, b in rb.items()}}, f"manifest_{REGC_SCORER}.json"))
    out_vol.commit()
    return {"folder": ent["folder"], "published": pubs, "units": {v: int(len(o["rows"]["key"])) for v, o in out.items()}, "gates": gates}


def committed(spec: str) -> tuple[str, bytes, str, str]:
    """<repo-relative path>@<commit>: the file's bytes AS COMMITTED (any branch of this repository); returns (path, bytes, sha256, full commit)."""
    rel, _, c = spec.rpartition("@")
    if not rel or not c:
        raise SystemExit(f"{spec}: expected <repo-relative path>@<commit>")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=REPO).stdout  # noqa: E731
    full = git("rev-parse", f"{c}^{{commit}}").decode().strip()
    b = git("show", f"{full}:{rel}")
    return rel, b, hashlib.sha256(b).hexdigest(), full


@app.local_entrypoint()
def main(phase: str, index: str, fit_record: str = "", skip_list: str = "", readouts: str = "R,C", only: str = "", regc: bool = False, records_ref: str = "HEAD",
         supersedes: str = ""):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    head = git("rev-parse", "HEAD")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    ip, ib, isha, icommit = committed(index)
    idx = json.loads(ib)
    ents = idx["folders"]
    if only:
        ents = [e for e in ents if e["folder"] in set(only.split(","))]
    names = [e["folder"] for e in ents]
    if len(set(names)) != len(names) or idx.get("n", idx.get("n_folders")) != len(idx["folders"]) or not ents:
        raise SystemExit("index: duplicate folders, n differs from its folder list, or nothing selected")
    for e in ents:  # every registered pack's landing record is committed with the index's sha256
        r_ = e["record"]
        if hashlib.sha256(subprocess.run(["git", "show", f"{r_['commit']}:{r_['path']}"], capture_output=True, check=True, cwd=REPO).stdout).hexdigest() != r_["sha256"]:
            raise SystemExit(f"{e['folder']}: its landing record is not committed with the index's sha256")
    rref = subprocess.run(["git", "rev-parse", f"{records_ref}^{{commit}}"], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    rdir = "experiments/2026-10-14_q35_sol_avg_POSTHOC/results"
    listing = subprocess.run(["git", "ls-tree", "--name-only", f"{rref}:{rdir}"], capture_output=True, text=True, check=True, cwd=REPO).stdout.split()
    for e in ents:  # the registered CUT pack of the original folder: the latest committed landing record with stage done at records_ref (cf_load's rule)
        regs = []
        for n_ in sorted(listing):
            if re.fullmatch(rf"scaleup507_{re.escape(e['original_folder'])}_\d{{4}}-\d\d-\d\d_\d{{6}}\.json", n_):
                rb_ = subprocess.run(["git", "show", f"{rref}:{rdir}/{n_}"], capture_output=True, check=True, cwd=REPO).stdout
                r_ = json.loads(rb_)
                if r_.get("stage") == "done":
                    regs.append((n_, r_, hashlib.sha256(rb_).hexdigest()))
        if not regs:
            raise SystemExit(f"{e['original_folder']}: no committed registered Sol landing record (stage done) at {records_ref}")
        rp_, rr_, rs_ = regs[-1]
        mreg = re.fullmatch(rf"{re.escape(OUTVOL)}:/(scaleup/{re.escape(e['original_folder'])}/(?:sol|sol_r2))/", str(rr_["outputs"]["path"]))
        if not mreg or not re.fullmatch(r"[0-9a-f]{64}", str(rr_["outputs"].get("manifest_sha256"))):
            raise SystemExit(f"{rp_}: no registered output path / manifest sha256")
        e["cut_pack"] = {"dir": mreg.group(1), "manifest_sha256": rr_["outputs"]["manifest_sha256"], "landing_record": {"path": f"{rdir}/{rp_}", "commit": rref, "sha256": rs_}}
    res_dir = HERE / "results" / "cutfull"
    res_dir.mkdir(parents=True, exist_ok=True)
    cites = {"git": head, "index": {"path": ip, "commit": icommit, "sha256": isha}, "registered_records_ref": {"ref": records_ref, "commit": rref}}
    if phase == "skip":
        builds = list(skip_build.map(ents))
        gate = skip_gate.remote(builds)
        lst = {"what": "Sol (a) C skip list rebuilt from the cutfull staged inputs (registered rule, unchanged; probe_shift.cutfull_skip); per folder and epoch "
                       "key, the DROPPED units [[irid, call_index], ...] (the registered skip format) for the rebuilt (cutfull) and the registered (32k) staged inputs",
               "rule": "09 RULE @ 36b9b47df, Full-length lead §2", "cites": cites, "fixed_tokens": builds[0]["fixed_tokens"] if builds else None,
               "supersedes": ({"path": f"experiments/2026-10-14_q35_sol_avg_POSTHOC/results/cutfull/{Path(supersedes).name}",
                               "sha256": hashlib.sha256((res_dir / Path(supersedes).name).read_bytes()).hexdigest(),
                               "note": "count denominator corrected; decisions unchanged (0 mismatches)"} if supersedes else None),
               "fixed_ids": builds[0]["fixed_ids"] if builds else None, "gate": gate,
               "folders": {b["folder"]: {"rel": b["rel"], "staged": b["staged"], "cut_pack": b["cut_pack"], "orphans_unscored": b["orphans_unscored"],
                                         "rebuilt": {k: [[r[0], r[1]] for r in v if r[2]] for k, v in b["decisions"]["rebuilt"].items() if k in set(b["pack_epochs"])},
                                         "registered": {k: [[r[0], r[1]] for r in v if r[2]] for k, v in b["decisions"]["registered"].items() if k in set(b["pack_epochs"])}}
                           for b in builds}}
        dest = res_dir / f"sola_uncut_skiplist_{stamp}{'' if gate['passes'] else '_GATE_FAILED'}.json"
        dest.open("x").write(json.dumps(lst))
        print(json.dumps({"gate_passes": gate["passes"],
                          "by_folder": {k: {x: v[x] for x in ("epochs", "pack_epochs", "units_compared", "mismatches", "registered_full_count_matches_pack", "rebuilt_units",
                                                               "rebuilt_dropped", "past_32768", "negative_test", "passes")}
                                        | {"registered_only_epochs": len(v["registered_only_epochs"]), "cutfull_only_epochs": len(v["cutfull_only_epochs"])}
                                        for k, v in gate["by_folder"].items()}}, indent=1), flush=True)
        print(dest.name, hashlib.sha256(dest.read_bytes()).hexdigest())
        if not gate["passes"]:
            raise SystemExit("skip-list gate failed (STOP): no C score")
        return
    if phase != "score":
        raise SystemExit("--phase skip | score")
    fp, fb, fsha, fcommit = committed(fit_record)
    fr = json.loads(fb)
    pr_ = PROBES.get(fr.get("probe"))
    if pr_ is None or not str(fr.get("status", "")).startswith("PASSED") or not fr["weights"].get("released") \
            or fr["fit_spec"]["cut"] != "none" or not fr["gates"]["refit_match"]["passes"] or fr["variant"]["name"] not in pr_["variants"] \
            or (fr.get("recipe") or {"name": "a"})["name"] != pr_["rc"]:
        raise SystemExit("the fit record is not a PASSED sol_a_uncut / c_asis_uncut fit (cut none, refit-match passed, weights released, its variants, its recipe)")
    ch_ = fr["variant"]["layer_choice_per_fold"]
    if len(ch_) != 5 or any(len(c) != 2 for c in ch_) or ch_ != fr["fit_spec"]["layer_choice_a"]:
        raise SystemExit("the fit record's layer choice is not five pairs, or differs between its variant block and fit spec")
    sola = {"tag": fr["tag"], "files": dict(fr["weights"]["tight_used_by_scorer"]), "fold_models": fr["weights"]["fold_models_sha256"], "choice": ch_,
            "variant": fr["variant"]["name"], "scorer": f"{pr_['prefix']}_{fr['variant']['name']}", "name": fr["probe"], "rc": pr_["rc"], "label": fr["label"]}
    cites["fit_record"] = {"path": fp, "commit": fcommit, "sha256": fsha}
    rd = set(readouts.split(","))
    if not rd <= {"R", "C"} or "R" not in rd:
        raise SystemExit("--readouts R or R,C")
    skip_by = {}
    if "C" in rd:
        sp, sb, ssha, scommit = committed(skip_list)
        sl = json.loads(sb)
        if not sl["gate"]["passes"] or sl["cites"]["index"]["sha256"] != isha:
            raise SystemExit("the skip list did not pass its gate, or was built on another index")
        miss = [n for n in names if n not in sl["folders"]]
        if miss:
            raise SystemExit(f"the skip list lacks folders: {miss[:5]}")
        skip_by = {n: {k: sl["folders"][n][k] for k in ("rebuilt", "registered", "staged")} for n in names}
        cites["skip_list"] = {"path": sp, "commit": scommit, "sha256": ssha, "rule": "09 RULE @ 36b9b47df, Full-length lead §2"}
    if regc and "C" not in rd:
        raise SystemExit("--regc needs --readouts R,C")
    outs = list(score_folder.starmap([(e, sola, skip_by.get(e["folder"]), cites, stamp, regc) for e in ents], return_exceptions=True))
    errs = {e["folder"]: f"{type(o).__name__}: {str(o)[:300]}" for e, o in zip(ents, outs) if isinstance(o, BaseException)}
    ok = [o for o in outs if not isinstance(o, BaseException)]
    rec = {"what": f"{sola['name']} scores on the cutfull sets: shas, counts and gates only", "label": sola["label"], "stamp": stamp, "cites": cites, "readouts": sorted(rd),
           "errors": errs, "folders": {o["folder"]: o for o in ok}}
    pointer = {"what": "pointer to the sol_a_uncut (and, with --regc, the registered sol_c) score files: Volume path, sha256, bytes", "volume": OUTVOL, "stamp": stamp,
               "cites": cites, "files": [{"path": f"{q['path'].split(':', 1)[1]}{fn}", "sha256": s_, "bytes": q["file_bytes"][fn]}
                                         for o in ok for q in o["published"] for fn, s_ in q["files"].items()]
               + [{"path": f"{q['path'].split(':', 1)[1]}{q['manifest']['name']}", "sha256": q["manifest"]["sha256"], "bytes": q["manifest"]["bytes"]}
                  for o in ok for q in o["published"]]}
    want_pubs = 2 if regc else 1
    bad = sorted(set(errs) | {o["folder"] for o in ok if len(o["published"]) != want_pubs})
    tagn = f"{sola['scorer']}_{stamp}_{uuid.uuid4().hex[:8]}"
    dest = res_dir / f"sola_uncut_score_{tagn}{'' if not bad else '_INCOMPLETE'}.json"
    dest.open("x").write(json.dumps(rec, indent=1))
    pdest = res_dir / f"sola_uncut_score_pointer_{tagn}.json"
    pdest.open("x").write(json.dumps(pointer, indent=1))
    print(json.dumps({"published_folders": sum(len(o["published"]) == want_pubs for o in ok), "of": len(ents), "not_published": bad, "errors": errs,
                      "gates_failed": {o["folder"]: [k for k, g in o["gates"].items() if not g["passes"]] for o in ok if not o["published"]},
                      "units": {o["folder"]: o["units"] for o in ok if o["published"]}}, indent=1), flush=True)
    print(dest.name, hashlib.sha256(dest.read_bytes()).hexdigest())
    print(pdest.name, hashlib.sha256(pdest.read_bytes()).hexdigest())
    if bad:
        raise SystemExit(f"{len(bad)} folders not published")
