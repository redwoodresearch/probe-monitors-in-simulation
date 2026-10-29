"""RULE.md "own_a tight gap fill on nm360": the 7 nm360 folders absent from 2e's config/owna_arms.json @ 724e3dbf0, scored with
2e's OWN code (run_owna507_score.py @ 724e3dbf0, imported from the pinned bytes; its functions run in-process with .local()):
skips_b + gate_b (corrected B, bit for bit, and the scale-up path on honest B) once, then per arm skips_arm + score_arm, which
gates (staged keys = Sol's, skip rows = Sol's, no own_a training main, fold_r36 = Sol's fold, rows aligned with Sol's per-row
files, H200, finite) and writes convaudit_work:/scaleup/<arm>/own_a/ in 2e's layout (fold -1 on every row: score NaN,
score_by_model = the five 164934 fold models; fold_r36 informational), create-only. Before anything is written, every target of
every arm is checked absent. After each arm publishes, a create-only sidecar writer_famexp.json names the writer (this job,
commit, app id) and the gate results. Keys, shas, counts and gate results only: never a score.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run <this file> --r36 <runs36.json>
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-p2probes_owna-gap_kill-2026-11-13"
BUILD = Path("/tmp/famexp_owna_gap_build")
OA = ("724e3dbf0", "experiments/2026-10-15_fpr_sweep_507_prep/run_owna507_score.py", "7ebfc56f2321fab09a0fe1c8e526154fb8ffb09247faf3f3ecab1b55d42acbc2")
ARMS_CFG = HERE / "config" / "owna_gap_arms.json"
OUTPUTS = ["own_a_tight__avg_top2.npz", "own_a_tight__cmd_A.npz", "own_a_tight__avg_top2.json", "manifest_owna.json", "writer_famexp.json"] + [
    f"training_mains__own_a_tight__f{f}.json" for f in range(5)]
OVERLAP_NOTE = ("9d006b058", "experiments/2026-10-14_p2_family_expansion/results/s4r_overlap_and_unhold_note_2026-10-15.json")
PINS_EXTRA = {"nm014_ruling_09": {"commit": "51a612b29", "path": "experiments/2026-10-14_priority1_spec_09/RULE.md",
                                  "section": "S4 source map (4r part) registered, and honest_b|nm014, item 2"},
              "setup_failures_v4": {"commit": "396da80fd", "path": "experiments/2026-10-14_q3_overnight_chain/results/hygiene/setup_failures_nm360_v4_2026-10-15.json",
                                    "sha256": "5427019e4d60d63c1c1da0dc547256b0a9eafa8d6b2da5dddbeff24ee207230e"}}

if modal.is_local():
    BUILD.mkdir(exist_ok=True)
    raw = subprocess.run(["git", "-C", str(REPO), "show", f"{OA[0]}:{OA[1]}"], capture_output=True, check=True).stdout
    if hashlib.sha256(raw).hexdigest() != OA[2]:
        raise SystemExit("run_owna507_score.py: pinned bytes differ")
    (BUILD / "owna507.py").write_bytes(raw)

app = modal.App(APP)
WORK = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2)
VOLS = {"/sol": modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2).read_only(),
        "/b": modal.Volume.from_name("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", version=2).read_only(),
        "/d": modal.Volume.from_name("probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", version=2).read_only(),
        "/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only(), "/work": WORK}
img = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3", "transformers==5.19.0", "jinja2")
       .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift").add_local_dir(BUILD, "/root/x2e"))


def _plain(x):
    """JSON-plain copy of a return value (numpy scalars -> Python), so a client without numpy can read it."""
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


@app.function(image=img, cpu=(16, 16), memory=(65536, 65536), timeout=3600, retries=0, scaledown_window=2, volumes=VOLS)
def go(cfg: dict, stamp: str, git: str, app_id: str) -> dict:
    """2e's driver, in-process: skips, the corrected-B gate (nothing scored or written unless it passes), then the arms."""
    sys.path.insert(0, "/root/x2e")
    import owna507 as oa

    if hashlib.sha256(Path(oa.__file__).read_bytes()).hexdigest() != OA[2]:
        raise SystemExit("owna507: imported bytes differ from the pin")
    oa.out_vol = WORK  # score_arm's reload()/commit() act on this container's /work mount (the same volume name as 2e's)
    WORK.reload()
    present = {a["arm"]: [fn for fn in OUTPUTS if (Path("/work/scaleup") / a["arm"] / "own_a" / fn).exists()] for a in cfg["arms"]}
    if any(present.values()) or any((Path("/work/scaleup") / a["arm"] / "own_a").exists() for a in cfg["arms"]):
        raise SystemExit(f"create-only: an own_a target or folder already exists, nothing written: {json.dumps({k: v for k, v in present.items() if v})}")
    arms = []
    for a in cfg["arms"]:
        mb = (Path("/work") / "scaleup" / a["arm"] / a["sol_dir"] / "manifest_sol.json").read_bytes()
        if oa.sha_b(mb) != a["sol_manifest_sha256"]:
            raise SystemExit(f"{a['arm']}: Sol manifest sha256 differs from the pinned one")
        sm = json.loads(mb)
        arms.append({"arm": a["arm"], "staged_run": sm["dump"]["staged_run"], "meta_arm": sm["dump"]["meta_arm"], "staged_sha": sm["dump"]["sha256"]})
    skb = oa.skips_b.local(cfg["b_want"], cfg["b_shas"])
    gb = oa.gate_b.local(cfg["epochs_b"], skb["skip"], cfg["choice"], cfg, cfg["fold_t196"])
    rec = {"stage": "gate_b", "fixed_tokens": skb["fixed_tokens"], "gate_b": gb}
    if not gb["pass"]:
        raise SystemExit(f"corrected-B gate failed, nothing scored or written: {json.dumps({k: v for k, v in gb.items() if k != 'weights_sha256'}, default=str)}")
    cfg = cfg | {"gate_b_summary": {"exact": gb["exact"], "scale_up_path_honest_b": gb["scale_up_path_honest_b"]}}
    outs = []
    for a, a2 in zip(cfg["arms"], arms):
        s = oa.skips_arm.local(a2)
        r = oa.score_arm.local(a, s, cfg["choice"], cfg, cfg["main_of"], cfg["fold_r36_of_main"], cfg["train"], stamp, git)
        if r.get("published"):
            g = r["gates"]
            side = {"writer": {"job": "famexp own_a tight gap fill (run_owna_gap.py)", "commit": git, "app_id": app_id, "stamp": stamp,
                               "code": f"{OA[1]} @ {OA[0]} (sha256 {OA[2]}), score_arm and gate_b run in-process"},
                    "gates": {"corrected_b_rebuild_bit_for_bit": gb["exact"], "scale_up_path_honest_b": gb["scale_up_path_honest_b"],
                              "rows_aligned_with_sol_by_key_unit_call_index": g["rows_aligned_with_sol"], "finite": g["finite"], "all_arm_gates": g},
                    "manifest_owna_sha256": r["manifest_sha256"], "files": r["files"]}
            sb = json.dumps(side, indent=1, default=str).encode()
            with (Path("/work/scaleup") / a["arm"] / "own_a" / "writer_famexp.json").open("xb") as fh:
                fh.write(sb)
            WORK.commit()
            r = r | {"writer_sidecar_sha256": hashlib.sha256(sb).hexdigest()}
        outs.append(r)
    return _plain(rec | {"stage": "done", "arms": outs, "published": sum(o.get("published", False) for o in outs), "of": len(outs)})


@app.local_entrypoint()
def main(r36: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout
    if git("status", "--porcelain", "--untracked-files=no").strip() or git("rev-parse", "HEAD") != git("rev-parse", "origin/aswin/p2-family-expansion"):
        raise SystemExit("uncommitted tracked changes, or HEAD is not on origin")
    head = git("rev-parse", "HEAD").decode().strip()
    sys.path.insert(0, str(BUILD))
    import owna507 as oa

    raw = {}
    for k, (kind, src, s) in oa.PINS.items():  # 2e's own pins, read as 2e's main reads them (git or local path; sha256 checked)
        b = git("show", src.replace("HEAD:", f"{OA[0]}:")) if kind == "git" else Path(r36 if k == "r36" else src).read_bytes()
        if oa.sha_b(b) != s:
            raise SystemExit(f"{k}: sha256 mismatch")
        raw[k] = json.loads(b)
    rec_ = raw["cb_record"]
    choice = rec_["layer_choice_own_a"]
    if choice != [[49, 50], [49, 50], [50, 51], [49, 50], [49, 50]]:
        raise SystemExit("layer_choice_own_a differs from part (c)'s")
    tav, cmd_a = rec_["saved"]["tight"]["avg"], rec_["saved"]["cmd"]["cmdA"]
    need = [f"{L}|{f}" for f in range(5) for L in choice[f]]
    fold_of_main = {}
    for r in raw["r36"]:
        if fold_of_main.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("r36: a main in two folds")
    t196 = {c["key"]: c["main"] for c in raw["t196"]["cells"]}
    fold_t196 = {t: fold_of_main[m] for t, m in t196.items()}
    mf = raw["b_manifest"]  # corrected B + C epochs, exactly as 2e's main builds them
    epochs = [{k: e[k] for k in ("key", "world", "task", "arm", "extraction_run", "staged_inputs_run")} for e in mf["epochs"]]
    shas = {e["staged_inputs_file"]: e["staged_inputs_sha256"] for e in mf["epochs"]}
    up, st, ex = raw["c_upload"], raw["c_stage"], raw["c_extract"]
    excl = set(st["c"]["c_gate"]["excluded"])
    epochs += [{"key": f"{f_['path']}#e0", "world": "honest_c", "task": f_["task"], "arm": "honest_b", "extraction_run": ex["config"]["run"],
                "staged_inputs_run": ex["config"]["inputs"]} for f_ in up["files"] if f"{f_['path']}#e0" not in excl]
    shas |= {f"{ex['config']['inputs']}/inputs/{a}": s for a, s in ex["config"]["inputs_sha256"].items()}
    if len({e["key"] for e in epochs}) != len(epochs) or len({(e["world"], e["task"]) for e in epochs}) != len(epochs):
        raise SystemExit("duplicate epochs in corrected B + C")
    want = {}
    for e in epochs:
        want.setdefault(e["staged_inputs_run"], {}).setdefault(e["arm"], []).append(e["key"])
    tr_tasks = sorted({e["task"] for e in epochs if e["world"] in ("run1", "honest_a")})
    train_files = [json.dumps({"probe": "own_a_tight", "fold_model": f, "training_mains": sorted({t196[t] for t in tr_tasks if fold_t196[t] != f}),
                               "source": "corrected B (correctedb_manifest_v1) run1 and honest_a runs of T196 tasks with r36 fold != f; 164934 @ cc53afc3a"}, indent=1)
                   for f in range(5)]
    main_of = {c["key"]: c["main"] for c in raw["cells311"]["cells"]} | {c["key"]: c["main"] for c in raw["cellsnm"]["cells"]}
    arms_cfg = json.loads(ARMS_CFG.read_text())["arms"]
    for a in arms_cfg:  # the extraction record each Sol manifest names: sha256, run and attempt (as 2e's main)
        er = a["extract_record"]
        eb = Path(er["path"]).read_bytes()
        rj = json.loads(eb)
        if oa.sha_b(eb) != er["sha256"] or rj["config"]["run"] != er["run"] or rj["config"]["attempt"] != er["attempt"]:
            raise SystemExit(f"{a['arm']}: the extraction record differs from the pinned one")
    if len({a["arm"] for a in arms_cfg}) != len(arms_cfg) or len(arms_cfg) != 7:
        raise SystemExit("the gap config must list the 7 arms once each")
    rec_path = "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_scores_2026-10-14_164934.json"
    cfg = {"arms": arms_cfg, "choice": choice, "epochs_b": epochs, "b_want": want, "b_shas": shas, "fold_t196": fold_t196,
           "main_of": main_of, "fold_r36_of_main": fold_of_main, "train": {"files": train_files, "mains": [json.loads(x)["training_mains"] for x in train_files]},
           "weights_R": {k: {"path": tav[k]["weights"], "sha256": tav[k]["weights_sha256"]} for k in need},
           "ref_R": {k: {"path": tav[k]["rows"], "sha256": tav[k]["rows_sha256"]} for k in need},
           "weights_C": {"tight": cmd_a["own_a_tight_weights"], "outer": cmd_a["own_a_weights"]},
           "ref_C": {"path": cmd_a["rows|own_a_tight"]["path"], "sha256": cmd_a["rows|own_a_tight"]["sha256"]},
           "record": {"commit": "cc53afc3a", "path": rec_path, "sha256": oa.PINS["cb_record"][2]}}
    ov = json.loads(git("show", f"{OVERLAP_NOTE[0]}:{OVERLAP_NOTE[1]}"))
    known = sorted({r["key"] for r in ov["rows"] if r["s4r_status"] == "staged"})
    v4 = git("show", f"{PINS_EXTRA['setup_failures_v4']['commit']}:{PINS_EXTRA['setup_failures_v4']['path']}")
    if oa.sha_b(v4) != PINS_EXTRA["setup_failures_v4"]["sha256"]:
        raise SystemExit("setup failures v4: sha256 mismatch")
    r09 = git("show", f"{PINS_EXTRA['nm014_ruling_09']['commit']}:{PINS_EXTRA['nm014_ruling_09']['path']}")
    pins = PINS_EXTRA | {"nm014_ruling_09": PINS_EXTRA["nm014_ruling_09"] | {"sha256": oa.sha_b(r09)}, "owna507": {"commit": OA[0], "path": OA[1], "sha256": OA[2]},
                         "arms_config": {"path": str(ARMS_CFG.relative_to(REPO)), "sha256": oa.sha_b(ARMS_CFG.read_bytes())}, "two_e_pins": {k: v[2] for k, v in oa.PINS.items()}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = go.remote(cfg, stamp, head, app.app_id or "")
    rec = {"label": "own_a tight gap fill on nm360 (2e's code and layout): shas, counts and gate results only", "git": head, "stamp (PT)": stamp, "app": APP,
           "app_id": app.app_id, "pins": pins, "known_overlaps_s4r_vs_s4": {"source": {"commit": OVERLAP_NOTE[0], "path": OVERLAP_NOTE[1]}, "n": len(known), "keys": known,
           "status": "known overlaps by key (09's unhold relayed about 10:55 PT; analysis gated on the S4 source map 4r part, registered @ 51a612b29)"}} | out
    dest = HERE / "results" / f"owna_gap_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1, default=str))
    print(dest, json.dumps({"stage": out.get("stage"), "published": out.get("published"), "of": out.get("of"),
                            "arms": [{"arm": a["arm"], "published": a.get("published"), "rows": a.get("rows"), "runs": a.get("runs"),
                                      "manifest_sha256": a.get("manifest_sha256")} for a in out.get("arms", [])]}))
