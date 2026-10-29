"""4f's independent recheck of the scale-up X_err (RULE.md here). PRELIMINARY. Modal CPU only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with "numpy==2.3.3" \
      modal run --detach experiments/2026-10-14_xerr_scaleup_recheck/run_recheck_scaleup.py --mode selftest
  ... --mode parse
  ... --mode full --target experiments/2026-10-14_xerr_scaleup_09/results/<record>.json@<sha>

The local entrypoint embeds the git-pinned inputs (git show, sha256-checked), refuses to run from a dirty or unpushed
tree, launches one container, and writes results/<mode>_<stamp>.json. The container writes the same record to the
Volume first. The console shows only verdicts, counts and max |diff|; values live only in the JSON.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # inside the container the script sits at /root
APP = "probe-shift_aswin-surya_sub-p2probes_xrecheck_kill-2026-11-13"
SCRATCH_IN = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/"
                  "823cc0a0-23fc-47f8-b84f-0087b7bffcbd/scratchpad/p2/xr_in")

# name -> (commit, path, sha256): read with `git show`, checked, embedded
GIT_PINS = {
    "v6_record": ("0ed287c95", "experiments/2026-10-14_xerr_prelim_09/results/xerr_v6_2026-10-14_161122.json",
                  "e789144b62f71e0fb168dc3106656c2b5f29aadafe12df0f86dd1f74441d7645"),
    "v7_phase1": ("6514cf401", "experiments/2026-10-14_xerr_prelim_09/results/xerr_v7_phase1_2026-10-14_180820.json",
                  "afcf8fe38e77bbb408980b255ef7e553535f731c77deefb7793f07998bf6550e"),
    "v7_phase2": ("e109b2a95", "experiments/2026-10-14_xerr_prelim_09/results/xerr_v7_phase2_2026-10-14_181336.json",
                  "adc74fff9d6207646b77b678e8fd3dbb383f80eac31c4d9abcace868029c6115"),
    "fibs_record": ("d74ccaba0", "experiments/2026-10-14_p2_fibs_pair/results/fibs_scores_2026-10-14_124948.json",
                    "366d594cd719d52e9bb11d34bf4d3491b6cbd82d64f61efa73f476773cb49b1a"),
    "refusal_harm_record": ("a8d786425", "experiments/2026-10-14_p2_refusal_harm/results/scores_2026-10-14_145748.json",
                            "d0b7f028f8ac3795dd946a261f384f474748a296772fb1f2f2f22cecaac6c961"),
    "highstakes_record": ("3d04b7936", "experiments/2026-10-14_p2_highstakes/results/scores_2026-10-14_133519.json",
                          "2d251dee71a4a0366bd9549fcc3726511fe576aafce0bb6a913353d7fc06f38a"),
    "asis_record": ("74992a373", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/asis_scores_2026-10-14_142228.json",
                    "59c051a4ff9cc40811745a119251224f031cdaf479b8d3fb909c1059ab94da31"),
    "strategy_bits_t196": ("89fc226f1", "experiments/2026-10-14_approach_provenance_09/results/strategy_bits_t196.json",
                           "9a5a4e7120037f929e85d1efd9c7cf3ba35756349d8135f15f4cfe1e8775cfc0"),
    "t196": ("debef0af2", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json",
             "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb"),
    "plus311": ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json",
                "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b"),
    "nm_cells_v1": ("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json",
                    "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e"),
    "nm_built": ("6feeee4b7", "experiments/2026-10-14_q3_new_mains_launch/config/nm_built.json",
                 "93d9b538710b6bd930e7df01b0f5296d417550a0f0a35bcb6eb696166cf39a04"),
    "nofold_exclusion": ("548525c3c", "experiments/2026-10-14_p2_family_expansion/results/nofold_exclusion_2026-10-14_204632.json",
                         "239b92e1ad9ad7b9144291e35a53fa59e131094e16f5f2af8d2152f4b9ca5d50"),
    "il_keylists": ("2e75fa8c5", "experiments/2026-10-14_il_launch_09/results/il_keylists_2026-10-14.json",
                    "9e73c6977e89f77751e7dc86a4aeaaa6d435be4875b41a6e411ca217d61f5879"),
    "baseline_compare": ("7917a3ae5", "experiments/2026-10-14_xerr_scaleup_recheck/results/full_2026-10-15_010552.json",
                         "cf21c3cccca9c5b499aa7db2aca7f49c50204a75de6ea24a68ce7b90e5b71035"),
    "simh_runs": ("573054a4d", "experiments/2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717/runs.json",
                  "f420ed2401dcc776cbbcfd414bbe9d6d1132a3bbd012b529a33aba7fa5ecd068"),
}
# local, git-ignored copies pinned by sha256 (r36 lives in many scratchpads; the spec pins it by sha only)
LOCAL_PINS = {"runs36": (SCRATCH_IN / "runs36.json",
                         "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96")}
# Volume files: name -> (path in the container, sha256)
VOL_PINS = {
    "sol_cmd": ("/sol/xfer/asis_2026-10-14_135701/rows_2026-10-14_142228/a_tight__cmd_A.npz",
                "2e3b60e8e92dd744311773900a5b41b3dd2b48efadc29c9efcd665438a390cc3"),
    "open_manifest": ("/p2/famexp/2026-10-14_163027/pack_open_2026-10-14_180501/manifest.json",
                      "a0765577c34182c48f7827492f9b745c76201da57c03088686d7f2f434c01e9f"),
    "sealed_manifest": ("/p2/famexp/2026-10-14_163027/pack_sim_sealed_2026-10-14_181123/manifest.json",
                        "6795cba0598256e92cd93435216edf45c31b865919d553a8282e9bcbafe79ea6"),
}
# the spec this harness implements (RULE.md here); checked locally at launch, recorded in every output
SPEC_PINS = {
    "scaleup_rule": ("96b8abb77", "experiments/2026-10-14_xerr_scaleup_09/RULE.md",
                     "687679ea6b62550bdb4a677434e767687b9d908ff30b8f03401718997923f114"),
    "xerr_prelim_rule": ("6054f46b9", "experiments/2026-10-14_xerr_prelim_09/RULE.md",
                         "e2b8cea03c65492c2cf2eb11b04fb4138ecb001857a698b4813e00cb37103511"),
    "priority1_spec": ("ac36f6785", "experiments/2026-10-14_priority1_spec_09/RULE.md",
                       "0e6ad0d939ac44be208216890123005103476d9f752e42bba7f8acb3a868d121"),
}
CODE = [REPO / "src/probe_shift/xerr_recheck.py", REPO / "src/probe_shift/xerr_recheck_modes.py",
        REPO / "src/probe_shift/xerr_recheck_full.py", REPO / "src/probe_shift/xerr_recheck_6s.py",
        Path(__file__).resolve(), HERE / "RULE.md"]
NEEDS = {"selftest": ["v6_record", "v7_phase1", "v7_phase2", "fibs_record", "refusal_harm_record", "highstakes_record",
                      "asis_record", "strategy_bits_t196", "t196"],
         "parse": ["plus311", "nm_cells_v1", "nm_built", "simh_runs", "runs36", "asis_record", "nofold_exclusion"]}
NEEDS["full"] = sorted(set(NEEDS["selftest"]) | set(NEEDS["parse"]) | {"il_keylists"})
NEEDS["synth"] = NEEDS["selftest"]
NEEDS["full"] = sorted(set(NEEDS["full"]) | {"baseline_compare"})  # per-key comparison with the first full comparison
NEEDS["diag"] = NEEDS["full"]
NEEDS["recheck6s"] = NEEDS["full"]  # Revision 6s new keys (xerr_recheck_6s.py): values-free record

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
         .add_local_python_source("probe_shift"))
work_vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2)
p2_vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2).read_only()
sol_vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2).read_only()
VOLS = {"/work": work_vol, "/p2": p2_vol, "/sol": sol_vol}


def _stamp() -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")


def _body(mode: str, payload: dict) -> dict:
    import os
    import traceback

    from probe_shift import xerr_recheck as core
    from probe_shift import xerr_recheck_modes as modes

    log = []
    import re as _re

    def note(s):
        if mode == "recheck6s":  # values-free mode: decimals in any message are masked (counts stay)
            s = _re.sub(r"[-+]?(\d+\.\d*|\.\d+)([eE][-+]?\d+)?|\d+[eE][-+]?\d+|nan|inf", "<num>", str(s))
        log.append(s)
        print(s, flush=True)

    inp = modes.Inputs(payload["embedded"], payload["vol_pins"])
    stamp = _stamp()
    env = {"label": "PRELIMINARY; 4f independent scale-up X_err recheck", "mode": mode, "stamp_pt": stamp,
           "harness_git": payload["git"], "code_sha256": payload["code_sha256"], "spec_sha256": payload["spec"],
           "target": {k: v for k, v in (payload.get("target") or {}).items() if k != "bytes"} or None}
    try:
        if mode == "selftest":
            res = modes.selftest(inp, note)
        elif mode == "parse":
            res = modes.parse(inp, "/work/scaleup", note)
        else:
            from probe_shift import xerr_recheck_full as fullmod

            if mode == "recheck6s":
                from probe_shift import xerr_recheck_6s as m6s

                res = m6s.recheck6s(inp, {k: v for k, v in payload["target"].items() if k != "bytes"}, note)
            elif mode == "synth":
                res = fullmod.synth(inp, note)
            elif mode == "diag":
                res = fullmod.diag(inp, {k: v for k, v in payload["target"].items() if k != "bytes"}, note,
                                   combos=payload.get("combos"))
            else:
                res = fullmod.full(inp, {k: v for k, v in payload["target"].items() if k != "bytes"}, note)
        env.update(result=res, fatal=None)
    except core.Stop as e:
        env.update(result=None, fatal="Stop", reason=str(e))
    except Exception as e:  # noqa: BLE001 -- reported loudly with the traceback; never a silent pass
        env.update(result=None, fatal=type(e).__name__, reason=str(e), traceback=traceback.format_exc()[-6000:])
    if mode in ("diag", "recheck6s") and env.get("fatal"):  # blinded modes: no raw message or traceback is kept
        env["reason"] = f"withheld in {mode} mode (messages may carry values); exception {env['fatal']}"
        env.pop("traceback", None)
    env["inputs_read"] = inp.ledger
    env["log"] = log

    def conv(x):
        import numpy as np

        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        raise TypeError(type(x).__name__)

    text = json.dumps(env, indent=1, default=conv, allow_nan=True)
    os.makedirs("/work/xrecheck_4f", exist_ok=True)
    out = f"/work/xrecheck_4f/{mode}_{stamp}.json"
    with open(out, "x") as fh:  # created exclusively; never overwrites
        fh.write(text)
    work_vol.commit()
    env = json.loads(text)
    env["volume_record"] = {"volume": "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14",
                            "path": out.replace("/work", "", 1), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    return env


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes=VOLS)
def run_selftest(payload: dict) -> dict:
    return _body("selftest", payload)


@app.function(image=image, cpu=2, memory=4096, timeout=1200, retries=0, volumes=VOLS)
def run_parse(payload: dict) -> dict:
    return _body("parse", payload)


@app.function(image=image, cpu=8, memory=32768, timeout=3600, retries=0, volumes=VOLS)
def run_full(payload: dict) -> dict:
    return _body("full", payload)


@app.function(image=image, cpu=8, memory=32768, timeout=3600, retries=0, volumes=VOLS)
def run_diag(payload: dict) -> dict:
    return _body("diag", payload)


@app.function(image=image, cpu=8, memory=32768, timeout=3600, retries=0, volumes=VOLS)
def run_recheck6s(payload: dict) -> dict:
    return _body("recheck6s", payload)


@app.function(image=image, cpu=8, memory=32768, timeout=1800, retries=0, volumes=VOLS)
def run_synth(payload: dict) -> dict:
    return _body("synth", payload)


def _git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True).stdout


def _on_origin(sha: str) -> bool:
    out = _git("branch", "-r", "--contains", sha).decode()
    return any(line.strip().startswith("origin/") for line in out.splitlines())


def _walk(x):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from _walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v)


def _check_target(target: str) -> dict:
    """Full mode refuses unless the target record exists at a commit reachable from origin and is a non-TEST final or
    interim scale-up record."""
    if not target or "@" not in target:
        raise SystemExit("full mode needs --target <path>@<sha>")
    path, sha = target.rsplit("@", 1)
    if not path.startswith("experiments/2026-10-14_xerr_scaleup_09/results/") or "test" in Path(path).name.lower():
        raise SystemExit(f"target path is not a non-TEST record of 09's scale-up job: {path}")
    _git("fetch", "origin")
    full = _git("rev-parse", "--verify", f"{sha}^{{commit}}").decode().strip()
    if not _on_origin(full):
        raise SystemExit(f"target commit {sha} is not reachable from any origin/ branch")
    raw = _git("show", f"{full}:{path}")
    rec = json.loads(raw)
    spec, res = rec.get("spec"), rec.get("result")
    if not isinstance(spec, dict) or not isinstance(res, dict) or not res:
        raise SystemExit("target lacks a spec or a non-empty result")
    mode = spec.get("mode")
    if mode not in ("final", "interim") or spec.get("parse_only") is True:
        raise SystemExit(f"target is not a final or interim estimate record (spec.mode {mode!r}, parse_only "
                         f"{spec.get('parse_only')!r})")
    labels = " ".join(str(x) for x in (rec.get("label"), res.get("label"), spec.get("note"), (res.get("record") or {}).get("mode")))
    if "TEST" in labels or (res.get("record") or {}).get("mode") not in (None, mode):
        raise SystemExit("target carries a TEST marker or an inconsistent record mode")
    for d in _walk(rec):
        if d.get("test_standin") is True or d.get("example") is True or d.get("test") is True or any(
                "/scaleup_TEST" in str(d.get(f, "")) for f in ("path", "local_path", "folder", "root")):
            raise SystemExit("target names a TEST stand-in, example, test marker or /scaleup_TEST input")
    g = res.get("gates") or {}
    if (g.get("V") or {}).get("pass") is not True or (g.get("V_xerr") or {}).get("pass") is not True:
        raise SystemExit("target result lacks passed gates V and V_xerr (not a scale-up estimate record)")
    if not isinstance(spec.get("arms"), dict) or not isinstance(spec.get("counted"), dict):
        raise SystemExit("target spec lacks arms / counted (not a scale-up estimate record)")
    return {"path": path, "commit": full, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": raw, "spec": spec}


def _d9_census(spec: dict) -> dict:
    """Independent input census: d9's committed label, run-table, counted-manifest, approach and registry files for the
    +311 pool on origin/aswin/q3-overnight-chain, against the paths the target's spec names."""
    import re

    ref = "origin/aswin/q3-overnight-chain"
    head = _git("rev-parse", ref).decode().strip()
    names = _git("ls-tree", "-r", "--name-only", ref, "--", "experiments/2026-10-14_q3_overnight_chain/").decode().split()
    pat = re.compile(r"(labels_su311|attempt_labels_311|run_table_su311|run_table_311|run_table_snap_su311|"
                     r"manifest_su311_q\d+_(main|run2|sim)|approach_provenance_su311|registry_S_new)")
    found = sorted(n for n in names if pat.search(n) and "TEST" not in n and "E2E" not in n)
    named = {d.get("path") for d in _walk(spec) if isinstance(d.get("path"), str)}
    return {"branch": ref, "head": head, "d9_files_matching": len(found),
            "on_branch_not_in_target_spec": [n for n in found if n not in named],
            "in_target_spec_not_on_branch_head": sorted(n for n in named if n.startswith("experiments/2026-10-14_q3_overnight_chain/")
                                                        and n not in set(names))}


def _embed_target_inputs(spec: dict) -> dict:
    """Every {commit, path, sha256} entry in the target's spec is embedded from git; {local_path, sha256} from disk."""
    out = {}
    for d in _walk(spec):
        if d.get("commit") and d.get("path") and d.get("sha256"):
            b = _git("show", f"{d['commit']}:{d['path']}")
            if hashlib.sha256(b).hexdigest() != d["sha256"]:
                raise SystemExit(f"target input {d['commit']}:{d['path']} differs from its sha256")
            out[f"git:{d['commit']}:{d['path']}"] = {"commit": d["commit"], "path": d["path"], "sha256": d["sha256"], "bytes": b}
        elif d.get("local_path") and d.get("sha256") and Path(d["local_path"]).exists():
            b = Path(d["local_path"]).read_bytes()
            if hashlib.sha256(b).hexdigest() != d["sha256"]:
                raise SystemExit(f"target input {d['local_path']} differs from its sha256")
            out[f"local:{d['local_path']}"] = {"commit": "local", "path": d["local_path"], "sha256": d["sha256"], "bytes": b}
    return out


@app.local_entrypoint()
def main(mode: str = "selftest", target: str = "", override: str = "", combos: str = ""):
    if mode not in ("selftest", "parse", "full", "synth", "diag", "recheck6s"):
        raise SystemExit(f"unknown mode {mode}")
    dirty = _git("status", "--porcelain", "--", *[str(p.relative_to(REPO)) for p in CODE]).decode().strip()
    if dirty:
        raise SystemExit(f"harness files are not committed:\n{dirty}")
    head = _git("rev-parse", "HEAD").decode().strip()
    if not _on_origin(head):
        raise SystemExit("HEAD is not on origin: push the harness commit before running")
    spec = {}
    for name, (commit, path, sha) in SPEC_PINS.items():
        got = hashlib.sha256(_git("show", f"{commit}:{path}")).hexdigest()
        if got != sha:
            raise SystemExit(f"spec pin {name} differs: {got}")
        spec[name] = {"commit": commit, "path": path, "sha256": sha}
    embedded = {}
    for name in NEEDS[mode]:
        if name in GIT_PINS:
            commit, path, sha = GIT_PINS[name]
            b = _git("show", f"{commit}:{path}")
        else:
            path, sha = LOCAL_PINS[name]
            commit, b = "local (git-ignored copy)", Path(path).read_bytes()
            path = str(path)
        if hashlib.sha256(b).hexdigest() != sha:
            raise SystemExit(f"local pin mismatch for {name}")
        embedded[name] = {"commit": commit, "path": path, "sha256": sha, "bytes": b}
    payload = {"embedded": embedded, "vol_pins": VOL_PINS, "git": head, "spec": spec,
               "code_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE}}
    if mode in ("full", "diag", "recheck6s"):
        t = _check_target(target)
        payload["target"] = {k: v for k, v in t.items() if k != "spec"}
        payload["target"]["d9_census"] = _d9_census(t["spec"])
        embedded.update(_embed_target_inputs(t["spec"]))
        embedded["__target__"] = {"commit": t["commit"], "path": t["path"], "sha256": t["sha256"], "bytes": t["bytes"]}
        ov = json.loads((REPO / override).read_text()) if override else {}
        embedded["__override__"] = {"commit": "local", "path": override or None, "sha256": None, "json": ov}
    elif target or override:
        raise SystemExit("--target and --override are only for --mode full / diag / recheck6s")
    if combos:
        if mode != "diag":
            raise SystemExit("--combos is only for --mode diag")
        payload["combos"] = json.loads(combos)
    fn = {"selftest": run_selftest, "parse": run_parse, "full": run_full, "synth": run_synth, "diag": run_diag,
          "recheck6s": run_recheck6s}[mode]
    rec = fn.remote(payload)
    out = HERE / "results" / f"{mode}_{rec['stamp_pt']}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "x") as fh:
        json.dump(rec, fh, indent=1)
    res = rec.get("result") or {}
    import re

    reason = re.sub(r"[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?|nan|inf", "<num>", str(rec.get("reason")))  # console: no numbers
    print(json.dumps({"mode": mode, "fatal": rec.get("fatal"), "reason": reason, "verdict": res.get("verdict"),
                      "volume_record": rec.get("volume_record"), "local_record": str(out.relative_to(REPO))}, indent=1))
    if mode == "selftest" and res:
        for k, v in res["summary"].items():
            if isinstance(v, dict) and "compared" in v:
                print(f"{k}: compared {v['compared']}, mismatches {v['n_mismatch']}, max |diff| {v['max_abs_diff']:.3g}")
        print("target leaves not produced here:", res["summary"]["target_leaves_not_produced_here"])
        print("comparator negative test:", res["summary"]["comparator_negative_test"]["pass"])
    if mode == "parse" and res:
        print(f"scale-up folders {res['n_scaleup_folders']}, verdict {res['verdict']}, findings {res['n_findings']}")
    if mode in ("synth", "full") and res:
        print(f"verdict {res.get('verdict')}")
    if mode == "recheck6s" and res:  # verdicts, counts and key paths only
        print(json.dumps({"totals": res.get("totals"), "priority_keys": res.get("priority_keys"),
                          "n_mismatched": len(res.get("mismatched_key_paths") or []),
                          "control_old_keys": {k: v for k, v in (res.get("control_old_keys") or {}).items()
                                               if k != "mismatched_key_paths"},
                          "fold_stratified_observed_sd_absent": {
                              k: v for k, v in (res.get("fold_stratified_observed_sd_absent") or {}).items()
                              if k != "mismatched_key_paths"},
                          "gate_old_keys": {k: (res.get("gate_old_keys_6r_map_on_6s") or {}).get(k)
                                            for k in ("verdict", "compared", "n_mismatch", "unmapped")},
                          "internal_checks": res.get("internal_checks")}, indent=1))
    if mode == "diag" and res:
        for row in res.get("combinations", []):
            print(json.dumps({k: row.get(k) for k in ("name", "of_baseline_mismatches_now_matching", "of_baseline_wording_now_matching",
                                                      "n_numeric_mismatched", "n_wording_mismatched", "matches_every_key", "stop")}))
    sys.stdout.flush()
    if rec.get("fatal") or (mode != "diag" and (res or {}).get("verdict") not in ("MATCH", "CLEAN", "PASS")):
        raise SystemExit(1)
