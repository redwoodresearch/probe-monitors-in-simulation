"""[^sim]'s attack-side share, given a committed source (2e's GO 10-20 ~02:2x; 09's ruling: the "about 5%" is UNSOURCED and NOT computed here).
POST HOC; not route C (attack runs are unchanged). NOT RUN until a short review PASS and refresh3's ARMED line.

The post prints "GPT-5.6-Sol wrote part of the output in about 42% of simulated attack runs". Writing's 10-17 08:33 relay gives 546 / 1,309,
from item (c) (run_detector_solfree_auroc.py @ d1db55766, blob bb7701bc6), whose values file (8282cb5e...) is not in git. This tool reruns
item (c)'s own reduce() on the attack detector only, imported from its blob, on the committed inputs item (c) pinned. Then it records:
- (a) reduce()'s counts per attack-detector cell (.detectors.attack.cells[<cell>].n_class1_sol_using / n_class1_all) and over the union of the
  cells' simulated runs (pooled). Expected per writing: 546 / 1,309 (the gate G_REPRODUCES_546_1309 is a recorded boolean, not a STOP);
- (b) the scope rule reduce() applied, cited by line in its blob;
- (c) the routed-runs variant from the usage record @ de0bf0301 (0d's 10-17 00:36 "routed runs only", 550 / 1,322). The usage record holds no
  definition of that query, so two filters on its own fields are stated here BEFORE the run, both are reported, and each is compared with
  550 / 1,322 as a boolean: R1 = family "sim attack" and routed true; R2 = R1 and arm "sim_attack@september" (the lead arm). The gap between
  (a) and each R is explained by key: simulated runs in only one of them, with their Sol verdicts.
The record bytes (counts and gates only; item (c)'s full attack record, with its AUROCs, goes to the Volume only: review O1) are built in the container,
written to the results Volume with exclusive create and committed, then returned. The laptop prints only gate results and booleans.
Usage (NOTE_attack_sol_share_2026-10-20.md): "$M" run experiments/2026-10-20_routec_simpool_numbers/run_attack_sol_share.py --out <results/...json>
"""

import hashlib
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-routecsim_atksol_kill-2026-11-19"
OUT_VOL = "probe-shift_aswin-surya_sub-routecsim_results_kill-2026-11-19"
OUT_NAME = re.compile(r"attack_sol_share_\d{4}-\d{2}-\d{2}_\d{6}\.json")
# name: (commit, path, sha256 of the git show bytes): item (c)'s script and the exact inputs its 10-17 08:2x run pinned
PINS = {
    "solfree_script": ("d1db55766", "experiments/2026-10-15_e2e_sim_pipeline/run_detector_solfree_auroc.py",
                       "a82879f4c4159352812ec69c23db40ee0d747e31bc7c1e4396c3813b5c76e36e"),
    "attack": ("80d75b27e", "experiments/2026-10-16_attack_refit3/results/attack_all_refit3_2026-10-16_141813.json",
               "df7171a98d82d28ae86653480be922d978fc36104c63f622b3a69fb43a074eac"),
    "scan": ("ff8ea35a7", "experiments/2026-10-16_attack_prompt_provenance/results/openai_usage_2026-10-16_185845_scan_rows.json",
             "0d0392fb773b1804a3ded3b4f3fd2f4dc8e7ae33b1ee660dd7200d0550410197"),
    "usage": ("de0bf0301", "experiments/2026-10-16_attack_prompt_provenance/results/openai_usage_reanalysis_2026-10-16_191817.json",
              "a0bcb2cef1ec21d598dda1795cc8b027eaf49235ba191f5d4c9643e108701102"),
    "header": ("4326d76e9", "experiments/2026-10-15_e2e_sim_pipeline/results/simh_cont_headers_2026-10-17_081840.json",
               "a7638c2d19f6a56f2953babca0ea98a7fd63854cbec21be6f1861ea452941e5d"),
}
EXPECTED = {"reduce_sol_n": (546, 1309), "routed_sol_n": (550, 1322)}  # writing 10-17 08:33 and 00:36 relays; compared as booleans only
LEAD_ARM = "sim_attack@september"
# reduce()'s scope rule, by line in item (c)'s blob bb7701bc6 (test_run_attack_sol_share.py checks each line's text at the blob)
SCOPE_LINES = {
    "cell sample: the simulated runs with a run score in the cell (problem c_simatk_vs_run1, prefix 'sim|')": "L142-L143 (rs = run_scores; keys = sorted(rs))",
    "pooled Sol-flagged count: the union over cells of simulated keys flagged SOL": "L180",
    "per run: Sol verdict by normalised eval path (record run_keys value, '#e<n>' and a leading 'runs/' dropped)": "L192",
    "n_class1_all and n_class1_sol_using": "L210-L211",
    "verdict sources: scan rows, the usage record and the header scan, every copy": "L119, L124, L128",
    "a path with no copy, or only incomplete copies, is UNKNOWN (not Sol-free, not Sol-using); routed or not plays no part": "L135",
}

image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.1.3")
app = modal.App(APP, image=image)
out_vol = modal.Volume.from_name(OUT_VOL, version=2, create_if_missing=True)
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


class Stop(SystemExit):
    """A gate failure: the message names the gate and keys, never a value."""


class _Lazy:
    def __getattr__(self, name):
        return _Lazy()

    def __call__(self, *a, **k):
        return _Lazy()


class _App:
    def __init__(self, *a, **k):
        pass

    def function(self, *a, **k):
        return lambda f: f

    def local_entrypoint(self, *a, **k):
        return lambda f: f


def load_pinned(name: str, src: bytes) -> types.ModuleType:
    """Import a pinned script from its bytes as a plain module, with a stand-in `modal` only while it imports (as run_routec_simpool_numbers)."""
    fake = types.ModuleType("modal")
    fake.App, fake.Image, fake.Volume = _App, _Lazy(), _Lazy()
    real = sys.modules.get("modal")
    sys.modules["modal"] = fake
    try:
        mod = types.ModuleType(f"pinned_{name}")
        mod.__file__ = f"/root/pinned_{name}.py"
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    finally:
        if real is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = real
    return mod


def check_pins(blobs: dict) -> None:
    for name, (_c, _p, want) in PINS.items():
        if sha(blobs[name]) != want:
            raise Stop(f"STOP: input {name} differs from its pin")


def build(blobs: dict, sf, prov: dict) -> tuple:
    """item (c)'s reduce() on the attack detector, its counts, the pooled union, and the routed variants (pure; tests call it). Returns the
    committed record (counts and gates only, review O1) and item (c)'s full attack record bytes (AUROCs etc.: Volume only, never committed)."""
    check_pins(blobs)
    rb = {"scan": blobs["scan"], "usage": blobs["usage"], "attack": blobs["attack"], "header": blobs["header"]}
    reply = sf.reduce(rb, ["attack"], {"label": sf.LABEL, "item_c_rerun_by": prov})  # item (c)'s gates 1-4 STOP inside, as on 10-17
    rc = json.loads(reply["record_bytes"])
    cells = rc["detectors"]["attack"]["cells"]
    per_cell = {c: {k: v[k] for k in ("n_class1_all", "n_class1_sol_using", "n_class1_sol_free", "n_class1_unknown")} for c, v in sorted(cells.items())}
    rec = json.loads(blobs["attack"])
    flags, _ = sf.sol_flags(json.loads(blobs["scan"]), json.loads(blobs["usage"]), json.loads(blobs["header"]))
    sim = sorted({k for c in rec["cells"].values() for k in c["problems"]["c_simatk_vs_run1"]["run_scores"] if k.split("|", 1)[0] == "sim"})
    path_of = {k: sf.norm(rec["run_keys"][k]) for k in sim}
    if len(set(path_of.values())) != len(sim):
        raise Stop("STOP: two simulated attack runs share an eval path")
    pooled = {"n_all": len(sim), "n_sol_using": sum(flags.get(p) is True for p in path_of.values()),
              "n_sol_free": sum(flags.get(p) is False for p in path_of.values()), "n_unknown": sum(flags.get(p) is None for p in path_of.values())}
    if pooled["n_sol_using"] != rc["n_sol_flagged_sim_runs"]["attack"]:
        raise Stop("STOP: the pooled Sol-using count differs from reduce()'s n_sol_flagged_sim_runs.attack")
    exp = EXPECTED["reduce_sol_n"]
    repro = {f"cell {c}": (v["n_class1_sol_using"], v["n_class1_all"]) == exp for c, v in per_cell.items()} | {
        "pooled": (pooled["n_sol_using"], pooled["n_all"]) == exp}
    # (c) routed variants, from the usage record's own fields; filters stated in the module docstring before the run
    usage = json.loads(blobs["usage"])
    det_paths = set(path_of.values())
    routed = {}
    for name, keep in (("R1 family 'sim attack' and routed", lambda r: r["family"] == "sim attack" and r["routed"] is True),
                       ("R2 = R1 and arm 'sim_attack@september'", lambda r: r["family"] == "sim attack" and r["routed"] is True and r["arm"] == LEAD_ARM)):
        rows = [r for r in usage["runs"] if keep(r)]
        if not rows:  # review O3: a schema or label mismatch must not read as a quiet False
            raise Stop(f"STOP: {name}: the filter keeps 0 rows of the usage record")
        paths = [sf.norm(r["path"]) for r in rows]
        if len(set(paths)) != len(paths):
            raise Stop(f"STOP: {name}: repeated eval paths in the usage record's routed rows")
        v = {sf.norm(r["path"]): sf.verdict(r.get("usage")) for r in rows}
        only_r, only_d = sorted(set(v) - det_paths), sorted(det_paths - set(v))
        routed[name] = {"kept_rows": len(rows), "n_runs": len(v), "n_sol_using": sum(x == "sol" for x in v.values()), "n_incomplete": sum(x == "incomplete" for x in v.values()),
                        "equals_relay_550_1322": (sum(x == "sol" for x in v.values()), len(v)) == EXPECTED["routed_sol_n"],
                        "gap_vs_reduce_pooled": {
                            "in_routed_only": {"n": len(only_r), "n_sol_by_usage_record": sum(v[p] == "sol" for p in only_r), "paths": only_r},
                            "in_reduce_only": {"n": len(only_d), "n_sol_by_item_c_flags": sum(flags.get(p) is True for p in only_d), "paths": only_d},
                            "in_both_with_different_sol_verdict": sorted(p for p in set(v) & det_paths if (v[p] == "sol") != (flags.get(p) is True))}}
    return {"what": __doc__.split("\nUsage")[0], "label": "POST HOC, descriptive; not route C; the attack-side Sol share's committed source",
            "gates": {"G_REPRODUCES_546_1309": ("PASS" if all(repro.values()) else
                                                f"FAIL in {sorted(k for k, ok in repro.items() if not ok)}") + f" ({sum(repro.values())} of {len(repro)})",
                      "G_ITEM_C_GATES": "PASS (reduce() returned: its self-test, positive control and model-name gates passed)",
                      "G_ROUTED_EQUALS_550_1322": {k: v["equals_relay_550_1322"] for k, v in routed.items()},
                      "G_ROUTED_ROWS": {k: f"kept {v['kept_rows']} rows" for k, v in routed.items()}},
            "a_reduce_counts": {"per_cell": per_cell, "pooled_union_of_cells": pooled, "reproduces_relay_546_1309": repro,
                                "post_wording_basis": "'about 42% of simulated attack runs' = n_class1_sol_using / n_class1_all (writing 10-17 08:33)"},
            "b_scope_rule": {"blob": "bb7701bc60276723e1a16c24e2c142634fff0626 (run_detector_solfree_auroc.py @ d1db55766)", "lines": SCOPE_LINES},
            "c_routed_variants": routed,
            "item_c_attack_record": {"sha256": sha(reply["record_bytes"]), "where": "Volume only (review O1): item (c) kept its AUROCs outside git"},
            "provenance": prov}, reply["record_bytes"]


def summary(rec: dict) -> dict:
    return {"gates": rec["gates"]}


@app.function(volumes={"/out": out_vol}, cpu=1, memory=8192, timeout=1200, retries=0)
def compute(blobs: dict, prov: dict) -> dict:
    if not OUT_NAME.fullmatch(prov["out_name"]):
        raise Stop("STOP: the record name is not attack_sol_share_<stamp>.json")
    check_pins(blobs)  # review O2: item (c)'s bytes are sha-checked here before they are executed
    sf = load_pinned("solfree_script", blobs["solfree_script"])
    rec, item_c = build(blobs, sf, prov)
    b = json.dumps(rec, indent=1, sort_keys=True).encode()
    with open(f"/out/{prov['out_name']}", "xb") as fh:  # durable before return
        fh.write(b)
    with open(f"/out/{prov['out_name'].removesuffix('.json')}_item_c_full.json", "xb") as fh:  # Volume only, never committed (review O1)
        fh.write(item_c)
    out_vol.commit()
    return {"record_bytes": b, "summary": summary(rec), "volume_path": f"{OUT_VOL}:/{prov['out_name']}"}


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, check=False, cwd=REPO)


@app.local_entrypoint()
def main(out: str):
    if git("status", "--porcelain", "--", str(Path(__file__).resolve())).stdout:
        raise Stop("STOP: the script is not committed")
    o = Path(out).resolve()
    if o.exists() or o.parent != HERE / "results" or not o.parent.is_dir() or not OUT_NAME.fullmatch(o.name):
        raise Stop("STOP: --out must be a new results/attack_sol_share_<stamp>.json in this folder")
    head = git("rev-parse", "HEAD").stdout.decode().strip()
    blobs = {}
    for name, (c, p, want) in PINS.items():
        r = git("show", f"{c}:{p}")
        if r.returncode or sha(r.stdout) != want:
            raise Stop(f"STOP: {name} @ {c} missing or differs from its pin")
        blobs[name] = r.stdout
    prov = {"git": head, "script_sha256": sha(Path(__file__).resolve().read_bytes()), "app": APP, "out_volume": OUT_VOL, "out_name": o.name,
            "inputs": {n: {"commit": c, "path": p, "sha256": s} for n, (c, p, s) in PINS.items()}}
    reply = compute.remote(blobs, prov)
    b = reply["record_bytes"]
    with open(o, "xb") as fh:
        fh.write(b)
    print(f"gates {json.dumps(reply['summary']['gates'], sort_keys=True)}")
    print(f"record {o.relative_to(REPO)} sha256 {sha(b)} (also on Volume {reply['volume_path']})")
