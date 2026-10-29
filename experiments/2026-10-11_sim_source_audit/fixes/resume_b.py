"""Fix (b) resume driver (STATUS_b.md): spawn only the GPU jobs (layer x readout) that still lack a record, on the
redeployed fix (b) app. Each grid script skips its existing records, but every spawned job loads its data first, so
re-spawning all 6 jobs to finish 2 records starts 6 GPU containers instead of 2. The tag is recomputed exactly as the
grid script computes it and must equal the expected tag, or nothing is spawned.

  stripped: run_claim_c_noagent_hswap.py grid,        records /score/claimcna_hswap_c2bdaa3014a8/   (part 4a)
  k26:      run_k26_hswap.py refit_claimc,            records /score/claimck26_hswap_bdbdc087a1ca/  (part 2)
  k26na:    run_k26_hswap.py refit_claimc "claimcna", records /score/claimcnad3_hswap_bdbdc087a1ca/ (part 4b)

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' python \
      experiments/2026-10-11_sim_source_audit/fixes/resume_b.py <job> <labels> [--dry-run] [--max-jobs N]

--max-jobs spawns only the first N missing jobs (in LAYERS x KINDS order), so a batch's containers stay inside the
watcher's margin; run it again for the rest.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import modal

FIG = Path(__file__).resolve().parents[3] / "experiments/2026-10-08_q3_onpolicy_figures"
JOBS = {  # job: (script, function, expected tag, record folder prefix, C grid name, extra spawn args)
    "stripped": ("run_claim_c_noagent_hswap.py", "grid", "hswap_c2bdaa3014a8", "claimcna_", "C_GRID", ()),
    "k26": ("run_k26_hswap.py", "refit_claimc", "hswap_bdbdc087a1ca", "claimck26_", "C_QWEN", ()),
    "k26na": ("run_k26_hswap.py", "refit_claimc", "hswap_bdbdc087a1ca", "claimcnad3_", "C_QWEN", ("claimcna",)),
}


def main(job: str, labels: str, dry_run: bool, max_jobs: int):
    script, fn_name, want, prefix, c_name, extra = JOBS[job]
    spec = importlib.util.spec_from_file_location(script[:-3], FIG / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # the grid script's own constants and attempted_run1
    att = mod.attempted_run1(labels)
    folder = f"{prefix}{want}"
    try:
        have = {Path(e.path).name for e in modal.Volume.from_name(mod.SCORE_VOL, version=2).listdir(folder)}
    except modal.exception.NotFoundError:
        have = set()  # part 4b has no records yet
    missing = {f"L{l}_{k}": [C for C in getattr(mod, c_name) if f"L{l}_{k}_C{C:g}.npz" not in have]
               for l in mod.LAYERS for k in mod.KINDS}
    missing = dict(list({j: cs for j, cs in missing.items() if cs}.items())[:max_jobs])
    print(json.dumps({"job": job, "folder": folder, "records_present": sum(n.endswith(".npz") and ".outscores." not in n for n in have),
                      "jobs_to_spawn": {j: [f"{C:g}" for C in cs] for j, cs in missing.items()}}))
    if dry_run or not missing:
        return
    src = (FIG / script).read_bytes()
    if script == "run_k26_hswap.py":  # as run_k26_hswap.__main__: the inventory gate, then the tag
        inv = modal.Function.from_name(mod.APP, "inventory_v2").remote()
        if not inv.get("passed"):
            raise SystemExit(f"inventory gate failed: {inv}")
        tag = "hswap_" + hashlib.sha256(src + inv["sha256"].encode() + json.dumps(att).encode()).hexdigest()[:12]
    else:  # as run_claim_c_noagent_hswap.__main__
        tag = "hswap_" + hashlib.sha256(src + json.dumps(att).encode()).hexdigest()[:12]
    if tag != want:
        raise SystemExit(f"tag {tag} != expected {want}; nothing spawned")
    fn = modal.Function.from_name(mod.APP, fn_name)
    calls = {}
    for j in missing:
        l, k = j[1:].split("_", 1)
        calls[j] = fn.spawn(tag, int(l), k, att, *extra).object_id
    print(json.dumps({"tag": tag, "attempted_run1": len(att), "calls": calls}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=sorted(JOBS))
    ap.add_argument("labels")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-jobs", type=int, default=6)
    main(**vars(ap.parse_args()))
