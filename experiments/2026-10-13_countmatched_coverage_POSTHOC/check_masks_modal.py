"""Codex round 2 check (RULE.md Amendment 2): do the last-token a_m keep masks built on the main scores file (this job)
equal d9's, built in the fix-full file's context and sliced to its first 5,620 runs? Compares each draw's and cell's
effective comparison mask (cell kept & draw kept & second real run). Also checks the recipe P2 gate references are finite.

  CMCOV_IN=<dir with the main last-token scores> PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' \\
      --with numpy modal run --detach experiments/2026-10-13_countmatched_coverage_POSTHOC/check_masks_modal.py
"""
import json
import math

import modal
import numpy as np
from run_cmcov_modal import AM, DRAWS, E2, LT_LOCAL, NEG, PAIRS, R1MET, STORED, G, kept, vol, work_vol

app = modal.App("probe-shift_aswin-surya_sub-cmcov_maskcheck_kill-2026-11-13")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scipy==1.16.2").add_local_python_source("probe_shift", "run_cmcov_modal")
if LT_LOCAL.is_file():
    image = image.add_local_file(LT_LOCAL, STORED["lt"][0])
FIXFULL = ("recipe1/stored/sce_fixfull_scores_2026-10-13_152618.npz", "48fc5eb52546f8884bf1083a3d8530a6c80624b52c619678edf72c28443826be")


def ctx(Z):
    arm, cell = Z["runs__arm"].astype(str), Z["runs__cell"].astype(str)
    G.update(arm=arm, fold=Z["runs__fold"].astype(int), cell=cell, pairing=[a for a in np.unique(arm) if "attack" in a])
    return arm


@app.function(image=image, cpu=4, memory=16384, timeout=900, retries=0, volumes={"/work": work_vol.read_only()})
def check() -> dict:
    Zm, Zf = np.load(vol(*STORED["lt"])), np.load(vol(*FIXFULL))
    nm, nf = len(Zm["runs__arm"]), len(Zf["runs__arm"])
    same_order = all(np.array_equal(Zm[f"runs__{k}"].astype(str), Zf[f"runs__{k}"][:nm].astype(str)) for k in ("arm", "key", "cell", "fold"))
    out = {"n_main": nm, "n_fixfull": nf, "same_order_first_n_main": bool(same_order), "masks": {}}
    for L in (48, 63):
        Na = np.load(vol(NEG + AM["lt", L][0], AM["lt", L][1]))
        ctx(Zf)
        full = [kept(Na[f"a|1.0|{d}"][..., :nf])[:nm] for d in range(DRAWS)]  # d9: the fix-full context, then sliced
        arm = ctx(Zm)
        mine = [kept(Na[f"a|1.0|{d}"][..., :nm]) for d in range(DRAWS)]  # this job: the main file's context
        for c in ("b", "b_fixed"):
            kx = kept(Zm[f"L{L}__{c}__nested_real"])
            for d in range(DRAWS):
                a, b = kx & full[d] & (arm == E2), kx & mine[d] & (arm == E2)
                out["masks"][f"L{L}|{c}|draw{d}"] = {"n_d9": int(a.sum()), "n_mine": int(b.sum()), "n_differ": int((a != b).sum())}
    met = json.loads(vol(*R1MET).read_text())["parts"]["P2"]["D_vs_a_m"]
    vals = [x for r in ("fpr_0.01", "fpr_0.05") for k in ("D_per_draw", "lower_end_min_max", "upper_end_min_max") for x in met[r][k]]
    out["recipe_refs"] = {"n": len(vals), "all_finite": all(isinstance(x, float | int) and math.isfinite(x) for x in vals)}
    out["pairs_checked"] = len(PAIRS)
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(check.remote(), indent=1))
