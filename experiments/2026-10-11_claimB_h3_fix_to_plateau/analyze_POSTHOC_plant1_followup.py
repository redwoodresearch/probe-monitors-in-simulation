"""POST HOC follow-up to the H3 plant-1 diagnosis (0d's Codex review, items 2 and 3; $0, laptop, stored numbers only).

2. The frozen-score tolerance: per run, |CPU float64 refit score - H3's GPU refit score| on the 300 plant-1 pair runs
   (plant1_diag_POSTHOC_2026-10-11_163142.json): max, median, p95, counts above 1e-4 / 1e-3 / 1e-2; whether any pair's
   shift (copy - twin) changes sign; the rank agreement of the run scores and of the pair shifts (Spearman), and the
   largest rank move; H3's stored fit report beside the CPU fits.
3. media_processing: within-environment plant-1 AUROC before (H2 unevened, step A's pipeline restricted to L58/L63) vs
   after (H3 evened), from plant1_shifts_POSTHOC_2026-10-11_163427.json's per-pair scores; per pair: shift before/after,
   the copy's share of the environment's twins it outscores before/after (its AUROC contribution), and the diagnosis
   flags (re-made, anchor line moved, planted output changed by a fix); how concentrated the drop is; and the (layer, C)
   each refit selected for that fold, before and after (all five folds shown).

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scipy \\
      python experiments/2026-10-11_claimB_h3_fix_to_plateau/analyze_POSTHOC_plant1_followup.py \\
      results/plant1_diag_POSTHOC_2026-10-11_163142.json results/plant1_shifts_POSTHOC_2026-10-11_163427.json
"""

import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).parent
ENV = "media_processing"
H2_A0, H3_LIN = "claimB/level1/h2_search_v1/plant1_k1/A0.npz", "claimB/h3_small_v1/04c3953b139e/refit/plant1.npz"
FROZEN = "claimB/h3_small_v1/04c3953b139e/refit/frozen_run_scores.json"


def main():
    diag = json.loads((HERE / sys.argv[1]).read_text())
    sh = json.loads((HERE / sys.argv[2]).read_text())
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    read = lambda p: b"".join(vol.read_file(p))  # noqa: E731
    out = {"label": "POST HOC: plant-1 follow-up items 2 and 3 (numbers only)", "inputs": {}}
    # item 2
    P = diag["pairs"]
    cpu = np.array([r["frozen"][t]["evened"] for r in P for t in ("copy", "twin")])
    gpu = np.array([r["frozen"][t]["h3_frozen_evened"] for r in P for t in ("copy", "twin")])
    ad = np.abs(cpu - gpu)
    s_cpu = np.array([r["frozen"]["copy"]["evened"] - r["frozen"]["twin"]["evened"] for r in P])
    s_gpu = np.array([r["frozen"]["copy"]["h3_frozen_evened"] - r["frozen"]["twin"]["h3_frozen_evened"] for r in P])
    rk_c, rk_g = np.argsort(np.argsort(cpu)), np.argsort(np.argsort(gpu))
    fz = json.loads(read(FROZEN))
    out["inputs"][FROZEN] = hashlib.sha256(read(FROZEN)).hexdigest()
    out["item2_frozen_tolerance"] = {
        "runs": int(len(ad)), "abs_diff_max": float(ad.max()), "abs_diff_median": float(np.median(ad)), "abs_diff_p95": float(np.percentile(ad, 95)),
        "above_1e-4": int((ad > 1e-4).sum()), "above_1e-3": int((ad > 1e-3).sum()), "above_1e-2": int((ad > 1e-2).sum()),
        "score_scale_sd": float(gpu.std()), "pair_shift_sign_changes": int((np.sign(s_cpu) != np.sign(s_gpu)).sum()),
        "pair_shift_abs_diff_max": float(np.abs(s_cpu - s_gpu).max()),
        "spearman_run_scores": float(spearmanr(cpu, gpu)[0]), "spearman_pair_shifts": float(spearmanr(s_cpu, s_gpu)[0]),
        "largest_run_rank_move": int(np.abs(rk_c - rk_g).max()),
        "by_environment_max_abs_diff": {e: float(max(abs(r["frozen"][t]["evened"] - r["frozen"][t]["h3_frozen_evened"]) for r in P if r["environment"] == e for t in ("copy", "twin")))
                                        for e in sorted({r["environment"] for r in P})},
        "cpu_fits": diag["frozen_fit_evals"], "h3_gpu_fits": fz.get("fits")}
    # item 3
    flags = {r["cell"]: r for r in P}
    cells = sorted(c for c in flags if flags[c]["environment"] == ENV)
    pp = sh["per_pair"]
    rows = []
    for v in ("before_A0_L58_L63", "after_stepA"):
        cop = np.array([pp[c][v]["copy"] for c in cells])
        twi = np.array([pp[c][v]["twin"] for c in cells])
        share = (cop[:, None] > twi[None, :]).mean(1) + 0.5 * (cop[:, None] == twi[None, :]).mean(1)
        rows.append((v, cop, twi, share))
    auc = {v: float(share.mean()) for v, _, _, share in rows}
    contrib = rows[1][3] - rows[0][3]  # per copy: change in the share of twins it outscores (sums to the AUROC change x n)
    order = np.argsort(contrib)
    drop = auc["after_stepA"] - auc["before_A0_L58_L63"]
    cum = np.cumsum(contrib[order]) / len(cells)
    out["item3_media_processing"] = {
        "pairs": len(cells), "within_env_auroc": auc, "auroc_change": drop,
        "re_made": sum(flags[c]["copy_changed_in_plan"] for c in cells), "anchor_moved": sum(not flags[c]["anchor_line_same"] for c in cells),
        "planted_output_changed_by_fix": sum(flags[c]["site_message_changed_by_evening"] for c in cells),
        "remade_ne_direct": sum(not flags[c]["remade_equals_direct_evening"] for c in cells),
        "pairs_needed_for_half_the_drop": int(np.argmax(cum <= drop / 2) + 1) if drop < 0 and (cum <= drop / 2).any() else None,
        "pairs_with_lower_share_after": int((contrib < 0).sum()), "pairs_with_higher_share_after": int((contrib > 0).sum()),
        "mean_shift_before": float(np.mean([pp[c]["before_A0_L58_L63"]["shift"] for c in cells])),
        "mean_shift_after": float(np.mean([pp[c]["after_stepA"]["shift"] for c in cells])),
        "twin_score_sd_before": float(rows[0][2].std()), "twin_score_sd_after": float(rows[1][2].std()),
        "per_pair": [{"cell": cells[k], "shift_before": pp[cells[k]]["before_A0_L58_L63"]["shift"], "shift_after": pp[cells[k]]["after_stepA"]["shift"],
                      "share_before": float(rows[0][3][k]), "share_after": float(rows[1][3][k]), "re_made": flags[cells[k]]["copy_changed_in_plan"],
                      "anchor_moved": not flags[cells[k]]["anchor_line_same"], "planted_output_changed_by_fix": flags[cells[k]]["site_message_changed_by_evening"]}
                     for k in order]}
    # selected (layer, C) per fold, before and after
    e2f = json.loads((HERE.parents[1] / "splits/folds_v2.json").read_text())["environment_folds"]["environment_to_fold"]
    picks = {}
    for name, path in (("before_A0", H2_A0), ("after_stepA", H3_LIN)):
        raw = read(path)
        out["inputs"][path] = hashlib.sha256(raw).hexdigest()
        z = np.load(io.BytesIO(raw))
        subs, folds, V = json.loads(str(z["subs"])), list(z["folds"]), z["val_auroc"][:, :, 0]
        norm = [(s["layer"], s["C"]) if isinstance(s, dict) else (s[0], s[1]) for s in subs]
        keep = [k for k, s in enumerate(norm) if s[0] in (58, 63)]
        picks[name] = {e: {"all": list(norm[int(V[:, folds.index(e2f[e])].argmax())]),
                           "L58_L63": list(norm[keep[int(V[keep, folds.index(e2f[e])].argmax())]])} for e in sorted(e2f) if e2f[e] in folds}
    out["selected_layer_C"] = picks
    out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"plant1_followup_items23_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out["item2_frozen_tolerance"].items() if k not in ("cpu_fits", "h3_gpu_fits")}, indent=1))
    print(json.dumps({k: v for k, v in out["item3_media_processing"].items() if k != "per_pair"}, indent=1))
    print(json.dumps(picks, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
