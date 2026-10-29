"""POST HOC convergence check of the halves selections (RULE.md in this folder). For each saved score array: the
validation-AUROC selection per label column as the analysis made it (unmasked argmax), whether the selected
(sub-configuration, column) entry passed amendment 3's gradient check (max(grad_full, grad_val) <= 1e-4·min(sqrt(C), 1)),
and the masked re-selection (unconverged entries at -inf). Score arrays only; each file's sha256 is checked against the
committed result that recorded it before it is read.

  uv run --no-project --python 3.12 --with numpy python \\
      experiments/2026-10-12_claimB_convergence_check/analyze_POSTHOC_convergence.py \\
      --halves <dir> --halves-e2 <dir> --q35 <dir> --q35-e2 <dir> --planted3 <dir> --planted35 <dir>
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
HALVES = "53da7f48:experiments/2026-10-11_claimB_halves/results/halves_2026-10-11_164208.json"
HALVES_E2 = "53da7f48:experiments/2026-10-11_claimB_halves/results/halves_e2_direct_POSTHOC_2026-10-11_172057.json"
Q35 = "770518ea:experiments/2026-10-11_claimB_halves_qwen35/results/halves_qwen35_2026-10-12_005827.json"
PLANTED = "8ebabc2e:experiments/2026-10-12_claimB_planted_oppoint/results/planted_oppoint_POSTHOC_2026-10-12_114022.json"
DIRS = ("dev_to_test", "test_to_dev")


def git_json(spec):
    return json.loads(subprocess.run(["git", "show", spec], capture_output=True, check=True, cwd=HERE).stdout)


def tol_of(C):
    return 1e-4 * min(C**0.5, 1.0)


def check(path, want):
    raw = Path(path).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        raise SystemExit(f"{path}: sha256 {got[:12]} differs from the recorded {want[:12]}")
    z = np.load(path, allow_pickle=False)
    subs = json.loads(str(z["subs"]))
    V, G = z["val_auroc"], z["grad"]
    if G.shape != (len(subs), 2, V.shape[1]) or not np.isfinite(G).all() or not np.isfinite(V).all():
        raise SystemExit(f"{path}: grad {G.shape} or non-finite values")
    tol = np.array([tol_of(C) for _, C in subs])
    gmax = np.maximum(G[:, 0, :], G[:, 1, :])
    ok = gmax <= tol[:, None]
    K = V.shape[1]
    best = V.argmax(0)
    keep = ok.any(0)
    bestm = np.where(ok, V, -np.inf).argmax(0)
    sel_ok = ok[best, np.arange(K)]
    b0 = int(best[0])
    return {"sha256": got, "columns": K, "candidates": len(subs),
            "col0": {"selected": subs[b0], "grad_full": float(G[b0, 0, 0]), "grad_val": float(G[b0, 1, 0]), "tol": float(tol[b0]),
                     "ratio_to_tol": float(gmax[b0, 0] / tol[b0]), "converged": bool(sel_ok[0]),
                     "masked_selected": subs[int(bestm[0])] if keep[0] else None, "masked_differs": bool(keep[0] and bestm[0] != best[0]),
                     "has_converged_entry": bool(keep[0]), "unconverged_of_18": int((~ok[:, 0]).sum())},
            "shuffles": {"selected_unconverged": int((~sel_ok[1:]).sum()), "of": K - 1,
                         "masked_selection_differs": int((keep[1:] & (bestm[1:] != best[1:])).sum()),
                         "no_converged_entry": int((~keep[1:]).sum()), "unconverged_entries": int((~ok[:, 1:]).sum()), "entries": int(ok[:, 1:].size)},
            "unconverged_by_candidate": {f"L{l}_C{C:g}": int((~ok[s]).sum()) for s, (l, C) in enumerate(subs)}}


def main():
    p = argparse.ArgumentParser()
    for a in ("halves", "halves_e2", "q35", "q35_e2", "planted3", "planted35"):
        p.add_argument("--" + a.replace("_", "-"), type=Path, required=True)
    a = p.parse_args()
    h, he, q, pl = git_json(HALVES)["inputs"], git_json(HALVES_E2)["inputs"], git_json(Q35)["inputs"], git_json(PLANTED)
    qk = "claimB/halves_qwen35_v1/a273e40df0db"
    out = {"label": "POST HOC convergence check (RULE.md): halves selections vs amendment 3's gradient mask", "results": {}}
    for d in DIRS:
        out["results"][f"halves_qwen3_32b/{d}"] = {
            "fixed_vs_e1": check(a.halves / d / "fixed_vs_e1.npz", h[f"{d}/fixed_vs_e1.npz"]),
            "e2_vs_e1": check(a.halves_e2 / d / "e2_vs_e1.npz", h[f"{d}/e2_vs_e1.npz"]),
            "plant2": check(a.halves / d / "plant2.npz", h[f"{d}/plant2.npz"]),
            "direct_e2_rerun": check(a.halves / "POSTHOC_e2" / d / "fixed_vs_e1.npz", he[f"POSTHOC_e2/{d}/fixed_vs_e1.npz"])}
        out["results"][f"halves_qwen35/{d}"] = {
            "fixed_vs_e1": check(a.q35 / d / "fixed_vs_e1.npz", q[f"{qk}/{d}/fixed_vs_e1.npz"]),
            "e2_vs_e1": check(a.q35_e2 / d / "e2_vs_e1.npz", q[f"{qk}/{d}/e2_vs_e1.npz"]),
            "plant2": check(a.q35 / d / "plant2.npz", q[f"{qk}/{d}/plant2.npz"])}
        out["results"][f"planted_check_qwen3_32b/{d}"] = {"fixed_vs_e1_refit": check(a.planted3 / d / "fixed_vs_e1.npz", pl["qwen3"]["inputs"][f"{d}/fixed_vs_e1.npz"])}
        out["results"][f"planted_check_qwen35/{d}"] = {"fixed_vs_e1_refit": check(a.planted35 / d / "fixed_vs_e1.npz", pl["qwen35"]["inputs"][f"{d}/fixed_vs_e1.npz"])}
    out["sources"] = {"halves": HALVES, "halves_e2": HALVES_E2, "q35": Q35, "planted": PLANTED}
    out["git"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"convergence_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    for key, comps in out["results"].items():
        for comp, r in comps.items():
            c0, sh = r["col0"], r["shuffles"]
            print(f"{key:34s} {comp:18s} col0 {c0['selected']} conv={c0['converged']} ratio={c0['ratio_to_tol']:.3g} "
                  f"masked_differs={c0['masked_differs']} | shuffles sel_unconv {sh['selected_unconverged']}/{sh['of']} "
                  f"masked_differs {sh['masked_selection_differs']} none_conv {sh['no_converged_entry']}")


if __name__ == "__main__":
    main()
