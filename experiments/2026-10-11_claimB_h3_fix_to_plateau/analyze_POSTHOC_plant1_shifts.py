"""POST HOC (H3 plant-1 diagnosis, step 2; $0 on the laptop, saved score arrays only): per-pair plant-1 shifts before
vs after the four fixes, for the planted probes themselves (each refit on the plant-1 comparison per held-out test
environment, then selected on validation). Before: H2's unevened fits (claimB/level1/h2_search_v1/plant1_k1/{A0,M2}.npz,
A0 also restricted to L58/L63 as H3's pipeline is); after: H3's evened fits (claimB/h3_small_v1/04c3953b139e/refit/
plant1{,_M2}.npz), real-label column. Per pair (cell): shift = s(copy) - s(twin). Reported: pooled AUROC (as the rule
reads it), the mean of the five within-environment AUROCs (fold scales differ), mean shift, pairs with shift > 0, sign
flips before -> after, the shift correlation; split by whether H3 re-made the pair (cells.json) and by the diagnosis
flags (anchor line moved; re-made != direct evening). Numbers only.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn \\
      python experiments/2026-10-11_claimB_h3_fix_to_plateau/analyze_POSTHOC_plant1_shifts.py results/plant1_diag_POSTHOC_2026-10-11_163142.json
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
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
PACK, PL1, SIM = "claimB/level1/packedB_2cfd1aec393c", "sim-attack-fixed-plant1", "sim-attack-fixed"
FILES = {"before_A0": "claimB/level1/h2_search_v1/plant1_k1/A0.npz", "before_M2": "claimB/level1/h2_search_v1/plant1_k1/M2.npz",
         "after_stepA": "claimB/h3_small_v1/04c3953b139e/refit/plant1.npz", "after_M2": "claimB/h3_small_v1/04c3953b139e/refit/plant1_M2.npz"}
TEST = ["clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"]


def main():
    diag = json.loads((HERE / sys.argv[1]).read_text())
    flags = {r["cell"]: r for r in diag["pairs"]}
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    read = lambda p: b"".join(vol.read_file(p))  # noqa: E731
    pruns = json.loads(read(f"{PACK}/runs.L63.json"))
    e2f = json.loads((HERE.parents[1] / "splits/folds_v2.json").read_text())["environment_folds"]["environment_to_fold"]
    env = np.array([e2f[m["environment"]] for m in pruns])
    idx = {(m["cell"], m["arm"]): i for i, m in enumerate(pruns)}
    cells = sorted(flags)
    out = {"label": "POST HOC: per-pair plant-1 shifts before vs after the four fixes (planted probes; numbers only)", "inputs": {}, "versions": {}}
    shifts = {}
    for name, path in list(FILES.items()) + [("before_A0_L58_L63", FILES["before_A0"])]:
        raw = read(path)
        out["inputs"][path] = hashlib.sha256(raw).hexdigest()
        z = np.load(io.BytesIO(raw))
        V, S, folds = z["val_auroc"][:, :, 0], z["test_run_score"][:, :, 0], list(z["folds"])
        subs = json.loads(str(z["subs"]))
        keep = np.array([(s["layer"] in (58, 63)) if isinstance(s, dict) and "layer" in s else True for s in subs]) if name.endswith("L58_L63") else np.ones(len(subs), bool)
        V, S = V[keep], S[keep]
        sel = np.full(len(pruns), np.nan)
        for i, f in enumerate(folds):
            fr = np.flatnonzero(env == f)
            sel[fr] = S[V[:, i].argmax(), fr]
        sc = {c: (sel[idx[(c, PL1)]], sel[idx[(c, SIM)]]) for c in cells}
        if not all(np.isfinite(v).all() for v in sc.values()):
            raise SystemExit(f"{name}: a pair has no score")
        d = np.array([sc[c][0] - sc[c][1] for c in cells])
        shifts[name] = d
        out.setdefault("per_pair", {c: {} for c in cells})
        for c, dd in zip(cells, d):
            out["per_pair"][c][name] = {"copy": float(sc[c][0]), "twin": float(sc[c][1]), "shift": float(dd)}
        y = np.r_[np.ones(len(cells)), np.zeros(len(cells))]
        x = np.r_[[sc[c][0] for c in cells], [sc[c][1] for c in cells]]
        per_env = []
        for e in TEST:
            ce = [c for c in cells if flags[c]["environment"] == e]
            per_env.append(roc_auc_score(np.r_[np.ones(len(ce)), np.zeros(len(ce))], np.r_[[sc[c][0] for c in ce], [sc[c][1] for c in ce]]))

        def sub(mask):
            m = np.array([mask(flags[c]) for c in cells])
            if m.sum() < 2:
                return None
            yy = np.r_[np.ones(m.sum()), np.zeros(m.sum())]
            xx = np.r_[[sc[c][0] for c, k in zip(cells, m) if k], [sc[c][1] for c, k in zip(cells, m) if k]]
            return {"pairs": int(m.sum()), "pooled_auroc": float(roc_auc_score(yy, xx)), "mean_shift": float(d[m].mean()), "shift_positive": int((d[m] > 0).sum())}
        out["versions"][name] = {"pooled_auroc": float(roc_auc_score(y, x)), "within_env_auroc_mean": float(np.mean(per_env)),
                                 "within_env_auroc": dict(zip(TEST, map(float, per_env))), "mean_shift": float(d.mean()), "shift_positive": int((d > 0).sum()),
                                 "re_made_pairs": sub(lambda r: r["copy_changed_in_plan"]), "kept_pack_pairs": sub(lambda r: not r["copy_changed_in_plan"]),
                                 "anchor_moved_pairs": sub(lambda r: not r["anchor_line_same"]), "remade_ne_direct_pairs": sub(lambda r: not r["remade_equals_direct_evening"])}
    for a, b in (("before_A0_L58_L63", "after_stepA"), ("before_M2", "after_M2")):
        out[f"{a}_vs_{b}"] = {"shift_correlation": float(np.corrcoef(shifts[a], shifts[b])[0, 1]),
                              "sign_flips_pos_to_neg": int(((shifts[a] > 0) & (shifts[b] <= 0)).sum()), "sign_flips_neg_to_pos": int(((shifts[a] <= 0) & (shifts[b] > 0)).sum())}
    out |= {"diag": sys.argv[1], "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"plant1_shifts_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    for k, v in out["versions"].items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk not in ("within_env_auroc",)})
    for k in [k for k in out if k.endswith(("after_stepA", "after_M2"))]:
        print(k, out[k])
    print("->", dst)


if __name__ == "__main__":
    main()
