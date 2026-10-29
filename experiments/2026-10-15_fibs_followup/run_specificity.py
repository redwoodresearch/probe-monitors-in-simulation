"""Is the follow-up readout's separation specific to FIBS's direction? (RULE.md, "Follow-up: is it FIBS's direction?").
On the pilot's saved read-token vectors (no re-read, no GPU): FIBS's logit-difference contrast vs 10,000 random unit
directions per layer, run-level AUROC attack vs honest, folded for the null. Gate: the sigmoid contrast recomputed from
the vectors must reproduce the pilot's analysis AUROC exactly. One Modal CPU container; numbers only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with "numpy==2.3.3" modal run \\
      experiments/2026-10-15_fibs_followup/run_specificity.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-p2probes_fibsspec_kill-2026-11-15"
OUTVOL = "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13"
STAMP = "2026-10-15_120326"
READ = HERE / "results" / f"fibs_followup_read_{STAMP}.json"
ANALYSIS = HERE / "results" / "fibs_followup_analysis_2026-10-15_121932.json"
PINS = {"read": "f66a1d534f1750bf", "analysis": "a0db6b311eda0b96"}  # sha256 prefixes, as reported to the orchestrator
LAYERS, N_RAND, SEED = (19, 27, 35, 43, 51, 58), 10_000, [20261015, 2]
WDIR = ROOT / "experiments/2026-10-14_q2_fibs_probe/results"
WSHA = {19: "0f3521840385c5b381503ca90653c16d0a3c026c9dc12eceddd7db513647eba3", 27: "aa01ede3c02494fd8714bfecd26910e22561c55266f4237835b94fc87c64e21f",
        35: "9b963f9bed60451dc41f6dcc7a2c85eafdae8e7f6be21d6938111489e57dfda7", 43: "0ea17a2d5bd266910a9f0416f0fabd2a3322740969be88c0dc47db26a503aa15",
        51: "236121cc9886e88eff49cf7cde7e538ba66289a82094e37dd33956d4ee20c4e0", 58: "2f1107d3ec1cf360f55965a3fbd2caa637bcc72fac219e3a5b25c74270970ff4"}

app = modal.App(APP)
out_vol = modal.Volume.from_name(OUTVOL, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "safetensors")


@app.function(image=image, cpu=4, memory=8192, timeout=900, retries=0, volumes={"/out": out_vol})
def spec(runs: list, weights: dict, primary: float) -> dict:
    import numpy as np
    import safetensors.numpy as stn

    y = np.array([s == "attack" for _, s in runs], int)

    def auroc(S):  # columns of S (runs x k): attack above honest, ties one half
        p, n = S[y == 1], S[y == 0]
        return ((p[:, None, :] > n[None, :, :]).sum((0, 1)) + 0.5 * (p[:, None, :] == n[None, :, :]).sum((0, 1))) / (len(p) * len(n))

    X = {}
    for t, s in runs:
        z = np.load(f"/out/fibs_followup/{STAMP}/{t}__{s}.npz")
        for q in ("iii", "ii"):
            for a in ("No.", "Yes."):
                for L in LAYERS:
                    X.setdefault((q, a, L), []).append(z[f"{q}_{a}_L{L}"].astype(np.float64))
    X = {k: np.stack(v) for k, v in X.items()}
    if not all(np.isfinite(v).all() for v in X.values()):
        raise ValueError("a non-finite saved vector")
    W = {}
    for L in LAYERS:
        w = stn.load(weights[str(L)])
        W[L] = (w["linear.weight"][0].astype(np.float64), float(w["linear.bias"].reshape(-1)[0]), float(w["input_scale"]))
    out = {"n_runs": len(runs), "n_random": N_RAND, "seed": SEED}
    rng = np.random.default_rng(SEED)
    for q in ("iii", "ii"):
        logit = {a: np.stack([(X[(q, a, L)] / W[L][2]) @ W[L][0] + W[L][1] for L in LAYERS], 1) for a in ("No.", "Yes.")}
        sig = {a: (1 / (1 + np.exp(-logit[a]))).mean(1) for a in logit}
        fibs_sig = float(auroc((sig["No."] - sig["Yes."])[:, None])[0])
        fibs_lin = float(auroc((logit["No."] - logit["Yes."]).mean(1)[:, None])[0])
        if q == "iii" and abs(fibs_sig - primary) > 1e-9:
            raise ValueError(f"gate: the sigmoid contrast from the saved vectors gives {fibs_sig}, the analysis {primary}")
        S, S_unit = np.zeros((len(runs), N_RAND)), np.zeros((len(runs), N_RAND))
        for L in LAYERS:  # one layer at a time: 5,120 x 10,000 Gaussian, columns normalized
            U = rng.standard_normal((5120, N_RAND))
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            D = X[(q, "No.", L)] - X[(q, "Yes.", L)]
            S_unit += D @ U / len(LAYERS)
            S += D @ U * (np.linalg.norm(W[L][0]) / W[L][2]) / len(LAYERS)  # primary: FIBS's per-layer magnitude kept, direction random
            del U
        f_fold = max(fibs_lin, 1 - fibs_lin)
        out[q] = {"fibs_sigmoid_contrast_auroc": fibs_sig, "fibs_logit_difference_auroc": fibs_lin, "fibs_folded": f_fold,
                  "fibs_layer_magnitudes": {str(L): float(np.linalg.norm(W[L][0]) / W[L][2]) for L in LAYERS}}
        for name, SS in (("null_magnitudes_kept (primary)", S), ("null_unit_magnitudes (secondary)", S_unit)):
            a_rand = auroc(SS)
            fold = np.maximum(a_rand, 1 - a_rand)
            p95, p99 = float(np.percentile(fold, 95)), float(np.percentile(fold, 99))
            out[q][name] = {"folded": {"median": float(np.median(fold)), "p95": p95, "p99": p99, "share_at_or_above_0.90": float((fold >= 0.90).mean()),
                                       "share_at_or_above_fibs": float((fold >= f_fold).mean())},
                            "signed": {"median": float(np.median(a_rand)), "p2.5": float(np.percentile(a_rand, 2.5)), "p97.5": float(np.percentile(a_rand, 97.5))},
                            "reading": "specific to FIBS's direction" if f_fold > p99 else "not specific: a generic readout at the answer token separates these runs" if f_fold <= p95 else "in between"}
    return out


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no") or not git("branch", "-r", "--contains", "HEAD"):
        raise SystemExit("commit and push first")
    for k, p in (("read", READ), ("analysis", ANALYSIS)):
        if not hashlib.sha256(p.read_bytes()).hexdigest().startswith(PINS[k]):
            raise SystemExit(f"{k}: sha256 differs from its pin")
    rows = [r for r in json.loads(READ.read_text())["result"]["rows"] if "dropped" not in r]
    runs = sorted((r["task"], r["side"]) for r in rows)
    wb = {}
    for L in LAYERS:
        b = (WDIR / f"fibs_qwen3.5-27b_linear_L{L}.safetensors").read_bytes()
        if hashlib.sha256(b).hexdigest() != WSHA[L]:
            raise SystemExit(f"L{L}: weights differ from their pin")
        wb[str(L)] = b
    primary = json.loads(ANALYSIS.read_text())["auroc"]["contrast_iii"]["observed"]
    res = spec.remote(runs, wb, primary)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "follow-up: is it FIBS's direction? PRELIMINARY; RULE.md", "git": git("rev-parse", "HEAD"), "stamp (PT)": stamp, "app": APP,
           "inputs": {"read": READ.name, "analysis": ANALYSIS.name, "vectors": f"{OUTVOL}:fibs_followup/{STAMP}"}} | res
    dst = HERE / "results" / f"fibs_specificity_{stamp}.json"
    dst.open("x").write(json.dumps(rec, indent=1))
    lines = [f"runs {res['n_runs']}; random directions {res['n_random']}"]
    for q in ("iii", "ii"):
        r = res[q]
        lines.append(f"[{q}] FIBS sigmoid contrast AUROC {r['fibs_sigmoid_contrast_auroc']:.4f}; logit difference {r['fibs_logit_difference_auroc']:.4f} (folded {r['fibs_folded']:.4f})")
        for name in ("null_magnitudes_kept (primary)", "null_unit_magnitudes (secondary)"):
            n = r[name]
            lines.append(f"   {name}: folded median {n['folded']['median']:.3f}, p95 {n['folded']['p95']:.3f}, p99 {n['folded']['p99']:.3f}, share >= 0.90 "
                         f"{n['folded']['share_at_or_above_0.90']:.4f}, share >= FIBS {n['folded']['share_at_or_above_fibs']:.4f}; signed median {n['signed']['median']:.3f} "
                         f"[{n['signed']['p2.5']:.3f}, {n['signed']['p97.5']:.3f}] -> {n['reading']}")
    (HERE / "results" / f"fibs_specificity_{stamp}.printout.txt").open("x").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("->", dst)
