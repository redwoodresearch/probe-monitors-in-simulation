"""RULE.md Amendment 2: why the Apollo D2 certificate gate failed, and the fix.
- 1(b) on the saved H200 rows (CPU): how far q2's loose sklearn fit moves under small input noise; max|grad G| at q2's stored coef.
- 1(c) one H100 re-read, DIAGNOSTIC ONLY (rows labelled on the Volume, never used for a result): Amendment 1's gate against the stored
  probe, and H100 vs H200 rows directly.
- 2 the fix: a converged (certified) float64 refit of the pre-registered recipe on the H200 rows, saved (scaler, coef, folded probe),
  with the held-out instructed-pair AUROC for it and for the stored coef; and the record the shared scorer reads.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run experiments/2026-10-14_p2_apollo/run_fix.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-p2probes_apfix_kill-2026-11-13"
APOLLO_REPO = Path.home() / "deception-detection"
APOLLO_COMMIT = "f8ec4010e74927394709dffa22b97bdf8cd5a62f"
MODEL, REV = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
Q2RUN, H200_STAMP = "runs/2026-10-09_191400", "2026-10-14_143716"
LAYER, READOUT, C, BAR = 31, "block_output", 0.1, 1e-6
CFG = {"chat_template": {"enable_thinking": False}, "reg_coeff": 10, "max_iter": 2000, "val_fraction": 0.2, "split_seed": 0, "blas_threads_per_fit": 2}

app = modal.App(APP)
reads = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_reads_kill-2026-11-13", version=2)
q2vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-q2-deception-qwen35_kill-2026-11-08").read_only()
gpu_image = (modal.Image.debian_slim(python_version="3.12")
             .uv_pip_install("torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy==2.3.3", "flash-linear-attention==0.5.2", "jinja2")
             .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))
cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .uv_pip_install("torch==2.8.0", "numpy==2.3.3", "scikit-learn==1.7.2", "threadpoolctl", "safetensors==0.8.0")
             .add_local_python_source("probe_shift"))


@app.function(image=gpu_image, gpu="H100!", cpu=(8, 8), memory=(65536, 65536), timeout=1200, retries=0,
              volumes={"/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only(), "/reads": reads})
def read_h100_diagnostic(stamp: str, train: list) -> dict:
    """Amendment 1's read_rows, unchanged except the GPU (H100) and the DIAGNOSTIC output folder."""
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.apollo import render_detect

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").model.eval()
    load_s, gpu = round(time.time() - t0), torch.cuda.get_device_name(0)
    device, det, got = base.embed_tokens.weight.device, [], {}

    def hook(module, args, output):
        got["x"] = (output[0] if isinstance(output, tuple) else output)[0, det].float()

    h = base.layers[LAYER].register_forward_hook(hook)
    rows, counts, lens = [], [], []
    t1 = time.time()
    try:
        for d in train:
            ids, det[:] = render_detect(tok, d, CFG["chat_template"])
            counts.append(len(det))
            lens.append(len(ids))
            if not det:
                continue
            with torch.inference_mode():
                base(input_ids=torch.tensor([ids], device=device), use_cache=False)
            rows.append(got["x"].to("cpu").numpy())
            if len(counts) % 100 == 0:
                print(f"{len(counts)}/{len(train)} dialogues, {time.time() - t1:.0f} s", flush=True)
    finally:
        h.remove()
    X = np.concatenate(rows).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError("non-finite rows")
    out = Path("/reads") / "apollo_cert_h100_DIAGNOSTIC" / stamp  # DIAGNOSTIC: never an analysis input (RULE.md Amendment 2)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"rows_L{LAYER}.npy", X)
    (out / "meta.json").write_text(json.dumps({"counts": counts, "gpu": gpu, "label": "DIAGNOSTIC ONLY"}))
    reads.commit()
    return {"gpu": gpu, "load_s": load_s, "forward_s": round(time.time() - t1), "n_dialogues": len(train), "n_rows": int(X.shape[0]),
            "longest_dialogue_tokens": max(lens), "dialogues_without_detected_tokens": sum(c == 0 for c in counts)}


@app.function(image=cpu_image, cpu=(8, 8), memory=(32768, 32768), timeout=1200, retries=0, volumes={"/reads": reads, "/q2": q2vol})
def analyze(h100_stamp: str, train: list) -> dict:
    import io

    import numpy as np
    import safetensors.numpy as stn
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    from probe_shift.apollo_sweep import split_pairs

    reads.reload()
    torch.set_num_threads(8)
    is_val = split_pairs(train, CFG["val_fraction"], CFG["split_seed"])
    y = np.array([t["label"] for t in train])

    def load(folder):
        d = Path("/reads") / folder
        raw = (d / f"rows_L{LAYER}.npy").read_bytes()
        Xa = np.load(io.BytesIO(raw))
        counts = json.loads((d / "meta.json").read_text())["counts"]
        if len(counts) != len(train) or sum(counts) != Xa.shape[0]:
            raise ValueError(f"{folder}: rows and counts disagree")
        st = np.concatenate([[0], np.cumsum(counts)])
        per = [Xa[st[i]: st[i + 1]] for i in range(len(train))]
        fit_rows = [i for i in range(len(train)) if not is_val[i] and counts[i]]
        val_rows = [i for i in range(len(train)) if is_val[i] and counts[i]]
        X = np.concatenate([per[i] for i in fit_rows])
        yt = np.concatenate([np.full(counts[i], y[i]) for i in fit_rows])
        return {"X": X, "yt": yt, "per": per, "val_rows": val_rows, "sha256": hashlib.sha256(raw).hexdigest(), "counts": counts}

    def q2_fit(X, yt):  # q2's exact settings (sweep.fit)
        with threadpool_limits(limits=CFG["blas_threads_per_fit"]):
            sc = StandardScaler().fit(X)
            Z32 = sc.transform(X)
            lr = LogisticRegression(C=1 / CFG["reg_coeff"], fit_intercept=False, random_state=42, max_iter=CFG["max_iter"]).fit(Z32, yt)
        return sc, Z32, lr

    def grad(Z, yt, w):  # grad of G = (sum logloss + ||w||^2 / (2C)) / n, no intercept, float64
        p = 1 / (1 + np.exp(-(Z @ w)))
        return (Z.T @ (p - yt) + w / C) / len(yt)

    def tight(Z, yt):
        Zt, ytt = torch.from_numpy(Z), torch.from_numpy(yt)
        w = torch.zeros(Z.shape[1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=1, max_iter=5000, tolerance_grad=1e-8, tolerance_change=0, history_size=50, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            zz = Zt @ w
            loss = ((torch.nn.functional.softplus(zz) - ytt * zz).sum() + (w @ w) / (2 * C)) / len(ytt)
            loss.backward()
            return loss

        opt.step(closure)
        return w.detach().numpy()

    rel = lambda a, b: float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))
    z = np.load(io.BytesIO((Path("/q2") / Q2RUN / "probes_block_output.npz").read_bytes()))
    coef_s, mean_s, scale_s = (z[k][LAYER].astype(np.float64) for k in ("coef", "scaler_mean", "scaler_scale"))

    # 1(b): sensitivity of the loose fit, on the H200 rows
    H = load(f"apollo_cert/{H200_STAMP}")
    if (len(H["yt"]), len(H["val_rows"])) != (2498, 114):  # q2's record: 2,498 train tokens, 114 val dialogues (Codex)
        raise ValueError(f"H200 rows: {len(H['yt'])} train tokens / {len(H['val_rows'])} val dialogues differ from q2's record")
    ytf = H["yt"].astype(np.float64)
    sc, Z32, lrA = q2_fit(H["X"], H["yt"])
    coefA = lrA.coef_[0].astype(np.float64)
    noise = {}
    for eps in (1e-3, 4e-3):
        rng = np.random.default_rng(20261014)
        Xn = (H["X"] * (1 + eps * rng.standard_normal(H["X"].shape))).astype(np.float32)
        _, _, lrN = q2_fit(Xn, H["yt"])
        noise[str(eps)] = {"coef_rel_move_vs_A": rel(lrN.coef_[0].astype(np.float64), coefA), "n_iter": int(lrN.n_iter_[0])}
    Z_own = Z32.astype(np.float64)
    Z_stored = (H["X"].astype(np.float64) - mean_s) / scale_s
    b = {"n_train_tokens": len(ytf), "coefA_n_iter": int(lrA.n_iter_[0]), "coefA_rel_vs_stored": rel(coefA, coef_s), "noise_refits": noise,
         "max|grad G| at stored coef, stored scaler": float(np.abs(grad(Z_stored, ytf, coef_s)).max()),
         "max|grad G| at stored coef, H200 rows' scaler": float(np.abs(grad(Z_own, ytf, coef_s)).max()),
         "max|grad G| at coef A (q2 settings, H200 rows)": float(np.abs(grad(Z_own, ytf, coefA)).max()),
         "scale rel diff, stored vs H200 rows (max / median)": [float(np.max(np.abs(sc.scale_ - scale_s) / scale_s)), float(np.median(np.abs(sc.scale_ - scale_s) / scale_s))]}

    # 2: the fix, a converged refit on the H200 rows
    wt = tight(Z_own, ytf)
    gt, g32 = float(np.abs(grad(Z_own, ytf, wt)).max()), float(np.abs(grad(Z_own, ytf, wt.astype(np.float32).astype(np.float64))).max())

    def val_auroc(w, mean, scale):  # q2's instructed_val_auroc: per-dialogue mean of the per-token decision
        s = np.array([(((H["per"][i].astype(np.float64) - mean) / scale) @ w).mean() for i in H["val_rows"]])
        return float(roc_auc_score(y[H["val_rows"]], s))

    va_t = val_auroc(wt, sc.mean_, sc.scale_)
    fix = {"max_abs_grad": gt, "max_abs_grad_float32": g32, "certified": bool(gt <= BAR and g32 <= BAR), "positive_control_val_auroc_ge_0.9": bool(va_t >= 0.9), "coef_rel_vs_stored": rel(wt, coef_s),
           "coef_rel_vs_coefA": rel(wt, coefA), "val_auroc_tight (H200 rows)": va_t,
           "val_auroc_stored (stored coef and scaler, H200 rows)": val_auroc(coef_s, mean_s, scale_s), "n_val_dialogues": len(H["val_rows"]),
           "h200_rows": {"path": f"apollo_cert/{H200_STAMP}/rows_L{LAYER}.npy", "sha256": H["sha256"]}}
    r, c = (wt / sc.scale_).astype(np.float32), np.float32(-float((sc.mean_ * wt / sc.scale_).sum()))
    probe = stn.save({"direction": r, "bias": np.array([c], np.float32), "coef": wt.astype(np.float32), "coef_float64": wt, "mean_float64": sc.mean_,
                      "scale_float64": sc.scale_, "mean": sc.mean_.astype(np.float32), "scale": sc.scale_.astype(np.float32)},
                     metadata={"layer": str(LAYER), "readout": READOUT, "fit": "tight float64 L-BFGS on the H200 rows (RULE.md Amendment 2)"})
    t_ = stn.load(probe)  # the scored artifact (Codex): float32 x . direction + bias vs the float64 standardized model, on the train rows
    lg_art = H["X"].astype(np.float32) @ t_["direction"] + t_["bias"][0]
    lg_ref = Z_own @ wt
    fix["artifact_logit_maxdiff"] = float(np.abs(lg_art.astype(np.float64) - lg_ref).max())
    fix["artifact_logit_scale"] = float(np.abs(lg_ref).max())

    # 1(c): the H100 re-read, DIAGNOSTIC only
    D = load(f"apollo_cert_h100_DIAGNOSTIC/{h100_stamp}")
    if D["counts"] != H["counts"]:  # the same dialogues and detected spans, row for row (Codex)
        raise ValueError("H100 and H200 count vectors differ: rows can't be aligned")
    scD, _, lrD = q2_fit(D["X"], D["yt"])
    gate = {"train_tokens": len(D["yt"]), "coef_rel_maxdiff": rel(lrD.coef_[0].astype(np.float64), coef_s), "n_iter": int(lrD.n_iter_[0]),
            "mean_max_elementwise_rel": float((np.abs(scD.mean_ - mean_s) / np.maximum(np.abs(mean_s), 1e-12)).max()),
            "scale_max_elementwise_rel": float((np.abs(scD.scale_ - scale_s) / scale_s).max()),
            "mean_float32_equal": bool(np.array_equal(scD.mean_.astype(np.float32), z["scaler_mean"][LAYER])),
            "scale_float32_equal": bool(np.array_equal(scD.scale_.astype(np.float32), z["scaler_scale"][LAYER]))}
    gate["passes"] = bool(gate["train_tokens"] == 2498 and gate["coef_rel_maxdiff"] <= 1e-6 and gate["mean_max_elementwise_rel"] <= 1e-6 and gate["scale_max_elementwise_rel"] <= 1e-6)
    XH, XD = np.concatenate(H["per"]).astype(np.float64), np.concatenate(D["per"]).astype(np.float64)
    cos = (XH * XD).sum(1) / (np.linalg.norm(XH, axis=1) * np.linalg.norm(XD, axis=1))
    rowrel = np.abs(XH - XD).max(1) / np.abs(XH).max(1)
    diag = {"gate_vs_stored": gate, "h100_vs_h200_rows": {"bit_equal_rows": int(np.all(XH == XD, axis=1).sum()), "n_rows": len(XH),
                                                          "min_cosine": float(cos.min()), "median_cosine": float(np.median(cos)),
                                                          "max_row_rel_diff": float(rowrel.max()), "median_row_rel_diff": float(np.median(rowrel))},
            "h100_rows": {"path": f"apollo_cert_h100_DIAGNOSTIC/{h100_stamp}/rows_L{LAYER}.npy", "sha256": D["sha256"], "label": "DIAGNOSTIC ONLY"}}
    return {"diagnosis_b": b, "diagnosis_c": diag, "fix": fix, "file": probe}


@app.local_entrypoint()
def main():
    from probe_shift import apollo

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if git("status", "--porcelain", "--", ".") or git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted changes")
    head = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", head):
        raise SystemExit("HEAD is not pushed")
    if subprocess.run(["git", "-C", str(APOLLO_REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() != APOLLO_COMMIT:
        raise SystemExit("deception-detection checkout is not at the pinned commit")
    train = apollo.instructed_pairs(APOLLO_REPO)
    a1 = json.loads((HERE / "results" / "convergence_2026-10-14_143716.json").read_text())
    if hashlib.sha256(json.dumps(train, sort_keys=True).encode()).hexdigest() != a1["train_sha256"]:  # the H200 rows' dialogues (Codex)
        raise SystemExit("the training dialogues differ from Amendment 1's")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rd = read_h100_diagnostic.remote(stamp, train)
    an = analyze.remote(stamp, train)
    res = HERE / "results"
    f = res / f"apollo_d2_L{LAYER}_fix_{stamp}.safetensors"
    f.open("xb").write(an["file"])
    fsha = hashlib.sha256(an["file"]).hexdigest()
    rec = {"label": "RULE.md Amendment 2 (diagnosis and fix)", "git": head, "app": APP, "stamp (PT)": stamp, "apollo_commit": APOLLO_COMMIT,
           "objective": "G = (sum logloss + ||w||^2 / (2C)) / n, C = 0.1, no intercept", "bar": BAR, "h100_read (DIAGNOSTIC ONLY)": rd,
           "diagnosis_b": an["diagnosis_b"], "diagnosis_c": an["diagnosis_c"], "fix": an["fix"] | {"file": {"path": f.name, "sha256": fsha}}}
    (res / f"fix_{stamp}.json").open("x").write(json.dumps(rec, indent=1))
    scorer = {"label": "Apollo D2 tight refit (RULE.md Amendment 2): the scorer's record", "git": head, "stamp (PT)": stamp,
              "directions": {"apollo_d2_tight": {"position": "block_output (per-token probe; mean rows exact up to rounding)", "layer": LAYER,
                                                 "passes": an["fix"]["certified"] and an["fix"]["positive_control_val_auroc_ge_0.9"], "gate_scope": "convergence certified; source-domain gate inherited from step 2 (stored probe); agent-format control outstanding",
                                                 "val_auroc_tight": an["fix"]["val_auroc_tight (H200 rows)"], "file": {"path": f.name, "sha256": fsha}}}}
    (res / f"fit_tight_{stamp}.json").open("x").write(json.dumps(scorer, indent=1))
    b, c_, fx = an["diagnosis_b"], an["diagnosis_c"], an["fix"]
    lines = [f"1(b) H200 rows: {json.dumps(b)}", f"1(c) H100 DIAGNOSTIC read: {json.dumps(rd)}", f"1(c) gate vs stored: {json.dumps(c_['gate_vs_stored'])}",
             f"1(c) H100 vs H200 rows: {json.dumps(c_['h100_vs_h200_rows'])}",
             (f"2 fix: max|grad G| {fx['max_abs_grad']:.3g} (float32 {fx['max_abs_grad_float32']:.3g}), certified {fx['certified']}; coef rel vs stored {fx['coef_rel_vs_stored']:.3g}; "
              f"val AUROC tight {fx['val_auroc_tight (H200 rows)']:.4f}, stored {fx['val_auroc_stored (stored coef and scaler, H200 rows)']:.4f} (n {fx['n_val_dialogues']})")]
    (res / f"fix_{stamp}.printout.txt").open("x").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"fix_{stamp}.json; fit_tight_{stamp}.json")
