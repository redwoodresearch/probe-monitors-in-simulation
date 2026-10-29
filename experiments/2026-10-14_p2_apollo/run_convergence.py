"""RULE.md Amendment 1: convergence certificate for the frozen Apollo D2 probe (block_output layer 31). One H200 re-reads the
training-side instructed pairs at layer 31 exactly as probe_shift.apollo_sweep.collect does and writes the per-token rows to a
Volume; a CPU container reproduces q2's fit (gate: coef, scaler and token count match the stored probe), then computes
max|grad G| at the STORED coef (float64; G = (sum logloss + ||w||^2 / (2C)) / n, C = 0.1, no intercept) and, over the bar, a
tight float64 L-BFGS refit folded into (direction, bias).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run experiments/2026-10-14_p2_apollo/run_convergence.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-p2probes_apcert_kill-2026-11-13"
APOLLO_REPO = Path.home() / "deception-detection"
APOLLO_COMMIT = "f8ec4010e74927394709dffa22b97bdf8cd5a62f"
MODEL, REV = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
Q2RUN = "runs/2026-10-09_191400"
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


@app.function(image=gpu_image, gpu=["H200"], cpu=(8, 8), memory=(65536, 65536), timeout=1200, retries=0,
              volumes={"/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only(), "/reads": reads})
def read_rows(stamp: str, train: list) -> dict:
    """apollo_sweep.collect's per-token path, block_output at LAYER only: rows = output[0][0, det].float() of decoder block LAYER."""
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
            if not det:  # as collect: no detected token, no rows
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
    out = Path("/reads") / "apollo_cert" / stamp
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"rows_L{LAYER}.npy", X)
    (out / "meta.json").write_text(json.dumps({"counts": counts, "gpu": gpu}))
    reads.commit()
    return {"gpu": gpu, "load_s": load_s, "forward_s": round(time.time() - t1), "n_dialogues": len(train), "n_rows": int(X.shape[0]),
            "longest_dialogue_tokens": max(lens), "dialogues_without_detected_tokens": sum(c == 0 for c in counts)}


@app.function(image=cpu_image, cpu=(8, 8), memory=(32768, 32768), timeout=1200, retries=0, volumes={"/reads": reads, "/q2": q2vol})
def certify(stamp: str, train: list) -> dict:
    import io

    import numpy as np
    import safetensors.numpy as stn
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    from probe_shift.apollo_sweep import split_pairs

    reads.reload()
    d = Path("/reads") / "apollo_cert" / stamp
    X_all = np.load(d / f"rows_L{LAYER}.npy")
    counts = json.loads((d / "meta.json").read_text())["counts"]
    if len(counts) != len(train) or sum(counts) != X_all.shape[0]:
        raise ValueError("rows and counts disagree")
    starts = np.concatenate([[0], np.cumsum(counts)])
    is_val = split_pairs(train, CFG["val_fraction"], CFG["split_seed"])
    y = np.array([t["label"] for t in train])
    fit_rows = [i for i in range(len(train)) if not is_val[i] and counts[i]]  # as sweep(): train side, dialogues with tokens
    X = np.concatenate([X_all[starts[i]: starts[i + 1]] for i in fit_rows])
    yt = np.concatenate([np.full(counts[i], y[i]) for i in fit_rows])  # integer labels, as sweep() (Codex)

    raw = (Path("/q2") / Q2RUN / "probes_block_output.npz").read_bytes()
    z = np.load(io.BytesIO(raw))
    coef_s, mean_s, scale_s = z["coef"][LAYER].astype(np.float64), z["scaler_mean"][LAYER].astype(np.float64), z["scaler_scale"][LAYER].astype(np.float64)
    res_q2 = json.loads((Path("/q2") / Q2RUN / "result.json").read_text())
    row_q2 = next(r for r in res_q2["by_probe"] if r["readout"] == READOUT and r["layer"] == LAYER)

    with threadpool_limits(limits=CFG["blas_threads_per_fit"]):  # q2's fit, reproduced (sweep.fit), scaling included
        scaler = StandardScaler().fit(X)
        Zf32 = scaler.transform(X)
        lr = LogisticRegression(C=1 / CFG["reg_coeff"], fit_intercept=False, random_state=42, max_iter=CFG["max_iter"]).fit(Zf32, yt)
    rel = lambda a, b: float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))
    gate = {"train_tokens": len(yt), "train_tokens_stored": row_q2["train_tokens"], "n_iter_refit": int(lr.n_iter_[0]), "n_iter_stored": row_q2["n_iter"],
            "coef_rel_maxdiff": rel(lr.coef_[0].astype(np.float64), coef_s),
            "mean_float32_equal": bool(np.array_equal(scaler.mean_.astype(np.float32), z["scaler_mean"][LAYER])),
            "scale_float32_equal": bool(np.array_equal(scaler.scale_.astype(np.float32), z["scaler_scale"][LAYER])),
            "mean_max_elementwise_rel": float((np.abs(scaler.mean_ - mean_s) / np.maximum(np.abs(mean_s), 1e-12)).max()),
            "scale_max_elementwise_rel": float((np.abs(scaler.scale_ - scale_s) / scale_s).max())}
    # a fit-consistency gate (Codex): the token count, the refit coef and the float32-rounded scaler must match the stored probe
    gate["passes"] = bool(gate["train_tokens"] == gate["train_tokens_stored"] and gate["coef_rel_maxdiff"] <= 1e-6
                          and gate["mean_max_elementwise_rel"] <= 1e-6 and gate["scale_max_elementwise_rel"] <= 1e-6)
    out = {"gate": gate, "npz_sha256": hashlib.sha256(raw).hexdigest(), "n": len(yt)}
    if not gate["passes"]:
        return out | {"stopped": "rows not reproduced: no certificate claimed (RULE.md Amendment 1)"}

    Z = Zf32.astype(np.float64)  # as sklearn's lbfgs received it
    yt = yt.astype(np.float64)

    def grad(w):  # grad of G = (sum logloss + ||w||^2 / (2C)) / n, no intercept
        p = 1 / (1 + np.exp(-(Z @ w)))
        return (Z.T @ (p - yt) + w / C) / len(yt)

    g_stored = float(np.abs(grad(coef_s)).max())
    out["stored"] = {"max_abs_grad": g_stored, f"max|grad G| <= {BAR}": g_stored <= BAR}
    if g_stored > BAR:
        torch.set_num_threads(8)
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
        wt = w.detach().numpy()
        gt = float(np.abs(grad(wt)).max())
        g32 = float(np.abs(grad(wt.astype(np.float32).astype(np.float64))).max())  # after float32 serialization (Codex)
        mu, sc = scaler.mean_, scaler.scale_
        r, c = (wt / sc).astype(np.float32), np.float32(-float((mu * wt / sc).sum()))
        out["tight"] = {"max_abs_grad": gt, "max_abs_grad_float32_serialized": g32, "certified": gt <= BAR, "status": "movement check pending (catch on B, as-run vs tight)",
                        "coef_rel_change_vs_stored": rel(wt, coef_s),
                        "file": stn.save({"direction": r, "bias": np.array([c], np.float32), "coef": wt.astype(np.float32), "coef_float64": wt, "mean": mu.astype(np.float32),
                                          "scale": sc.astype(np.float32)}, metadata={"layer": str(LAYER), "readout": READOUT, "fit": "tight float64 L-BFGS"})}
    return out


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
    train = apollo.instructed_pairs(APOLLO_REPO)  # as run_step2_qwen35.build(), train side only
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rd = read_rows.remote(stamp, train)
    ce = certify.remote(stamp, train)
    rec = {"label": "RULE.md Amendment 1 (convergence certificate)", "git": head, "app": APP, "stamp (PT)": stamp, "apollo_commit": APOLLO_COMMIT,
           "train_sha256": hashlib.sha256(json.dumps(train, sort_keys=True).encode()).hexdigest(), "objective": "G = (sum logloss + ||w||^2 / (2C)) / n, C = 0.1, no intercept",
           "bar": BAR, "read": rd} | {k: v for k, v in ce.items() if k != "tight"}
    res = HERE / "results"
    if "tight" in ce:
        f = res / f"apollo_d2_L{LAYER}_tight_{stamp}.safetensors"
        f.open("xb").write(ce["tight"]["file"])
        rec["tight"] = {k: v for k, v in ce["tight"].items() if k != "file"} | {"file": {"path": f.name, "sha256": hashlib.sha256(ce["tight"]["file"]).hexdigest()}}
    dest = res / f"convergence_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1))
    lines = [f"read: {rd}", f"gate: {ce['gate']}"]
    if "stopped" in ce:
        lines.append(f"STOPPED: {ce['stopped']}")
    else:
        lines.append(f"stored coef: n {ce['n']}, max|grad G| {ce['stored']['max_abs_grad']:.3g}, certified {ce['stored'][f'max|grad G| <= {BAR}']}")
        if "tight" in rec:
            lines.append(f"tight: max|grad G| {rec['tight']['max_abs_grad']:.3g}, certified {rec['tight']['certified']}, coef rel change {rec['tight']['coef_rel_change_vs_stored']:.3g}; {rec['tight']['status']}")
    dest.with_suffix(".printout.txt").open("x").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(dest.name)
