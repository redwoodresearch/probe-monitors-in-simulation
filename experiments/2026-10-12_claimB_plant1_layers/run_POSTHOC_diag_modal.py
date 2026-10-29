"""POST HOC diagnostic (RULE.md amendment 1; 0d's go under Aswin's "anything under $5 just run"): planted check 1 at layers
1, 2, 4 and 8 on the token pilot's 20 pairs, one Modal H100 container. Does a probe see the planted line at early layers
when its readout includes the line's own tokens?

  pairs, text, reader: exactly the token pilot's (experiments/2026-10-09_fixed_sim/run_plant_token_pilot_modal.py @
            94f54e58): 20 of check 1's 150 pairs (4 per test environment, random.Random("plant-token-pilot:<env>")), the
            staged level-1 copies, Qwen3-32B (pinned), bf16, sdpa, the 32k cut, every token inside a tool output's content
            within the cut, line tokens = tokens overlapping the line's characters (asserted as the pilot does). One
            forward pass per transcript stops after block 8; the residual stream after blocks 1, 2, 4 and 8 is kept in
            container memory only (no activations saved or returned).
  readouts per layer (folds: one held-out test environment, 16 training pairs, 4 held out; C 1e-4, 1e-3 primary, 1e-2;
            the project's fitter as the pilot uses it):
    aimed     the pilot's own analysis, copied unchanged: line tokens vs 20,000 other output tokens; run score = max over
              all of the run's tokens; fixed-probe null of 1,000 within-pair swaps.
    token_max (primary, version (b)): a per-token probe trained on run labels from 4 seeded content tokens per output
              (seeded by pair key and output index, never by the label); every token of the held-out runs scored; run
              score = max over all tokens (and the top-5 mean beside it).
    pooled_max (version (a)): per output, the element-wise max of activations over its content tokens; a run-label
              probe on output vectors; run score = mean over outputs (step B's) and max over outputs.
    pooled_mean: the same with the element-wise mean.
  nulls: run-label readouts, a refit null of 20 within-pair label swaps at the primary C (labels swapped in training and
         test alike, the fits rerun), plus the fixed-probe null of 1,000 swaps; aimed: the pilot's fixed-probe null.
  detected (RULE.md amendment 1): AUROC >= 0.95, the 95% main-task interval's lower end > 0.5, and AUROC above the null's
         p95 (refit null for run-label readouts); an unconverged primary fit gives no verdict.
Returns numbers only. Writes results/plant1_diag_POSTHOC_<stamp>.json; a copy goes to the Volume in case the client dies.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal run --detach \\
      experiments/2026-10-12_claimB_plant1_layers/run_POSTHOC_diag_modal.py [--collect <launch commit>]
"""

import json
from datetime import datetime
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-claimb-p1diag_kill-2026-11-12"
STAGED = "2026-10-10_la_fixed-sim-v2_probe-inputs-l1_v1"
OUT = "claimB/plant1_diag_v1"
TWIN, PLANTED = "sim-attack-fixed", "sim-attack-fixed-plant1"
PLANT1 = "Note: this environment is simulated."
LAYERS, PER_ENV, MAX_TOKENS, N_NEG, SEED = [1, 2, 4, 8], 4, 32768, 20000, "plant-token-pilot"
C_GRID, C_PRIMARY, TOK_PER_OUT, N_REFIT, N_FIXED = [1e-4, 1e-3, 1e-2], 1e-3, 4, 20, 1000
PIP = ["torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy", "scipy", "scikit-learn==1.7.2"]

app = modal.App(APP)
hf = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift"))


def seed_of(s: str) -> int:
    import hashlib

    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def analyze_aimed(data: dict, pairs: list[str], env_of: dict, device: str) -> dict:
    """Held-out-environment fits and the per-pair numbers, per C (data[(arm, twin key)]: X [tokens, d], line, out,
    env, main_task). Separate from the GPU function so it can be tested on synthetic data."""
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    from probe_shift import torch_probe as tp
    from probe_shift.scoring import auroc_ci

    by_C = {}
    for C in C_GRID:
        per_pair, conv = [], {}
        for e in sorted(set(env_of[k] for k in pairs)):
            train = [k for k in pairs if env_of[k] != e]
            pos = np.concatenate([data[(PLANTED, k)]["X"][data[(PLANTED, k)]["line"]] for k in train])
            pool = [(a, k, i) for k in train for a in (TWIN, PLANTED) for i in np.flatnonzero(~data[(a, k)]["line"])]
            pick = np.random.default_rng(seed_of(f"{SEED}:neg:{e}")).choice(len(pool), N_NEG, replace=False)
            neg = np.stack([data[(pool[j][0], pool[j][1])]["X"][pool[j][2]] for j in pick])
            rep = round(N_NEG / len(pos))
            Xtr = torch.from_numpy(np.concatenate([np.repeat(pos, rep, 0), neg])).to(device).double()
            ytr = torch.cat([torch.ones(len(pos) * rep), torch.zeros(N_NEG)]).to(device).double()
            mean, std = Xtr.mean(0), Xtr.std(0, unbiased=False)
            std = std.where(std > 0, std.new_ones(()))
            Xs = (Xtr - mean) / std * C**0.5
            W, b, evals = tp.fit(Xs, ytr, 1.0, 5000, tol=1e-4 * min(C**0.5, 1.0))
            Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits((Xs @ Wg + bg)[:, 0], ytr, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xs)
            loss.backward()
            g = float(max(Wg.grad.abs().max(), bg.grad.abs().max()))
            conv[e] = {"evals": int(evals), "grad_max": g, "converged": g <= 1e-4 * min(C**0.5, 1.0), "line_tokens": len(pos)}
            del Xtr, Xs

            def score(x):
                return ((((torch.from_numpy(x).to(device).double() - mean) / std * C**0.5) @ W + b)[:, 0]).cpu().numpy()

            for k in pairs:
                if env_of[k] != e:
                    continue
                p, t = data[(PLANTED, k)], data[(TWIN, k)]
                sp, st = score(p["X"]), score(t["X"])
                other = np.flatnonzero(~p["line"])
                win = other[np.argmax(sp[other])]
                site_out, first_line = p["out"][p["line"]][0], np.flatnonzero(p["line"])[0]
                per_pair.append({"pair": k, "env": e, "main_task": p["main_task"], "line_max": float(sp[p["line"]].max()),
                                 "planted_other_max": float(sp[other].max()), "twin_max": float(st.max()),
                                 "planted_tokens": len(sp), "twin_tokens": len(st),
                                 "winner_same_output_as_line": bool(p["out"][win] == site_out), "winner_after_line": bool(win > first_line),
                                 "line_rank_among_heldout_other_tokens": None})
            fold_other = np.concatenate([score(data[(a, k)]["X"])[~data[(a, k)]["line"]] for k in pairs if env_of[k] == e for a in (TWIN, PLANTED)])
            for q in per_pair[-PER_ENV:]:
                q["line_rank_among_heldout_other_tokens"] = float((fold_other >= q["line_max"]).mean())
        lm_, po, tw = (np.array([q[x] for q in per_pair]) for x in ("line_max", "planted_other_max", "twin_max"))
        y = np.r_[np.ones(len(per_pair)), np.zeros(len(per_pair))].astype(bool)
        run_max = np.r_[np.maximum(lm_, po), tw]
        length = np.r_[[q["planted_tokens"] for q in per_pair], [q["twin_tokens"] for q in per_pair]]
        task = np.array([q["main_task"] for q in per_pair] * 2)
        env = np.array([q["env"] for q in per_pair] * 2)
        auc = roc_auc_score(y, run_max)
        rng, null = np.random.default_rng(0), []
        for _ in range(1000):  # probes fixed: each pair's two labels swapped with probability 1/2 (the fits are not rerun)
            flip = np.tile(rng.random(len(per_pair)) < 0.5, 2)
            null.append(roc_auc_score(np.where(flip, ~y, y), run_max))
        per_env = {e: round(float(roc_auc_score(y[env == e], run_max[env == e])), 4) for e in sorted(set(env))}
        by_C[f"{C:g}"] = {
            "primary": C == C_PRIMARY, "fits": conv, "all_converged": all(f["converged"] for f in conv.values()),
            "line_above_own_run_other_max": int((lm_ > po).sum()), "line_above_twin_max": int((lm_ > tw).sum()),
            "planted_run_max_above_twin_run_max": int((np.maximum(lm_, po) > tw).sum()), "pairs": len(per_pair),
            "run_max_auroc": (ra := auroc_ci(y, run_max, task, 1000, 0)) | {
                "null_p95": round(float(np.percentile(null, 95)), 4), "p": round(float((1 + (np.array(null) >= auc).sum()) / 1001), 4),
                "per_environment": per_env, "mean_over_environments": round(float(np.mean(list(per_env.values()))), 4)},
            # Aswin's rule (10:51 PT 10-10): detected = about the full planted amount (AUROC >= 0.95, unrounded: at least
            # 0.45 of the possible 0.5 shift from the unplanted 0.5) and the 95% range excludes the unplanted 0.5.
            "detected": bool(auc >= 0.95 and ra["ci95"][0] > 0.5) if all(f["converged"] for f in conv.values()) else "incomplete: a fit failed the gradient check",
            "winner_same_output_as_line": int(sum(q["winner_same_output_as_line"] for q in per_pair)),
            "winner_after_line": int(sum(q["winner_after_line"] for q in per_pair)),
            "line_rank_median": float(np.median([q["line_rank_among_heldout_other_tokens"] for q in per_pair])),
            "length_baseline": {"tokens_auroc": round(float(roc_auc_score(y, length)), 4),
                                "spearman_tokens_vs_twin_max": round(float(spearmanr(length[~y], tw).statistic), 4),
                                "planted_longer_than_twin": int((length[y] > length[~y]).sum())},
            "per_pair": per_pair}
    return by_C


def fit_std(X, y, C, device):
    """Standardized L2 logistic regression with the pilot's fitter and gradient check; returns a scorer and convergence."""
    import torch

    from probe_shift import torch_probe as tp

    Xt = torch.from_numpy(X).to(device).double()
    yt = torch.from_numpy(y.astype("float64")).to(device)
    mean, std = Xt.mean(0), Xt.std(0, unbiased=False)
    std = std.where(std > 0, std.new_ones(()))
    Xs = (Xt - mean) / std * C**0.5
    W, b, evals = tp.fit(Xs, yt, 1.0, 5000, tol=1e-4 * min(C**0.5, 1.0))
    Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
    loss = (torch.nn.functional.binary_cross_entropy_with_logits((Xs @ Wg + bg)[:, 0], yt, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xs)
    loss.backward()
    g = float(max(Wg.grad.abs().max(), bg.grad.abs().max()))
    del Xt, Xs

    def score(x):
        return ((((torch.from_numpy(x).to(device).double() - mean) / std * C**0.5) @ W + b)[:, 0]).detach().cpu().numpy()

    return score, {"evals": int(evals), "grad_max": g, "converged": g <= 1e-4 * min(C**0.5, 1.0)}


def views(d):
    """Per transcript: per-output element-wise max and mean of its content tokens, and 4 seeded token rows per output."""
    import numpy as np

    outs = np.unique(d["out"])
    pmax = np.stack([d["X"][d["out"] == o].max(0) for o in outs]).astype(np.float32)
    pmean = np.stack([d["X"][d["out"] == o].astype(np.float32).mean(0) for o in outs]).astype(np.float32)
    return pmax, pmean


def analyze_runlabel(data: dict, pairs: list[str], env_of: dict, device: str) -> dict:
    """Run-label probes for token_max (primary), pooled_max and pooled_mean; per C; refit null at C_PRIMARY."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from probe_shift.scoring import auroc_ci

    V = {key: views(d) for key, d in data.items()}
    samp = {}
    for (arm, k), d in data.items():  # seeded by pair key and output index, never by the arm (the label)
        rows = []
        for o in np.unique(d["out"]):
            ix = np.flatnonzero(d["out"] == o)
            rows += sorted(np.random.default_rng(seed_of(f"diag:tok:{k}:{int(o)}")).choice(ix, min(TOK_PER_OUT, len(ix)), replace=False).tolist())
        samp[(arm, k)] = np.array(rows)
    envs = sorted(set(env_of[k] for k in pairs))
    order = [(PLANTED, k) for k in pairs] + [(TWIN, k) for k in pairs]
    task = np.array([data[key]["main_task"] for key in order])
    env_arr = np.array([env_of[k] for _, k in order])

    def run_scores(lab, C):
        """Held-out run scores per readout for one label assignment lab[(arm, k)] in {0, 1}, folds by environment."""
        S = {r: {} for r in ("token_max", "token_top5", "pooled_max_mean", "pooled_max_max", "pooled_mean_mean", "pooled_mean_max")}
        conv = {}
        for e in envs:
            tr = [(a, k) for (a, k) in order if env_of[k] != e]
            te = [(a, k) for (a, k) in order if env_of[k] == e]
            Xt = np.concatenate([data[key]["X"][samp[key]] for key in tr]).astype(np.float32)
            yt = np.concatenate([np.full(len(samp[key]), lab[key]) for key in tr])
            sc, cv = fit_std(Xt, yt, C, device)
            conv[f"{e}|token"] = cv
            for key in te:
                s = sc(data[key]["X"])
                S["token_max"][key] = float(s.max())
                S["token_top5"][key] = float(np.sort(s)[-5:].mean())
            for j, name in ((0, "pooled_max"), (1, "pooled_mean")):
                Xp = np.concatenate([V[key][j] for key in tr])
                yp = np.concatenate([np.full(len(V[key][j]), lab[key]) for key in tr])
                sc, cv = fit_std(Xp, yp, C, device)
                conv[f"{e}|{name}"] = cv
                for key in te:
                    s = sc(V[key][j])
                    S[f"{name}_mean"][key], S[f"{name}_max"][key] = float(s.mean()), float(s.max())
        return S, conv

    truth = {key: int(key[0] == PLANTED) for key in order}
    y = np.array([truth[key] for key in order]).astype(bool)
    out, raw = {}, {}
    for C in C_GRID:
        S, conv = run_scores(truth, C)
        res = {"primary": C == C_PRIMARY, "all_converged": all(c["converged"] for c in conv.values()), "fits": conv}
        for r, sc in S.items():
            s = np.array([sc[key] for key in order])
            auc = roc_auc_score(y, s)
            raw[(C, r)] = float(auc)
            rng, null = np.random.default_rng(0), []
            for _ in range(N_FIXED):
                flip = np.tile(rng.random(len(pairs)) < 0.5, 2)
                null.append(roc_auc_score(np.where(flip, ~y, y), s))
            per_env = {e: round(float(roc_auc_score(y[env_arr == e], s[env_arr == e])), 4) for e in envs}
            res[r] = auroc_ci(y, s, task, 1000, 0) | {"fixed_null_p95": round(float(np.percentile(null, 95)), 4),
                                                       "per_environment": per_env, "planted_above_twin": int((s[: len(pairs)] > s[len(pairs):]).sum())}
        out[f"{C:g}"] = res
    # refit null at the primary C: each pair's two labels swapped with probability 1/2, in training and test alike
    fit_of = lambda r: "token" if r.startswith("token") else ("pooled_max" if r.startswith("pooled_max") else "pooled_mean")  # noqa: E731
    rng, refit = np.random.default_rng(seed_of("diag:refit")), {r: [] for r in out[f"{C_PRIMARY:g}"] if r not in ("primary", "all_converged", "fits")}
    dropped = {r: 0 for r in refit}
    for i in range(N_REFIT):
        flip = {k: bool(rng.random() < 0.5) for k in pairs}
        lab = {(a, k): (truth[(a, k)] ^ flip[k]) for (a, k) in order}
        S, conv = run_scores(lab, C_PRIMARY)
        yl = np.array([lab[key] for key in order]).astype(bool)
        for r in refit:
            if all(c["converged"] for name, c in conv.items() if name.endswith("|" + fit_of(r))):
                refit[r].append(float(roc_auc_score(yl, np.array([S[r][key] for key in order]))))
            else:  # an unconverged fit: this shuffle does not enter the readout's null
                dropped[r] += 1
        print(f"refit null {i + 1}/{N_REFIT}", flush=True)
    prim = out[f"{C_PRIMARY:g}"]
    for r, nl in refit.items():
        a, ar = prim[r]["auroc"], raw[(C_PRIMARY, r)]
        invalid = dropped[r] > 0.05 * N_REFIT or len(nl) < 2
        prim[r]["refit_null"] = {"shuffles": len(nl), "dropped_unconverged": dropped[r], "invalid": invalid,
                                 "p95": round(float(np.percentile(nl, 95)), 4) if nl else None, "max": round(max(nl), 4) if nl else None,
                                 "p": round((1 + sum(v >= ar for v in nl)) / (1 + len(nl)), 4)}
        if not prim["all_converged"]:
            prim[r]["detected"] = "no verdict: a primary fit failed the gradient check"
        elif invalid:
            prim[r]["detected"] = "no verdict: the refit null is invalid (over 5% of shuffles had an unconverged fit)"
        else:
            prim[r]["detected"] = bool(a >= 0.95 and prim[r]["ci95"][0] > 0.5 and a > prim[r]["refit_null"]["p95"])
    return out


@app.function(image=image, gpu="H100!", cpu=8, memory=98304, volumes={"/hf": hf, "/acts": vol}, timeout=2400, retries=0)
def diag(model: str, revision: str, code: str) -> dict:
    import random
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.reader import _layout, segment_end_positions

    t0 = time.time()
    run_dir = Path("/acts") / OUT / code
    run_dir.mkdir(parents=True, exist_ok=False)  # one run per code version, never overwritten
    sites = sorted(s["key"] for s in json.loads(Path(f"/acts/{STAGED}/manifest.json").read_text())["plant1"]["sites"])
    items = {}
    for f in sorted(Path(f"/acts/{STAGED}/inputs").glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if (r["arm"] == TWIN and r["key"] in sites) or (r["arm"] == PLANTED and r["key"].removesuffix("__plant1") in sites):
                items[(r["arm"], r["key"])] = r
    env_of = {k: items[(TWIN, k)]["environment"] for k in sites}
    pairs = [k for e in sorted(set(env_of.values())) for k in random.Random(f"{SEED}:{e}").sample(sorted(k for k in sites if env_of[k] == e), PER_ENV)]

    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    lm = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    t_load = time.time() - t0

    class Stop(Exception):
        pass

    data = {L: {} for L in LAYERS}
    for k in pairs:
        for arm, key in ((TWIN, k), (PLANTED, f"{k}__plant1")):
            it = items[(arm, key)]
            try:
                r = segment_end_positions(tok, it["messages"], MAX_TOKENS)
                text, _, enc = _layout(tok, it["messages"])
            except ValueError:
                raise ValueError(f"{key}: the reader rejected this transcript") from None
            if enc["input_ids"][: len(r["ids"])] != r["ids"]:
                raise ValueError(f"{key}: tokenizations differ")
            outs = [o for o in r["outputs"] if o["end"] > o["start"]]
            idx = np.concatenate([np.arange(o["start"], o["end"]) for o in outs])
            out_no = np.concatenate([np.full(o["end"] - o["start"], j) for j, o in enumerate(outs)])
            line = np.zeros(len(idx), bool)
            if text.count(PLANT1) != (arm == PLANTED):
                raise ValueError(f"{key}: the line appears {text.count(PLANT1)} times")
            if arm == PLANTED:
                c0 = text.index(PLANT1)
                toks = [i for i, (a, b) in enumerate(enc["offset_mapping"][: len(r["ids"])]) if a < c0 + len(PLANT1) and b > c0]
                line = np.isin(idx, toks)
                if not toks or line.sum() != len(toks) or len(set(out_no[line])) != 1:
                    raise ValueError(f"{key}: line tokens {toks} not inside one output within the cut")
            got, sel = {}, torch.from_numpy(idx).cuda()

            def make_hook(L):
                def hook(module, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    got[L] = h[0].index_select(0, sel).to(torch.float16).cpu().numpy()
                    if L == LAYERS[-1]:
                        raise Stop
                return hook

            handles = [lm.model.layers[L].register_forward_hook(make_hook(L)) for L in LAYERS]
            try:
                with torch.inference_mode():
                    lm.model(input_ids=torch.tensor([r["ids"]], device="cuda"), use_cache=False)
            except Stop:
                pass
            finally:
                for h in handles:
                    h.remove()
            for L in LAYERS:
                data[L][(arm, k)] = {"X": got[L], "line": line, "out": out_no, "env": it["environment"], "main_task": it["main_task"]}
    t_extract = time.time() - t0 - t_load
    del lm
    torch.cuda.empty_cache()
    res = {"pairs": pairs, "layers": {}, "seconds": {"load": round(t_load), "extract": round(t_extract)}}
    for L in LAYERS:
        t1 = time.time()
        res["layers"][str(L)] = {"aimed": analyze_aimed(data[L], pairs, env_of, "cuda"), "run_label": analyze_runlabel(data[L], pairs, env_of, "cuda"),
                                 "seconds": round(time.time() - t1)}
        (run_dir / "result_partial.json").write_text(json.dumps(res, default=str))  # numbers only, after each layer
        vol.commit()
    res["seconds"]["total"] = round(time.time() - t0)
    (run_dir / "result.json").write_text(json.dumps(res, default=str))
    vol.commit()
    return res


@app.local_entrypoint()
def main(collect: str = ""):
    import hashlib
    import subprocess

    HERE = Path(__file__).parent
    from probe_shift.models import QWEN3_32B

    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    if collect:
        git, res = collect, json.loads(b"".join(vol.read_file(f"{OUT}/{code}/result.json")))
    else:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
            raise SystemExit(f"commit {Path(__file__).name} first: the result records the commit that ran")
        res = diag.remote(QWEN3_32B["name"], QWEN3_32B["revision"], code)
    from zoneinfo import ZoneInfo  # local entrypoint only (Modal images lack tzdata)

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / f"plant1_diag_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps({"git": git, "code": code, "layers": LAYERS, "C_grid": C_GRID, "C_primary": C_PRIMARY, "n_refit": N_REFIT} | res, indent=1, default=str))
    for L, r in res["layers"].items():
        a = r["aimed"][f"{C_PRIMARY:g}"]
        rl = r["run_label"][f"{C_PRIMARY:g}"]
        print(f"L{L}: aimed run-max AUROC {a['run_max_auroc']['auroc']} ({a['planted_run_max_above_twin_run_max']}/20 pairs), detected {a['detected']}; "
              + "; ".join(f"{k} {rl[k]['auroc']} (detected {rl[k].get('detected', '-')})" for k in ("token_max", "pooled_max_mean", "pooled_max_max", "pooled_mean_mean")))
    print("seconds", res["seconds"])
