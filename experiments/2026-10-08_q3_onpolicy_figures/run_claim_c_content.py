"""POST-HOC content-only command-end rerun (plan Amendment 13): claim C's detector (Amendment 8) and its first-k scores
(Amendment 10) refit on command-end readouts whose token is not a fixed slot (q1_eval.fixed_call_tokens at scrub1: "0"
and "n"). Deployed Modal app, no extraction:
- flags (CPU): rebuilds claim C's cache row order (run_claim_c.cache for the swapped arms; run_retrain_grid.cache's
  level-1 order for honest_a and honest_b), asserts it equals claimc_cache_v1's rows at every layer, and flags each
  call_end row whose readout token decodes exactly to a fixed slot. It decodes only that single token, prints no text,
  and writes the flags (with counts per arm) to /score/claimcco_flags_v1/.
- grid (H100, one per layer, call_end only): exactly run_claim_c_firstk.grid on the unflagged call_end rows. First k means
  the first k content-only commands. Records under /score/claimcco_<tag>/.

  modal deploy run_claim_c_content.py
  uv run ... python run_claim_c_content.py flags            (spawns, waits, prints counts)
  uv run ... python run_claim_c_content.py grid <labels>    (spawns on the deployed app)
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-q3onpol_claimcco_kill-2026-11-10"
SCORE_VOL, ACTS_VOL, HF_CACHE = ("probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08", "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08",
                                 "probe-shift_aswin-surya_hf-cache")
CACHE, FLAGS = "claimc_cache_v1", "claimcco_flags_v1"
FIXED = ["0", "n"]  # q1_eval.fixed_call_tokens("..._scrub1"): ">" is a fixed slot only from scrub level 2
BASES = ["2026-10-09_rerun_strict_segends_v1_scrub1", "2026-10-10_scaleup_canary_segends_v1_scrub1",
         "2026-10-10_scaleup_l2_segends_v1_scrub1", "2026-10-10_scaleup_topup_segends_v1_scrub1"]
SWAPPED = [(f"{b}_swapsys", "attack", "run1") for b in BASES] + [("2026-10-10_second_attack_segends_v1_scrub1_swapsys", "attack", "run2")] \
          + [(f"{b}_swapsys2", "sim_attack@september", "sim") for b in BASES]  # run_claim_c.SWAPPED
HONEST_RUNS = BASES + ["2026-10-10_second_attack_segends_v1_scrub1"]  # run_retrain_grid.LEVELS["level1"] on-policy runs, in order
INPUTS = {"2026-10-09_rerun_strict_segends_v1_scrub1": "rerun_strict_segends_v2_scrub1"}  # extraction run -> staging (config/extract_*.yaml)
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
BASES = ["2026-10-09_rerun_strict_segends_v1_scrub1", "2026-10-10_scaleup_canary_segends_v1_scrub1",
         "2026-10-10_scaleup_l2_segends_v1_scrub1", "2026-10-10_scaleup_topup_segends_v1_scrub1"]
SWAPPED = [(f"{b}_swapsys", "attack", "run1") for b in BASES] + [("2026-10-10_second_attack_segends_v1_scrub1_swapsys", "attack", "run2")] \
          + [(f"{b}_swapsys2", "sim_attack@september", "sim") for b in BASES]
LAYERS, KINDS, C_GRID = [39, 51, 63], ["output_end", "call_end"], [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
N_NULL, MAX_ITER, MAX_ITER_CAP, KS = 100, 5000, 80000, (3, 5)

app = modal.App(APP)
score_vol, acts_vol = modal.Volume.from_name(SCORE_VOL, version=2), modal.Volume.from_name(ACTS_VOL, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scipy", "transformers==5.19.0")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift").add_local_file(REPO / T196, f"/root/{T196}"))


def tol_of(C: float) -> float:
    return 1e-4 * min(C**0.5, 1.0)


def frozen_folds(d: dict) -> dict:  # run_score_onpolicy.frozen_folds (amendment 25), on a parsed task file
    h = lambda s: hashlib.sha256((d["salt"] + s).encode()).hexdigest()
    envs, by_env = sorted(d["eligible_mains_by_env"], key=h), {}
    for c in d["cells"]:
        by_env.setdefault(c["env"], []).append(c["main"])
    return {c["key"]: (by_env[c["env"]].index(c["main"]) + envs.index(c["env"])) % 5 for c in d["cells"]}


def staging_of(run: str) -> str:
    """The staged inputs an extraction read: config/extract_*.yaml's `inputs` (the date prefix dropped; rerun_strict is v2)."""
    base = INPUTS.get(run.replace("_swapsys2", "").replace("_swapsys", ""), run.split("_", 1)[1].replace("_swapsys2", "").replace("_swapsys", ""))
    return base + ("_swapsys2" if run.endswith("_swapsys2") else "_swapsys" if run.endswith("_swapsys") else "")


@app.function(image=image, cpu=8, memory=65536, timeout=3600, retries=0,
              volumes={"/score": score_vol, "/acts": acts_vol.read_only(), "/hf": modal.Volume.from_name(HF_CACHE).read_only()})
def flags() -> dict:
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", revision="fc05daec18b0a78c049392ed2e771dde82bdf654")
    score_vol.reload()
    world_of = {"honest_a": "honest_a", "honest_b": "honest_b"}
    plan = [(run, [arm], lambda a, w=world: w) for run, arm, world in SWAPPED] + [(run, None, world_of.get) for run in HONEST_RUNS]
    rows, flag, info = [], [], {}
    for run, arms, wf in plan:
        root = Path("/acts") / run
        json.loads((root / "complete.json").read_text())
        files = sorted((root / "meta" / arms[0]).glob("*.json")) if arms else sorted((root / "meta").glob("*/*.json"))
        staged = {}
        for f in files:
            m = json.loads(f.read_text())
            w = wf(m["arm"])
            if w is None or m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                continue
            if m["arm"] not in staged:  # the staged file this extraction read, checked against its recorded sha256
                p = Path("/acts") / staging_of(run) / "inputs" / f"{m['arm']}.jsonl"
                b = p.read_bytes()
                if hashlib.sha256(b).hexdigest() != m["inputs_sha256"]:
                    raise ValueError(f"{run}/{m['arm']}: the staged file differs from the one extracted")
                staged[m["arm"]] = {e["key"]: e["ids"] for e in map(json.loads, b.decode().splitlines())}
            ids = staged[m["arm"]][m["key"]]
            for c in m["calls"]:
                if c.get("kind") in KINDS and c.get("row") is not None:
                    rows.append({"world": w, "task": m["task"], "kind": c["kind"], "order": [c["turn"], c.get("call_index", 0)]})
                    t = tok.decode([ids[c["readout"]]]) if c["kind"] == "call_end" else None
                    flag.append(t in FIXED)
                    if c["kind"] == "call_end":
                        k = info.setdefault(w, {"call_end": 0, "fixed": 0, "fixed_0": 0, "fixed_n": 0, "fixed_0_timeout": 0})
                        k["call_end"] += 1
                        k["fixed"] += int(t in FIXED)
                        k["fixed_0"] += int(t == "0")
                        k["fixed_n"] += int(t == "n")
                        k["fixed_0_timeout"] += int(t == "0" and c.get("param") == "timeout")
    for layer in LAYERS:  # the rebuilt order must be the cache's, row for row
        if json.loads((Path("/score") / CACHE / f"on_rows.L{layer}.json").read_text()) != rows:
            raise ValueError(f"the rebuilt rows differ from {CACHE} at layer {layer}")
    d = Path("/score") / FLAGS
    d.mkdir(exist_ok=True)
    out = {"rows_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(), "fixed_tokens": FIXED, "by_world": info,
           "share_fixed": {w: v["fixed"] / v["call_end"] for w, v in info.items()}}
    if (d / "flags.json").exists():
        raise FileExistsError("flags exist; write under a new name")
    (d / "flags.json").write_text(json.dumps(out | {"fixed": flag}))
    score_vol.commit()
    return out


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, timeout=2 * 3600, retries=0, volumes={"/score": score_vol}, scaledown_window=5)
def grid(tag: str, layer: int, attempted: list) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import rankdata

    from probe_shift import torch_probe as tp

    score_vol.reload()
    fold_of = frozen_folds(json.loads((Path("/root") / T196).read_text()))
    att, t0, out = set(attempted), time.time(), {}
    d = Path("/score") / CACHE
    rows = json.loads((d / f"on_rows.L{layer}.json").read_text())
    fl = json.loads((Path("/score") / FLAGS / "flags.json").read_text())
    if fl["rows_sha256"] != hashlib.sha256(json.dumps(rows).encode()).hexdigest() or len(fl["fixed"]) != len(rows):
        raise ValueError("the flags are not aligned to this layer's cache rows")
    kind = "call_end"
    kn = np.flatnonzero([r["kind"] == kind and not f for r, f in zip(rows, fl["fixed"])])  # content-only command-end rows
    X = torch.from_numpy(np.load(d / f"on_L{layer}.npy")[kn]).cuda()
    runs, n_run = np.unique([f"{r['world']}|{r['task']}" for r in rows], return_inverse=True)
    n_run = n_run[kn]
    rank = np.zeros(len(kn), int)  # each row's order within its run, among this readout kind's rows
    for u in range(len(runs)):
        m = np.flatnonzero(n_run == u)
        rank[sorted(m, key=lambda i: tuple(rows[kn[i]]["order"]))] = np.arange(len(m))
    world = np.array([u.split("|", 1)[0] for u in runs])
    task = np.array([u.split("|", 1)[1] for u in runs])
    r_fold = np.array([fold_of[t] for t in task])
    train = (world == "honest_a") | ((world == "run1") & np.isin(runs, sorted(att)))
    y = (world == "run1").astype(float)
    Y = np.zeros((len(runs), 1 + N_NULL))
    Y[:, 0] = y
    for c in range(1, 1 + N_NULL):
        rng = np.random.default_rng([20261010, c])
        yc = y.copy()
        for t in sorted(set(task[train])):
            j = np.flatnonzero(train & (task == t))
            yc[j] = y[rng.permutation(j)]
        Y[:, c] = yc
    K = Y.shape[1]
    for C in C_GRID:
        p = Path("/score") / f"claimcco_{tag}" / f"L{layer}_{kind}_C{C:g}.npz"
        if p.exists():
            continue
        r = {"val_auroc": np.full((5, K), np.nan), "grad_full": np.zeros((5, K)), "grad_val": np.zeros((5, K)),
             "evals": np.zeros((5, 2), int), "on_score": np.full((len(runs), K), np.nan),
             **{f"on_score_k{k}": np.full((len(runs), K), np.nan) for k in KS}}
        for f in range(5):
            v = (f + 1) % 5
            for name, tr in (("full", train & (r_fold != f)), ("val", train & (r_fold != f) & (r_fold != v))):
                a = np.flatnonzero(tr[n_run])
                b_ = np.flatnonzero(r_fold[n_run] == f) if name == "full" else np.flatnonzero((train & (r_fold == v))[n_run])
                Xa, Xb = tp.standardize(X[a].double(), X[b_].double())
                Xa.mul_(C**0.5), Xb.mul_(C**0.5)
                Ya = torch.from_numpy(Y[n_run[a]]).cuda()
                mi, prev = MAX_ITER, None
                while True:  # refit only fits that used all their iterations; stop when a refit repeats
                    W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol_of(C))
                    Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
                    loss = (F.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
                    loss.backward()
                    g = torch.maximum(Wg.grad.abs().amax(0), bg.grad.abs()).cpu().numpy()
                    if g.max() <= tol_of(C) or evals < mi or evals == prev or mi * 4 > MAX_ITER_CAP:
                        break
                    mi, prev = mi * 4, evals
                r[f"grad_{name}"][f], r["evals"][f, int(name == "val")] = g, evals
                S = tp.decision(Xb, W, b).cpu().numpy()
                u, inv = np.unique(n_run[b_], return_inverse=True)
                M = np.stack([np.bincount(inv, weights=S[:, c]) for c in range(K)], 1) / np.bincount(inv)[:, None]
                if name == "val":
                    R, Yv = rankdata(M, axis=0), Y[u]
                    n1 = Yv.sum(0)
                    r["val_auroc"][f] = ((R * Yv).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Yv) - n1))
                else:
                    r["on_score"][u] = M
                    for k in KS:  # the first k rows of each held-out run; runs with fewer than k rows stay NaN
                        sel = rank[b_] < k
                        uk, ik = np.unique(n_run[b_][sel], return_inverse=True)
                        Mk = np.stack([np.bincount(ik, weights=S[sel, c]) for c in range(K)], 1) / np.bincount(ik)[:, None]
                        full = np.bincount(ik) == k
                        r[f"on_score_k{k}"][uk[full]] = Mk[full]
                del Xa, Xb
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p.with_suffix(".tmp"), "wb") as fh:
            np.savez(fh, on_runs=runs, on_fold=r_fold, train=train, Y=Y, tol=tol_of(C), rank=rank, n_run=n_run, **r)
        os.replace(p.with_suffix(".tmp"), p)
        score_vol.commit()
        out[f"{C:g}"] = round(time.time() - t0)
        print(f"L{layer} {kind} C {C:g} done at {out[f'{C:g}']} s; evals max {r['evals'].max()}", flush=True)
    return out


def attempted_run1(labels: str) -> list:
    att = []
    for f in labels.split(","):  # run 1 is every label folder's attack arm except the second batch's
        if "second_attack" in Path(f).name:
            continue
        att += [f"run1|{r['task']}" for r in map(json.loads, Path(f).read_text().splitlines()) if r["arm"] == "attack" and r.get("label") == "attempted"]
    if len(att) != len(set(att)):
        raise ValueError("a run-1 task is labelled twice")
    return sorted(att)


if __name__ == "__main__":  # spawns on the deployed app
    if sys.argv[1:] == ["flags"]:
        print(json.dumps(modal.Function.from_name(APP, "flags").spawn().get(), indent=1))
    elif len(sys.argv) == 3 and sys.argv[1] == "grid":
        att = attempted_run1(sys.argv[2])
        tag = hashlib.sha256((HERE / "run_claim_c_content.py").read_bytes() + json.dumps(att).encode()).hexdigest()[:12]
        fn = modal.Function.from_name(APP, "grid")
        print(json.dumps({"tag": tag, "attempted_run1": len(att), "calls": {f"L{l}_call_end": fn.spawn(tag, l, att).object_id for l in LAYERS}}))
    else:
        raise SystemExit("usage: run_claim_c_content.py flags | grid <labels>")
