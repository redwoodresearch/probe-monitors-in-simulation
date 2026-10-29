"""Option (a), PLAN.md addendum 1 (EXPLORATORY; frozen choices; no null; numbers only). One Modal CPU container.

Sets: attribution D (100 fixed-sim test runs vs their real e1 twins) and E (50 real e1 runs, planted R2-undone vs
unplanted), from their stored per-token projections (`claimB/attribution_tokens_v1/{d,e}`); and the token pilot's 20
check-1 pairs (planted copy vs twin), from its L63 activations, projected here through the frozen realism probe. That
probe is the meaning control's refit of step A's fixed-vs-e1 comparison (C 0.001, the pair's environment held out),
re-created on CPU by `run_attribution_tokens_modal.refit_probes`'s fit (copied; no selection).
Gates (any failure raises): (1) the re-created L63 probe scores the D runs' pack output_end rows within 0.05 of D's
stored projections at the end tokens, correlation >= 0.9995, per environment; (2) the reader's non-empty clipped output
spans reproduce each pilot transcript's saved row count and (step, action) list; (3) pilot projections at each kept
output's last row match the pack's output_end scores within 0.05.
Readings per set (L63; D and E also at step A's layer): output-end mean; w = 10 max-of-rolling-means within outputs;
offsets j in {0, 1, 2, 4, 8, 16} before each output's last content token (all eligible outputs, and the common outputs
with >= 17 tokens). Only outputs whose output_end token is inside the 32k cut. AUROC (pooled, per environment), paired
win rate, 95% intervals from 2,000 main-task resamples (default_rng(20261011)). Result to the Volume first.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal run --detach \\
      experiments/2026-10-11_claimB_maxwin_rescore/run_a_subset_modal.py [--collect]
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else HERE  # /root in the container has one parent
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
# the meaning control's constants (2026-10-10_claimB_meaning_control/run_meaning_control_modal.py:41-44)
PACK, C, REAL, SIM = "claimB/level1/packedB_2cfd1aec393c", 1e-3, "real-attack-e1", "sim-attack-fixed"
A_LAYER = {"clinical_trial": 58, "media_processing": 58, "tiktok": 58, "shared_dotfiles": 63, "user_tracking": 63}
ATTR, MAX_TOKENS = "claimB/attribution_tokens_v1", 32768
# the token pilot's constants (2026-10-09_fixed_sim/run_plant_token_pilot_modal.py)
STAGED, PILOT, PLANTED = "2026-10-10_la_fixed-sim-v2_probe-inputs-l1_v1", "2026-10-10_la_fixed-sim-v2_plant1-token-pilot_qwen3-32b_L63_v1", "sim-attack-fixed-plant1"
W, OFFSETS, COMMON, DRAWS, SEED, TOL = 10, (0, 1, 2, 4, 8, 16), 17, 2000, 20261011, 0.05
OUT = f"claimB/maxwin_a_v1/{SCRIPT_SHA[:12]}"

app = modal.App("probe-shift_aswin-surya_sub-claimb-maxwin-a_kill-2026-11-11")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
hf = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.8.0", "transformers==4.56.2", "numpy", "scikit-learn==1.7.2")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))


def refit(XP, t, pruns, env_to_fold, e):
    """refit_probes' fit for one held-out environment at L63, on CPU (same data, standardization, C, iteration rule)."""
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp

    arm_of = np.array([pruns[i]["arm"] for i in t["run"]])
    efold = np.array([env_to_fold[pruns[i]["environment"]] for i in t["run"]])
    tr = np.isin(arm_of, [REAL, SIM]) & (efold != env_to_fold[e])
    Xa = torch.from_numpy(XP[tr]).double()
    Ya = torch.from_numpy((arm_of[tr] == SIM).astype(float)[:, None]).double()
    mean, std = Xa.mean(0), Xa.std(0, unbiased=False)
    std = std.where(std > 0, std.new_ones(()))
    Xa = (Xa - mean) / std * C**0.5
    tol, mi, prev = 1e-4 * min(C**0.5, 1.0), 5000, None
    while True:
        Wt, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol)
        if evals < mi or evals == prev or mi * 4 > 80000:
            break
        mi, prev = mi * 4, evals
    Wg, bg = Wt.clone().requires_grad_(True), b.clone().requires_grad_(True)
    loss = (torch.nn.functional.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
    loss.backward()
    grad = float(max(Wg.grad.abs().max(), bg.grad.abs().max()))
    if grad > tol:
        raise ValueError(f"{e}: the refit did not converge (gradient {grad:.2e})")
    m, s = mean.numpy(), std.numpy()
    w, b0 = Wt.detach().numpy()[:, 0], float(b.detach().numpy()[0])
    return lambda X: ((np.asarray(X, np.float64) - m) / s * C**0.5) @ w + b0


def summaries(outs: list) -> dict:
    """Run scores from a run's kept outputs (each an array of its content tokens' projections, in order)."""
    import numpy as np

    r = {"out_end_mean": float(np.mean([o[-1] for o in outs])) if outs else None}
    wins = [np.convolve(o, np.ones(W) / W, "valid").max() if len(o) >= W else o.mean() for o in outs]
    r["w10_max"] = float(max(wins)) if wins else None
    long = [o for o in outs if len(o) >= COMMON]
    for j in OFFSETS:
        el = [o[-1 - j] for o in outs if len(o) >= j + 1]
        r[f"off{j}"] = float(np.mean(el)) if el else None
        r[f"off{j}_common"] = float(np.mean([o[-1 - j] for o in long])) if long else None
    return r


def read_set(pairs: list, rng) -> dict:
    """pairs: (main_task, environment, pos_scores, neg_scores). AUROC (pooled, per environment), win rate, intervals."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    out = {}
    names = list(pairs[0][2])
    common = [p for p in pairs if all(p[2][k] is not None and p[3][k] is not None for k in names)]
    for name in names:
        for label, use in (("eligible", [p for p in pairs if p[2][name] is not None and p[3][name] is not None]), ("common", common)):
            if len(use) < 2:
                out[f"{name}|{label}"] = {"pairs": len(use)}
                continue
            tasks = sorted({p[0] for p in use})
            pos, neg = np.array([p[2][name] for p in use]), np.array([p[3][name] for p in use])
            task_of = np.array([tasks.index(p[0]) for p in use])

            def stats(ix, pos=pos, neg=neg):
                y = np.r_[np.ones(len(ix)), np.zeros(len(ix))]
                t = np.sort(neg[ix])[::-1][int(np.floor(0.01 * len(ix)))]  # halves' thr(): analyze_halves.py:55-61 @ 05dd923d
                return (roc_auc_score(y, np.r_[pos[ix], neg[ix]]), float(np.mean((pos[ix] > neg[ix]) + 0.5 * (pos[ix] == neg[ix]))),
                        float((pos[ix] > t).mean()), float((neg[ix] > t).mean()))

            auc, win, catch, fa = stats(np.arange(len(use)))
            boots = []
            for _ in range(DRAWS):
                pick = rng.integers(0, len(tasks), len(tasks))
                ix = np.concatenate([np.flatnonzero(task_of == k) for k in pick])
                boots.append(stats(ix))
            bo = np.array(boots)
            per_env = {e: roc_auc_score(np.r_[np.ones(n), np.zeros(n)], np.r_[pos[m], neg[m]])
                       for e in sorted({p[1] for p in use}) for m in [np.array([p[1] == e for p in use])] for n in [int(m.sum())] if n >= 2}
            out[f"{name}|{label}"] = {"pairs": len(use), "main_tasks": len(tasks), "auroc": auc, "auroc_ci95": np.percentile(bo[:, 0], [2.5, 97.5]).tolist(),
                                      "win_rate": win, "win_ci95": np.percentile(bo[:, 1], [2.5, 97.5]).tolist(), "catch_1pct": catch,
                                      "catch_1pct_ci95": np.percentile(bo[:, 2], [2.5, 97.5]).tolist(), "achieved_fa": fa, "per_env_auroc": per_env,
                                      "per_env_mean": float(np.mean(list(per_env.values()))) if per_env else None}
    return out


@app.function(image=image, cpu=4, memory=16384, volumes={"/acts": vol, "/hf": hf}, timeout=30 * 60, retries=0)
def run(env_to_fold: dict, git: str) -> dict:
    import time

    import numpy as np
    from transformers import AutoTokenizer

    from probe_shift.q1_eval import step_table
    from probe_shift.reader import segment_end_positions

    t0, acts = time.time(), Path("/acts")
    res = {"label": "option (a), PLAN.md addendum 1: EXPLORATORY, frozen choices, no null (numbers only)", "git": git, "script_sha256": SCRIPT_SHA}
    pruns = json.loads((acts / PACK / "runs.L63.json").read_text())
    t = step_table(pruns, "output_end")
    XP = np.load(acts / PACK / "L63_output_end.npy")
    if len(XP) != len(t["run"]):  # refit_probes indexes the packed rows in step-table order
        raise ValueError("pack rows and step table differ in length")
    row_of = {(pruns[i]["arm"], pruns[i]["key"], int(s), int(a)): r for r, (i, s, a) in enumerate(zip(t["run"], t["step"], t["action"]))}

    def load(sub):
        got = []
        for f in sorted((acts / ATTR / sub).glob("*.npz")):
            if f.name.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            got.append((json.loads(str(z["meta"])), {k: z[k] for k in z.files if k != "meta"}))
        return got

    D, E = load("d"), load("e")
    cells = json.loads((acts / ATTR / "cells.json").read_text())
    if len({json.dumps(m["provenance"], sort_keys=True) for m, _ in D + E}) != 1:
        raise ValueError("D and E files with different provenance")
    if sorted((m["cell"], m["arm"]) for m, _ in D) != sorted((c, a) for c in cells["D"] for a in (SIM, REAL)):
        raise ValueError("D files do not match cells.json exactly")
    if sorted((m["cell"], m["version"]) for m, _ in E) != sorted((c, v) for c in cells["E"] for v in ("planted", "unplanted")):
        raise ValueError("E files do not match cells.json exactly")
    plan = json.loads((acts / PILOT / "meta.json").read_text())
    pilot_envs = set()
    staged = {}
    for f in sorted((acts / STAGED / "inputs").glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if (r["arm"] == SIM and r["key"] in plan["pairs"]) or (r["arm"] == PLANTED and r["key"].removesuffix("__plant1") in plan["pairs"]):
                staged[(r["arm"], r["key"])] = r
                pilot_envs.add(r["environment"])
    probes = {e: refit(XP, t, pruns, env_to_fold, e) for e in sorted(pilot_envs | {m["environment"] for m, _ in D})}
    res["seconds_refits"] = round(time.time() - t0)

    gate1 = {}
    for e in sorted({m["environment"] for m, _ in D}):
        mine, stored = [], []
        for m, z in D:
            if m["environment"] != e:
                continue
            for s in m["spans"]:
                if s["kind"] == "output" and s["kept"]:
                    mine.append(probes[e](XP[row_of[(m["arm"], m["key"], s["step"], s["action"])]][None])[0])
                    stored.append(float(z["proj_L63"][s["end_token"]]))
        mine, stored = np.array(mine), np.array(stored)
        gate1[e] = {"outputs": len(mine), "max_abs_diff": float(np.abs(mine - stored).max()), "corr": float(np.corrcoef(mine, stored)[0, 1])}
        if gate1[e]["max_abs_diff"] > TOL or gate1[e]["corr"] < 0.9995:
            raise ValueError(f"gate 1 failed for {e}: {gate1[e]}")
    res["gate1"] = gate1

    tok = AutoTokenizer.from_pretrained(plan["model"], revision=plan["revision"])
    pilot_outs, gate3 = {}, {"outputs": {}, "unmatched": {}, "max_abs_diff": 0.0}
    for k in plan["pairs"]:
        for arm, key in ((SIM, k), (PLANTED, f"{k}__plant1")):
            it, saved = staged[(arm, key)], plan["transcripts"][f"{arm}__{k}"]
            r = segment_end_positions(tok, it["messages"], MAX_TOKENS)
            outs = [o for o in r["outputs"] if o["end"] > o["start"]]
            if sum(o["end"] - o["start"] for o in outs) != saved["rows"] or [[o["step"], o["action"]] for o in outs] != saved["outputs"]:
                raise ValueError(f"gate 2 failed for {arm} {key[:40]}")
            kept = {(p["step"], p["action"]) for p in r["positions"] if p["kind"] == "output_end" and p["kept"]}
            X = np.load(acts / PILOT / f"{arm}__{k[:150]}.npy")
            proj = probes[it["environment"]](X)
            runs, at = [], 0
            for o in outs:
                seg, at = proj[at: at + o["end"] - o["start"]], at + o["end"] - o["start"]
                if (o["step"], o["action"]) in kept:
                    runs.append(seg)
                    row = row_of.get((arm, key, o["step"], o["action"]))
                    if row is None:  # counted; any unmatched output fails gate 3
                        gate3["unmatched"][arm] = gate3["unmatched"].get(arm, 0) + 1
                        continue
                    d = abs(seg[-1] - probes[it["environment"]](XP[row][None])[0])
                    gate3["outputs"][arm] = gate3["outputs"].get(arm, 0) + 1
                    gate3["max_abs_diff"] = max(gate3["max_abs_diff"], float(d))
            pilot_outs[(arm, k)] = (it["main_task"], it["environment"], runs)
    if not gate3["outputs"].get(SIM) or not gate3["outputs"].get(PLANTED) or gate3["unmatched"] or gate3["max_abs_diff"] > TOL:
        raise ValueError(f"gate 3 failed: {gate3}")
    res["gate2"], res["gate3"] = "passed (40 transcripts)", gate3

    def outs_of(m, z, layer):
        p = z[f"proj_L{layer}"]
        return [p[s["span"][0]: s["span"][1]].astype(np.float64) for s in m["spans"] if s["kind"] == "output" and s["kept"] and s["span"][1] > s["span"][0]]

    rng = np.random.default_rng(SEED)
    raw = {}  # set -> [(main_task, environment, positive run's outputs, negative run's outputs)]
    dd = {(m["cell"], m["arm"]): (m, z) for m, z in D}
    ee = {(m["cell"], m["version"]): (m, z) for m, z in E}
    for lname in ("L63", "Lsel"):
        lay = (lambda m: 63) if lname == "L63" else (lambda m: A_LAYER[m["environment"]])
        raw[f"D|{lname}"] = [(dd[(c, SIM)][0]["main_task"], dd[(c, SIM)][0]["environment"], outs_of(*dd[(c, SIM)], lay(dd[(c, SIM)][0])),
                              outs_of(*dd[(c, REAL)], lay(dd[(c, REAL)][0]))) for c in cells["D"]]
        raw[f"E|{lname}"] = [(ee[(c, "planted")][0]["main_task"], ee[(c, "planted")][0]["environment"], outs_of(*ee[(c, "planted")], lay(ee[(c, "planted")][0])),
                              outs_of(*ee[(c, "unplanted")], lay(ee[(c, "unplanted")][0]))) for c in cells["E"]]
    raw["check1|L63"] = [(*pilot_outs[(PLANTED, k)], pilot_outs[(SIM, k)][2]) for k in plan["pairs"]]
    res["counts"] = {}
    for name, rows in raw.items():
        outs = [o for r in rows for o in r[2] + r[3]]
        res["counts"][name] = {"pairs": len(rows), "outputs": len(outs), f"outputs_ge_{COMMON}": sum(len(o) >= COMMON for o in outs),
                               "outputs_skipped_by_offset": {str(j): sum(len(o) < j + 1 for o in outs) for j in OFFSETS},
                               "outputs_shorter_than_w": sum(len(o) < W for o in outs)}
    res["readings"] = {name: read_set([(mt, e, summaries(po), summaries(ne)) for mt, e, po, ne in rows], rng) for name, rows in raw.items()}
    c1 = res["readings"]["check1|L63"]
    a, b = c1["w10_max|eligible"], c1["out_end_mean|eligible"]
    res["trigger_b"] = {"w10_minus_out_end_auroc": a["auroc"] - b["auroc"], "w10_win": a["win_rate"], "out_end_win": b["win_rate"],
                        "fires": bool(a["auroc"] - b["auroc"] >= 0.05 - 1e-9 and a["win_rate"] >= b["win_rate"] - 1e-9)}
    res["seconds_total"] = round(time.time() - t0)
    (acts / OUT).mkdir(parents=True, exist_ok=True)
    (acts / OUT / "result.json").write_text(json.dumps(res, indent=1))
    vol.commit()
    return res


@app.local_entrypoint()
def main(collect: bool = False):
    if collect:
        res = json.loads(b"".join(vol.read_file(f"{OUT}/result.json")))
    else:
        dirty = subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
        if dirty:
            raise SystemExit("commit this script first: the result records the commit that ran")
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
        folds = json.loads((REPO / "splits" / "folds_v2.json").read_text())["environment_folds"]["environment_to_fold"]
        res = run.remote(folds, git)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"maxwin_a_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("gate1", "gate2", "gate3", "counts", "trigger_b", "seconds_refits", "seconds_total")}, indent=1))
    for name, rd in res["readings"].items():
        for k, v in rd.items():
            if "auroc" in v:
                print(f"{name:12s} {k:24s} pairs {v['pairs']:3d} tasks {v['main_tasks']:3d} AUROC {v['auroc']:.3f} [{v['auroc_ci95'][0]:.3f}, {v['auroc_ci95'][1]:.3f}]"
                      f" win {v['win_rate']:.3f} [{v['win_ci95'][0]:.3f}, {v['win_ci95'][1]:.3f}] catch1% {v['catch_1pct']:.3f}"
                      f" [{v['catch_1pct_ci95'][0]:.3f}, {v['catch_1pct_ci95'][1]:.3f}] fa {v['achieved_fa']:.3f} per-env mean {v['per_env_mean']}")
    print("->", dst)
