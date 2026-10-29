"""Planted check 1, a closer look (exploratory; Aswin about 21:30 PT 10-10 via 0d: "shouldn't it be higher, why isn't it
working? how do the activations look?"; under his $5 rule). One Modal H100, no forward pass: everything reads stored
layer-63 activations of Qwen3-32B.

Two probes, each refit once per held-out test environment (the environment's own fold), at layer 63:
  real_vs_sim  step A's comparison (fixed sim vs real e1, run labels) refit at layer 63 with C = 0.001 (A chose C 0.001 in
               every fold; layer 63 in 6 of 10 folds, 58 in the others) on the output_end rows of step B's level-1 pack
               (claimB/level1/packedB_2cfd1aec393c, A's rows), the grid's standardization, fitter and refit loop;
  aimed        the token pilot's probe (run_plant_token_pilot_modal.py, C 0.001): the planted line's tokens against 20,000
               seeded other tokens, trained on the pilot's pairs from the other environments.
Outputs (numbers only; no text, no weights):
  traces       for 5 of the pilot's 20 pairs (seed "plant-trace-2026-10-10"), both probes' scores at every content token
               of the planted output and the 3 outputs on each side, for the planted copy and its twin, with output ends
               and the line's tokens marked (token rows re-derived with the reader, checked against the pilot's meta);
  sites        for all 150 planted check 1 pairs, real_vs_sim at the planted output's end-of-output readout (the output
               holding the line, found by tokenizing the planted copy as the pilot did; checked: every earlier output's row
               equals the twin's) and the run score
               (mean over outputs), planted minus twin, against the spread of real e1 run and output scores in the same
               environment;
  cosines      between the two probes' directions in activation space (w / std), per environment, with references:
               real_vs_sim across environments, and 1/sqrt(d) for unrelated directions.
Writes the result to the Volume first (claimB/plant_trace_<git8>.json), then results/plant_trace_<stamp>.json.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal run --detach \\
      experiments/2026-10-09_fixed_sim/run_plant_trace_modal.py [--collect <git8>]
"""

import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
sys.path.insert(0, str(HERE) if modal.is_local() else "/root")
import run_plant_token_pilot_modal as P  # noqa: E402  (the pilot's constants and recipe)

APP = "probe-shift_aswin-surya_sub-claimb_plant-trace_kill-2026-11-10"
PACK, LAYER, C = "claimB/level1/packedB_2cfd1aec393c", 63, 1e-3
REAL, SIM = "real-attack-e1", "sim-attack-fixed"
SHOW_SEED, N_SHOW, CONTEXT = "plant-trace-2026-10-10", 5, 3
MAX_ITER, MAX_ITER_CAP = 5000, 80000

app = modal.App(APP)
image = P.image.add_local_file(HERE / "run_plant_token_pilot_modal.py", "/root/run_plant_token_pilot_modal.py")


def tol_of(c: float) -> float:
    return 1e-4 * min(c**0.5, 1.0)


@app.function(image=image, gpu="H100!", cpu=8, memory=131072, volumes={"/acts": P.vol, "/hf": P.hf}, timeout=3600, retries=0)
def trace(env_to_fold: dict, git: str) -> dict:
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from probe_shift import torch_probe as tp
    from probe_shift.q1_eval import step_table
    from probe_shift.reader import _layout, segment_end_positions

    dev = "cuda"
    run = Path("/acts") / P.RUN
    meta = json.loads((run / "meta.json").read_text())
    if meta["staged"] != P.STAGED or meta["layer"] != LAYER:
        raise ValueError("the pilot dump is not the expected staged run or layer")
    pairs = meta["pairs"]
    items = {}
    for f in sorted(Path(f"/acts/{P.STAGED}/inputs").glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            k = r["key"].removesuffix("__plant1")
            if (k in pairs and r["arm"] in (P.TWIN, P.PLANTED)) or r["arm"] == P.PLANTED:
                items[(r["arm"], k)] = r
    tok = AutoTokenizer.from_pretrained(meta["model"], revision=meta["revision"])
    data = {}
    for k in pairs:
        for arm in (P.TWIN, P.PLANTED):
            it, m = items[(arm, k)], meta["transcripts"][f"{arm}__{k}"]
            r = segment_end_positions(tok, it["messages"], P.MAX_TOKENS)
            outs = [o for o in r["outputs"] if o["end"] > o["start"]]
            out_no = np.concatenate([np.full(o["end"] - o["start"], j) for j, o in enumerate(outs)])
            X = np.load(run / f"{arm}__{k[:150]}.npy")
            steps = [[o["step"], o["action"]] for o in outs]
            if len(X) != m["rows"] or len(out_no) != len(X) or steps != [list(s) for s in m["outputs"]]:
                raise ValueError(f"{arm} {k}: re-derived token rows differ from the pilot's meta")
            idx = np.concatenate([np.arange(o["start"], o["end"]) for o in outs])
            line = np.zeros(len(X), bool)
            if arm == P.PLANTED:  # recompute the line's rows as the pilot did and compare with its meta
                text, _, enc = _layout(tok, it["messages"])
                c0 = text.index(P.PLANT1)
                toks = [i for i, (a, b) in enumerate(enc["offset_mapping"][: len(r["ids"])]) if a < c0 + len(P.PLANT1) and b > c0]
                line = np.isin(idx, toks)
            if np.flatnonzero(line).tolist() != m["line_rows"]:
                raise ValueError(f"{arm} {k}: line rows differ from the pilot's meta")
            data[(arm, k)] = {"X": X, "line": line, "out": out_no, "steps": steps, "env": it["environment"], "main_task": it["main_task"]}
    env_of = {k: data[(P.TWIN, k)]["env"] for k in pairs}

    def aimed(e):  # the pilot's training for held-out environment e, at C 0.001 (copied from P.analyze)
        train = [k for k in pairs if env_of[k] != e]
        pos = np.concatenate([data[(P.PLANTED, k)]["X"][data[(P.PLANTED, k)]["line"]] for k in train])
        pool = [(a, k, i) for k in train for a in (P.TWIN, P.PLANTED) for i in np.flatnonzero(~data[(a, k)]["line"])]
        pick = np.random.default_rng(P.seed_of(f"{P.SEED}:neg:{e}")).choice(len(pool), P.N_NEG, replace=False)
        neg = np.stack([data[(pool[j][0], pool[j][1])]["X"][pool[j][2]] for j in pick])
        rep = round(P.N_NEG / len(pos))
        Xtr = torch.from_numpy(np.concatenate([np.repeat(pos, rep, 0), neg])).to(dev).double()
        ytr = torch.cat([torch.ones(len(pos) * rep), torch.zeros(P.N_NEG)]).to(dev).double()
        mean, std = Xtr.mean(0), Xtr.std(0, unbiased=False)
        std = std.where(std > 0, std.new_ones(()))
        Xs = (Xtr - mean) / std * C**0.5
        W, b, evals = tp.fit(Xs, ytr, 1.0, 5000, tol=tol_of(C))
        Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)  # the pilot's gradient check
        loss = (torch.nn.functional.binary_cross_entropy_with_logits((Xs @ Wg + bg)[:, 0], ytr, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xs)
        loss.backward()
        g = float(max(Wg.grad.abs().max(), bg.grad.abs().max()))
        del Xtr, Xs
        return mean, std, W, b, {"evals": int(evals), "grad_max": g, "converged": g <= tol_of(C), "line_tokens": len(pos)}

    runs = json.loads((Path("/acts") / PACK / f"runs.L{LAYER}.json").read_text())
    t = step_table(runs, "output_end")
    XP = np.load(Path("/acts") / PACK / f"L{LAYER}_output_end.npy")
    arm_of = np.array([runs[i]["arm"] for i in t["run"]])
    efold = np.array([env_to_fold[runs[i]["environment"]] for i in t["run"]])

    def real_vs_sim(f):  # A's comparison refit with the grid's standardization, fitter and refit loop
        tr = np.isin(arm_of, [REAL, SIM]) & (efold != f)
        Xa = torch.from_numpy(XP[tr]).to(dev).double()
        Ya = torch.from_numpy((arm_of[tr] == SIM).astype(float)[:, None]).to(dev).double()
        mean, std = Xa.mean(0), Xa.std(0, unbiased=False)
        std = std.where(std > 0, std.new_ones(()))
        Xa = (Xa - mean) / std * C**0.5
        mi, prev = MAX_ITER, None
        while True:
            W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol_of(C))
            if evals < mi or evals == prev or mi * 4 > MAX_ITER_CAP:
                break
            mi, prev = mi * 4, evals
        Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
        loss.backward()
        grad = float(max(Wg.grad.abs().max(), bg.grad.abs().max()))
        del Xa
        return mean, std, W, b, {"evals": int(evals), "grad_max": grad, "converged": grad <= tol_of(C), "train_rows": int(tr.sum())}

    def score(probe, x):
        mean, std, W, b = probe[:4]
        return ((((torch.from_numpy(np.asarray(x)).to(dev).double() - mean) / std * C**0.5) @ W + b)[:, 0]).cpu().numpy()

    def direction(probe):
        return (probe[2][:, 0] / probe[1]).cpu().numpy()

    envs = sorted(set(env_of.values()))
    rvs = {e: real_vs_sim(env_to_fold[e]) for e in envs}
    aim = {e: aimed(e) for e in envs}
    out = {"git": git, "layer": LAYER, "C": C, "pack": PACK, "pilot_run": P.RUN, "envs": envs,
           "fits": {e: {"real_vs_sim": rvs[e][4], "aimed": aim[e][4]} for e in envs}}
    out["incomplete"] = not all(rvs[e][4]["converged"] and aim[e][4]["converged"] for e in envs)  # a fit missed its gradient check

    def cos(u, v):
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

    d = {e: (direction(rvs[e]), direction(aim[e])) for e in envs}
    out["cosines"] = {"real_vs_sim_vs_aimed": {e: round(cos(*d[e]), 4) for e in envs},
                      "real_vs_sim_across_envs": [round(cos(d[a][0], d[b][0]), 4) for i, a in enumerate(envs) for b in envs[i + 1:]],
                      "aimed_across_envs": [round(cos(d[a][1], d[b][1]), 4) for i, a in enumerate(envs) for b in envs[i + 1:]],
                      "unrelated_reference_1_over_sqrt_d": round(1 / len(d[envs[0]][0]) ** 0.5, 4)}

    show = random.Random(SHOW_SEED).sample(sorted(pairs), N_SHOW)
    out["traces"] = []
    for k in show:
        e, p, w = env_of[k], data[(P.PLANTED, k)], data[(P.TWIN, k)]
        site = int(p["out"][p["line"]][0])
        site_step = p["steps"][site]
        tw_site = w["steps"].index(site_step)
        rec = {"pair_main_task": p["main_task"], "env": e, "site_step_action": site_step}
        for name, d_, s0 in (("planted", p, site), ("twin", w, tw_site)):
            keep = (d_["out"] >= s0 - CONTEXT) & (d_["out"] <= s0 + CONTEXT)
            idx = np.flatnonzero(keep)
            rec[name] = {"real_vs_sim": [round(float(v), 4) for v in score(rvs[e], d_["X"][idx])],
                         "aimed": [round(float(v), 4) for v in score(aim[e], d_["X"][idx])],
                         "output": (d_["out"][idx] - s0).tolist(), "line": d_["line"][idx].tolist()}
        out["traces"].append(rec)

    # all 150 planted check 1 pairs at the site output, real_vs_sim of the pair's environment
    s_all = {}
    for e in sorted({runs[i]["environment"] for i in t["run"]} & set(envs)):
        m = np.array([runs[i]["environment"] == e for i in t["run"]])
        s_all.update(dict(zip(np.flatnonzero(m), score(rvs[e], XP[m]))))
    def site_of(it):  # (step, action) of the output holding the planted line, as the pilot finds it
        r = segment_end_positions(tok, it["messages"], P.MAX_TOKENS)
        text, _, enc = _layout(tok, it["messages"])
        c0 = text.index(P.PLANT1)
        first = min(i for i, (a, b) in enumerate(enc["offset_mapping"][: len(r["ids"])]) if a < c0 + len(P.PLANT1) and b > c0)
        o = next((o for o in r["outputs"] if o["start"] <= first < o["end"]), None)
        return None if o is None else (o["step"], o["action"])

    by_run = {}
    for j, i in enumerate(t["run"]):
        if j in s_all:
            by_run.setdefault(int(i), []).append((int(t["step"][j]), int(t["action"][j]), j))
    key_of = {i: runs[i]["key"] for i in range(len(runs))}
    idx_of = {(runs[i]["arm"], runs[i]["key"]): i for i in range(len(runs))}
    sites, missing, missing_twin, prefix_equal = [], 0, 0, 0
    for i, rows in by_run.items():
        if runs[i]["arm"] != P.PLANTED:
            continue
        tw = idx_of.get((SIM, key_of[i].removesuffix("__plant1")))
        if tw is None or tw not in by_run:
            missing_twin += 1
            continue
        tw_rows = {(s, a): j for s, a, j in by_run[tw]}
        site = site_of(items[(P.PLANTED, key_of[i].removesuffix("__plant1"))])
        here = {(s, a): j for s, a, j in by_run[i]}
        if site is None or site not in here or site not in tw_rows:
            missing += 1
            continue
        s, a = site
        j = here[site]
        before = sorted(q for q in here if q < site)
        prefix_equal += before == sorted(q for q in tw_rows if q < site) and all(np.array_equal(XP[here[q]], XP[tw_rows[q]]) for q in before)
        e = runs[i]["environment"]
        sites.append({"env": e, "site_delta": float(s_all[j] - s_all[tw_rows[(s, a)]]),
                      "run_delta": float(np.mean([s_all[x] for *_, x in by_run[i]]) - np.mean([s_all[x] for *_, x in by_run[tw]])),
                      "outputs": len(by_run[i])})
    spread = {}
    for e in envs:
        real_runs = [r for r in by_run if runs[r]["arm"] == REAL and runs[r]["environment"] == e]
        spread[e] = {"real_run_score_sd": float(np.std([np.mean([s_all[x] for *_, x in by_run[r]]) for r in real_runs])),
                     "real_output_score_sd": float(np.std([s_all[x] for r in real_runs for *_, x in by_run[r]])), "real_runs": len(real_runs)}
    for q in sites:
        q["site_delta_in_run_sd"] = q["site_delta"] / spread[q["env"]]["real_run_score_sd"]
        q["run_delta_in_run_sd"] = q["run_delta"] / spread[q["env"]]["real_run_score_sd"]
    out["sites"] = {"pairs": len(sites), "site_not_found": missing, "twin_not_found": missing_twin, "pairs_with_identical_earlier_rows": int(prefix_equal), "site_delta_positive": int(sum(q["site_delta"] > 0 for q in sites)),
                    "run_delta_positive": int(sum(q["run_delta"] > 0 for q in sites)),
                    "median": {x: round(float(np.median([q[x] for q in sites])), 4) for x in ("site_delta", "run_delta", "site_delta_in_run_sd", "run_delta_in_run_sd", "outputs")},
                    "spread": spread, "per_pair": sites}
    p = Path("/acts/claimB") / f"plant_trace_{git[:8]}.json"
    p.write_text(json.dumps(out))  # numbers only; a copy in case the laptop client dies
    P.vol.commit()
    return out


@app.local_entrypoint()
def main(collect: str = ""):
    import numpy as np  # noqa: F401

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if collect:
        res = json.loads(b"".join(P.vol.read_file(f"claimB/plant_trace_{collect}.json")))
    else:
        folds = json.loads((REPO / "splits" / "folds_v2.json").read_text())["environment_folds"]["environment_to_fold"]
        res = trace.remote(folds, git)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"plant_trace_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("cosines", "fits")}, indent=1))
    print(json.dumps({k: v for k, v in res["sites"].items() if k != "per_pair"}, indent=1))
    print("->", HERE / "results" / f"plant_trace_{stamp}.json")
