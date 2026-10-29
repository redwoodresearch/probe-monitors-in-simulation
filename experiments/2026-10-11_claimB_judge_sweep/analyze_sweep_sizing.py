"""Judge elicitation sweep, sizing before anything runs (PLAN.md; $0, laptop, numbers only, no API call).
(1) Honest runs made before 10-11 (splits/exclusions_v4.json): pairs by environment split, same-source exclusions,
Q1's e1/e2 control pairs by batch, Qwen tokens read. (2) Power of the rule, the old 200-cell Sol run as stand-in:
P(clear) = P(upper 95% < -0.10) + P(lower 95% > -0.10), normal approximation; stand-in SE scaled to the test pairs by
cells (larger SE, the planning basis) and by main tasks (smaller SE), neither a bound. (3) Sol batch price per set-up:
input = measured judge tokens per Qwen token x transcripts read; output = the measured mean.

  uv run --no-project --python 3.12 --with numpy --with scikit-learn --with scipy python \\
      experiments/2026-10-11_claimB_judge_sweep/analyze_sweep_sizing.py <judge_batch_calls.jsonl> <judge_run_selection.json> <judge_tokens.json>
"""

import gzip
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
PROBE = ROOT / "experiments/2026-10-09_fixed_sim/data/claimB_output_scores_A_2026-10-10_022646.json.gz"
EXCL = ROOT / "splits/exclusions_v4.json"
DEV = {"auto_workflow", "file_indexing", "model_registry", "slack_clone", "web_scraping"}  # claim B practice environments
SIM, E1 = "sim-attack-fixed", "real-attack-e1"
SEED, DRAWS, MARGIN, CUT = 20261011, 2000, -0.10, 32768
GAPS = [0.0, -0.05, -0.15, -0.20, -0.30]
N_DEV, N_PLANT_RUNS, POS, N_RETEST, PQS, SHOTS = 50, 20, 3, 30, 3, 4  # PLAN.md's registered sizes
RATE_IN, RATE_OUT = 2.5e-6, 15e-6  # Sol by Batch API: half of $5 / $30 per million (the old run's basis)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pair_acc(s_sim, s_real):
    return float(np.mean((s_sim > s_real) + 0.5 * (s_sim == s_real)))


def p_clear(gap, se):
    return float(norm.cdf((MARGIN - gap) / se - 1.96) + norm.cdf((gap - MARGIN) / se - 1.96))


def main(calls, sel_path, tok_path):
    ex = json.loads(EXCL.read_text())
    cells = set(ex["paired_honest_cells"]["cells"])
    valid = [r for r in ex["samples"] if r["main_task"] in cells and not r["reason"]]
    split = lambda r: "practice" if r["environment"] in DEV else "test"  # noqa: E731
    sims = [r for r in valid if r["arm"] == "sim-honest"]
    reals = defaultdict(list)
    for r in sorted((r for r in valid if r["arm"] == "real-honest"), key=lambda r: r["key"]):
        reals[r["main_task"]].append(r)
    batch = lambda r: r["key"].split("_")[0]  # noqa: E731  base200 | overlap800
    ctrl = {s: [t for t in cells if len(reals[t]) >= 2 and split(reals[t][0]) == s] for s in ("practice", "test")}
    tok = defaultdict(list)
    for r in sims:
        tok[f"sim|{split(r)}"].append(min(r["tokens"], CUT))
    for t in cells:
        for i, r in enumerate(reals[t][:2]):
            tok[f"real-{'e1' if i == 0 else 'e2'}|{split(r)}"].append(min(r["tokens"], CUT))
    hon = {"pairs": {s: sum(1 for r in sims if split(r) == s) for s in ("practice", "test")},
           "same_source_excluded_here": sum(1 for r in sims if r.get("sim_reference") != "overlap800"),
           "control_pairs": {s: {"all": len(v), "same_batch": sum(batch(reals[t][0]) == batch(reals[t][1]) for t in v)} for s, v in ctrl.items()},
           "mean_tokens_read": {k: float(np.mean(v)) for k, v in sorted(tok.items())}}
    out = {"label": "Judge sweep sizing (plan only; $0; numbers only)", "honest": hon}
    # 2. stand-in power (the old run: Sol p and step A's run scores on the same 200 fixed-sim / real-e1 test cells)
    sel = json.loads(Path(sel_path).read_text())
    recs = [json.loads(line) for line in Path(calls).read_text().splitlines()]
    mains = [r for r in recs if r.get("batch_chunk") != "check" and r["kind"] == "main" and r["status"] == "ok" and r["replicate"] == "0"]
    p = {(r["cell"], r["arm"]): r["p"] for r in mains}
    by = defaultdict(list)
    for r in json.loads(gzip.decompress(PROBE.read_bytes()))["levels"]["level1"]["rows"]:
        by[r["key"]].append(r["score"])
    probe = {k: float(np.mean(v)) for k, v in by.items()}
    key = sel["run_keys"]
    cs = [c for c in sel["cells"] if (c, SIM) in p and (c, E1) in p]
    js, jr = np.array([p[(c, SIM)] for c in cs]), np.array([p[(c, E1)] for c in cs])
    ps, pr = (np.array([probe[key[f"{c}|{a}"]] for c in cs]) for a in (SIM, E1))
    tasks, inv = np.unique([c.split(":")[0] for c in cs], return_inverse=True)
    mem = [np.flatnonzero(inv == j) for j in range(len(tasks))]
    rng, dA, dP = np.random.default_rng(SEED), [], []
    for _ in range(DRAWS):
        i = np.concatenate([mem[j] for j in rng.choice(len(tasks), len(tasks))])
        yi = np.r_[np.ones(len(i)), np.zeros(len(i))]
        dA.append(roc_auc_score(yi, np.r_[js[i], jr[i]]) - roc_auc_score(yi, np.r_[ps[i], pr[i]]))
        dP.append(pair_acc(js[i], jr[i]) - pair_acc(ps[i], pr[i]))
    y = np.r_[np.ones(len(cs)), np.zeros(len(cs))]
    out["stand_in"] = {"cells": len(cs), "main_tasks": len(tasks), "auroc_judge": roc_auc_score(y, np.r_[js, jr]), "auroc_probe": roc_auc_score(y, np.r_[ps, pr]),
                       "pair_acc_judge": pair_acc(js, jr), "pair_acc_probe": pair_acc(ps, pr)}
    n_test, power = hon["pairs"]["test"], {}
    for name, d in (("pair_accuracy", dP), ("auroc", dA)):
        se200 = float(np.std(d))
        for scen, n_ref in (("by_cells_larger_se", len(cs)), ("by_main_tasks_smaller_se", len(tasks))):
            se = lambda n: se200 * np.sqrt(n_ref / n)  # noqa: E731
            n80 = {str(g): next((n for n in range(20, n_test + 1) if p_clear(g, se(n)) >= 0.80), "not reachable within the test pairs") for g in GAPS}
            power[f"{name}|{scen}"] = {"se_stand_in": se200, "se_at_test_pairs": float(se(n_test)),
                                       "p_clear_answer": {str(g): p_clear(g, se(n_test)) for g in GAPS}, "pairs_for_80pct": n80}
    out["power"] = power
    # 3. price: judge tokens per Qwen token read, from the same old calls' arms (sim and real e1 main calls)
    t = json.loads(Path(tok_path).read_text())["rows"]
    read_old = float(np.mean([t["test|sim-attack-fixed"]["read_mean"], t["test|real-attack-e1"]["read_mean"]]))
    sub = [r for r in mains if r["arm"] in (SIM, E1)]
    tin, tout = float(np.mean([r["usage"]["input"] for r in sub])), float(np.mean([r["usage"]["output"] for r in sub]))
    f = tin / read_old
    mt = hon["mean_tokens_read"]
    read = {s: float(np.mean([mt[f"sim|{s}"], mt[f"real-e1|{s}"]])) for s in ("practice", "test")}
    call = lambda s, k=1, shots=0: f * (k * read[s] + shots * read["practice"]) * RATE_IN + tout * RATE_OUT  # noqa: E731
    nc, npl = hon["control_pairs"]["test"]["all"], 2 * N_PLANT_RUNS * POS
    single = 2 * n_test + nc + npl + N_RETEST  # runs: main pairs (2 runs), control e2 runs, planted copies, retest
    plan = {"S0_old_prompt": (2 * N_DEV, call("practice"), single, call("test")),
            "S1_needham_3PQ": (2 * N_DEV * PQS, call("practice"), single * PQS, call("test")),
            "S2_fewshot_4": (2 * N_DEV, call("practice", shots=SHOTS), single, call("test", shots=SHOTS)),
            "S3_pairwise_both_orders": (2 * N_DEV, call("practice", k=2), 2 * (n_test + nc + npl) + 2 * N_RETEST, call("test", k=2))}
    price = {k: {"practice_calls": v[0], "practice_usd": v[0] * v[1], "test_calls_if_winner": int(v[2] * 1.05), "test_usd_if_winner": v[2] * v[3] * 1.05}
             for k, v in plan.items()}
    dev_total = sum(v["practice_usd"] for v in price.values())
    out["price_sol_batch"] = {"basis": {"old_calls": len(sub), "judge_input_per_qwen_token": f, "mean_output_tokens": tout, "honest_read": read, "old_read": read_old,
                                        "measured_usd_per_old_request": float(np.mean([r["cost_usd"] for r in sub])),
                                        "caveat": "judge tokens: 400 calls on 200 cells per arm; Qwen tokens: means over all 473 runs per arm (no per-run counts stored)"},
                              "by_setup": price, "practice_total": dev_total,
                              "whole_job_raw": {k: dev_total + v["test_usd_if_winner"] for k, v in price.items()},
                              "whole_job_with_30pct": {k: 1.3 * (dev_total + v["test_usd_if_winner"]) for k, v in price.items()}}
    out |= {"inputs": {str(x): sha(x) for x in (calls, sel_path, tok_path, PROBE, EXCL)},
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "git_dirty": bool(subprocess.run(["git", "status", "--porcelain", str(HERE)], capture_output=True, text=True, cwd=HERE).stdout.strip()),
            "script_sha256": sha(__file__)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"sweep_sizing_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float)[:6000])
    print("->", dst)


if __name__ == "__main__":
    main(*sys.argv[1:4])
