"""Re-pricing after stage 1's labels check (post hoc; PURCHASE.md: "If any rate is below 80% of today's, the rest waits
and is re-priced"). The orchestrator relayed: re-price at $0 using the unpaired tasks' own original runs. Keys, labels and
stored scores only; $0. Three parts:

1. Task mix. Transfer and destructive positives per real run for:
   - the 163 same-batch tasks' first two runs (the sizing's donors);
   - all 274 unpaired tasks' overlap800 runs;
   - the first 50 of the seeded order (stage 1's tasks), on their original runs and on the new runs.
   Also the chance that 50 tasks drawn at random from the 274 average as low as the first 50 did on their original runs.
2. The pooled lean test re-priced. analyze_lean_sizing.py's method unchanged, except that each donor copy of a 163 task
   is thinned. Each positive is kept with probability f (per monitor and arm), and its catches thin with it
   (hypergeometric). Two levels of f:
   - unpaired: the 274 unpaired tasks' own original rates over the 163's, real and simulated;
   - stage 1: the new real runs' rates over the 163's, with the simulated side at the first 50 tasks' simulated rate.
   With f = 1 it must reproduce the committed sizing at the same k.
3. Zoo C preview. Credential positives per original matched run (Aswin's labels, run_id and credential only) for the same
   task sets, against today's 3.57 and the stage-1 floor of 2.86. C's sizing already projects new runs from the task's own
   overlap800 runs in its pessimistic model, which is this task-mix assumption.

  uv run --no-project --with numpy python experiments/2026-10-11_q2_lean_newruns/analyze_reprice_POSTHOC.py <per_task_catch_196.json> \\
      <recal_2026-10-10_225753.json> <stage1_labels_real_2026-10-11_190808.json> <screening_labels_v1.csv> <batch_spec_2026-10-11_151207.json>
"""
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("lean", HERE / "analyze_lean_sizing.py"); L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
SHA = L.SHA | {"stage1_labels_real_2026-10-11_190808.json": "40c005d24ba6a7b8a4434dad1cbdc98e97a9c1c156fbbf11b46c20f34b81eb87", "screening_labels_v1.csv": "edfeb7d43a104113453d5a768964b87b0ee8fc1bd9a6e286276e9b8fd58ae224",
               "batch_spec_2026-10-11_151207.json": "ce22799937e35d7db7634c618d0e1ab2a0d72963982797761b958cf7eacc6c4a"}
COMMITTED = HERE / "results/lean_sizing_2026-10-11_153740.json"
KS, TARGET = (230, 274), "1%"
MONS = {"transfer": "transfer", "destructive_raw": "destructive"}


def thin(c, p, f, g):
    """Keep each positive with probability f; catches among the kept are hypergeometric."""
    if f >= 1:
        return c, p
    kept = g.binomial(p.astype(int), f)
    cc = np.array([g.hypergeometric(int(ci), int(pi - ci), int(ki)) if ki else 0 for ci, pi, ki in zip(c, p, kept)], float)
    return cc, kept.astype(float)


def main(m1_p, recal_p, stage1_p, labels_p, spec_p):
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    for f in (Path(__file__).resolve(), HERE / "analyze_lean_sizing.py", COMMITTED):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    got = {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (m1_p, recal_p, stage1_p, labels_p, spec_p)}
    for k, v in SHA.items():
        if got[k] != v:
            raise SystemExit(f"{k} differs from its pinned sha256")
    full = json.loads(Path(recal_p).read_text()); rows = full["rows"]
    stage1 = json.loads(Path(stage1_p).read_text()); order = json.loads(Path(spec_p).read_text())["task order"]
    if hashlib.sha256("\n".join(order).encode()).hexdigest() != "b804d4d44f1ea8d716ab30b0bb099e7d9e59ef908d427de013cf3d36b0d00c8b":
        raise SystemExit("task order differs")
    first50 = set(order[:50])
    sim_tasks = {r["task"] for r in rows if r["world"] == 1}
    runs = defaultdict(set)
    for r in rows:
        if r["world"] == 0:
            runs[r["task"]].add(r["run"])
    first2 = {t: sorted(v)[:2] for t, v in runs.items() if len(v) >= 2}
    same = sorted(t for t, (a, b) in first2.items() if "|overlap800_" in a and "|overlap800_" in b)
    unpaired = sorted(sim_tasks - set(same))
    if len(same) != 163 or len(unpaired) != 274 or not first50 <= set(unpaired):
        raise SystemExit("cohort differs")
    o800 = {t: sorted(x for x in runs[t] if "|overlap800_" in x) for t in sim_tasks}
    pos_run = {m: Counter() for m in MONS}; pos_sim = {m: Counter() for m in MONS}  # positives per real run, per simulated task
    for r in rows:
        for m in MONS:
            if r[m] == 1 and r["world"] == 0:
                pos_run[m][r["run"]] += 1
            elif r[m] == 1:
                pos_sim[m][r["task"]] += 1
    real_runs = {"163 first two": [x for t in same for x in first2[t]], "274 unpaired, overlap800 runs": [x for t in unpaired for x in o800[t]],
                 "first 50, overlap800 runs": [x for t in sorted(first50) for x in o800[t]]}
    sim_sets = {"163": same, "274 unpaired": unpaired, "first 50": sorted(first50)}
    mix = {m: {"real positives a run": {k: sum(pos_run[m][x] for x in v) / len(v) for k, v in real_runs.items()} | {
                   "first 50, new runs (stage 1)": stage1["new real runs (stage 1; 100 runs, 50 pairs; direct labeler)"][MONS[m]]["positives per run"]},
               "simulated positives a task": {k: sum(pos_sim[m][t] for t in v) / len(v) for k, v in sim_sets.items()}} for m in MONS}
    # how unusual are the first 50? draws of 50 of the 274 unpaired tasks, per-task mean of positives a run (original runs)
    g = np.random.default_rng([L.SEED, 50])
    for m in MONS:
        per_task = np.array([np.mean([pos_run[m][x] for x in o800[t]]) for t in unpaired]); obs = np.mean([np.mean([pos_run[m][x] for x in o800[t]]) for t in sorted(first50)])
        draws = np.array([per_task[g.choice(len(unpaired), 50, replace=False)].mean() for _ in range(20_000)])
        mix[m]["first 50 vs random 50 of the 274 (task means of original runs)"] = {"observed": float(obs), "random-50 median": float(np.median(draws)),
                                                                                     "share of random 50s at or below": float((draws <= obs).mean())}
    f = {"unpaired": {m: {"real": mix[m]["real positives a run"]["274 unpaired, overlap800 runs"] / mix[m]["real positives a run"]["163 first two"],
                          "simulated": mix[m]["simulated positives a task"]["274 unpaired"] / mix[m]["simulated positives a task"]["163"]} for m in MONS},
         "stage 1": {m: {"real": mix[m]["real positives a run"]["first 50, new runs (stage 1)"] / mix[m]["real positives a run"]["163 first two"],
                         "simulated": mix[m]["simulated positives a task"]["first 50"] / mix[m]["simulated positives a task"]["163"]} for m in MONS}}
    out = {"label": "POST HOC (after stage 1's labels check; no verdict)", "git": head, "inputs sha256": got, "task mix": mix, "thinning f": f, "pooled lean, re-priced": {}}
    # 2. the pooled lean test re-priced, at the primary target
    q = L.TARGETS[TARGET]
    m1 = L.m1_counts(m1_p, f"caught_{TARGET[:-1]}pct"); T1 = len(m1["real_run1"][0])
    W1 = np.vstack([np.ones(T1), np.random.default_rng([L.SEED, 1]).poisson(1.0, (L.DRAWS, T1))])
    g1_rs, g1_rr = L.gap(*m1["real_run1"], *m1["simulated"], W1), L.gap(*m1["real_run1"], *m1["real_run2"], W1)
    sols = [L.sol_counts(full, n, q) for n in MONS]; T = len(sols[0][0])
    W = np.vstack([np.ones(T + L.MAXK), np.random.default_rng([L.SEED, 23]).poisson(1.0, (L.DRAWS, T + L.MAXK))])
    committed = json.loads(COMMITTED.read_text())["targets"][TARGET]["by k"]
    for level in ("none (f = 1, must reproduce)", "unpaired", "stage 1"):
        res = {}
        for k in KS:
            se_rs, se_rr, rhos, sets0 = [], [], [], None
            for j in range(L.SETS):
                don = np.random.default_rng([L.SEED, 7, k, j]).integers(0, T, k)
                gt = np.random.default_rng([L.SEED, 9, k, j])
                aug = []
                for (name, (_, c)) in zip(MONS, sols):
                    a_ = {}
                    for arm in L.ARMS:
                        fa = 1.0 if level.startswith("none") else f[level][name]["simulated" if arm == "simulated" else "real"]
                        cc, pp = thin(c[arm][0][don], c[arm][1][don], min(fa, 1.0), gt)  # f above 1 is capped (no thickening)
                        a_[arm] = (np.concatenate([c[arm][0], cc]), np.concatenate([c[arm][1], pp]))
                    aug.append(a_)
                Wk = W[:, :T + k]
                G_rs = (g1_rs + sum(L.gap(*c["real_run1"], *c["simulated"], Wk) for c in aug)) / 3
                G_rr = (g1_rr + sum(L.gap(*c["real_run1"], *c["real_run2"], Wk) for c in aug)) / 3
                se_rs.append(G_rs[1:].var()); se_rr.append(G_rr[1:].var()); rhos.append(np.corrcoef(G_rs[1:], G_rr[1:])[0, 1])
                if j == 0:
                    sets0 = (aug, Wk, G_rs, G_rr)
            S_rs, S_rr, rho = float(np.sqrt(np.mean(se_rs))), float(np.sqrt(np.mean(se_rr))), float(np.mean(rhos))
            gz = np.random.default_rng(L.SEED); z1, z2 = gz.standard_normal(L.MC), gz.standard_normal(L.MC)
            x, yv = S_rs * z1, S_rr * (rho * z1 + np.sqrt(1 - rho ** 2) * z2)
            p_both = float(((np.abs(x) < L.M - 1.96 * S_rs) & (np.abs(yv) < L.M - 1.96 * S_rr)).mean())
            aug, Wk, G_rs, G_rr = sets0
            pd = {}
            for arm, unpl in (("simulated", G_rs), ("real_run2", G_rr)):
                hits = []
                for mons in L.plant([(m1, W1)] + [(c, Wk) for c in aug], arm, L.COPIES):
                    Gp = sum(L.gap(*c["real_run1"], *c[arm], W_) for c, W_ in mons) / 3
                    hits.append(L.detected(float(Gp[0]), tuple(L.ci(Gp[1:])), float(unpl[0]), L.PLANT))
                pd[arm] = float(np.mean(hits))
            pt = L.p_testable(pd["simulated"]) * L.p_testable(pd["real_run2"])
            res[str(k)] = {"SE G_RS": S_rs, "SE G_RR": S_rr, "P(both pass)": p_both, "P(plants testable)": pt, "P(clear)": p_both * pt}
            if level.startswith("none") and abs(res[str(k)]["P(clear)"] - committed[str(k)]["P(clear)"]) > 1e-12:
                raise SystemExit(f"f = 1 doesn't reproduce the committed sizing at k = {k}")
            print(level, k, {kk: round(v, 3) for kk, v in res[str(k)].items()}, flush=True)
        out["pooled lean, re-priced"][level] = res
    # 3. zoo C preview from Aswin's labels (run_id and credential only)
    cred = Counter()
    with open(labels_p, newline="") as fh:
        for r in csv.DictReader(fh):
            cred[r["run_id"]] += r["credential"] == "1"
    matched = [x for t in sim_tasks for x in o800[t]]
    c_sets = {"today's 800 matched runs": matched} | real_runs
    out["zoo C preview (credential positives a run, original runs)"] = {k: sum(cred[x] for x in v) / len(v) for k, v in c_sets.items()} | {
        "today (C sizing)": 3.57, "stage-1 floor (80%)": 0.8 * 3.57}
    if round(out["zoo C preview (credential positives a run, original runs)"]["today's 800 matched runs"], 2) != 3.57:
        raise SystemExit("credential positives on today's matched runs don't reproduce 3.57 a run")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"reprice_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "inputs sha256"}, indent=1)); print(dest)


if __name__ == "__main__":
    main(*sys.argv[1:6])
