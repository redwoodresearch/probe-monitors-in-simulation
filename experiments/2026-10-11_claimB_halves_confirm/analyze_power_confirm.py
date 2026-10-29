"""$0 power check (0d relaying Aswin, 10-11 about 23:30 PT): could the environment-halves strict-alarm result (item (A),
aswin/claimb-halves; pooled fixed sim 5.4% flagged vs second real e2 1.6% flagged by the same sim-trained detector at the
1% threshold set on held-out e1) be confirmed on fresh, COMPARABLE runs scored by the FROZEN detectors at the FROZEN
thresholds? Numbers only; reads score arrays (no activations, no text) and run-ID lists.

1. Gate: rebuild item (A)'s per-direction scores (the sim-trained fixed-vs-e1 detector as selected, POSTHOC_e2 arrays)
   and reproduce its numbers exactly (dev->test sim 33/473, e2 8/473; test->dev sim 18/466, e2 7/466). Thresholds are
   then frozen at the 1% point of each direction's held-out e1 runs, as written.
2. Power for a comparable fresh design: N new cells per direction, each one fresh fixed-sim run and one fresh real attack
   run on a held-out (never-trained) environment. Fresh data are simulated by resampling held-out main tasks (all their
   cells) from item (A)'s own flags at the frozen threshold: task-clustered, true rates = item (A)'s observed rates, with
   e2 standing in for the fresh real run. The test on each simulated set: gap = sim flagged - real flagged, 95% interval
   from a task bootstrap (threshold frozen, not re-set); "beyond" if the lower bound > 0, "within 10 pts" if the upper
   bound < 0.10 (item (A)'s registered strict-alarm readings; its refit null doesn't apply to a frozen detector).
   Per direction and pooled (both directions, N each).
3. 0d's independent-runs binomial (scratchpad power_b/power.py, ps 0.054, pr 0.016, seed 20261011) reproduced beside it.
4. Counts only for lean230 stage 1 + zoo C stage 1 (SEALED): the two run-list files are loaded, and only the ID, task,
   environment and status fields are kept; no other field is used, computed on or printed.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy \\
      python experiments/2026-10-11_claimB_halves_confirm/analyze_power_confirm.py
"""

import collections
import hashlib
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
OUT = "claimB/halves_v1/5b07509a98fc"  # item (A)'s run_halves_modal.py sha256[:12] @ aswin/claimb-halves 05dd923d
PH_OUT = f"{OUT}/POSTHOC_e2"
SIM, E1, E2 = "sim-attack-fixed", "real-attack-e1", "real-attack-e2"
TEST = ["clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"]
DEV = ["auto_workflow", "file_indexing", "model_registry", "slack_clone", "web_scraping"]
DIRS = {"dev_to_test": (DEV, TEST), "test_to_dev": (TEST, DEV)}
EXPECT = {"dev_to_test": (33, 473, 8, 473), "test_to_dev": (18, 466, 7, 466)}  # item (A) + POSTHOC result files
SHA = {"runs.json": "1f55e447610e669e3d27726b8bcec24027798aa5d7fc477898833b684664bc27",
       "dev_to_test/fixed_vs_e1.npz": "c68284a61ba2bcf5c657d5f55509b018d3af751b413dad9eabb43c9007ca8e0c",
       "test_to_dev/fixed_vs_e1.npz": "42d8b2455b7c228e5b50fab108e761ba91e84fdf336665715da2d2fc47a984f3",
       "POSTHOC_e2/dev_to_test/fixed_vs_e1.npz": "9210c7391bc4b8cf89650a52b0e63e75caea034e0047c0f25194b12757246fd6",
       "POSTHOC_e2/test_to_dev/fixed_vs_e1.npz": "4ab399cc937b01098b5f44100642e94c4d9c26c13b58c3729330026094339687"}
SEED, REPS, BOOT = 20261012, 1000, 1000
SIZES = [25, 50, 100, 150, 200, 300, 466]
LEAN_IDS = "origin/aswin/claimb-q2-sims-lean230:experiments/2026-10-11_q2_sim_honest_lean230/results/lean230_stage1_run_ids.json"
ZOOC = "origin/main:experiments/2026-10-07_phase2_attack_check/results/zoo_c_runs_oct11c_stage1_2026-10-11_183251.json"


def thr(neg):  # item (A)'s rule: flag scores strictly above the (floor(1% n)+1)-th highest negative score
    s = np.sort(neg)[::-1]
    return s[int(np.floor(0.01 * len(s)))]


def load():
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    raw = {}
    for k in SHA:
        b = b"".join(vol.read_file(f"{OUT}/{k}"))
        if hashlib.sha256(b).hexdigest() != SHA[k]:
            raise SystemExit(f"{k}: sha256 differs from item (A)'s recorded input")
        raw[k] = b
    runs = json.loads(raw["runs.json"])["runs"]
    env = np.array([r["environment"] for r in runs])
    arm = np.array([r["arm"] for r in runs])
    task = np.array([r["main_task"] for r in runs])
    cell = np.array([str(r["cell"]) for r in runs])
    score = {}
    for d in DIRS:
        zo, zp = np.load(io.BytesIO(raw[f"{d}/fixed_vs_e1.npz"])), np.load(io.BytesIO(raw[f"POSTHOC_e2/{d}/fixed_vs_e1.npz"]))
        b = int(zo["val_auroc"][:, 0].argmax())
        if int(zp["val_auroc"][:, 0].argmax()) != b:
            raise SystemExit(f"{d}: rerun picks another sub-configuration")
        score[d] = zp["test_run_score"][b, :, 0]
    return env, arm, task, cell, score


def per_task_flags(env, arm, task, cell, score):
    """Gate + frozen thresholds; per held-out main task, its paired cells' flags: an array (cells x 2) of (sim flagged,
    e2 flagged) for the same cell, per direction."""
    out, gate = {}, {}
    for d, (_, held_envs) in DIRS.items():
        m = np.isin(env, held_envs)
        s, a, t = score[d][m], arm[m], task[m]
        th = thr(s[a == E1])
        fs, fe = s[a == SIM] > th, s[a == E2] > th
        got = (int(fs.sum()), int((a == SIM).sum()), int(fe.sum()), int((a == E2).sum()))
        gate[d] = {"got": got, "expected": EXPECT[d], "pass": got == EXPECT[d], "threshold": float(th),
                   "e1_fpr": float((s[a == E1] > th).mean())}
        c = cell[m]
        flag = {(cc, aa): bool(ss > th) for ss, aa, cc in zip(s, a, c) if aa in (SIM, E2)}
        paired = sorted({cc for (cc, aa) in flag if aa == SIM} & {cc for (cc, aa) in flag if aa == E2})
        if len(paired) != got[1] or len(paired) != got[3]:
            raise SystemExit(f"{d}: sim and e2 cells don't pair one to one")
        task_of = {cc: tt for cc, tt in zip(c, t)}
        by = collections.defaultdict(list)
        for cc in paired:
            by[task_of[cc]].append((flag[(cc, SIM)], flag[(cc, E2)]))
        out[d] = [np.array(by[x], dtype=float) for x in sorted(by)]
    return out, gate


def draw_set(T, n_cells, rng):
    """Main tasks resampled with replacement (all their paired cells) until exactly n_cells cells: the last task keeps
    a random subset of its cells (the same cells for both arms). Rows: (sim flagged, cells, e2 flagged, cells)."""
    rows, k = [], 0
    while k < n_cells:
        cells = T[rng.integers(len(T))]
        need = n_cells - k
        if len(cells) > need:
            cells = cells[rng.choice(len(cells), need, replace=False)]
        rows.append((cells[:, 0].sum(), len(cells), cells[:, 1].sum(), len(cells)))
        k += len(cells)
    return np.array(rows, dtype=float)


def readings(Ds, rng):
    """Gap and its task-bootstrap 95% interval (frozen threshold) on simulated fresh sets Ds (one tasks x 4 array per
    direction); the bootstrap resamples tasks within each direction, then pools the counts (fixed allocation)."""
    tot = sum(D.sum(0) for D in Ds)
    g = tot[0] / tot[1] - tot[2] / tot[3]
    acc = np.zeros((BOOT, 4))
    for D in Ds:
        acc += D[rng.integers(len(D), size=(BOOT, len(D)))].sum(1)
    gb = acc[:, 0] / acc[:, 1] - acc[:, 2] / acc[:, 3]
    lo, hi = np.percentile(gb, [2.5, 97.5])
    return g, lo, hi


def power(T_by_dir, n, dirs, rng):
    beyond = within = both = 0
    for _ in range(REPS):
        _, lo, hi = readings([draw_set(T_by_dir[d], n, rng) for d in dirs], rng)
        beyond += lo > 0
        within += hi < 0.10
        both += (lo > 0) and (hi < 0.10)
    return {"P_beyond": beyond / REPS, "P_within_10pts": within / REPS, "P_both": both / REPS,
            "P_clear": (beyond + within - both) / REPS}


def binom_0d(n_sim, n_real, rng, reps=20000, ps=0.054, pr=0.016):
    s, r = rng.binomial(n_sim, ps, reps) / n_sim, rng.binomial(n_real, pr, reps) / n_real
    se = np.sqrt(np.maximum(s * (1 - s) / n_sim + r * (1 - r) / n_real, 1e-9))
    return float(np.mean(s - r - 1.96 * se > 0))


def sealed_counts():
    lean = json.loads(subprocess.run(["git", "show", LEAN_IDS], capture_output=True, text=True, check=True).stdout)["runs"]
    zoo = json.loads(subprocess.run(["git", "show", ZOOC], capture_output=True, text=True, check=True).stdout)["rows"]
    lean_env = {r["main"]: r["environment"] for r in lean}  # IDs, task and environment only
    zoo_by = collections.Counter((r["task"], r["env"]) for r in zoo if r["status"] == "success")
    res = {"lean230_stage1_runs": len(lean), "zooc_stage1_success_runs": sum(zoo_by.values()),
           "zooc_tasks_matching_lean230": sum(1 for t in lean_env if any(k[0] == t for k in zoo_by))}
    for d, (_, held_envs) in DIRS.items():
        tasks = [t for t, e in lean_env.items() if e in held_envs]
        res[d] = {"lean230_tasks_never_trained": len(tasks), "zooc_runs_on_them": sum(v for (t, e), v in zoo_by.items() if t in tasks),
                  "by_environment": dict(sorted(collections.Counter(lean_env[t] for t in tasks).items()))}
    return res


def main():
    env, arm, task, cell, score = load()
    T, gate = per_task_flags(env, arm, task, cell, score)
    if not all(g["pass"] for g in gate.values()):
        print(json.dumps(gate, indent=1))
        raise SystemExit("gate failed: item (A)'s numbers are not reproduced; nothing else is computed")
    rng = np.random.default_rng(SEED)
    table = {}
    for n in SIZES:
        table[n] = {d: power(T, n, [d], rng) for d in DIRS}
        table[n]["pooled"] = power(T, n, list(DIRS), rng)
    rngb = np.random.default_rng(20261011)  # 0d's seed
    binom = {f"{a}_vs_{b}": binom_0d(a, b, rngb) for a, b in [(50, 50), (50, 274), (100, 274), (230, 274), (274, 274)]}
    binom_same = {n: binom_0d(2 * n, 2 * n, rngb) for n in SIZES}  # pooled: exactly 2n sims and 2n reals, independent runs
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out = {"label": "POWER (planning, $0): fresh comparable confirmation of item (A)'s strict-alarm result with frozen detectors and thresholds",
           "gate": gate, "per_direction_tasks": {d: {"main_tasks": len(T[d]), "paired_cells": int(sum(len(x) for x in T[d])),
                                                     "sim_flagged": int(sum(x[:, 0].sum() for x in T[d])), "real_flagged": int(sum(x[:, 1].sum() for x in T[d]))} for d in DIRS},
           "settings": {"reps": REPS, "boot": BOOT, "seed": SEED, "sizes_cells_per_direction": SIZES,
                        "real_stand_in": "e2 flagged by the sim-trained detector (item (A) POSTHOC)"},
           "power_task_clustered": {str(k): v for k, v in table.items()},
           "binomial_0d_reproduced": binom, "binomial_independent_pooled_same_sizes": {str(k): v for k, v in binom_same.items()},
           "sealed_counts": sealed_counts(), "inputs": SHA, "git": git,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"power_confirm_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("gate", "per_direction_tasks", "binomial_0d_reproduced", "sealed_counts")}, indent=1))
    for n, v in table.items():
        print(n, {d: (round(x["P_beyond"], 3), round(x["P_clear"], 3)) for d, x in v.items()}, "binom pooled", round(binom_same[n], 3))
    print("->", dst)


if __name__ == "__main__":
    main()
