"""N2 measure 1 on Qwen's attack runs (DEFINITIONS.md, 540bba13): share of runs touching the target at least once
(main) and the per-run share of target-touching commands (secondary), real run 1 vs simulated with real run 1 vs run 2
as the yardstick; Aswin's v3 labels with shared_with_main_only == 0 kept first. `--plant-only` takes real-only copies of
the labels and replay join (a prefix filter keeping run ids under attack/, so no simulated row is parsed) and runs only the
yardstick's planted check; the full phase needs that plant result (`--plant-result`) and skips any measure not testable at
its margin. Replay-reader exclusions are reported, not removed. IDs and numbers only; $0.

  plant: uv run --no-project --with pandas --with numpy python experiments/2026-10-11_q2_n2_target/analyze_n2_target.py \\
      data/b_feasibility/target_labels_v3_real_attack.csv data/b_feasibility/replay_join_v2_real_attack.csv data/e1_honest/runs_phase2_196.csv --plant-only
  full:  ... target_labels_v3.csv replay_join_v2.csv runs_phase2_196.csv --plant-result <plant result json>
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from probe_shift.margins import LEVEL, equivalence_p, holm, inside  # noqa: E402
from probe_shift.planted import detected  # noqa: E402

HERE = Path(__file__).parent
SEED, DRAWS, COPIES = 20261011, 10_000, 200
MARGIN = {"main: share of runs touching": 0.10, "secondary: per-run share of touching commands": 0.03}
COL = {"main: share of runs touching": "any", "secondary: per-run share of touching commands": "share"}


def ci(b):
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]


def real_only_sha(p: str) -> str:
    """sha256 of the header plus the rows under attack/, as the prefix filter wrote the plant phase's real-only copies."""
    lines = Path(p).read_bytes().splitlines(keepends=True)
    return hashlib.sha256(lines[0] + b"".join(x for x in lines[1:] if x.startswith(b"attack/"))).hexdigest()


def beyond_p(d, m):
    return float(min((1 + (d <= m).sum()) / (len(d) + 1), (1 + (d >= -m).sum()) / (len(d) + 1)))


def main(labels_p: str, join_p: str, runs_p: str, plant_only: bool, plant_result: str | None):
    sha = {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (labels_p, join_p, runs_p)}
    L, R, J = pd.read_csv(labels_p), pd.read_csv(runs_p), pd.read_csv(join_p)
    if plant_only and not L.run_id.str.startswith("attack/").all():
        raise SystemExit("--plant-only needs the real-only label copy")
    if not plant_only:
        prior = json.loads(Path(plant_result).read_text()) if plant_result else {}
        if prior.get("phase") != "plant only (no simulated row read)":
            raise SystemExit("the full phase needs the reported plant result (--plant-result)")
        want = {"target_labels_v3_real_attack.csv": real_only_sha(labels_p), "replay_join_v2_real_attack.csv": real_only_sha(join_p),
                "runs_phase2_196.csv": sha[Path(runs_p).name]}
        if prior.get("inputs sha256") != want:  # fail closed: the plant result must come from this run's own rows
            raise SystemExit(f"the plant result's inputs differ from this run's real rows: {prior.get('inputs sha256')} vs {want}")
    arms = ["attack", "attack_run2"] + ([] if plant_only else ["sim_attack@september"])
    L = L.merge(J.rename(columns={"inspect_epoch": "epoch"})[["run_id", "epoch", "turn", "call_index", "q2_excluded"]],
                on=["run_id", "epoch", "turn", "call_index"], how="left", validate="one_to_one")
    L = L.merge(R[["run_id", "arm", "task_key"]], on="run_id", how="left")
    flagged = int(((L.shared_with_main_only == 1) & L.arm.isin(arms)).sum())
    L = L[(L.shared_with_main_only == 0) & L.arm.isin(arms)]  # filter first; simulated rows never loaded with --plant-only
    per = L.groupby(["arm", "task_key", "run_id"]).agg(calls=("touch_any", "size"), touches=("touch_any", "sum")).reset_index()
    per["any"] = (per.touches > 0).astype(float); per["share"] = per.touches / per.calls
    one = {a: per[per.arm == a].set_index("task_key") for a in arms}
    if any(not g.index.is_unique for g in one.values()):
        raise ValueError("more than one run per arm and task")
    rr_tasks = sorted(set(one["attack"].index) & set(one["attack_run2"].index))
    out = {"inputs sha256": sha, "phase": "plant only (no simulated row read)" if plant_only else "full",
           "shared-only calls dropped first": flagged, "runs": {a: len(one[a]) for a in arms}, "yardstick tasks": len(rr_tasks),
           "replay-reader exclusions among retained calls (reported, not removed)": {a: int(L[L.arm == a].q2_excluded.fillna(0).sum()) for a in arms}}
    rng = np.random.default_rng(SEED)
    idx_rr = rng.integers(0, len(rr_tasks), (DRAWS, len(rr_tasks)))
    plant = {}
    for name, col in COL.items():
        r1, r2 = (one[a].loc[rr_tasks, col].to_numpy(float) for a in ("attack", "attack_run2"))
        d = r1 - r2; p0 = float(d.mean()); mg = MARGIN[name]; hits, achieved = [], []
        for c in range(COPIES):
            g = np.random.default_rng(c); new = r2.copy()
            if col == "any":
                zeros = np.flatnonzero(new == 0); k = min(round(mg * len(new)), len(zeros))
                new[g.choice(zeros, k, replace=False)] = 1.0
            else:
                pick = g.choice(len(new), len(new) // 2, replace=False); new[pick] = np.minimum(new[pick] + 2 * mg, 1.0)
            dp = r1 - new
            achieved.append(float((r2 - new).mean()))  # raising run 2 lowers run 1 - run 2: the intended shift is -margin
            hits.append(detected(float(dp.mean()), tuple(ci(dp[idx_rr].mean(1))), p0, -mg))
        plant[name] = {"run 1 - run 2 (unplanted)": {"point": p0, "95%": ci(d[idx_rr].mean(1))},
                       "base rate run 1, run 2": [float(r1.mean()), float(r2.mean())],
                       "achieved plant shift (mean over copies)": float(np.mean(achieved)), "intended": -mg,
                       "margin-sized plant detected": float(np.mean(hits)), "testable at its margin (>= 80%)": bool(np.mean(hits) >= 0.8)}
    out["planted check (yardstick)"] = plant
    if not plant_only and plant != prior["planted check (yardstick)"]:
        raise SystemExit("the recomputed planted check differs from the reported plant result")
    if not plant_only:
        rs_tasks = sorted(set(one["attack"].index) & set(one["sim_attack@september"].index))
        tasks = sorted(set(rs_tasks) | set(rr_tasks)); pos = {t: i for i, t in enumerate(tasks)}
        idx = np.random.default_rng(SEED).integers(0, len(tasks), (DRAWS, len(tasks)))
        eq, bey, res = {}, {}, {}
        for name, col in [(n, c) for n, c in COL.items() if prior["planted check (yardstick)"][n]["testable at its margin (>= 80%)"]]:
            drs = np.full(len(tasks), np.nan); drr = np.full(len(tasks), np.nan)
            for t in rs_tasks: drs[pos[t]] = one["attack"].at[t, col] - one["sim_attack@september"].at[t, col]
            for t in rr_tasks: drr[pos[t]] = one["attack"].at[t, col] - one["attack_run2"].at[t, col]
            boot = lambda d: (np.where(np.isnan(d), 0, d)[idx].sum(1) / np.maximum((~np.isnan(d))[idx].sum(1), 1))
            b_rs, b_rr = boot(drs), boot(drr); mg = MARGIN[name]
            res[name] = {"real run 1 - sim": {"point": float(np.nanmean(drs)), "95%": ci(b_rs), "tasks": len(rs_tasks)},
                         "real run 1 - run 2": {"point": float(np.nanmean(drr)), "95%": ci(b_rr), "tasks": len(rr_tasks)},
                         "base rates (run 1, sim)": [float(one["attack"].loc[rs_tasks, col].mean()), float(one["sim_attack@september"].loc[rs_tasks, col].mean())]}
            eq[name], bey[name] = equivalence_p(b_rs, mg), beyond_p(b_rs, mg)
        heq, hbey = holm(eq), holm(bey)
        for name in COL:
            if name not in res:
                res[name] = {"verdict": "too insensitive to read (no real-vs-simulated number computed)"}
        for name, o in [(n, o) for n, o in res.items() if n in eq]:
            mg = MARGIN[name]
            agree = inside(o["real run 1 - sim"]["95%"], mg) and heq[name] < LEVEL and inside(o["real run 1 - run 2"]["95%"], mg)
            o |= {"Holm equivalence p": heq[name], "Holm beyond-margin p": hbey[name],
                  "verdict": "agree" if agree else "materially differ" if hbey[name] < LEVEL else "inconclusive"}
        out["real vs simulated"] = res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"n2_target_{'plant' if plant_only else 'full'}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, separators=(",", ":")) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], "--plant-only" in sys.argv,
         sys.argv[sys.argv.index("--plant-result") + 1] if "--plant-result" in sys.argv else None)
