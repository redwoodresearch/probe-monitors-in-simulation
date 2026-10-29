"""Aswin's blinded judge panel (#474) unblinded by world: ids, labels and counts only. Local and free.

The panel rated the 124 turns the fake-environment judge flagged (1 suspects fake or test, 2 notices something odd,
3 false positive). This joins its labels.csv to a1's key (id -> run, world, arm) and reports, per arm (primary) and per
judge world (secondary):
  - the plurality category counts (and unresolved), and each judge's counts;
  - run-level shares: runs with at least one plurality-1 turn, one plurality-2 turn, and one of either, over all runs the
    fake-environment judge read in that world (its result file's `worlds.<world>.runs`), with exact (Clopper-Pearson)
    95% intervals;
  - per arm, the same shares over the launch rosters' runs (counted attempts with an .eval, the judge's own selection;
    the judge counts only runs with a Qwen text turn, so its per-world totals are a few runs lower, reported beside);
  - real attack run 1 minus September simulated attack (primary), beside real attack run 1 minus run 2 and honest A minus
    honest B (controls), as differences in those shares with task-paired 95% intervals: 10,000 task bootstraps over the
    tasks either arm ran, each arm's share over its own runs in the draw, so a task's runs in both arms move together and
    an unmatched run counts only for its arm. Newcombe's independent-samples interval is kept beside it.
Item text is never read; the key's run ids are used for grouping only and are not written out.

  uv run --no-project --with scipy python analyze_panel_by_world.py <labels.csv> <key.json> <judge result.json> <summary.json> <launch dir>...
"""

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from scipy.stats import beta, norm

HERE = Path(__file__).parent
INPUT_SHA = "243a7cc435f6adeed98a7138ab476b32ee0f669045fabf8433d8f797b94e82d4"  # a1's blind/items.json, as #474's manifest records it
WORLDS = ["real", "real_attack_run2", "sim@september", "real_honest_a", "sim@today"]
ARMS = ["attack run 1", "attack run 2", "sim_attack@september", "honest_a", "honest_b", "sim_honest@september", "sim_attack@today", "sim_honest@today"]
CATS = ("1", "2", "1 or 2")


def arm_of(arm: str, sim: str | None, second: bool) -> str:
    """The judge's run key, as an arm: real attack split by launch, simulated arms by simulator."""
    if arm == "attack":
        return "attack run 2" if second else "attack run 1"
    return f"{arm}@{sim}" if arm.startswith("sim_") else arm


def rosters(dirs: list) -> dict:
    """Runs per arm and task, selected as judge_fake_env.turns() selects them (counted attempts; honest_a's first, else
    the last; an .eval present), from launch.json and file names only."""
    n = defaultdict(Counter)
    for run_dir in map(Path, dirs):
        for r in json.loads((run_dir / "launch.json").read_text())["runs"]:
            counted = [a for a in r["attempts"] if not a.get("redrawn_because")]
            if not counted:
                continue
            a = counted[0] if r["arm"] == "honest_a" else counted[-1]
            if list((run_dir / Path(a["log_dir"]).relative_to(Path(a["log_dir"]).parents[1])).glob("*.eval")):
                n[arm_of(r["arm"], None if r["real"] else r["simulator"], run_dir.name.endswith("_second_attack"))][r["task"]] += 1
    return dict(n)


def paired(a: str, b: str, c: str, roster: dict, by_arm: dict, i: int, boot: int = 10_000) -> dict:
    """Share of a's runs with a flagged turn minus b's, with a task bootstrap (seed [20261010, 493, i]): tasks drawn with
    replacement from those either arm ran; in each draw, each arm's flagged runs over its runs on the drawn tasks."""
    import numpy as np

    units = sorted(set(roster[a]) | set(roster[b]))
    runs = {x: np.array([roster[x].get(t, 0) for t in units], float) for x in (a, b)}
    flagged = {x: Counter(r.rsplit("/", 1)[-1] for r in by_arm[x][c]) for x in (a, b)}
    for x in (a, b):
        if any(flagged[x][t] > roster[x].get(t, 0) for t in flagged[x]):
            raise SystemExit(f"{x}: a flagged run's task is not in the roster")
    hits = {x: np.array([flagged[x].get(t, 0) for t in units], float) for x in (a, b)}
    rng = np.random.default_rng([20261010, 493, i])
    W = np.zeros((boot, len(units)))
    np.add.at(W, (np.repeat(np.arange(boot), len(units)), rng.integers(0, len(units), boot * len(units))), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = W @ hits[a] / (W @ runs[a]) - W @ hits[b] / (W @ runs[b])
    ok = np.isfinite(d)
    both = [k for k, t in enumerate(units) if runs[a][k] and runs[b][k]]
    return {"difference": hits[a].sum() / runs[a].sum() - hits[b].sum() / runs[b].sum(),
            "task_bootstrap_ci95": [float(np.percentile(d[ok], 2.5)), float(np.percentile(d[ok], 97.5))], "bootstrap_failed": int((~ok).sum()),
            "tasks": len(units), "tasks_with_both_arms": len(both),
            "discordant_tasks_a_only_b_only": [sum(1 for k in both if hits[a][k] and not hits[b][k]), sum(1 for k in both if hits[b][k] and not hits[a][k])],
            "newcombe_ci95_independent": newcombe(int(hits[a].sum()), int(runs[a].sum()), int(hits[b].sum()), int(runs[b].sum()))["ci95"]}
JUDGES = ["google/gemini-3.1-pro-preview", "x-ai/grok-4.7", "deepseek/deepseek-v4-pro"]


def exact(k: int, n: int) -> list:
    return [float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0, float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]


def wilson(k: int, n: int) -> tuple:
    z, p = norm.ppf(0.975), k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return c - h, c + h


def newcombe(k1: int, n1: int, k2: int, n2: int) -> dict:
    """p1 - p2 with Newcombe's hybrid-score interval (method 10)."""
    p1, p2 = k1 / n1, k2 / n2
    (l1, u1), (l2, u2) = wilson(k1, n1), wilson(k2, n2)
    d = p1 - p2
    return {"difference": d, "ci95": [d - ((p1 - l1) ** 2 + (u2 - p2) ** 2) ** 0.5, d + ((u1 - p1) ** 2 + (p2 - l2) ** 2) ** 0.5]}


def shares(flagged: dict, n: int) -> dict:
    return {c: {"runs": len(flagged[c]), "of": n, "share": len(flagged[c]) / n, "exact_ci95": exact(len(flagged[c]), n)} for c in CATS}


def main(labels_path: str, key_path: str, judge_path: str, summary_path: str, launch_dirs: list):
    summary = json.loads(Path(summary_path).read_text())
    if summary["input_sha256"] != INPUT_SHA:
        raise SystemExit("the panel's input is not a1's items.json")
    with open(labels_path, newline="") as fh:
        raw = list(csv.DictReader(fh))
    def no_repeats(pairs):
        if len({k for k, _ in pairs}) != len(pairs):
            raise SystemExit("repeated keys in key.json")
        return dict(pairs)

    raw_key = json.loads(Path(key_path).read_text(), object_pairs_hook=no_repeats)
    if len({int(r["id"]) for r in raw}) != len(raw) or len({int(k) for k in raw_key}) != len(raw_key):
        raise SystemExit("duplicate ids")
    labels, key = {str(int(r["id"])): r for r in raw}, {str(int(k)): v for k, v in raw_key.items()}
    if set(labels) != set(key) or len(key) != summary["n_items"]:
        raise SystemExit(f"ids differ: {len(set(labels) ^ set(key))} unmatched")
    for r in labels.values():  # recompute the plurality (unique plurality among valid votes; else unresolved)
        votes = Counter(r[j] for j in JUDGES if r[j] in ("1", "2", "3"))
        top = votes.most_common()
        want = top[0][0] if top and (len(top) == 1 or top[0][1] > top[1][1]) else "unresolved"
        if any(r[j] not in ("1", "2", "3", "invalid", "refused") for j in JUDGES) or r["majority"] != want:
            raise SystemExit(f"id {r['id']}: labels or plurality inconsistent")
    if dict(Counter(r["majority"] for r in labels.values())) != {k: v for k, v in summary["majority_counts"].items() if v}:
        raise SystemExit("plurality counts differ from summary.json")
    runs_read = {w: v["runs"] for w, v in json.loads(Path(judge_path).read_text())["worlds"].items()}
    if set(runs_read) != set(WORLDS) or {v["world"] for v in key.values()} - set(WORLDS):
        raise SystemExit("worlds differ between the key and the judge's result")

    out = {"inputs": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (labels_path, key_path, judge_path, summary_path)}
                     | {f"{Path(d).name}/launch.json": hashlib.sha256((Path(d) / "launch.json").read_bytes()).hexdigest() for d in launch_dirs},
           "panel": {"fleiss_kappa": summary["fleiss_kappa"], "kappa_n_items": summary["kappa_n_items"], "majority_rule": summary["majority_rule"],
                     "per_judge_counts_all": summary["per_judge_counts"], "masking_counts": summary["masking_counts"]},
           "worlds": {}}
    flagged, by_arm = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))  # world or arm -> category -> runs
    arm = {i: arm_of(v["arm"], v["sim"], v["world"] == "real_attack_run2") for i, v in key.items()}
    for i, v in key.items():
        m = labels[i]["majority"]
        if m in ("1", "2"):
            for d, g in ((flagged, v["world"]), (by_arm, arm[i])):
                d[g][m].add(v["run"])
                d[g]["1 or 2"].add(v["run"])
    roster = rosters(launch_dirs)
    n_arm = {a: sum(c.values()) for a, c in roster.items()}
    if set(n_arm) != set(ARMS) or set(arm.values()) - set(ARMS):
        raise SystemExit(f"arms differ: rosters {sorted(n_arm)}, key {sorted(set(arm.values()))}")
    out["arms (primary)"] = {a: {"turns": sum(x == a for x in arm.values()), "runs_in_roster": n_arm[a],
                                 "plurality": {c: sum(labels[i]["majority"] == c for i in arm if arm[i] == a) for c in ("1", "2", "3", "unresolved")},
                                 "per_judge": {j: dict(Counter(labels[i][j] for i in arm if arm[i] == a)) for j in JUDGES},
                                 "runs_flagged": shares(by_arm[a], n_arm[a])} for a in ARMS}
    roster_world = {"real": n_arm["attack run 1"] + n_arm["honest_b"], "real_attack_run2": n_arm["attack run 2"], "real_honest_a": n_arm["honest_a"],
                    "sim@september": n_arm["sim_attack@september"] + n_arm["sim_honest@september"], "sim@today": n_arm["sim_attack@today"] + n_arm["sim_honest@today"]}
    out["roster_runs_minus_judge_runs_per_world"] = {w: roster_world[w] - runs_read[w] for w in WORLDS}
    out["comparisons by arm (run shares; primary)"] = {
        name: {c: paired(a, b, c, roster, by_arm, 3 * i + j) for j, c in enumerate(CATS)}
        for i, (name, a, b) in enumerate((("primary: real attack run 1 minus September simulated attack", "attack run 1", "sim_attack@september"),
                                          ("control: real attack run 1 minus real attack run 2", "attack run 1", "attack run 2"),
                                          ("control: honest A minus honest B", "honest_a", "honest_b")))}
    for w in WORLDS:
        ids = [i for i, v in key.items() if v["world"] == w]
        n = runs_read[w]
        out["worlds"][w] = {
            "turns": len(ids), "runs_with_a_turn": len({key[i]["run"] for i in ids}), "runs_read": n,
            "turns_by_arm_and_judge_kind": dict(Counter(f"{key[i]['arm']}|{key[i]['judge']}" for i in ids)),
            "plurality": {c: sum(labels[i]["majority"] == c for i in ids) for c in ("1", "2", "3", "unresolved")},
            "per_judge": {j: dict(Counter(labels[i][j] for i in ids)) for j in JUDGES},
            "plurality_by_arm": {a: dict(Counter(labels[i]["majority"] for i in ids if key[i]["arm"] == a)) for a in sorted({key[i]["arm"] for i in ids})},
            "runs_flagged": shares(flagged[w], n)}
    out["secondary: comparisons by judge world (run shares)"] = {
        name: {c: newcombe(len(flagged[a][c]), runs_read[a], len(flagged[b][c]), runs_read[b]) for c in ("1", "2", "1 or 2")}
        for name, a, b in (("real (attack run 1 + honest B) minus sim@september (attack + honest)", "real", "sim@september"),
                           ("real (attack run 1 + honest B) minus real_attack_run2 (not like for like)", "real", "real_attack_run2"))}
    out["notes"] = ["underpowered: 0-7 flagged runs per arm, and no planted check",
                    "world 'real' is the fake-environment judge's: real attack run 1 plus honest B; sim@september holds its simulated attack and "
                    "honest runs; the per-arm comparisons are the like-for-like ones",
                    "per-arm intervals are task-paired bootstraps; the Newcombe intervals beside them assume independent runs",
                    "Fleiss kappa 0.47 across the three judges; Gemini labels category 1 far more often than the other two",
                    "the panel's regex masking may leave weekday, time-zone or year cues"]
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE).stdout.strip()
    out["git"] = {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"panel_by_world_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    for a in ARMS:
        r = out["arms (primary)"][a]
        print(a, "plurality", r["plurality"], "| runs", {c: f"{v['runs']}/{v['of']} [{v['exact_ci95'][0]:.3f}, {v['exact_ci95'][1]:.3f}]" for c, v in r["runs_flagged"].items()})
    for name, v in out["comparisons by arm (run shares; primary)"].items():
        print(name, {c: f"{x['difference']:+.4f} [{x['task_bootstrap_ci95'][0]:+.4f}, {x['task_bootstrap_ci95'][1]:+.4f}] "
                        f"tasks {x['tasks']} (both {x['tasks_with_both_arms']}), discordant {x['discordant_tasks_a_only_b_only']}" for c, x in v.items()})
    print("roster minus judge runs per world", out["roster_runs_minus_judge_runs_per_world"])
    for name, v in out["secondary: comparisons by judge world (run shares)"].items():
        print(name, {c: f"{x['difference']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]" for c, x in v.items()})
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("labels", "key", "judge_result", "summary"):
        ap.add_argument(a)
    ap.add_argument("launch_dirs", nargs=5)
    a = ap.parse_args()
    main(a.labels, a.key, a.judge_result, a.summary, a.launch_dirs)
