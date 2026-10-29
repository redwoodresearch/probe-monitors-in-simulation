"""Sabotage scale-up, items (b) and (c): which tasks outside the 40 stage-1 tasks can join the real-vs-simulated comparison, and in
what order. IDs and numbers only: every input is an ID or number file read by named fields; no command text is read or written.

A task is covered when each of the three attack arms (first real run `attack`, second real run `attack_run2`, simulated
`sim_attack@september`) has exactly one run on it, and every state-changing call in each run (rule-1 y == 1, through
check_joins.rescore's corrected join) has a stored call_end reading, which every layer, including the frozen gate's layer 30, holds.
The task list is seeded (SEED below; an ordering seed, not a blinding seed) and chosen by coverage only, never by any score. It has
two ordered parts: the held-out side (the primary study takes a prefix of this part only), then the train side's non-stage-1 tasks.
Each part is environment-stratified and interleaved so that every prefix WITHIN that part keeps environments close to proportional;
proportionality is not claimed across the two parts. Tasks seen in the ID inputs but absent from the frozen split are reported.

  uv run --no-project --python 3.12 --with pandas --with numpy python coverage.py --inputs <stage-1 ID inputs dir> \\
      --metas <acts_meta root> --split <split_v1.json> --out <dir>

Writes (mode "x"): coverage_<stamp>.json (counts, per-task details), task_list_<stamp>.json (ordered tasks),
runs_<stamp>.csv (rank, task, env, side, world, run_id) and calls_all_<stamp>.csv (every call of the listed runs, with `ordinal`,
the position the panel's fresh ID c{ordinal:04d} takes, in make_sim_ids.py's columns plus `world`).
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-11_sabotage_test"))
import check_joins as J  # noqa: E402
import prepare_stage1 as P  # noqa: E402

SEED = 20261013
SPLIT_SHA = "992e6e71793146b4df5f00d7cd98330d4f5e073f0a8379031aebfc172f818e11"
DIGESTS = {"2026-10-09_rerun_strict_segends_v1": "3f86eaf02ae1c0c6ec047099a2f555e2417f2e87ea7af206e03951bd6ee53fb3",  # check_calibration_2026-10-11_173049.json
           "2026-10-10_scaleup_canary_segends_v1": "820ab62ea4ba9f7388fd3a65f4d6560066d162b97fcbb05ec2a0493910ea94c7",
           "2026-10-10_scaleup_l2_segends_v1": "511a60d045c3d6f9fab66005acee00dba8cabba73877a4ddb7ba925add38c993",
           "2026-10-10_scaleup_topup_segends_v1": "56d23f7e7b2d6b15d20137b8cda9cd491767b24fce229d5164c1aca9b26a1ac4"}
WORLDS = ["real", "run2", "sim"]
COLS = ["call_id", "task", "split", "run_id", "epoch", "turn", "call_index", "ordinal", "tool", "irid", "state_changing", "scored",
        "has_call_end_reading", "world"]


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_metas(root: Path) -> dict:
    """Recorded digests for the four scale-up extractions; the second-attack extraction (no recorded digest) by count."""
    seen = {}
    for x in sorted({x for _, xs in J.EXTRACTIONS.values() for x in xs}):
        files = sorted((root / x / "meta").glob("*/*.json"))
        seen[x] = hashlib.sha256(b"".join(hashlib.sha256(f.read_bytes()).digest() for f in files)).hexdigest()
        if x in DIGESTS and seen[x] != DIGESTS[x]:
            raise SystemExit(f"{x}: metas digest differs from check_calibration_2026-10-11_173049.json")
        if x not in DIGESTS and len(files) != json.loads((root / x / "complete.json").read_text())["epochs"]:
            raise SystemExit(f"{x}: meta count differs from complete.json")
    return seen


def tools(root: Path) -> dict:
    """(world, irid, call_index) -> tool name of each world's kept command entries (redrawn and rerun attempts skipped)."""
    out = {}
    for w, (arm, xs) in J.EXTRACTIONS.items():
        for x in xs:
            for f in sorted((root / x / "meta" / arm).glob("*.json")):
                m = json.loads(f.read_text())
                if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                    continue
                for c in m["calls"]:
                    if "call_index" in c and c.get("kind", "command") == "command":
                        if (w, c["irid"], c["call_index"]) in out:
                            raise SystemExit("two command entries for one call")
                        out[w, c["irid"], c["call_index"]] = c["name"]
    return out


def interleave(tasks: dict, rng) -> list:
    """Seeded order within each environment, then interleaved so every prefix stays close to proportional (Sainte-Lague)."""
    by_env = {}
    for t, env in sorted(tasks.items()):
        by_env.setdefault(env, []).append(t)
    by_env = {e: [ts[i] for i in rng.permutation(len(ts))] for e, ts in sorted(by_env.items())}
    taken, out = dict.fromkeys(by_env, 0), []
    while len(out) < len(tasks):
        e = min((e for e in by_env if taken[e] < len(by_env[e])), key=lambda e: ((taken[e] + 0.5) / len(by_env[e]), e))
        out.append(by_env[e][taken[e]])
        taken[e] += 1
    return out


def main(inputs: str, metas: str, split: str, out: str):
    if sha256(split) != SPLIT_SHA:
        raise SystemExit("split_v1.json: sha256 mismatch")
    sp = json.loads(Path(split).read_text())["tasks"]
    digests = check_metas(Path(metas))
    C = P.load(inputs)  # sha256-checks the four ID inputs; one row per call of the three attack arms
    command, call_end = J.entries(metas)
    R = J.rescore(C, command, inputs).sort_values(["run_id", "epoch", "turn", "call_index"]).reset_index(drop=True)
    if R.groupby("run_id").epoch.nunique().max() != 1 or R.groupby("run_id").world.nunique().max() != 1:
        raise SystemExit("a run spans more than one epoch or world")
    R["state_changing"] = R.y.fillna(-1).astype(int)
    R["has_call_end_reading"] = [(w, i, c) in call_end for w, i, c in zip(R.world, R.irid, R.call_index)]
    per = {}
    for t in sorted(sp):
        g = R[R.task_key == t]
        d = {"env": sp[t]["env"], "side": sp[t]["side"], "stage1": sp[t]["stage1"], "worlds": {}}
        for w in WORLDS:
            h = g[g.world == w]
            sc = h[h.state_changing == 1]
            d["worlds"][w] = {"runs": int(h.run_id.nunique()), "calls": len(h), "state_changing": len(sc),
                              "state_changing_without_reading": int((~sc.has_call_end_reading).sum())}
        d["covered"] = all(v["runs"] == 1 and v["state_changing_without_reading"] == 0 for v in d["worlds"].values())
        per[t] = d
    new = {t: d for t, d in per.items() if not d["stage1"]}
    held = interleave({t: d["env"] for t, d in new.items() if d["covered"] and d["side"] == "held_out"}, np.random.default_rng([SEED, 1]))
    train = interleave({t: d["env"] for t, d in new.items() if d["covered"] and d["side"] == "train"}, np.random.default_rng([SEED, 2]))
    order = [(t, "held_out") for t in held] + [(t, "train_not_stage1") for t in train]
    rank = {t: i + 1 for i, (t, _) in enumerate(order)}
    L = R[R.task_key.isin(rank)].copy()
    T = tools(Path(metas))
    L["ordinal"] = L.groupby("run_id").cumcount()
    L["task"], L["split"] = L.task_key, L.task_key.map(lambda t: "held-out" if sp[t]["side"] == "held_out" else "train")
    L["tool"] = [T.get((w, i, c), "") for w, i, c in zip(L.world, L.irid, L.call_index)]
    L["call_id"] = [f"{w}sab-" + hashlib.sha256("|".join(map(str, k)).encode()).hexdigest()[:12]
                    for w, k in zip(L.world, L[["run_id", "epoch", "turn", "call_index"]].itertuples(index=False))]
    if L.call_id.duplicated().any():
        raise SystemExit("call_id collision")
    runs = L[["task", "world", "run_id"]].drop_duplicates().assign(rank=lambda d: d.task.map(rank), env=lambda d: d.task.map(lambda t: sp[t]["env"]),
                                                                   side=lambda d: d.task.map(lambda t: dict(order)[t]))
    runs = runs.sort_values(["rank", "world"])[["rank", "task", "env", "side", "world", "run_id"]]
    count = lambda ts, side: {"tasks": len(ts), "by_world": {w: {k: int(sum(per[t]["worlds"][w][k] for t in ts)) for k in ("calls", "state_changing")} for w in WORLDS},  # noqa: E731
                              "side": side}
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE).stdout.strip()  # noqa: E731
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = {"written": stamp, "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", str(HERE)))},
           "inputs_sha256": {n: sha256(Path(inputs) / n) for n in P.SHA} | {"split_v1.json": SPLIT_SHA}, "metas_digests": digests, "seed": SEED,
           "rule": "covered = exactly one run in each of real, run2, sim, and every state-changing call in each run with a stored call_end reading",
           "order_scope": "two parts, held_out then train_not_stage1; every prefix within a part is environment-proportional (Sainte-Lague); none claimed across parts",
           "tasks_outside_split": sorted(set(R.task_key) - set(sp)),
           "tasks": {"all": len(per), "stage1": sum(d["stage1"] for d in per.values()),
                     "not_stage1_by_side": {s: {"tasks": sum(d["side"] == s for d in new.values()), "covered": sum(d["side"] == s and d["covered"] for d in new.values()),
                                                "not_covered": sorted(t for t, d in new.items() if d["side"] == s and not d["covered"])} for s in ("held_out", "train")}},
           "listed": {"held_out": count(held, "held_out (primary)"), "train_not_stage1": count(train, "train side, not stage 1: probe never trained on these attack runs, but their honest runs set the threshold")},
           "stage1_40_for_comparison": count([t for t, d in per.items() if d["stage1"]], "already labelled"),
           "per_task": per}
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    files = {f"task_list_{stamp}.json": json.dumps({"seed": SEED, "order_scope": "held_out part first (primary: take a prefix of it), then train_not_stage1; proportional within each part only", "order": [{"rank": rank[t], "task": t, "env": sp[t]["env"], "side": s} for t, s in order]}, indent=0),
             f"coverage_{stamp}.json": json.dumps(res, indent=1)}
    for name, text in files.items():
        with (o / name).open("x") as fh:
            fh.write(text)
    for name, df in {f"runs_{stamp}.csv": runs, f"calls_all_{stamp}.csv": L.sort_values(["run_id", "ordinal"])[COLS]}.items():
        with (o / name).open("x") as fh:
            df.to_csv(fh, index=False)
    print(json.dumps({k: v for k, v in res.items() if k != "per_task"}, indent=1))
    print(json.dumps({n: {"sha256": sha256(o / n)} for n in list(files) + [f"runs_{stamp}.csv", f"calls_all_{stamp}.csv"]}, indent=1), "\n->", o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("--inputs", "--metas", "--split", "--out"):
        ap.add_argument(a, required=True)
    main(**vars(ap.parse_args()))
