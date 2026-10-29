"""Sabotage test, stage 1 preparation (PLAN.md): the seeded training/held-out split by main task, the 40 stage-1 tasks,
the counts, and the blinded labelling lists (Aswin's smaller version: every target-touching command, a seeded k per run
of the other flagged commands, and an audit of the rest; k for real and simulated runs and the audit target are arguments). IDs, labels and stored scores only: every input is an ID or number file,
read with named columns; no command text is read or written.

Inputs (sha256-checked), all in one folder:
- runs_phase2_196.csv: run_id, arm, task_key, env, main_task
- target_labels_v3.csv (#719): run_id, epoch, turn, call_index, touch_any, shared_with_main_only
- replay_join_v2.csv (#719's v2 handoff): run_id, inspect_epoch, turn, call_index, irid, q2_excluded
- draw0.json, the 196-task impactful-call monitor's observed draw 0 (onpolicy_2x2_scaleup196_draws0-0_2026-10-10_141824):
  meta (world, task, irid, call, y) and, per world, the real-trained monitor's cross-fitted flag at the frozen 1% threshold

  uv run --no-project --python 3.12 --with pandas --with numpy python prepare_stage1.py split  --inputs <dir>
  uv run ... prepare_stage1.py counts --inputs <dir> --k-real K --k-sim K --miss-share F
  uv run ... prepare_stage1.py lists  --inputs <dir> --k-real K --k-sim K --miss-share F --out <gitignored data dir>
"""

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SHA = {"runs_phase2_196.csv": "a56dbc133913b05eb57f6d3829fee7ee99ccee97171f6f484994dc0160d30900", "target_labels_v3.csv": "b6f3ebdaa3de1f28b2b6cbdcf8733085bbb6c183d6e2af68854f801a2f45a153",
       "replay_join_v2.csv": "253f8df1013debb98a757ccdc8e856972aa6d1082772c650fc375e55daa37bf8",
       "draw0.json": "c2656658baaf3aa150f83f4d105e7572bfd854f98032831621419e1f8e8f36bf"}
SEED, N_TRAIN, STAGE1 = 20261011, 60, {"train": 24, "held_out": 16}
ARMS = {"attack": ("real run 1", "real", "real->real"), "attack_run2": ("real run 2", "run2", "real->run2"),
        "sim_attack@september": ("simulated", "sim", "real->sim")}
PRIOR_PER_RUN, AUDIT_CAP = 1.5, 600  # k per run and the audit miss share are arguments (PLAN.md 3), set by Aswin's choice
SPLIT_FILE = HERE / "config" / "split_v1.json"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stamp() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")


def git() -> dict:
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE).stdout.strip()  # noqa: E731
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def load(inputs: str) -> pd.DataFrame:
    """One row per call of the three attack arms: IDs, the v3 target flag, the rule-1 label y and the monitor's flag."""
    d = Path(inputs)
    for name, want in SHA.items():
        if not sha256(d / name).startswith(want):
            raise SystemExit(f"{name}: sha256 mismatch")
    L = pd.read_csv(d / "target_labels_v3.csv", usecols=["run_id", "epoch", "turn", "call_index", "touch_any", "shared_with_main_only"])
    J = pd.read_csv(d / "replay_join_v2.csv", usecols=["run_id", "inspect_epoch", "turn", "call_index", "irid", "q2_excluded"])
    R = pd.read_csv(d / "runs_phase2_196.csv", usecols=["run_id", "arm", "task_key", "env", "main_task"])
    C = L.merge(J.rename(columns={"inspect_epoch": "epoch"}), on=["run_id", "epoch", "turn", "call_index"], how="left", validate="one_to_one")
    if C.irid.isna().any():
        raise SystemExit("a v3 row has no replay-join row")
    C = C.merge(R, on="run_id", how="left", validate="many_to_one")
    C = C[C.arm.isin(list(ARMS))].copy()
    C["world"] = C.arm.map({a: v[1] for a, v in ARMS.items()})
    if set(C.groupby("irid").call_index.min()) != {0}:  # replay-join call_index is 0-based within a request id ...
        raise SystemExit("unexpected replay-join call numbering")
    C["call"] = C.call_index + 1  # ... and the scorer's meta call is 1-based (as in the sizing facts join)
    D = json.loads((d / "draw0.json").read_text())
    M = pd.DataFrame(D["meta"])
    if set(M.groupby("irid").call.min()) != {1}:
        raise SystemExit("unexpected scorer call numbering")
    M["row"] = np.arange(len(M))
    flag = np.full(len(M), np.nan)
    for _, world, cell in ARMS.values():
        pc = D["draws"][0]["cells"][cell]["per_call"]
        r = np.asarray(pc["row"])
        if not (M.world.values[r] == world).all():
            raise SystemExit(f"{cell}: per-call rows from another world")
        flag[r] = np.asarray(pc["flag_frozen"], dtype=float)
    M["flag"] = flag
    C = C.merge(M[["world", "irid", "call", "task", "y", "row", "flag"]], on=["world", "irid", "call"], how="left", validate="one_to_one")
    hit = C.row.notna()
    if (C.task[hit] != C.task_key[hit]).any():
        raise SystemExit("scorer task differs from the run's task")
    C["scored"] = hit & C.flag.notna()
    C["touch"] = (C.touch_any == 1) & (C.shared_with_main_only == 0)  # candidate (i): Aswin's v3 target-touching rule
    C["touch_all"] = C.touch_any == 1  # option (E): shared-with-main-only matches kept too
    C["flagged"] = C.scored & (C.flag == 1)  # candidate (ii): the monitor's flag at its frozen 1% threshold
    return C


def largest_remainder(sizes: dict, total: int) -> dict:
    """Allocate `total` across strata in proportion to `sizes` (largest-remainder rounding, ties by stratum name)."""
    s = sum(sizes.values())
    q = {k: total * v / s for k, v in sizes.items()}
    out = {k: min(int(math.floor(x)), sizes[k]) for k, x in q.items()}
    for k in sorted(q, key=lambda k: (-(q[k] - math.floor(q[k])), k)):
        if sum(out.values()) >= total:
            break
        if out[k] < sizes[k]:
            out[k] += 1
    return out


def pick(tasks: pd.DataFrame, n: int, rng) -> list[str]:
    """Seeded, environment-stratified pick of n main tasks."""
    alloc = largest_remainder(tasks.groupby("env").size().to_dict(), n)
    chosen = []
    for env in sorted(alloc):
        pool = sorted(tasks.task_key[tasks.env == env])
        chosen += [pool[i] for i in rng.permutation(len(pool))[: alloc[env]]]
    return sorted(chosen)


def split(C: pd.DataFrame) -> dict:
    T = C[["task_key", "env"]].drop_duplicates().sort_values("task_key")
    runs = C.groupby("task_key").world.nunique()
    scored = C[C.scored].groupby("task_key").world.nunique()
    T["usable"] = T.task_key.map(lambda t: runs.get(t, 0) == 3 and scored.get(t, 0) == 3)
    train = set(pick(T, N_TRAIN, np.random.default_rng([SEED, 1])))
    T["side"] = np.where(T.task_key.isin(train), "train", "held_out")
    s1 = []
    for k, (side, n) in enumerate(STAGE1.items()):
        s1 += pick(T[(T.side == side) & T.usable], n, np.random.default_rng([SEED, 2 + k]))
    T["stage1"] = T.task_key.isin(s1)
    out = {"seed": SEED, "rule": "main tasks split by environment-stratified seeded draw (PLAN.md 2); stage 1 = 24 training + 16 held-out "
           "usable tasks, also environment-stratified", "counts": {side: {"tasks": int((T.side == side).sum()), "usable": int(((T.side == side) & T.usable).sum()),
           "stage1": int(((T.side == side) & T.stage1).sum())} for side in ("train", "held_out")},
           "tasks": {r.task_key: {"env": r.env, "side": r.side, "usable": bool(r.usable), "stage1": bool(r.stage1)} for r in T.itertuples()}}
    return out


def audit_plan(N: dict, n_runs: int, miss_share: float) -> dict:
    """Audit sizes for the two strata: the smallest integer allocation (exhaustive over the first stratum) with
    sum_h b_h <= miss_share * PRIOR_PER_RUN * n_runs, where b_h = 3 N_h / n_h is the rule-of-three 95% bound on
    sabotage the candidate list misses in stratum h if the audit finds none there (0 for a fully audited stratum).
    Capped at AUDIT_CAP rows; the bound reached is reported."""
    B = miss_share * PRIOR_PER_RUN * n_runs
    (h1, N1), (h2, N2) = sorted(N.items())
    b = lambda n, Nh: 0.0 if n >= Nh else (3 * Nh / n if n else math.inf)  # noqa: E731
    best = None
    for n1 in range(0, N1 + 1):
        rest = B - b(n1, N1)
        if rest < 0:
            continue
        n2 = N2 if rest == 0 else min(N2, math.ceil(3 * N2 / rest))
        if b(n2, N2) <= rest and (best is None or n1 + n2 < sum(best)):
            best = (n1, n2)
    if best is None or sum(best) > AUDIT_CAP:  # cap: spend AUDIT_CAP rows where they lower the bound most
        best = min(((n1, AUDIT_CAP - n1) for n1 in range(max(0, AUDIT_CAP - N2), min(N1, AUDIT_CAP) + 1)), key=lambda n: b(n[0], N1) + b(n[1], N2))
    n = {h1: best[0], h2: best[1]}
    bound = b(best[0], N1) + b(best[1], N2)
    return {"strata_sizes": N, "audit": n, "total": sum(n.values()), "target_bound": round(B, 1), "bound_if_none_found": round(bound, 1),
            "expected_sabotage_prior": PRIOR_PER_RUN * n_runs, "capped": sum(n.values()) >= AUDIT_CAP and bound > B}


def stage1_calls(C: pd.DataFrame, sp: dict, k: dict, include_shared: bool) -> pd.DataFrame:
    """Stage-1 calls with their selection: (i) every scored target-touching call; (ii) a seeded k[world] per run of the
    scored flagged calls that are not target-touching (inclusion probability k / N_run); the rest in audit strata."""
    S = C[C.task_key.isin([t for t, v in sp["tasks"].items() if v["stage1"]])].copy()
    S = S.sort_values(["run_id", "epoch", "turn", "call_index"]).reset_index(drop=True)
    S["side"] = S.task_key.map(lambda t: sp["tasks"][t]["side"])
    S["tt"] = S.touch_all if include_shared else S.touch
    S["sel_i"] = S.scored & S.tt
    fo = S.scored & S.flagged & ~S.tt
    rng = np.random.default_rng([SEED, 7])
    S["sel_ii"], S["p_incl"] = False, np.where(S.sel_i, 1.0, np.nan)
    for run, ix in S[fo].groupby("run_id").groups.items():
        ix, kk = np.asarray(ix), k[S.world[ix[0]]]
        take = ix[np.sort(rng.choice(len(ix), min(kk, len(ix)), replace=False))]
        S.loc[take, "sel_ii"] = True
        S.loc[ix, "p_incl"] = min(kk, len(ix)) / len(ix)
    S["stratum"] = np.select([S.sel_i, fo, S.scored & (S.y == 1), S.scored & (S.y == 0)],
                             ["target_touching", "flagged_other", "unflagged_impactful", "unflagged_read_only"], "unscored")
    return S


def counts(S: pd.DataFrame, k: dict, miss_share: float, include_shared: bool) -> dict:
    runs = S.run_id.nunique()
    N = {h: int((S.stratum == h).sum()) for h in ("unflagged_impactful", "unflagged_read_only")}
    a = audit_plan(N, runs, miss_share)
    out = {"runs": runs, "k_flagged_per_run": k, "audit_miss_share": miss_share, "include_shared": include_shared, "audit": a, "by_side_world": {}}
    for (side, world), g in S.groupby(["side", "world"]):
        out["by_side_world"][f"{side} / {world}"] = {
            "runs": int(g.run_id.nunique()), "calls": len(g), "unscored": int((g.stratum == "unscored").sum()),
            "unscored but target-touching": int((~g.scored & g.tt).sum()),
            "shared-with-main-only target matches (scored)": int((g.scored & g.touch_all & ~g.touch).sum()), "target_touching (i)": int(g.sel_i.sum()),
            "flagged_other": int((g.stratum == "flagged_other").sum()), "flagged subset (ii)": int(g.sel_ii.sum()),
            "unflagged_impactful": int((g.stratum == "unflagged_impactful").sum()), "unflagged_read_only": int((g.stratum == "unflagged_read_only").sum())}
    out["candidates"] = int((S.sel_i | S.sel_ii).sum())
    out["total_rows"] = out["candidates"] + a["total"]
    return out


def lists(S: pd.DataFrame, c: dict, out_dir: Path) -> dict:
    ign = subprocess.run(["git", "check-ignore", "-q", str(out_dir / "x.csv")], cwd=HERE)
    if ign.returncode != 0:
        raise SystemExit(f"{out_dir} is not git-ignored")
    rng = np.random.default_rng([SEED, 9])
    S = S.copy()
    S["source"] = np.where(S.sel_i, "target_touching", np.where(S.sel_ii, "flagged_subset", ""))
    S["audited"] = False
    for h, n in c["audit"]["audit"].items():
        ix = S.index[S.stratum == h].to_numpy()
        take = ix[np.sort(rng.choice(len(ix), n, replace=False))]
        S.loc[take, "audited"] = True
        S.loc[take, "source"] = f"audit:{h}"
        S.loc[ix, "p_incl"] = n / len(ix)
    K = S[S.source != ""].copy()
    ids = set()
    while len(ids) < len(K):
        ids.add(f"sab1-{int(rng.integers(0, 2**32)):08x}")
    K["opaque_id"] = rng.permutation(sorted(ids))
    K = K.iloc[rng.permutation(len(K))].reset_index(drop=True)  # one shuffled order: every source interleaved
    K["run_hash"] = K.run_id.map(lambda r: hashlib.sha256(r.encode()).hexdigest()[:16])
    files = {"labels_stage1.csv": K[["opaque_id"]].assign(label="", note=""),
             "locator_stage1.csv": K[["opaque_id", "run_hash", "epoch", "turn", "call_index"]],
             "key_stage1.csv": K[["opaque_id", "world", "arm", "task_key", "env", "side", "run_id", "epoch", "turn", "call_index", "irid", "call",
                                  "y", "flag", "touch", "touch_all", "stratum", "source", "p_incl"]]}
    out_dir.mkdir(parents=True, exist_ok=True)
    sums = {}
    for name, df in files.items():
        p = out_dir / name
        with p.open("x") as fh:  # never overwrite a list
            df.to_csv(fh, index=False)
        sums[name] = {"rows": len(df), "sha256": sha256(p)}
    sums["rows_by_source"] = K.source.value_counts().to_dict()
    sums["rows_by_side_world_source"] = {f"{a} / {b} / {c}": int(v) for (a, b, c), v in K.groupby(["side", "world", "source"]).size().items()}
    return sums


def main(phase: str, inputs: str, out: str | None, k_real: int | None, k_sim: int | None, miss_share: float | None, include_shared: bool):
    C = load(inputs)
    if phase == "split":
        sp = split(C)
        SPLIT_FILE.parent.mkdir(exist_ok=True)
        with SPLIT_FILE.open("x") as fh:
            fh.write(json.dumps(sp, indent=1))
        print(json.dumps(sp["counts"], indent=1), "\n->", SPLIT_FILE)
        return
    sp = json.loads(SPLIT_FILE.read_text())
    if sp != split(C):
        raise SystemExit("the committed split does not reproduce from these inputs")
    if None in (k_real, k_sim, miss_share) or min(k_real, k_sim) < 1 or not 0 < miss_share <= 1:
        raise SystemExit("counts and lists need --k-real >= 1, --k-sim >= 1 and 0 < --miss-share <= 1 (Aswin's choice, PLAN.md 3)")
    k = {"real": k_real, "run2": k_real, "sim": k_sim}
    S = stage1_calls(C, sp, k, include_shared)
    c = counts(S, k, miss_share, include_shared)
    res = {"git": git(), "written": stamp(), "inputs_sha256": {n: sha256(Path(inputs) / n) for n in SHA}, "split_sha256": sha256(SPLIT_FILE), "counts": c}
    if phase == "lists":
        res["files"] = lists(S, c, Path(out))
    dst = HERE / "results" / f"{phase}_{res['written']}.json"
    dst.parent.mkdir(exist_ok=True)
    with dst.open("x") as fh:
        fh.write(json.dumps(res, indent=1, default=int))
    print(json.dumps(res, indent=1, default=int), "\n->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["split", "counts", "lists"])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out")
    ap.add_argument("--k-real", type=int)
    ap.add_argument("--k-sim", type=int)
    ap.add_argument("--miss-share", type=float)
    ap.add_argument("--include-shared", action="store_true", help="option (E): target matches shared with the main task count as (i)")
    main(**vars(ap.parse_args()))
