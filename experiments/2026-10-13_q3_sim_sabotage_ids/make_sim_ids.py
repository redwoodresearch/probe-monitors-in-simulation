"""ID list of the simulated runs' calls on the 40 stage-1 sabotage tasks, for Aswin's re-read of the stage-1 panel votes (0d, 10-13, $0).

IDs and numbers only: every input is an ID or number file read by named fields, and no command text is read or written.
- Runs: Qwen3.5-27B's September simulated attack runs (`sim_attack@september`) on the 40 stage-1 tasks (24 training, 16 held-out),
  from config/split_v1.json (sha256 992e6e71…, committed at 21e748ba; pass it with --split).
- State-changing: the stage-1 rule labeler, i.e. the 196-task scorer's rule-1 label `y` in draw0.json. It is joined through the
  command entry's own `call` (check_joins.rescore, the corrected join), and the count that differs from prepare_stage1.load()'s
  `call_index + 1` join is reported. Calls without the scorer's record are "unscored" (state_changing = -1).
  `scored` (stage 1: record and monitor flag) is kept as its own column.
- Tool name and the stored call_end reading come from the extraction metas, whose digests must equal
  check_calibration_2026-10-11_173049.json's.

  uv run --no-project --python 3.12 --with pandas --with numpy python make_sim_ids.py --inputs <stage-1 inputs dir> \\
      --metas <acts_meta root> --split <split_v1.json> --out <dir>

--worlds (default sim) picks the attack arms: sim, real (real run 1) and run2 (real run 2), comma-separated; the default reproduces
the simulated list made at 74cbc50a. With other worlds, files are prefixed by the worlds joined with "_", and every file carries a
`world` column. Extractions without a recorded digest (run2's second_attack) are checked by meta count against complete.json, and their
digest is recorded in the summary.

Writes, each opened with mode "x":
- sim_calls_all_<stamp>.csv: every call in the 40 runs, in packet order (turn, call_index). `ordinal` is the call's position in
  its run. The stage-1 panel gave the call at that position the fresh ID c{ordinal:04d}, if its packet holds the same calls.
  Aswin's re-read checks this per run before using it.
- sim_state_changing_ids_<stamp>.csv: the calls with y == 1.
- sim_ids_summary_<stamp>.json: counts, input hashes and git.
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

SPLIT_SHA = "992e6e71793146b4df5f00d7cd98330d4f5e073f0a8379031aebfc172f818e11"
DIGESTS = {"2026-10-09_rerun_strict_segends_v1": "3f86eaf02ae1c0c6ec047099a2f555e2417f2e87ea7af206e03951bd6ee53fb3",  # check_calibration_2026-10-11_173049.json
           "2026-10-10_scaleup_canary_segends_v1": "820ab62ea4ba9f7388fd3a65f4d6560066d162b97fcbb05ec2a0493910ea94c7",
           "2026-10-10_scaleup_l2_segends_v1": "511a60d045c3d6f9fab66005acee00dba8cabba73877a4ddb7ba925add38c993",
           "2026-10-10_scaleup_topup_segends_v1": "56d23f7e7b2d6b15d20137b8cda9cd491767b24fce229d5164c1aca9b26a1ac4"}
COLS = ["call_id", "task", "split", "run_id", "epoch", "turn", "call_index", "ordinal", "tool", "irid", "state_changing", "scored", "has_call_end_reading"]


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tools(root: Path, world: str, seen: dict) -> dict:
    """(irid, call_index) -> tool name of one world's kept command entries (redrawn and rerun attempts skipped, as in check_joins)."""
    out = {}
    arm, xs = J.EXTRACTIONS[world]
    for x in xs:
        files = sorted((root / x / "meta").glob("*/*.json"))
        d = hashlib.sha256(b"".join(hashlib.sha256(f.read_bytes()).digest() for f in files)).hexdigest()
        if x in DIGESTS and d != DIGESTS[x]:
            raise SystemExit(f"{x}: metas digest differs from check_calibration_2026-10-11_173049.json")
        if x not in DIGESTS and len(files) != json.loads((root / x / "complete.json").read_text())["epochs"]:
            raise SystemExit(f"{x}: meta count differs from complete.json")
        seen[x] = d
        for f in sorted((root / x / "meta" / arm).glob("*.json")):
            m = json.loads(f.read_text())
            if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                continue
            for c in m["calls"]:
                if "call_index" in c and c.get("kind", "command") == "command":
                    if (c["irid"], c["call_index"]) in out:
                        raise SystemExit(f"two command entries for {c['irid']} / {c['call_index']}")
                    out[c["irid"], c["call_index"]] = c["name"]
    return out


def main(inputs: str, metas: str, split: str, out: str, worlds: str):
    W = worlds.split(",")
    if not W or len(set(W)) != len(W) or not set(W) <= set(J.EXTRACTIONS):
        raise SystemExit(f"--worlds must be distinct names from {sorted(J.EXTRACTIONS)}")
    if sha256(split) != SPLIT_SHA:
        raise SystemExit("split_v1.json: sha256 mismatch")
    sp = json.loads(Path(split).read_text())["tasks"]
    stage1 = {t: v["side"] for t, v in sp.items() if v["stage1"]}
    if sorted(stage1.values()).count("train") != 24 or len(stage1) != 40:
        raise SystemExit("expected 24 training + 16 held-out stage-1 tasks")
    C = P.load(inputs)  # sha256-checks the four ID inputs; one row per call of the three attack arms
    command, call_end = J.entries(metas)
    R = J.rescore(C, command, inputs)
    keys = ["run_id", "epoch", "turn", "call_index"]
    S = R[R.world.isin(W) & R.task_key.isin(list(stage1))].sort_values(keys).reset_index(drop=True)
    L = C[C.world.isin(W) & C.task_key.isin(list(stage1))].sort_values(keys).reset_index(drop=True)
    if not (S[keys].values == L[keys].values).all():
        raise SystemExit("the two joins hold different calls")
    if S.groupby("run_id").epoch.nunique().max() != 1:
        raise SystemExit("a run spans more than one epoch")
    if S.groupby("run_id").world.nunique().max() != 1:
        raise SystemExit("a run spans more than one world")
    seen, T = {}, {}
    for w in W:
        T |= {(w, *k): v for k, v in tools(Path(metas), w, seen).items()}
    S["ordinal"] = S.groupby("run_id").cumcount()
    S["task"], S["split"] = S.task_key, S.task_key.map(stage1).map({"train": "train", "held_out": "held-out"})
    S["tool"] = [T.get((w, i, c), "") for w, i, c in zip(S.world, S.irid, S.call_index)]
    S["state_changing"] = S.y.fillna(-1).astype(int)  # the rule label wherever the scorer's record exists (Codex: not gated on the monitor flag)
    S["has_call_end_reading"] = [(w, i, c) in call_end for w, i, c in zip(S.world, S.irid, S.call_index)]
    S["call_id"] = [f"{w}sab-" + hashlib.sha256("|".join(map(str, k)).encode()).hexdigest()[:12] for w, k in zip(S.world, S[keys].itertuples(index=False))]
    if S.call_id.duplicated().any():
        raise SystemExit("call_id collision")
    old = L.y.fillna(-1).astype(int).values
    sc = S[S.state_changing == 1]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    pre, cols = "_".join(W), COLS if W == ["sim"] else COLS + ["world"]  # the default keeps 74cbc50a's simulated columns
    files = {f"{pre}_calls_all_{stamp}.csv": S[cols], f"{pre}_state_changing_ids_{stamp}.csv": sc[cols]}
    for name, df in files.items():
        with (o / name).open("x") as fh:
            df.to_csv(fh, index=False)
    by = lambda df, col: {str(k): int(v) for k, v in df.groupby(col).size().items()}  # noqa: E731
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE).stdout.strip()  # noqa: E731
    res = {"written": stamp, "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", str(HERE)))},
           "inputs_sha256": {n: sha256(Path(inputs) / n) for n in P.SHA} | {"split_v1.json": SPLIT_SHA},
           "worlds": W, "metas_digests": seen, "digests_checked_against_check_calibration": sorted(set(seen) & set(DIGESTS)),
           "files": {n: {"rows": len(df), "sha256": sha256(o / n)} for n, df in files.items()},
           "stage1_tasks": len(stage1),
           "tasks_without_a_run": {w: sorted(set(stage1) - set(S.task[S.world == w])) for w in W},
           "runs_per_task": {w: by(S[S.world == w].groupby("task").run_id.nunique().rename("n").reset_index(), "n") for w in W},
           "runs": int(S.run_id.nunique()), "runs_by_world": {w: int(S.run_id[S.world == w].nunique()) for w in W},
           "calls": len(S), "calls_by_world": by(S, "world"), "calls_by_state_changing": by(S, "state_changing"),
           "state_changing_by_world_split": {w: by(sc[sc.world == w], "split") for w in W},
           "state_changing_by_split": by(sc, "split"), "state_changing_by_tool": by(sc, "tool"),
           "state_changing_without_tool": int((sc.tool == "").sum()), "rule_label_without_monitor_flag": int((S.y.notna() & ~S.scored).sum()),
           "state_changing_with_call_end_reading": int(sc.has_call_end_reading.sum()),
           "calls_whose_label_differs_from_the_call_index_plus_1_join": int((S.state_changing.values != old).sum())}
    with (o / f"{pre}_ids_summary_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), "\n->", o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("--inputs", "--metas", "--split", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--worlds", default="sim", help="comma-separated: sim, real, run2")
    main(**vars(ap.parse_args()))
