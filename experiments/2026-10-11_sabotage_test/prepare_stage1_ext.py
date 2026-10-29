"""Insurance option (C), held back (PLAN.md 3, "Extension"): the flagged calls that take each TRAINING real run (real run 1
and real run 2 of the 24 stage-1 training tasks) from k = 12 to k = 20. IDs and numbers only; the existing lists are not
touched.

The 12-per-run subset is reproduced exactly from prepare_stage1.py (checked against key_stage1.csv). From each run's
remaining flagged calls outside the target-touching set, min(20, N) - min(12, N) more are drawn now, without replacement,
with a separate seed, before any label exists. Release depends on the first labels (PLAN.md), so the union is not treated
as a uniform 20-subset. Weights for the run's flagged calls outside the target-touching set:
- not released: each of the original rows stands for N / min(12, N) calls; the extension is never labelled.
- released: conditional on the original rows, the extension is a uniform draw from the run's other flagged calls, so
  each original row stands for itself (weight 1) and each extension row for (N - min(12, N)) / (min(20, N) - min(12, N)).
  This is conditionally unbiased given the first draw, whatever triggered the release.

  uv run --no-project --python 3.12 --with pandas --with numpy python prepare_stage1_ext.py --inputs <dir> --key <key_stage1.csv> \\
      --out <gitignored data dir>
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import prepare_stage1 as P

K_BEFORE, K_AFTER, EXT_SEED = 12, 20, [P.SEED, 11]


def main(inputs: str, key: str, out: str):
    out_dir = Path(out).resolve()
    if subprocess.run(["git", "check-ignore", "-q", str(out_dir / "x.csv")], cwd=P.HERE).returncode != 0:
        raise SystemExit(f"{out_dir} is not git-ignored")
    C = P.load(inputs)
    sp = json.loads(P.SPLIT_FILE.read_text())
    S = P.stage1_calls(C, sp, {"real": K_BEFORE, "run2": K_BEFORE, "sim": 3}, False)
    K = pd.read_csv(key, dtype={"opaque_id": str})
    listed = set(zip(K.world, K.irid, K.call_index))
    mine = set(zip(S.world[S.sel_ii], S.irid[S.sel_ii], S.call_index[S.sel_ii]))
    if mine != set(zip(K.world[K.source == "flagged_subset"], K.irid[K.source == "flagged_subset"], K.call_index[K.source == "flagged_subset"])):
        raise SystemExit("the 12-per-run subset does not reproduce key_stage1.csv")
    rng = np.random.default_rng(EXT_SEED)
    T = S[(S.side == "train") & S.world.isin(["real", "run2"]) & (S.stratum == "flagged_other")]
    rows, weights = [], []
    for run, g in T.groupby("run_id", sort=True):
        n = len(g)
        rest = g[~g.sel_ii]
        add = min(K_AFTER, n) - min(K_BEFORE, n)
        take = rest.iloc[np.sort(rng.choice(len(rest), add, replace=False))] if add > 0 else rest.iloc[:0]
        if set(zip(take.world, take.irid, take.call_index)) & listed:
            raise SystemExit("an extension row is already listed")
        rows.append(take)
        kb, ka = min(K_BEFORE, n), min(K_AFTER, n)
        weights.append({"run_id": run, "world": g.world.iloc[0], "task_key": g.task_key.iloc[0], "flagged_other": n, "k_before": kb, "k_after": ka,
                        "w_original_not_released": n / kb, "w_original_released": 1.0, "w_extension_released": (n - kb) / (ka - kb) if ka > kb else None})
    E = pd.concat(rows, ignore_index=True)
    ids, old = set(), set(K.opaque_id)
    while len(ids) < len(E):
        i = f"sab1-{int(rng.integers(0, 2**32)):08x}"
        if i not in old:
            ids.add(i)
    E["opaque_id"] = rng.permutation(sorted(ids))
    E = E.iloc[rng.permutation(len(E))].reset_index(drop=True)
    E["run_hash"] = E.run_id.map(lambda r: hashlib.sha256(r.encode()).hexdigest()[:16])
    W = pd.DataFrame(weights)
    E = E.merge(W[["run_id", "w_extension_released"]], on="run_id", how="left").assign(source="flagged_extension")
    files = {"labels_stage1_ext.csv": E[["opaque_id"]].assign(label="", note=""),
             "locator_stage1_ext.csv": E[["opaque_id", "run_hash", "epoch", "turn", "call_index"]],
             "key_stage1_ext.csv": E[["opaque_id", "world", "arm", "task_key", "env", "side", "run_id", "epoch", "turn", "call_index", "irid", "call",
                                      "y", "flag", "touch", "stratum", "source", "w_extension_released"]],
             "weights_stage1_ext.csv": W}
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"k_before": K_BEFORE, "k_after": K_AFTER, "seed": EXT_SEED, "rows_by_world": E.world.value_counts().to_dict(),
           "runs": len(W), "runs_below_k_after": int((W.flagged_other < K_AFTER).sum()), "files": {}}
    for name, df in files.items():
        p = out_dir / name
        with p.open("x") as fh:
            df.to_csv(fh, index=False)
        res["files"][name] = {"rows": len(df), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    res["git"], res["written"] = P.git(), P.stamp()
    dst = P.HERE / "results" / f"lists_ext_{res['written']}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(res, indent=1, default=int))
    print(json.dumps(res, indent=1, default=int), "\n->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    main(**vars(ap.parse_args()))
