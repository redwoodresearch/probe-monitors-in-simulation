"""Selection-aware null for the run-level re-read (AMENDMENT_POSTHOC_output_catch_2026-10-11.md, the null's protocol;
POST HOC). One Modal CPU container reads d9's grid files for follow-up (c) (run_claimB_b_grid.py at 857cb3cc, output_end,
level 1) and refits nothing.

For each of the grid's 101 label columns (column 0 the real labels, then 100 shuffles flipped per main task with the
grid's seed), it repeats the grid's "best" choice per held-out environment (layer x C, converged fits only, the grid's
own rule) and takes the chosen probes' saved test run scores. Gates, before any catch value: column 0's choices equal the
grid result's, and the test-environment AUROC null reproduces the grid result's null summary. Then each column gets the
run-level re-read's measure (analyze_POSTHOC_catch_quantile.catch: per environment, leave-one-task-out thresholds on the
label-0 runs, q2's rule at 5% and 1%), on the test environments: gap = catch - the reference's own flag rate. Column 0
is checked against the published run-level catches (which came from (c)'s per-output refit, not the saved run scores);
if it matches, the published values are compared with the null, and otherwise column 0 is the like-for-like observed
value and both are reported. Writes the result to the Volume first, then results/selection_null_POSTHOC_<stamp>.json.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with pyyaml modal deploy \
      experiments/2026-10-10_claimB_onpolicy/run_POSTHOC_selection_null_modal.py
  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with pyyaml python \
      experiments/2026-10-10_claimB_onpolicy/run_POSTHOC_selection_null_modal.py --grid-result <claimB_onpolicy_grid_*.json>
  (spawns the deployed function and waits; if the client dies, the result is on the Volume under selnull_POSTHOC/).
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

CLAIMB = Path.home() / "probe-shift-claimb-b"
GRID_FILE = CLAIMB / "experiments/2026-10-10_claimB_probe/run_claimB_b_grid.py"
HERE = Path(__file__).parent
PINNED, READOUT, LEVEL, COMPS = "857cb3cc", "output_end", "level1", ("fixed_vs_e1", "e2_vs_e1")
sys.path.insert(0, str(GRID_FILE.parent) if modal.is_local() else "/root")
import run_claimB_b_grid as G  # noqa: E402  (d9's module, unchanged)

APP = "probe-shift_aswin-surya_sub-claimb_selnull_kill-2026-11-10"
app = modal.App(APP)
MEASURE = HERE / "analyze_POSTHOC_catch_quantile.py"
image = G.image.add_local_file(GRID_FILE, "/root/run_claimB_b_grid.py").add_local_file(MEASURE, "/root/analyze_POSTHOC_catch_quantile.py")
OUT = "/acts/claimB/level1/selnull_POSTHOC"


def column_scores(cfg: dict, comp: str):
    """The grid's best choice per (fold, column) over output_end x layer x C, and each column's chosen test run scores."""
    import numpy as np

    _, runs, _, sc, Y, env = G.setup(cfg, LEVEL, comp, G.LAYERS[0])
    L, Cs = G.LAYERS, G.C_GRID
    R = {(l, C): dict(np.load(G.done_path(cfg, LEVEL, comp, READOUT, l, C))) for l in L for C in Cs}
    folds = R[(L[0], Cs[0])]["folds"].tolist()
    V = np.stack([np.stack([R[(l, C)]["val_auroc"] for C in Cs]) for l in L])  # [layer, C, fold, column]
    ok = np.stack([np.stack([np.maximum(R[(l, C)]["grad_full"], R[(l, C)]["grad_val"]) <= R[(l, C)]["tol"] for C in Cs]) for l in L])
    V = np.where(ok, V, -np.inf)  # converged fits only, real labels and shuffles alike (the grid's rule)
    keep = ok.any((0, 1)).all(0)
    c_ix, top = V.argmax(1), V.max(1)
    best = top.argmax(0)  # [fold, column]
    c_of = c_ix[best, np.arange(len(folds))[:, None], np.arange(Y.shape[1])[None, :]]
    s = np.full(Y.shape, np.nan)
    for i, f in enumerate(folds):
        m = env == f
        for k in np.flatnonzero(keep):
            s[m, k] = R[(L[best[i, k]], Cs[c_of[i, k]])]["test_run_score"][m, k]
    choices0 = {int(f): [READOUT, L[best[i, 0]], Cs[c_of[i, 0]]] for i, f in enumerate(folds)}
    r_in = np.flatnonzero(sc.runs_in)
    r_test = r_in[np.isin([runs[i]["environment"] for i in r_in], G.TEST_ENVS)]
    if not keep[0]:
        raise ValueError(f"{comp}: column 0 (the real labels) has a fold with no converged fit")
    aucs = G.col_auc(Y[r_test][:, keep], s[r_test][:, keep])
    return runs, sc, Y, s, keep, choices0, G.null_summary(aucs[0], aucs[1:])


@app.function(image=image, cpu=4, memory=16384, volumes={"/acts": G.vol}, timeout=1800, retries=0, scaledown_window=5)
def null(cfg: dict, expected: dict, published: dict, tag: str) -> dict:
    import collections
    import hashlib
    import os

    import numpy as np
    from analyze_POSTHOC_catch_quantile import catch

    dst = f"{OUT}/selection_null_{tag}.json"
    if os.path.exists(dst):
        raise ValueError(f"{dst} exists; this tag has run")
    code = {n: hashlib.sha256(Path(f).read_bytes()).hexdigest() for n, f in (("job", __file__), ("measure", "/root/analyze_POSTHOC_catch_quantile.py"),
                                                                              ("grid", "/root/run_claimB_b_grid.py"))}
    out = {"tag": tag, "label": "POST HOC: selection-aware null for the run-level re-read (no refit)", "code_sha256": code, "comparisons": {}}
    cols_by, labels_by = {}, {}
    for comp in COMPS:
        runs, sc, Y, s, keep, choices0, auc_null = column_scores(cfg, comp)
        if choices0 != {int(k): v for k, v in expected[comp]["choices"].items()}:
            raise ValueError(f"{comp}: column 0's choices {choices0} differ from the grid result's")
        if auc_null != expected[comp]["null"]:
            raise ValueError(f"{comp}: AUROC null {auc_null} differs from the grid result's {expected[comp]['null']}")
        labels_by[comp] = {k: frozenset(runs[i]["main_task"] for i in np.flatnonzero(sc.runs_in) if runs[i]["arm"] == "real-attack-e1" and Y[i, k] == 1)
                           for k in np.flatnonzero(keep)}  # tasks whose real run 1 is flipped to label 1 in column k
        stats = {}
        for k in np.flatnonzero(keep):
            P = collections.defaultdict(dict)  # {env: {task: {label: score}}}
            for i in np.flatnonzero(sc.runs_in):
                r = runs[i]
                if r["environment"] in G.TEST_ENVS:
                    P[r["environment"]].setdefault(r["main_task"], {})[int(Y[i, k])] = float(s[i, k])
            pairs = {env: [(t, d[0], d[1]) for t, d in sorted(by.items())] for env, by in P.items()}
            if any(len(d) != 2 for by in P.values() for d in by.values()):
                raise ValueError(f"{comp} column {k}: a task lacks one label")
            stats[int(k)] = {f"{a:.0%}": (lambda x: {"catch": x[0], "reference_rate": x[1], "gap": x[0] - x[1]})(catch(pairs, sorted(pairs), a))
                             for a in (0.05, 0.01)}
        cols_by[comp] = stats
        out["comparisons"][comp] = {"columns_kept": int(keep.sum()), "shuffles_dropped": int((~keep[1:]).sum()),
                                    "null_invalid": bool((~keep[1:]).sum() > 5), "auroc_null_reproduced": auc_null, "column0_choices": choices0,
                                    "column0": stats.get(0), "null": {}, "per_column": {str(k): v for k, v in sorted(stats.items())},
                                    "column0_matches_published": {a: bool(abs(stats[0][a]["catch"] - published[comp][a]["catch"]) < 1e-9
                                                                          and abs(stats[0][a]["reference_rate"] - published[comp][a]["reference_rate"]) < 1e-9)
                                                                  for a in ("5%", "1%")}}
        for a in ("5%", "1%"):
            g = [v[a]["gap"] for k, v in stats.items() if k > 0]
            at_least = sum(x >= stats[0][a]["gap"] - 1e-12 for x in g)  # shuffles at or above the observed (column 0) gap
            out["comparisons"][comp]["null"][a] = {"gap_p95": float(np.percentile(g, 95)), "gap_max": float(max(g)), "gap_mean": float(np.mean(g)), "shuffles": len(g),
                                                   "shuffles_at_or_above_observed": int(at_least), "p": (1 + at_least) / (1 + len(g))}
    both = sorted(set(cols_by["fixed_vs_e1"]) & set(cols_by["e2_vs_e1"]) - {0})
    if any(labels_by["fixed_vs_e1"][k] != labels_by["e2_vs_e1"][k] for k in both + [0]):
        raise ValueError("the two comparisons' shuffles do not flip the same tasks")
    out["contrast_null_invalid"] = len(both) < 95
    for a in ("5%", "1%"):  # the main contrast's null: same flips in both comparisons (asserted above)
        d = [cols_by["fixed_vs_e1"][k][a]["catch"] - cols_by["e2_vs_e1"][k][a]["catch"] for k in both]
        obs = cols_by["fixed_vs_e1"][0][a]["catch"] - cols_by["e2_vs_e1"][0][a]["catch"]
        at_least = sum(x >= obs - 1e-12 for x in d)
        out.setdefault("contrast_null", {})[a] = {"p95": float(np.percentile(d, 95)), "max": float(max(d)), "shuffles": len(d), "observed": obs,
                                                  "shuffles_at_or_above_observed": int(at_least), "p": (1 + at_least) / (1 + len(d)),
                                                  "per_column": {str(k): v for k, v in zip(both, d)}}
    os.makedirs(OUT, exist_ok=True)
    with open(dst, "x") as fh:  # durable before the return; never overwritten
        fh.write(json.dumps(out, indent=1))
    G.vol.commit()
    return out


def main(grid_result: str, packs: str):
    import hashlib
    import subprocess

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CLAIMB, capture_output=True, text=True, check=True).stdout.strip()
    assert head.startswith(PINNED), head
    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__)), str(HERE / "analyze_POSTHOC_catch_quantile.py")],
                      capture_output=True, text=True, check=True).stdout:
        raise SystemExit("commit the job and the measure first")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    grid = json.loads(Path(grid_result).read_text())
    folds = json.loads((CLAIMB / "splits" / "folds_v2.json").read_text())
    cfg = {"packs": dict(x.split("=") for x in packs.split(",")), "seed": G.SEED, "bootstrap": G.BOOT, "tag": grid["meta"]["code_tag"],
           "main_task_to_fold": folds["task_folds"]["main_task_to_fold"], "environment_to_fold": folds["environment_folds"]["environment_to_fold"]}
    if grid["meta"]["packs"] != cfg["packs"]:
        raise SystemExit(f"the grid result was run on packs {grid['meta']['packs']}, not {cfg['packs']}")
    expected = {c: {"choices": grid["levels"][LEVEL][c]["by_readout"][READOUT]["best"]["choices"],
                    "null": grid["levels"][LEVEL][c]["by_readout"][READOUT]["best"]["test_envs"]["null"]} for c in COMPS}
    published = {"fixed_vs_e1": {"5%": {"catch": 17 / 120, "reference_rate": 9 / 120}, "1%": {"catch": 8 / 120, "reference_rate": 5 / 120}},
                 "e2_vs_e1": {"5%": {"catch": 4 / 120, "reference_rate": 9 / 120}, "1%": {"catch": 3 / 120, "reference_rate": 5 / 120}}}  # the re-read's 005649 result
    tag = hashlib.sha1(json.dumps({"grid": Path(grid_result).name, "git": git, "expected": expected}, sort_keys=True).encode()).hexdigest()[:12]
    call = modal.Function.from_name(APP, "null").spawn(cfg, expected, published, tag)  # the deployed app; the Volume holds the result too
    print("spawned", call.object_id, "tag", tag, flush=True)
    res = call.get()
    local = {"job": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "measure": hashlib.sha256(MEASURE.read_bytes()).hexdigest(),
             "grid": hashlib.sha256(GRID_FILE.read_bytes()).hexdigest()}
    if res["code_sha256"] != local:
        raise SystemExit(f"the deployed code differs from the committed files: {res['code_sha256']} vs {local}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"selection_null_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps({"git": git, "grid_result": Path(grid_result).name} | res, indent=1))
    print(json.dumps({c: {k: v[k] for k in ("columns_kept", "shuffles_dropped", "null_invalid", "column0", "null")} for c, v in res["comparisons"].items()}, indent=1))  # per_column values are in the file
    print("contrast null:", {a: {k: v for k, v in x.items() if k != "per_column"} for a, x in res["contrast_null"].items()})
    print("->", dst)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-result", required=True)
    ap.add_argument("--packs", default="level0=47e205bcde77,level1=e191dd596f80")
    a = ap.parse_args()
    main(a.grid_result, a.packs)
