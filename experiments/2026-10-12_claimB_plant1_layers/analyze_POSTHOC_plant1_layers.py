"""POST HOC ($0, score arrays only; RULE.md here, committed before this ran): planted check 1 per layer and readout from
step B's stored level-1 grids. Imports d9's grid code at 857cb3cc (the version whose hash is the grid tag d903c93e9322)
from a git archive, points its /acts paths at a local mirror of the needed Volume files, gates on reproducing the stored
summaries, then calls its `summarize` once per (readout, layer).

  uv run --no-project --python 3.12 --with numpy --with scipy --with scikit-learn --with torch --with 'modal>=1.4.3' python \\
      experiments/2026-10-12_claimB_plant1_layers/analyze_POSTHOC_plant1_layers.py --acts <local mirror> --tree <empty dir> \\
      --b-summary <claimB_b_grid_2026-10-10_050016_d9-857cb3cc.json> --maxa-summary <claimB_maxa_grid_2026-10-10_154626_d9-857cb3cc.json>
"""
import argparse
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
PIN, TAG = "857cb3cc", "d903c93e9322"
GRIDS = {"output_end": "2cfd1aec393c", "output_end_mean": "2cfd1aec393c", "output_end_max": "3adf85556e82"}
SUMMARY_OF = {"output_end": "b", "output_end_mean": "b", "output_end_max": "maxa"}
WANT_SUMMARY = {"b": "32b67b40fca2830a", "maxa": "d34a4df491cfb9f7"}  # sha256 prefixes in results_manifest_v1 / v9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--b-summary", type=Path, required=True)
    ap.add_argument("--maxa-summary", type=Path, required=True)
    a = ap.parse_args()
    summaries = {}
    for k, p in (("b", a.b_summary), ("maxa", a.maxa_summary)):
        raw = p.read_bytes()
        if not hashlib.sha256(raw).hexdigest().startswith(WANT_SUMMARY[k]):
            raise SystemExit(f"{p.name} does not match its S3 manifest")
        summaries[k] = json.loads(raw)
    tar = subprocess.run(["git", "archive", PIN, "experiments/2026-10-10_claimB_probe/run_claimB_b_grid.py", "src/probe_shift",
                          "splits/folds_v2.json"], capture_output=True, check=True, cwd=HERE.parents[1]).stdout
    a.tree.mkdir(parents=True, exist_ok=True)
    tarfile.open(fileobj=io.BytesIO(tar)).extractall(a.tree, filter="data")
    sys.path[:0] = [str(a.tree / "src"), str(a.tree / "experiments/2026-10-10_claimB_probe")]
    import run_claimB_b_grid as G  # noqa: E402

    code = (a.tree / "experiments/2026-10-10_claimB_probe/run_claimB_b_grid.py").read_bytes() + (a.tree / "src/probe_shift/torch_probe.py").read_bytes() \
        + (a.tree / "src/probe_shift/q1_eval.py").read_bytes()
    if hashlib.sha256(code).hexdigest()[:12] != TAG:
        raise SystemExit("the archived grid code does not hash to the grid tag")
    acts = str(a.acts.resolve())
    G.Path = lambda p, *rest: pathlib.Path(str(p).replace("/acts", acts, 1), *rest)  # the module's /acts paths -> the mirror
    folds = json.loads((a.tree / "splits/folds_v2.json").read_text())
    base = {"seed": G.SEED, "bootstrap": G.BOOT, "tag": TAG, "main_task_to_fold": folds["task_folds"]["main_task_to_fold"],
            "environment_to_fold": folds["environment_folds"]["environment_to_fold"]}
    out = {"label": "POST HOC (RULE.md): planted check 1 per layer x readout from step B's stored level-1 grids; run score = mean of output scores",
           "pin": PIN, "tag": TAG, "layers": G.LAYERS, "C_grid": G.C_GRID, "gate": {}, "rows": [], "inputs": {}}
    for rd, pack in GRIDS.items():
        cfg = base | {"packs": {"level1": pack}}
        for l in G.LAYERS:
            for C in G.C_GRID:
                p = G.done_path(cfg, "level1", "plant1", rd, l, C)
                out["inputs"][str(p).replace(acts + "/", "")] = hashlib.sha256(p.read_bytes()).hexdigest()
            rp = pathlib.Path(acts) / "claimB" / "level1" / f"packedB_{pack}" / f"runs.L{l}.json"
            out["inputs"][str(rp).replace(acts + "/", "")] = hashlib.sha256(rp.read_bytes()).hexdigest()
        allL = G.summarize(cfg, "level1", "plant1", G.LAYERS, G.C_GRID, [rd])["by_readout"][rd]["best"]["test_envs"]
        stored = summaries[SUMMARY_OF[rd]]["levels"]["level1"]["plant1"]["by_readout"][rd]["best"]["test_envs"]
        ok = allL["test"]["auroc"] == stored["test"]["auroc"] and allL["null"]["null_p95"] == stored["null"]["null_p95"]
        out["gate"][rd] = {"recomputed": allL["test"]["auroc"], "stored": stored["test"]["auroc"], "null_p95": [allL["null"]["null_p95"], stored["null"]["null_p95"]], "match": ok}
        print("gate", rd, out["gate"][rd], flush=True)
        if not ok:
            raise SystemExit(f"gate failed for {rd}: no per-layer table")
    for rd, pack in GRIDS.items():
        cfg = base | {"packs": {"level1": pack}}
        for l in G.LAYERS:
            r = G.summarize(cfg, "level1", "plant1", [l], G.C_GRID, [rd])["by_readout"][rd]["best"]
            t = r["test_envs"]
            row = {"readout": rd, "layer": l, "auroc": t["test"]["auroc"], "ci95": t["test"]["ci95"], "n": t["test"]["n"],
                   "null_p95": t["null"]["null_p95"], "null_p": t["null"]["p"], "shuffles": t["null"]["shuffles"],
                   "C_per_fold": {k: v[2] for k, v in r["choices"].items()}, "null_invalid": r["null_invalid"]}
            # summarize() reports 4 decimals: a value within 0.0005 of a threshold gets no verdict (needs an unrounded recompute)
            near = abs(row["auroc"] - 0.95) < 5e-4 or abs(row["ci95"][0] - 0.5) < 5e-4 or abs(row["auroc"] - row["null_p95"]) < 5e-4
            if row["null_invalid"] or near:
                row["detected"] = None
                row["no_verdict_reason"] = "invalid null (over 5% of shuffles unscorable)" if row["null_invalid"] else "within 0.0005 of a threshold at 4 decimals"
            else:
                row["detected"] = bool(row["auroc"] >= 0.95 and row["ci95"][0] > 0.5 and row["auroc"] > row["null_p95"])
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
    out["git"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / f"plant1_layers_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print("->", f"results/plant1_layers_POSTHOC_{stamp}.json")


if __name__ == "__main__":
    main()
