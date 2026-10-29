"""PLAN.md amendment 5, step 1 (local, from run_plant_oracle_modal.py's scores): the nested choice and the numbers.
Per variant: run-level AUROC with run score = max over outputs (the bar: 0.95) and = mean, each with a 95% main-task
interval, per-environment AUROCs and their mean (each fold's probe has its own offset), and a fixed-probe null (1,000 draws swapping a pair's two labels with probability 1/2; the choice is not
rerun); the site-level AUROC (site output vs the twin's same output); the share of pairs whose planted copy's max beats
its twin's; the share of planted copies whose highest output is the site. Also every candidate's pooled run-max AUROC
(not selection-adjusted). Writes results/plant_oracle_<stamp>.json.

  uv run python experiments/2026-10-09_fixed_sim/analyze_plant_oracle.py data/plant_oracle_scores_<stamp>.pkl.gz
  uv run python experiments/2026-10-09_fixed_sim/analyze_plant_oracle.py data/plant_oracle_<tag>/ <launch record .json>
"""

import gzip
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

from probe_shift.scoring import auroc_ci

HERE = Path(__file__).parent
DRAWS, SEED, BAR = 1000, 0, 0.95


def load(path, launches):
    """The launcher's combined file, or the workers' Volume copies (a directory of L<layer>.pkl.gz, fetched one exact file at
    a time) when the laptop client died before writing it. `launches` (JSON, directory input only): the launch record the
    worker files lack, {"staging_manifest_sha256": ..., "layers": {"<layer>": {"commit": ..., "app": ...}}}."""
    if not Path(path).is_dir():
        return pickle.loads(gzip.decompress(Path(path).read_bytes()))
    import run_plant_oracle_modal as R

    launch = json.loads(Path(launches).read_text())
    layers = [layer for layer in R.LAYERS if (Path(path) / f"L{layer}.pkl.gz").exists()]  # a timed-out layer has no file
    if sorted(map(int, launch["layers"])) != layers:
        raise ValueError(f"launch record covers {sorted(launch['layers'])}, files {layers}")
    outs = [pickle.loads(gzip.decompress((Path(path) / f"L{layer}.pkl.gz").read_bytes())) for layer in layers]
    return {"git": launch["layers"], "readouts": R.READOUTS, "layers": layers, "C_grid": R.C_GRID,
            "layers_missing": [layer for layer in R.LAYERS if layer not in layers],
            "sites": {"manifest_sha256": launch["staging_manifest_sha256"]}, "layers_out": outs}


def main(path, launches=None):
    d = load(path, launches)
    L = {o["layer"]: o for o in d["layers_out"]}
    base = L[d["layers"][0]]
    runs, run_of = base["runs"], base["run"]
    for o in L.values():  # every layer's pack holds the same rows in the same order
        if not (np.array_equal(o["run"], run_of) and np.array_equal(o["step"], base["step"]) and o["runs"] == runs):
            raise ValueError(f"L{o['layer']}: rows differ from L{base['layer']}")
    envs = sorted(map(str, set(base["env"])))
    keys = sorted(runs)
    y = np.array([runs[i]["arm"].endswith("plant1") for i in keys])
    task = np.array([runs[i]["main_task"] for i in keys])
    pair = np.array([runs[i]["key"].removesuffix("__plant1") for i in keys])
    pos_of = {i: n for n, i in enumerate(keys)}
    inv = np.array([pos_of[i] for i in run_of])
    swaps = [np.flatnonzero(pair == p) for p in sorted(set(pair))]
    rng = np.random.default_rng(SEED)

    def scores_of(sel, variant):
        s = np.full(len(run_of), np.nan)
        for e, (rd, layer, C) in sel.items():
            t = L[layer]["results"][f"{rd}|{variant}|{C:g}"]["test"][e]
            s[t["rows"]] = t["score"]
        if np.isnan(s).any():
            raise ValueError("an output got no test score")
        return s

    def run_scores(s):
        mx, mean = np.full(len(keys), -np.inf), np.zeros(len(keys))
        np.maximum.at(mx, inv, s)
        np.add.at(mean, inv, s)
        return {"max": mx, "mean": mean / np.bincount(inv)}

    def null(score):
        obs, vals = roc_auc_score(y, score), []
        for _ in range(DRAWS):
            yy = y.copy()
            for g in swaps:
                if rng.random() < 0.5:
                    yy[g] = yy[g[::-1]]
            vals.append(roc_auc_score(yy, score))
        vals = np.array(vals)
        return {"null_p95": round(float(np.percentile(vals, 95)), 4), "null_range": [round(float(x), 4) for x in np.percentile(vals, [2.5, 97.5])],
                "p": round(float((1 + (vals >= obs).sum()) / (1 + DRAWS)), 4)}

    out = {"source": str(path), "git_of_run": d["git"], "staging_manifest_sha256": d["sites"]["manifest_sha256"], "bar": BAR,
           "layers": d["layers"], "layers_missing": d.get("layers_missing", []), "variants": {}}
    print("layers", d["layers"], "missing", out["layers_missing"])
    for variant in ("site", "twin_all"):
        cand = [(rd, layer, C) for rd in d["readouts"] for layer in d["layers"] for C in d["C_grid"]]
        sel, masked = {}, 0
        for e in envs:
            best, best_v = None, -np.inf
            for rd, layer, C in cand:
                r = L[layer]["results"][f"{rd}|{variant}|{C:g}"]
                if not all(v for k, v in r["converged"].items() if k.startswith(f"{e}|")):
                    masked += 1
                    continue
                v = np.mean([r["val"][f"{e}|{x}"] for x in envs if x != e])
                if v > best_v:  # strict: ties keep the earlier candidate
                    best, best_v = (rd, layer, C), v
            sel[e] = best
        s = scores_of(sel, variant)
        rs = run_scores(s)
        site = base["is_site"]
        site_y, site_s = base["planted"][site], s[site]
        top_is_site = [site[inv == n][np.argmax(s[inv == n])] for n in range(len(keys)) if y[n]]
        pp = [(g[y[g]][0], g[~y[g]][0]) for g in swaps]  # (planted, twin) positions in keys
        pw = [rs["max"][a] > rs["max"][b] for a, b in pp]
        site_win = [s[site & (inv == a)][0] > s[site & (inv == b)][0] for a, b in pp]
        res = {"choices": {e: list(v) for e, v in sel.items()}, "candidates_masked_unconverged": masked,
               "runs": {"planted": int(y.sum()), "twin": int((~y).sum())},
               "site_level": auroc_ci(site_y, site_s, task[inv[site]], 1000, 0) | {"planted_site_above_twin_site": round(float(np.mean(site_win)), 4)},
               "planted_copy_top_output_is_site": round(float(np.mean(top_is_site)), 4),
               "pairs_planted_max_above_twin_max": round(float(np.mean(pw)), 4),
               # a tie: the planted copy's highest output comes before the line, where both copies hold the same outputs
               "pairs_win_tie_loss": [int(sum(rs["max"][a] > rs["max"][b] for a, b in pp)), int(sum(rs["max"][a] == rs["max"][b] for a, b in pp)),
                                      int(sum(rs["max"][a] < rs["max"][b] for a, b in pp))],
               "planted_copy_top_output_is_site_count": int(np.sum(top_is_site))}
        for agg in ("max", "mean"):
            res[f"run_{agg}"] = auroc_ci(y, rs[agg], task, 1000, 0) | null(rs[agg])
        env_run = np.array([runs[i]["environment"] for i in keys])
        per_env = {e: round(float(roc_auc_score(y[env_run == e], rs["max"][env_run == e])), 4) for e in envs}
        res["run_max"] |= {"per_environment": per_env, "mean_over_environments": round(float(np.mean(list(per_env.values()))), 4)}
        # Aswin's rule (10:51 PT 10-10): detected when the estimate shifts by about the planted amount (here the 0.95 bar:
        # at least 0.45 of the possible 0.5 shift from the unplanted 0.5, a twin against itself; unrounded) and its 95%
        # range excludes the unplanted estimate.
        res["detected"] = bool(roc_auc_score(y, rs["max"]) >= BAR and res["run_max"]["ci95"][0] > 0.5)
        table = []
        for rd, layer, C in cand:
            r = L[layer]["results"][f"{rd}|{variant}|{C:g}"]
            sc = np.full(len(run_of), np.nan)
            for e in envs:
                sc[r["test"][e]["rows"]] = r["test"][e]["score"]
            table.append({"readout": rd, "layer": layer, "C": C, "all_converged": all(r["converged"].values()),
                          "run_max_auroc": round(float(roc_auc_score(y, run_scores(sc)["max"])), 4)})
        res["per_candidate_not_selection_adjusted"] = table
        out["variants"][variant] = res
        print(variant, "| choices", res["choices"], "| masked", masked)
        print(f"  run max  {res['run_max']}\n  run mean {res['run_mean']}\n  site     {res['site_level']}")
        print(f"  pairs planted max > twin max {res['pairs_planted_max_above_twin_max']} (win/tie/loss {res['pairs_win_tie_loss']}); "
              f"planted copy's top output is the site "
              f"{res['planted_copy_top_output_is_site']}; detected (Aswin's rule, bar {BAR}): {res['detected']}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"plant_oracle_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
