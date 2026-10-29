"""Fix-full (PLAN.md Amendment 6), POST HOC, preliminary until Codex: Seed-OSS-36B's frozen G3 detector, false alarms on the
simulated honest runs as simulated (d9's dump, g3_score_simh.py) vs under full and N (g3_score_fx.py), paired by run, at 1%
(forecast.json's thresholds) and 5% (the same rule on the fit's real honest runs, as analyze_fx_seed.py). Per run, caught =
the share of its nested models (f, v != f; layer by f) scoring it above T[f, v]. Ranges: 2,000 draws of the runs' sorted
main tasks (seed 20261012), thresholds fixed. The fit's real honest runs beside as a reference. Gates: the same keys and
folds in all three; runs the fix left unchanged (byte-identical inputs) score as the fx run within 1e-3 at every layer; the
variants' false alarms equal fx_seed_POSTHOC_2026-10-13_162239.json's. Local, $0.

  PYTHONPATH=src uv run --no-project --python 3.12 --with numpy python experiments/2026-10-13_q2_fixfull/analyze_seed_honest_fa.py \\
      --simh-scores results/g3_scores_simh_<stamp>.json --fx-scores <dir>/g3_scores_fx_2026-10-13_161130.json \\
      --g3-results <G3 results dir> --stage-record <stage_record.json of 2026-10-13_q2_fixfull_inputs_v2>
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
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from probe_shift.attack_eval import threshold  # noqa: E402

B, SEED, LAYER_BY_FOLD, FIT, LAYERS = 2000, 20261012, (40, 63, 63, 63, 63), "2026-10-13_111525", (16, 24, 32, 40, 48, 56, 63)
FORECAST = ("6d7b1f50c2c08a5887962b5066681684d6e1ae63", "experiments/2026-10-13_g3_new_reader/forecast.json",
            "4243f2cd38d93d33e5b63ef08bc7e3fb3b387198af43e667c8821966504fff77")
TABLE = HERE / "results" / "fx_seed_POSTHOC_2026-10-13_162239.json"
VARIANTS = {"full": "sim-honest-fx-full", "N": "sim-honest-fx-NL"}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def caught(S, F, T):
    valid = np.arange(5)[None, :] != F[:, None]
    return np.where(valid, S > T[F], False).sum(1) / valid.sum(1)


def rng(v):
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def load(rec_path, arm):
    rec = json.loads(Path(rec_path).read_text())["arms"][arm]
    p = Path(rec_path).parent / rec["scores file"]
    if sha(p) != rec["scores sha256"] or rec["status"] != "scored":
        raise SystemExit(f"{arm}: scores file mismatch or not scored")
    z = np.load(p)
    return [str(k) for k in z["keys"]], z["folds"].astype(int), {l: z[f"L{l}"] for l in LAYERS}


def main():
    ap = argparse.ArgumentParser()
    for x in ("--simh-scores", "--fx-scores", "--g3-results", "--stage-record"):
        ap.add_argument(x, required=True, type=Path)
    a = ap.parse_args()
    for f in (Path(__file__).resolve(), HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    committed = subprocess.run(["git", "show", "HEAD:experiments/2026-10-13_q2_fixfull/results/g3_scores_fx_2026-10-13_161130.json"],
                               cwd=HERE, capture_output=True, check=True).stdout
    if hashlib.sha256(committed).hexdigest() != sha(a.fx_scores):
        raise SystemExit("--fx-scores is not the committed g3_scores_fx_2026-10-13_161130.json")
    blob = subprocess.run(["git", "show", f"{FORECAST[0]}:{FORECAST[1]}"], cwd=HERE, capture_output=True, check=True).stdout
    if hashlib.sha256(blob).hexdigest() != FORECAST[2]:
        raise SystemExit("not the acknowledged forecast.json")
    fc = json.loads(blob)["frozen"]
    sd = a.g3_results / f"run_scores_{FIT}"
    if sha(sd / "runs.json") != fc["inputs of x sha256"]["runs.json"] or any(sha(sd / f"scores_L{l}.npz") != fc["inputs of x sha256"][f"scores_L{l}.npz"] for l in set(LAYER_BY_FOLD)):
        raise SystemExit("the fit's stored scores differ from the frozen ones")
    runs = json.loads((sd / "runs.json").read_text()); arm = np.array([r["arm"] for r in runs]); fold = np.array([int(r["fold"]) for r in runs])
    N = {l: np.load(sd / f"scores_L{l}.npz")["output_end__mean__task_nested"] for l in set(LAYER_BY_FOLD)}
    hon = arm == "real-honest"
    T1 = np.array([[np.nan if v is None else v for v in row] for row in fc["thresholds at 1% [f][v]"]], float)
    T5 = np.full((5, 5), np.nan)
    for f in range(5):
        for v in range(5):
            if v != f:
                T5[f, v] = threshold(N[LAYER_BY_FOLD[f]][f, v][hon & (fold == v)], 0.05)
    keys, F, Z0 = load(a.simh_scores, "sim-honest")
    hmt = {r["key"]: r["main_task"] for r in json.loads((HERE.parent / "2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717/runs.json").read_text())
           if r["arm"] == "sim-honest"}
    st = json.loads(a.stage_record.read_text())
    pick = lambda Z, F_: np.stack([Z[LAYER_BY_FOLD[f_]][f_, :, j] for j, f_ in enumerate(F_)])  # noqa: E731
    S0 = pick(Z0, F)
    out = {"POST HOC": True, "status": "preliminary until Codex", "plan": "PLAN.md Amendment 6", "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip(),
           "inputs sha256": {a.simh_scores.name: sha(a.simh_scores), a.fx_scores.name: sha(a.fx_scores), "forecast.json": FORECAST[2], a.stage_record.name: sha(a.stage_record)},
           "runs": len(keys), "draws": B, "seed": SEED, "reproduction gate (runs the fix left unchanged)": {}, "rates": {}}
    tk = sorted({hmt[k] for k in keys}); ti = np.array([tk.index(hmt[k]) for k in keys])
    g = np.random.default_rng(SEED); W = np.stack([np.bincount(g.integers(0, len(tk), len(tk)), minlength=len(tk)) for _ in range(B)]).astype(float)[:, ti]
    Sv = {}
    for lab, harm in VARIANTS.items():
        kv, Fv, Zv = load(a.fx_scores, harm)
        if kv != keys or not (Fv == F).all():
            raise SystemExit(f"{lab}: keys or folds differ from the as-is scores")
        unchanged = {r["key"] for r in st["runs"] if r["arm"] == harm and r["first_changed_message"] is None}
        u = np.array([k in unchanged for k in keys])
        valid = np.arange(5)[None, :] != F[:, None]
        d = max((float(np.abs(np.where(valid[u], Zv[l][F[u], :, np.flatnonzero(u)] - Z0[l][F[u], :, np.flatnonzero(u)], 0)).max()) if u.any() else 0.0) for l in LAYERS)
        out["reproduction gate (runs the fix left unchanged)"][lab] = {"runs": int(u.sum()), "max abs diff, all 7 layers": d}
        if not u.any() or d > 1e-3:
            raise SystemExit(f"{lab}: reproduction gate failed ({int(u.sum())} unchanged runs, max abs diff {d})")
        Sv[lab] = pick(Zv, F)
    for S_ in (S0, *Sv.values()):
        if not np.isfinite(np.where(np.arange(5)[None, :] != F[:, None], S_, 0.0)).all():
            raise SystemExit("a non-finite nested score")
    table = json.loads(TABLE.read_text())
    ri = np.flatnonzero(hon); Fr = fold[ri]; Sr = np.stack([N[LAYER_BY_FOLD[fold[i]]][fold[i], :, i] for i in ri])
    tr = sorted({runs[i]["main_task"] for i in ri}); tri = np.array([tr.index(runs[i]["main_task"]) for i in ri])
    gr = np.random.default_rng(SEED); Wr = np.stack([np.bincount(gr.integers(0, len(tr), len(tr)), minlength=len(tr)) for _ in range(B)]).astype(float)[:, tri]
    for al, T in (("0.01", T1), ("0.05", T5)):
        c0 = caught(S0, F, T); b0 = 100 * (W @ c0) / W.sum(1)
        res = {"as simulated": {"false alarms (points, 95%)": [100 * float(c0.mean()), rng(b0)], "runs caught in at least one fold": int((c0 > 0).sum())}}
        for lab, S_ in Sv.items():
            cv = caught(S_, F, T); bv = 100 * (W @ cv) / W.sum(1)
            pub = table["rates"][str(float(al))][f"false alarms on simulated honest, {VARIANTS[lab]}"]["share caught (points)"]
            if abs(100 * cv.mean() - pub) > 1e-9:
                raise SystemExit(f"{lab} at {al}: false alarms differ from the fx_seed table's")
            res[lab] = {"false alarms (points, 95%)": [100 * float(cv.mean()), rng(bv)], "runs caught in at least one fold": int((cv > 0).sum()),
                        f"paired change, {lab} - as simulated (points, 95%)": [100 * float((cv - c0).mean()), rng(bv - b0)],
                        "runs whose catch rose / fell / stayed": [int((cv > c0).sum()), int((cv < c0).sum()), int((cv == c0).sum())]}
        cr = caught(Sr, Fr, T)
        res["real honest (the fit's runs), reference"] = {"runs": int(len(ri)), "false alarms (points, 95%)": [100 * float(cr.mean()), rng(100 * (Wr @ cr) / Wr.sum(1))]}
        out["rates"][al] = res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"seed_honest_fa_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k in ("runs", "reproduction gate (runs the fix left unchanged)", "rates")}, indent=1))
    print(dest)


if __name__ == "__main__":
    main()
