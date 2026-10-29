"""Q4, RULE.md Amendment 8 test 5 (POST HOC; Modal CPU; counts and stored scores only, no text): does the mixed probe USE the
simulator tells? Per simulated run (by key): tells per output in the j = 21 window (test 4's count: the first 21 outputs of
`fixes.calls_of`; tells = outputs fix-full's R1/R2/R5 change; visible = the change survives a content strip, which only the
Qwen3.5 reader applies) and the run's catch by the mixed probe (c) at j = 21 (stored column-0 scores of the null job, the
as-run fits), on test 2's cohorts:
- catch = hits / 4 over the run's four nested models, at test 2's matched threshold (each model's second-real catch on its
  test fold = 50%, unit weights) and, beside it, at the nominal 1% (run_q4b_cells.metrics' rule).
- Readers: Qwen3.5 (q35_sol_cut32k vs q35_own) with VISIBLE tells per output; Qwen3-32B (q3_sol vs q3_own) with ALL tells.
- (a) within each dataset: the least-squares slope of catch on tells per output (per 0.1), task-bootstrap 95% interval.
- (b) Sol vs own simulated catch, raw and matched on tells per output (own runs reweighted to Sol's distribution over the
  bins 0, (0, 0.1], (0.1, 0.2], (0.2, 0.3], > 0.3): the difference, its share left after matching, task-bootstrap
  intervals (2,000 draws per dataset, independent).
Also prints test 4's per-output rates (tells / outputs, window and whole run) by dataset.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-13_q4_why_weaker_POSTHOC/run_q4e_tells_use.py
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_q4d_tells as T  # noqa: E402  (test 4's image, volumes and count_run, as run)

APP = "probe-shift_aswin-surya_sub-q4own_tellsuse_kill-2026-11-13"
IMAGE = T.image.add_local_file(HERE / "run_q4d_tells.py", "/root/run_q4d_tells.py")
NULL = {"q35_own": "q4b_null_work/2026-10-14_022926/q35_own", **{c: f"q4b_null_work/2026-10-14_023056/{c}" for c in ("q3_sol", "q3_own", "q35_sol_cut32k")}}
PAIRS = {"qwen35": ("q35_sol_cut32k", "q35_own", "visible"), "qwen3": ("q3_sol", "q3_own", "all")}
BINS, B, SEED, TARGET = (0.0, 0.1, 0.2, 0.3), 2000, 20261013, 0.5
H_, E2, SA = "real-honest", "real-attack-e2", "sim-attack"
FOLD_PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
app = modal.App(APP)


@app.function(image=IMAGE, cpu=2, memory=8192, timeout=1800, retries=0, volumes=T.VOLS)
def sol_counts(i: int) -> dict:
    keys = {r["key"] for r in json.loads(Path("/root/runs36.json").read_text()) if r["arm"] == "sim-attack"}
    out = {}
    for line in Path("/acts", T.SOL_STAGED, "inputs", f"shard_{i}.jsonl").open():
        r = json.loads(line)
        if r["arm"] == "sim-attack" and r["key"] in keys:
            out[r["key"]] = T.count_run(r["messages"])
    return out


@app.function(image=IMAGE, cpu=2, memory=8192, timeout=1800, retries=0, volumes=T.VOLS)
def own_counts(k: int, n: int) -> dict:
    sys.path.insert(0, "/root/repo/q4")
    import stage_q4b as S

    counted = S.counted_attempts("main")
    todo = sorted(x for x in counted if x[0] == "sim_attack@september" and counted[x]["launch"] in S.MAIN)[k::n]
    out = {}
    for arm_dir, task in todo:
        c = counted[(arm_dir, task)]
        s = S.load(c["eval"])
        out[f"{c['launch']}/{c['rel']}#e0"] = T.count_run(S.transcripts.reader_messages(s, S.transcripts.system_prompt(s)))
    return out


@app.function(image=IMAGE, cpu=2, memory=8192, timeout=1800, retries=0, volumes=T.VOLS)
def catches(cell: str) -> dict:
    """Per simulated run in the j = 21 cohort: catch at the matched 50% threshold and at the nominal 1% (unit weights)."""
    import numpy as np

    runs = json.loads(Path("/ev", "q4b_cells_work", cell, "runs.json").read_text())
    Z = np.load(Path("/ev", NULL[cell], "L48_c_scores.npz"))["j21"][..., 0].astype(np.float64), np.load(Path("/ev", NULL[cell], "L63_c_scores.npz"))["j21"][..., 0].astype(np.float64)
    arm, fold = np.array([r["arm"] for r in runs]), np.array([int(r["fold"]) for r in runs])
    out = {}
    for L, z in zip((48, 63), Z):
        fin = np.isfinite(z)
        cohort = np.ones(len(arm), bool)
        for f, v in FOLD_PAIRS:
            cohort &= ~((fold == f) & ~fin[f, v])
        cohort &= fin.any((0, 1))
        hm, hn = np.zeros(len(arm)), np.zeros(len(arm))
        for f, v in FOLD_PAIRS:
            s = np.where(cohort, z[f, v], np.nan)
            e = np.flatnonzero((arm == E2) & (fold == f) & np.isfinite(s))
            o = np.sort(s[e])[::-1]
            Tm = o[int(np.argmax(np.arange(1, len(o) + 1) >= TARGET * len(o) - 1e-9))]
            hv = np.sort(s[(arm == H_) & (fold == v) & np.isfinite(s)])[::-1]
            Tn = hv[int(np.floor(0.01 * len(hv)))]
            sf = np.nan_to_num(s, nan=-np.inf)
            hm[fold == f] += sf[fold == f] > Tm
            hn[fold == f] += sf[fold == f] > Tn
        sel = np.flatnonzero(cohort & (arm == SA))
        out[f"L{L}"] = {runs[i]["key"]: {"task": runs[i]["main_task"], "matched": hm[i] / 4, "nominal": hn[i] / 4} for i in sel}
    return out


def analyze(counts: dict, catch: dict, kind: str) -> dict:
    """counts: key -> count_run output; catch: key -> {task, matched, nominal}. Rates in the window."""
    import numpy as np

    keys = sorted(k for k in catch if k in counts)
    t = np.array([counts[k]["window"].get("tell_visible" if kind == "visible" else "tell", 0) / max(counts[k]["window"]["outputs"], 1) for k in keys])
    task = np.array([catch[k]["task"] for k in keys])
    y = {m: np.array([catch[k][m] for k in keys]) for m in ("matched", "nominal")}
    tasks = np.unique(task)
    rng = np.random.default_rng(SEED)
    W = np.vstack([np.ones(len(tasks))] + [np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(B)])[:, np.searchsorted(tasks, task)]
    b = np.digitize(t, BINS[1:], right=True) + (t > 0)  # 0: none; 1: (0, .1]; 2: (.1, .2]; 3: (.2, .3]; 4: > .3
    return {"keys": keys, "t": t, "y": y, "W": W, "bin": b, "missing_counts": len([k for k in catch if k not in counts])}


def slope(t, y, W):
    mt, my = (W * t).sum(1) / W.sum(1), (W * y).sum(1) / W.sum(1)
    return ((W * (t - mt[:, None]) * (y - my[:, None])).sum(1) / (W * (t - mt[:, None]) ** 2).sum(1)) * 0.1


def summ(x):
    import numpy as np

    return {"value": round(float(x[0]), 4), "ci95": [round(float(np.percentile(x[1:], q)), 4) for q in (2.5, 97.5)]}


@app.function(image=IMAGE, cpu=4, memory=16384, timeout=1800, retries=0)
def compare(sol_c: dict, own_c: dict, cat: dict, git: str) -> dict:
    import numpy as np

    out = {}
    for reader, (sc, oc, kind) in PAIRS.items():
        for L in ("L48", "L63"):
            S, O = analyze(sol_c, cat[sc][L], kind), analyze(own_c, cat[oc][L], kind)
            res = {"runs": {"sol": len(S["keys"]), "own": len(O["keys"])}, "missing_counts": {"sol": S["missing_counts"], "own": O["missing_counts"]},
                   "tells_per_output_mean": {"sol": round(float(S["t"].mean()), 4), "own": round(float(O["t"].mean()), 4)},
                   "bin_shares": {d: [round(float((X["bin"] == j).mean()), 4) for j in range(5)] for d, X in (("sol", S), ("own", O))}}
            for m in ("matched", "nominal"):
                rs, ro = ((X["W"] * X["y"][m]).sum(1) / X["W"].sum(1) for X in (S, O))
                # own reweighted to Sol's bin shares (bins Sol lacks in own get no weight); per draw
                ps = np.vstack([(S["W"] * (S["bin"] == j)).sum(1) / S["W"].sum(1) for j in range(5)])
                cb = np.vstack([np.where((O["W"] * (O["bin"] == j)).sum(1) > 0, (O["W"] * O["y"][m] * (O["bin"] == j)).sum(1) / np.maximum((O["W"] * (O["bin"] == j)).sum(1), 1e-12), np.nan) for j in range(5)])
                ok = ~np.isnan(cb)
                ro_m = np.nansum(ps * cb, 0) / np.where(ok, ps, 0).sum(0)
                raw, matched = rs - ro, rs - ro_m
                res[m] = {"slope_per_0.1": {"sol": summ(slope(S["t"], S["y"][m], S["W"])), "own": summ(slope(O["t"], O["y"][m], O["W"]))},
                          "catch": {"sol": summ(rs), "own": summ(ro), "own_matched_to_sol_tells": summ(ro_m)},
                          "diff_raw": summ(raw), "diff_matched": summ(matched), "share_left": summ(matched / raw)}
            out[f"{reader}|{L}"] = res
    return out | {"git": git}


@app.local_entrypoint()
def main():
    REPO = HERE.parents[1]
    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()  # noqa: E731
    if g("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    head = g("rev-parse", "HEAD")  # read once, before any spawn
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    cat_calls = {c: catches.spawn(c) for c in ("q35_sol_cut32k", "q35_own", "q3_sol", "q3_own")}
    own_c = {k: v for d in own_counts.starmap([(k, 16) for k in range(16)]) for k, v in d.items()}
    sol_c = {k: v for d in sol_counts.map(range(64)) for k, v in d.items()}
    cat = {c: h.get() for c, h in cat_calls.items()}
    rates = {}
    for d, C in (("sol", sol_c), ("own", own_c)):
        for scope in ("window", "whole"):
            o = sum(v[scope]["outputs"] for v in C.values())
            rates[f"{d}|{scope}"] = {"runs": len(C), "outputs": o, "tell_per_output": round(sum(v[scope].get("tell", 0) for v in C.values()) / o, 4),
                                     "tell_visible_per_output": round(sum(v[scope].get("tell_visible", 0) for v in C.values()) / o, 4)}
    res = {"label": "POST HOC (RULE.md Amendment 8 test 5); counts and stored scores only", "git": head, "stamp": stamp,
           "per_output_rates_all_sim_runs": rates, "compare": compare.remote(sol_c, own_c, cat, head)}
    vol = modal.Volume.from_name(T.OWN, version=2)
    import io

    with vol.batch_upload(force=True) as b:
        b.put_file(io.BytesIO(json.dumps(res, indent=1).encode()), f"q4e_work/tells_use_{stamp}.json")
    dest = HERE / "results" / f"q4e_tells_use_POSTHOC_{stamp}.json"
    dest.open("x").write(json.dumps(res, indent=1))
    print("git", head)
    print("per-output rates (all simulated runs):", json.dumps(rates))
    pc = lambda x: f"{x['value']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]"  # noqa: E731
    for k, r in res["compare"].items():
        if k == "git":
            continue
        print(f"== {k}: runs {r['runs']} missing counts {r['missing_counts']} tells/output {r['tells_per_output_mean']} bins {r['bin_shares']}")
        for m in ("matched", "nominal"):
            x = r[m]
            print(f"   {m:8s} slope/0.1 sol {pc(x['slope_per_0.1']['sol'])} own {pc(x['slope_per_0.1']['own'])} | catch sol {pc(x['catch']['sol'])} own {pc(x['catch']['own'])} "
                  f"own@sol-tells {pc(x['catch']['own_matched_to_sol_tells'])} | diff raw {pc(x['diff_raw'])} matched {pc(x['diff_matched'])} share left {pc(x['share_left'])}")
    print(dest.name)
