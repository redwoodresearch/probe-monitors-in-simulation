"""G2's scoring (PLAN.md here): Llama part 2's scorer as run (../2026-10-12_q2_llama_reader/run_score_fix_llama.py, sha256
0006c455…, at dddc8324) with only these changes: its app name, the repaired-plus arm (the plus extraction's runs), and the
prefix gate. In every repaired-plus run, the kept outputs before its first output that differs from the repaired arm (the
build record's) have the same tokens as in part 2, so their per-row probe scores must equal part 2's rows (fix939's
extraction for its 823 runs, part 1's for the other 116) within 1e-3 at every layer and nested model, over at least 1,000
rows; or nothing is written. --smoke reads the smoke extraction and reports the prefix check only, writing a record.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-13_q2_llama_leftover/run_score_plus_llama.py --run-id 2026-10-12_143129 \\
      --scores-dir <part 2's results/run_scores_2026-10-12_143129> --fix <fix939 per-run jsonl> --build <build record> [--plus-run <name>] [--smoke]

Part 2's docstring follows, unchanged.

Llama part 2's scoring (PLAN.md step C here): ../2026-10-12_q2_olmo_reader/run_score_fix_olmo.py with only its names,
layers (20..79 of 80) and the run (an argument: Llama's fit run id) changed. The frozen probes of that run score fix939's
823 re-rendered runs (this extraction's arm sim-attack-fix939) and, as the gate, re-score the 939 as-simulated runs. One Modal CPU
container; saved activations and probes only, no fit, no model.

Per run in test fold f and layer l: each nested model (f, v), v != f, scores the run's R2 rows as
((x - mean) / std) . w + b in float32 from the float16 rows, and the run score is the mean over its rows, as the fit pools
them. The local side checks the re-scored as-simulated runs against the stored nested scores (scores_L<l>.npz,
output_end__mean__task_nested) within 1e-3 at every layer and nested model, or writes nothing; then it writes the fixed
runs' nested scores (npz) and a record (json) to results/.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-12_q2_llama_reader/run_score_fix_llama.py --run-id <Llama fit run id> \\
      --scores-dir <results/run_scores_<run id>> --fix <fix939 per-run jsonl>
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-llama-leftover_score_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-q2-llama_acts_kill-2026-11-12"
WORK = "probe-shift_aswin-surya_sub-q2-llama_work_kill-2026-11-12"
EXTRACT, FIX_LABEL, PLUS_LABEL = "2026-10-12_la_output-end-r2j_llama33-70b_alllayers_v1", "sim-attack-fix939", "sim-attack-plus"
MIN_PREFIX_ROWS = 1000
LAYERS, TOL = (20, 30, 40, 50, 60, 70, 79), 1e-3
PIN = {"fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl": "5f6ecbc42b1c6bb9586cb58e474d74c5c1d914bedb95d80a5a07dd078b0e5ebd"}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
work = modal.Volume.from_name(WORK, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, volumes={"/acts": acts, "/work": work}, cpu=8, memory=32768, timeout=3600, retries=0)
def score(sim: list[dict], run_id: str, plus_run: str, first: dict) -> dict:
    """sim: the as-simulated runs, {i, filename, fold, key, fix}. Returns, per layer, nested scores [5, 5, len(sim)] for the
    as-simulated rows, the re-rendered rows and the repaired-plus rows (NaN where a run has none), the runs found, and the
    prefix gate: per layer, the largest |plus - repaired| per-row score over the prefix rows, and their number."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    dirs = {"sim-attack": Path("/acts") / EXTRACT, FIX_LABEL: Path("/acts") / f"{EXTRACT}_fix939", PLUS_LABEL: Path("/acts") / plus_run}
    metas, pos = {}, {}
    for label in ("sim-attack", FIX_LABEL, PLUS_LABEL):
        for r in sim:
            p = dirs[label] / "meta" / label / f"{r['filename']}.json"
            if p.exists():
                m = json.loads(p.read_text())
                metas[(label, r["filename"])] = (m["chunk"], m["row_start"], m["n_rows"], m["key"])
                pos[(label, r["filename"])] = [(q["step"], q["action"], q["row"]) for q in m["positions"] if q["kind"] == "output_end" and q["kept"]]
    fix_found = sorted(f for (lab, f) in metas if lab == FIX_LABEL)
    prefix = []  # (j, fold, plus chunk, plus row, repaired label, repaired chunk, repaired row)
    for j, r in enumerate(sim):
        if (PLUS_LABEL, r["filename"]) not in metas:
            continue
        rl = FIX_LABEL if r["fix"] else "sim-attack"
        a, b = pos[(PLUS_LABEL, r["filename"])], pos[(rl, r["filename"])]
        if [x[:2] for x in a] != [x[:2] for x in b]:
            raise ValueError("a repaired-plus run's kept outputs differ from the repaired run's")
        f0 = tuple(first[r["key"]])
        cp, sp = metas[(PLUS_LABEL, r["filename"])][:2]; cr, sr = metas[(rl, r["filename"])][:2]
        prefix += [(j, r["fold"], cp, sp + x[2], rl, cr, sr + y[2]) for x, y in zip(a, b) if x[:2] < f0]
    if any(("sim-attack", r["filename"]) not in metas for r in sim):
        raise ValueError("an as-simulated run has no meta in the extraction")

    def one_layer(layer):
        with np.load(Path("/work") / "runs" / run_id / f"probes_L{layer}.npz") as zf:
            z = {k: zf[k] for k in zf.files}  # once: an npz re-reads its zip member on every access
        out = {lab: np.full((5, 5, len(sim)), np.nan) for lab in ("sim-attack", FIX_LABEL, PLUS_LABEL)}
        by_chunk = {}
        for j, r in enumerate(sim):
            for lab in ("sim-attack", FIX_LABEL, PLUS_LABEL):
                if (lab, r["filename"]) in metas:
                    c, s0, n, _ = metas[(lab, r["filename"])]
                    by_chunk.setdefault((lab, c), []).append((j, lab, s0, n, r["fold"]))
        for (lab_, c), items in by_chunk.items():
            X = np.load(dirs[lab_] / "by_layer" / f"L{layer}" / f"{c}.npy")  # whole file: row-indexed mmap is slow on a Volume
            for j, lab, s0, n, f in items:
                if n == 0:
                    continue
                x = X[s0:s0 + n].astype(np.float32)
                for v in range(5):
                    if v == f:
                        continue
                    k = f"output_end__task_nested_{f}_{v}__"
                    s = ((x - z[k + "mean"]) / z[k + "std"]) @ z[k + "w"] + np.float32(z[k + "b"])
                    out[lab][f, v, j] = float(np.mean(s.astype(np.float64)))
        need, got, diffs = {}, {}, []
        for _, _, cp, rp, rl, cr, rr in prefix:  # each chunk file read once; only the prefix rows kept
            need.setdefault((PLUS_LABEL, cp), set()).add(rp); need.setdefault((rl, cr), set()).add(rr)
        for (lab, ch), rws in need.items():
            X = np.load(dirs[lab] / "by_layer" / f"L{layer}" / f"{ch}.npy")
            got.update({(lab, ch, r_): X[r_].astype(np.float32) for r_ in rws})
            del X
        for j, f, cp, rp, rl, cr, rr in prefix:
            xp, xr = got[(PLUS_LABEL, cp, rp)], got[(rl, cr, rr)]
            for v in range(5):
                if v != f:
                    k = f"output_end__task_nested_{f}_{v}__"
                    sp_, sr_ = (((x - z[k + "mean"]) / z[k + "std"]) @ z[k + "w"] + np.float32(z[k + "b"]) for x in (xp, xr))
                    diffs.append(abs(float(sp_) - float(sr_)))
        out["prefix max"] = float("inf") if not np.isfinite(diffs).all() else max(diffs, default=0.0)  # a NaN fails the gate
        return layer, out
    with ThreadPoolExecutor(len(LAYERS)) as pool:
        res = dict(pool.map(one_layer, LAYERS))
    plus_found = sorted(f for (lab, f) in metas if lab == PLUS_LABEL)
    return {"fix filenames found": fix_found, "fix keys": sorted(metas[(FIX_LABEL, f)][3] for f in fix_found),
            "plus keys": sorted(metas[(PLUS_LABEL, f)][3] for f in plus_found), "prefix rows": len(prefix),
            "prefix max |plus - repaired| per-row score, by layer": {str(l): res[l].pop("prefix max") for l in LAYERS},
            "scores": {str(l): {lab: res[l][lab].tolist() for lab in res[l]} for l in LAYERS}}


@app.local_entrypoint()
def main(run_id: str, scores_dir: str, fix: str, build: str, plus_run: str = EXTRACT + "_plus", smoke: bool = False):
    import numpy as np
    me = Path(__file__).resolve()
    for f in (me, HERE / "PLAN.md", Path(build).resolve()):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    d, fix_p = Path(scores_dir), Path(fix)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    if sha(fix_p) != PIN[fix_p.name]:
        raise SystemExit("not the pinned fix939 per-run file")
    runs = json.loads((d / "runs.json").read_text())
    want = sorted({json.loads(l)["key"] for l in fix_p.open() if json.loads(l)["label"] == "FIX"})
    sim = [{"i": i, "filename": r["filename"], "fold": int(r["fold"]), "key": r["key"], "fix": r["key"] in set(want)} for i, r in enumerate(runs) if r["arm"] == "sim-attack"]
    if len(sim) != 939 or len(want) != 823 or not set(want) <= {r["key"] for r in sim}:
        raise SystemExit("not the 939 as-simulated runs and fix939's 823 re-rendered keys")
    brec = json.loads(Path(build).read_text())
    res = score.remote([{k: r[k] for k in ("i", "filename", "fold", "key", "fix")} for r in sim], run_id, plus_run, brec["first differing output (step, action)"])
    if res["fix keys"] != want:
        raise SystemExit(f"the extraction's re-rendered runs are not fix939's 823: {len(res['fix keys'])} found")
    prefix = {"run": plus_run, "rows": res["prefix rows"], "max |plus - repaired| per-row score, by layer (tolerance 1e-3)": res["prefix max |plus - repaired| per-row score, by layer"]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    worst = max(prefix["max |plus - repaired| per-row score, by layer (tolerance 1e-3)"].values())
    if smoke:  # the smoke extraction: the prefix check only, on exactly its 4 runs
        ok = "_test_" in plus_run and len(res["plus keys"]) == 4 and prefix["rows"] > 0 and worst <= TOL
        rec = {"git": head, "smoke prefix check": prefix, "plus runs found": len(res["plus keys"]), "passed": bool(ok),
               "inputs sha256": {Path(build).name: sha(Path(build))}}
        with open(dest := HERE / "results" / f"smoke_prefix_{stamp}.json", "x") as fh:
            fh.write(json.dumps(rec, indent=1) + "\n")
        print(json.dumps(rec, indent=1)); print(dest)
        if not ok:
            raise SystemExit("smoke prefix check failed (needs the _test_ run, its 4 runs, prefix rows, every difference within 1e-3)")
        return
    if res["plus keys"] != brec["plus keys"]:
        raise SystemExit(f"the extraction's repaired-plus runs are not the build's: {len(res['plus keys'])} found")
    if not worst <= TOL or prefix["rows"] < MIN_PREFIX_ROWS:
        raise SystemExit(f"prefix gate failed: {prefix}")
    gate, fixed = {}, {}
    idx = np.array([r["i"] for r in sim]); fold = np.array([r["fold"] for r in sim])
    for l in LAYERS:
        stored = np.load(d / f"scores_L{l}.npz")["output_end__mean__task_nested"][:, :, idx]
        mine = np.array(res["scores"][str(l)]["sim-attack"])
        m = np.zeros_like(mine, bool)
        for j, f in enumerate(fold):
            m[f, :, j] = True; m[f, f, j] = False
        diff = np.abs(mine[m] - stored[m])
        if not np.isfinite(stored[m]).all() or not np.isfinite(diff).all():
            raise SystemExit(f"L{l}: a missing stored or re-scored nested score")
        gate[str(l)] = float(diff.max())
        fixed[f"L{l}"] = np.array(res["scores"][str(l)][PLUS_LABEL])
    if max(gate.values()) > TOL:
        raise SystemExit(f"gate failed: max |re-scored - stored| by layer {gate}")
    has_fix = np.array([r["key"] in set(brec["plus keys"]) for r in sim])  # the build's runs, not inferred from finiteness
    for l in LAYERS:  # every repaired-plus run has every nested score
        m = np.zeros((5, 5, len(sim)), bool)
        for j, f in enumerate(fold):
            m[f, :, j] = has_fix[j]; m[f, f, j] = False
        if not np.isfinite(fixed[f"L{l}"][m]).all():
            raise SystemExit(f"L{l}: a repaired-plus run without a finite nested score")
    npz = HERE / "results" / f"plus_scores_llama_{stamp}.npz"
    np.savez(npz, run_index=idx, has_plus=has_fix, **fixed)
    rec = {"plan": "PLAN.md (this folder)", "git": head, "run": run_id, "extraction": [EXTRACT, f"{EXTRACT}_fix939", plus_run],
           "inputs sha256": {fix_p.name: sha(fix_p), "runs.json": sha(d / "runs.json"), Path(build).name: sha(Path(build))} | {f"scores_L{l}.npz": sha(d / f"scores_L{l}.npz") for l in LAYERS},
           "gate: max |re-scored - stored| nested score, by layer (tolerance 1e-3)": gate, "prefix gate": prefix,
           "as-simulated runs": len(sim), "with repaired-plus scores": int(has_fix.sum()), "unchanged (part 2's scores stand)": int((~has_fix).sum()),
           "scores file": npz.name, "scores sha256": sha(npz)}
    with open(dest := HERE / "results" / f"plus_scores_llama_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)); print(dest)
