"""G1 placebo (PLAN.md): frozen-probe scoring with #36 at layers 63 and 48. A copy of
`../2026-10-12_plantslip_transfer/q2_reply_score.py` (aswin/plantslip-q2-reply @ 27dd1702): P4's reader only, its
`score()` unchanged, at layer 63 (P4's comparison) and layer 48 (row 5's comparison, whose probe and stored scores are
pinned as `../2026-10-13_q2_row5_whitespace/score_whitespace.py` pins them on aswin/q2-row5-whitespace); no pins mode
(each probe file's sha256 is checked against its pin on every call).

Per run in test fold f, at layer l: each nested model (f, v), v != f, scores the run's kept output_end rows as
((x - mean) / std) . w + b in float32 from the float16 rows, and the run score is the mean over those rows.
  --which controls  P4's 20 seeded unplanted e2 controls (P4's staging record), re-extracted at layers 48 and 63; the
                    record holds the gate: at each layer, 80 re-scored nested scores against #36's stored scores,
                    tolerance 1e-3 logit.
  --which placebo   the placebo runs; needs --gate, this script's passing controls record.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-13_q2_placebo_plant/run_score_placebo.py --which <controls|placebo> --extract-run <run> \\
      --stage <P4's staging record (controls) | results/stage_placebo_*.json (placebo)> [--gate <controls record>]
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-placebo_score_kill-2026-11-13"
READERS = {  # P4's frozen detector, as q2_reply_score.py names it, now read at two layers
    "P4": {"reader": "Qwen/Qwen3-32B (#36)", "work": "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06", "run": "2026-10-06_181548",
           "acts": "probe-shift_aswin-surya_sub-la-activations", "layers": [48, 63], "fold_layers": [63, 63, 63, 63, 63],
           "scores": "/Users/aswinsurya/probe-distribution-shift-from-sims/.claude/worktrees/agent-a41654b9677ba3292/experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548"},
}
PIN = {"probes_L63.npz": "52180c44f8509b180f06f303c25927a97914446d499a34b95b3925d190e4e7d4",  # as P4's gate and planted records read it
       "probes_L48.npz": "9fd9989b88eacf8652cc6ccf653e105b03496cff19f8fca0f5a13843015bd104",  # as row 5's score_whitespace.py pins it
       "scores_L63.npz": "60f8e0d4b7b45baf46c6cf96f61677c7a0e966a3698a99a913763a4d763e6f10",
       "scores_L48.npz": "f6cf939e20134218cc2f0f7d9ecc1bdaf21aadfdb05b921d7361551017e70fc4",
       "runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96",
       "P4 stage": "f1a68c90a5d26799925bda1c0d574728a547cde6c2ec3b76bfa18c3691c75157"}  # q2_reply_stage_2026-10-12_135318.json (the controls)
CONTROLS_RUN, PLACEBO_RUN = "2026-10-13_q2_placebo_controls_qwen3-32b_l48-63_v1", "2026-10-13_q2_placebo_reply_qwen3-32b_l48-63_v2"
TOL = 1e-3

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")
VOLS = {f"/work_{p}": modal.Volume.from_name(r["work"], version=2).read_only() for p, r in READERS.items()}
VOLS |= {f"/acts_{p}": modal.Volume.from_name(r["acts"], version=2, create_if_missing=True).read_only() for p, r in READERS.items()}


def filename(key: str) -> str:  # the extractors' file stems
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes=VOLS, cpu=8, memory=32768, timeout=3600, retries=0)
def score(pair: str, extract_run: str, runs: list[dict]) -> dict:
    """runs: {j, key, fold, arm}. Returns per layer the nested scores [len(runs), 5] (NaN off the run's fold and where
    a run has no meta or no kept row), the sha256 of the probe files read, and the count of runs with no meta."""
    import numpy as np
    r_ = READERS[pair]
    root = Path(f"/acts_{pair}") / extract_run
    metas, missing = {}, 0
    for i, r in enumerate(runs):
        p = root / "meta" / r["arm"] / f"{filename(r['key'])}.json"
        if not p.exists():
            missing += 1
            continue
        m = json.loads(p.read_text())
        rows = sorted(q["row"] for q in m["positions"] if q["kind"] == "output_end" and q["kept"])
        metas[i] = (m["chunk"], m["row_start"], rows)
    out, shas = {}, {}
    for l in r_["layers"]:
        f = Path(f"/work_{pair}") / "runs" / r_["run"] / f"probes_L{l}.npz"
        shas[str(l)] = hashlib.sha256(f.read_bytes()).hexdigest()
        with np.load(f) as zf:
            z = {k: zf[k] for k in zf.files}
        S = np.full((len(runs), 5), np.nan)
        by_chunk = {}
        for i, (c, s0, rows) in metas.items():
            by_chunk.setdefault(c, []).append(i)
        for c, idx in by_chunk.items():
            X = np.load(root / "by_layer" / f"L{l}" / f"{c}.npy")  # whole file: row-indexed mmap is slow on a Volume
            for i in idx:
                _, s0, rows = metas[i]
                if not rows:
                    continue
                x = X[[s0 + q for q in rows]].astype(np.float32); f_ = runs[i]["fold"]
                for v in range(5):
                    if v != f_:
                        k = f"output_end__task_nested_{f_}_{v}__"
                        S[i, v] = float(np.mean((((x - z[k + "mean"]) / z[k + "std"]) @ z[k + "w"] + np.float32(z[k + "b"])).astype(np.float64)))
        out[str(l)] = S.tolist()
    return {"scores": out, "probes sha256": shas, "runs with no meta": missing}


@app.local_entrypoint()
def main(which: str, extract_run: str, stage: str, gate: str = ""):
    import numpy as np
    me = Path(__file__).resolve()
    for f in (me, HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    pair = "P4"; r_ = READERS[pair]; d = Path(r_["scores"])
    if any(sha(d / n) != PIN[n] for n in ("runs.json", "scores_L48.npz", "scores_L63.npz")):
        raise SystemExit("#36's stored scores changed")
    st = json.loads(Path(stage).read_text())
    if which == "controls":
        if extract_run != CONTROLS_RUN or sha(stage) != PIN["P4 stage"]:
            raise SystemExit("controls: P4's staging record and the controls extraction run")
        todo_src = st["controls"]
    elif which == "placebo":
        g = json.loads(Path(gate).read_text()) if gate else {}
        if extract_run != PLACEBO_RUN or g.get("which") != "controls" or g.get("gate") != "pass" or g.get("extract run") != CONTROLS_RUN \
                or g.get("stage sha256") != PIN["P4 stage"] or g.get("probes sha256") != {"48": PIN["probes_L48.npz"], "63": PIN["probes_L63.npz"]} \
                or any(g.get(f"gate L{l}") != "pass" or g.get(f"gate L{l}: comparisons (20 controls x 4 nested models)") != 80 for l in (48, 63)):
            raise SystemExit("placebo scoring needs the placebo extraction run and this script's passing controls record (--gate)")
        if st.get("gate") != "pass" or st.get("staged run (Volume probe-shift_aswin-surya_sub-la-activations)") != "2026-10-13_q2_placebo_reply_inputs_v2":
            raise SystemExit("placebo scoring needs the v2 staging record (PLAN.md Addendum 2), with a passing gate")
        todo_src = st["placebo"]
    else:
        raise SystemExit("--which controls|placebo")
    runs_all = json.loads((d / "runs.json").read_text())
    j_of = {(r["arm"], r["key"]): j for j, r in enumerate(runs_all)}
    todo = [{"j": j_of[(x["orig_arm"], x["key"])], "key": x["key"], "fold": int(runs_all[j_of[(x["orig_arm"], x["key"])]]["fold"]), "arm": x["arm"]}
            for x in todo_src]
    res = score.remote(pair, extract_run, todo)
    if res["probes sha256"] != {"48": PIN["probes_L48.npz"], "63": PIN["probes_L63.npz"]}:
        raise SystemExit("a probe file is not the pinned one")
    S = {l: np.array(v) for l, v in res["scores"].items()}
    rec = {"plan": "PLAN.md (this folder)", "git": head, "which": which, "pair": pair, "reader": r_["reader"], "frozen run": r_["run"],
           "extract run": extract_run, "stage file": Path(stage).name, "stage sha256": sha(stage), "probes sha256": res["probes sha256"],
           "stored scores sha256": {n: PIN[n] for n in ("runs.json", "scores_L48.npz", "scores_L63.npz")},
           "runs": len(todo), "runs with no meta": res["runs with no meta"]} | ({"gate record": Path(gate).name, "gate record sha256": sha(gate)} if which == "placebo" else {})
    if which == "controls":  # the reproduction gate, at each layer: each control, every nested model v != f
        ok = len(todo) == 20 == len({t["key"] for t in todo})
        for l in ("48", "63"):
            stored = np.load(d / f"scores_L{l}.npz")["output_end__mean__task_nested"]
            diffs = [abs(S[l][i, v] - stored[t["fold"], v, t["j"]]) for i, t in enumerate(todo) for v in range(5) if v != t["fold"]]
            ok_l = bool(len(diffs) == 80 and np.isfinite(diffs).all() and max(diffs) <= TOL)
            rec |= {f"gate L{l}: max |re-scored - stored| nested score (tolerance 1e-3)": float(np.nanmax(diffs)) if diffs else None,
                    f"gate L{l}: comparisons (20 controls x 4 nested models)": len(diffs), f"gate L{l}": "pass" if ok_l else "fail"}
            ok = ok and ok_l
        rec["gate"] = "pass" if ok else "fail"
    npz = HERE / "results" / f"placebo_scores_{which}_{stamp}.npz"
    np.savez(npz, j=np.array([t["j"] for t in todo]), arm=np.array([t["arm"] for t in todo]), **{f"L{l}": v for l, v in S.items()})
    rec |= {"scores file": npz.name, "scores sha256": sha(npz)}
    with open(dest := HERE / "results" / f"placebo_scores_{which}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)); print(dest)
    if which == "controls" and rec["gate"] != "pass":  # recorded above; nothing more runs
        raise SystemExit("the reproduction gate failed")
