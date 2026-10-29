"""Why does Qwen3.5 lose most of Qwen3-32B's no-editor real-vs-sim separation (2026-10-08)?

Q1 output_end, no-editor variant, held-out environment: run AUROC 0.8715 for Qwen3-32B (q1_gpu_2026-10-06_190522) and
0.6113 for Qwen3.5 matched to Qwen3-32B's 32k cut (2026-10-06_215818_q1_qwen35_matched). Each run's score is the mean
of its non-editor output scores, each scored by the probe of the run's held-out environment fold at that fold's
chosen layer. This splits the outputs both readers scored (joined on run, step, action) by
  - tool of the output;
  - context length: the output's token position in Qwen3-32B's tokenization (the same outputs for both readers);
  - editor exposure: how many text_editor outputs the run showed before it;
  - whether a review note appeared in an earlier output (the real insert tool appends one; the simulator does not).
Per stratum and reader: the output-level AUROC (real vs sim), the run AUROC from that stratum's outputs alone, and the
stratum's contribution to the run-level standardized mean difference (real minus sim run means over the pooled
within-arm SD; contributions add up to the whole). The reader gap per stratum (Qwen3-32B minus Qwen3.5 output AUROC)
gets a 95% interval from 1000 main-task draws (seed 0) shared by both readers.
Checks first: the step scores reproduce each reader's saved run scores and headline AUROC.
Tool, exposure and note come from the staged level-0 reader messages (streamed from the shared Volume; nothing is
written there or kept locally), numbered as probe_shift.reader numbers them. Saved scores only, no compute on Modal.

  uv run --with pandas python experiments/2026-10-06_qwen35_replication/analyze_q1_no_editor_strata.py
"""

import hashlib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
INPUTS_VOL, INPUTS = "probe-shift_aswin-surya_sub-la-activations", "2026-10-06_la_segment-end_qwen3-32b_v1"  # level-0 reader messages
READERS = {
    "qwen3-32b": {"results": ROOT / "experiments" / "2026-10-06_q1_segment_end" / "results" / "q1_gpu_2026-10-06_190522.json",
                  "volume": "probe-shift_aswin-surya_sub-la-activations", "id": "2026-10-06_190522"},
    "qwen3.5": {"results": HERE / "results" / "q1" / "q1_gpu_2026-10-06_215818_q1_qwen35_matched.json",
                "volume": "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06", "id": "2026-10-06_215818_q1_qwen35_matched"},
}
COL, ARMS = "real_vs_sim__no_editor_outputs__held_out_environment", ("real-attack-e1", "sim-attack")
POS = "sim-attack"  # the Q1 real-vs-sim probes score high for sim: AUROCs here are sim over real, as in the saved headline
NOTE = re.compile(r"Review the changes and make sure they are as expected")  # the insert note and the rarer str_replace wording
N_BOOT, SEED = 1000, 0


def short_key(key: str) -> str:
    """The extraction's run key (run_extract_segment_end.py): keys over 240 bytes are cut to 150 characters plus a hash."""
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def auroc(y, s, W=None):
    """Mann-Whitney AUROC of y=1 over y=0, ties counted half, under row weights W (draws x rows; None: one draw of 1s)."""
    W = np.ones((1, len(y))) if W is None else W
    pos, neg = y == 1, y == 0
    o = np.argsort(s, kind="mergesort")
    ss, Wo = s[o], W[:, o]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    P, N = np.add.reduceat(Wo * pos[o], starts, axis=1), np.add.reduceat(Wo * neg[o], starts, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (P * (np.cumsum(N, axis=1) - N + 0.5 * N)).sum(1) / (W[:, pos].sum(1) * W[:, neg].sum(1))


def step_scores(name: str, r: dict, meta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The reader's no-editor held-out-environment output scores (each at its run's fold's chosen layer), and checks."""
    res = json.loads(r["results"].read_text())
    runs = pd.DataFrame(res["manifest"]["runs"], columns=["arm", "key"]).merge(meta, on=["arm", "key"], how="left", validate="one_to_one")
    if runs.environment.isna().any():
        raise ValueError(f"{name}: {int(runs.environment.isna().sum())} runs without an environment")
    h = res["headline_held_out_environment"]["output_end"]["real_vs_sim"]["no_editor_outputs"]
    layer_of = {int(f): l for f, l in h["chosen_layer_by_fold"].items()}
    env_fold = runs.environment.map(res["config"]["environment_to_fold"]).to_numpy()
    vol = modal.Volume.from_name(r["volume"], version=2)
    base = f"{res['config']['extraction_run']}__q1gpu/{r['id']}"
    parts = []
    for L in sorted(set(layer_of.values())):
        z = np.load(io.BytesIO(b"".join(vol.read_file(f"{base}/L{L}_step_scores_output_end.npz"))))
        use = np.isin(env_fold[z["run"]], [f for f, l in layer_of.items() if l == L]) & ~np.isnan(z[COL])
        parts.append(pd.DataFrame({"run": z["run"][use], "step": z["step"][use], "action": z["action"][use],
                                   "token": z["token"][use], "score": z[COL][use].astype(np.float64), "layer": L}))
    t = pd.concat(parts, ignore_index=True)
    t = t.join(runs[["arm", "key", "cell", "main_task"]], on="run")
    if not t.arm.isin(ARMS).all():
        raise ValueError(f"{name}: scored rows outside {ARMS}")
    mean = t.groupby("run").score.mean()
    saved = pd.Series(h["run_scores"], dtype=float)
    common = saved.dropna().index.intersection(mean.index)
    y = (runs.arm.loc[common] == POS).to_numpy().astype(int)
    checks = {"runs with a saved score": int(saved.notna().sum()), "runs rebuilt": int(len(mean)),
              "max abs run score difference (saved values are rounded to 5 decimals)": float((mean.loc[common] - saved.loc[common]).abs().max()),
              "run AUROC rebuilt": float(auroc(y, mean.loc[common].to_numpy())[0]), "run AUROC saved": h["run"]["auroc"],
              "layers by fold": layer_of}
    if (checks["max abs run score difference (saved values are rounded to 5 decimals)"] > 1e-4 or len(common) != saved.notna().sum()
            or abs(checks["run AUROC rebuilt"] - h["run"]["auroc"]) > 1e-3):
        raise ValueError(f"{name}: step scores do not reproduce the saved run scores: {checks}")
    return t, checks


def context() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per run (arm, key, cell, environment) of every staged run, and per tool output of the real and simulated attack runs: tool, earlier text_editor outputs, and whether an earlier
    output carried the review note. step and action as probe_shift.reader numbers them (step: agent turn, 0-based;
    action: the k-th tool message after it)."""
    vol = modal.Volume.from_name(INPUTS_VOL, version=2)
    shards = [e.path for e in vol.listdir(f"{INPUTS}/inputs") if e.path.endswith(".jsonl")]

    def scan(path):
        rows, runs = [], []
        for line in b"".join(vol.read_file(path)).decode().splitlines():
            it = json.loads(line)
            it["key"] = short_key(it["key"])
            runs.append({k: it[k] for k in ("arm", "key", "cell", "main_task", "environment")})
            if it["arm"] not in ARMS:
                continue
            step, action, n_editor, note_before = -1, 0, 0, False
            for m in it["messages"]:
                if m["role"] == "assistant":
                    step, action = step + 1, 0
                elif m["role"] == "tool":
                    rows.append({"key": it["key"], "arm": it["arm"], "step": step, "action": action, "tool": m.get("tool_name"),
                                 "editor_before": n_editor, "note_before": note_before})
                    n_editor += m.get("tool_name") == "text_editor"
                    note_before |= bool(NOTE.search(m.get("content") or ""))
                    action += 1
        return rows, runs

    with ThreadPoolExecutor(16) as pool:
        parts = list(pool.map(scan, shards))
    return pd.DataFrame([r for p, _ in parts for r in p]), pd.DataFrame([r for _, q in parts for r in q])


def main():
    ctx, meta = context()
    tabs, checks = {}, {}
    for name, r in READERS.items():
        tabs[name], checks[name] = step_scores(name, r, meta)
    K = ["key", "step", "action"]
    t = tabs["qwen3-32b"].merge(tabs["qwen3.5"][K + ["score", "token"]], on=K, how="inner", suffixes=("", "_q35"), validate="one_to_one")
    n_ctx = len(t)
    t = t.merge(ctx, on=["key", "arm", "step", "action"], how="left", validate="one_to_one")
    checks["joined"] = {"qwen3-32b outputs": len(tabs["qwen3-32b"]), "qwen3.5 outputs": len(tabs["qwen3.5"]), "common": n_ctx,
                        "missing from the transcript scan": int(t.tool.isna().sum()),
                        "text_editor among scored outputs": int((t.tool == "text_editor").sum())}
    if checks["joined"]["missing from the transcript scan"] or checks["joined"]["text_editor among scored outputs"]:
        raise ValueError(f"the transcript scan does not line up with the scored outputs: {checks['joined']}")
    print(json.dumps(checks, indent=1))
    t["score_q3"], y = t.score, (t.arm == POS).to_numpy().astype(int)
    top = t.tool.value_counts()
    t["tool_s"] = t.tool.where(t.tool.isin(top[top >= 500].index), "other")
    t["length"] = pd.cut(t.token, [0, 8192, 16384, 32768], right=False, labels=["0-8k", "8-16k", "16-32k"]).astype(str)
    t["exposure"] = pd.cut(t.editor_before, [0, 1, 4, 11, 10**9], right=False, labels=["0", "1-3", "4-10", "11+"]).astype(str)
    t["note"] = np.where(t.note_before, "after a review note", "no note yet")
    partitions = {"tool": "tool_s", "context length (Qwen3-32B tokens)": "length", "earlier text_editor outputs": "exposure",
                  "review note earlier in the run": "note"}
    tasks, inv = np.unique(t.main_task.to_numpy(), return_inverse=True)  # resample main tasks (the CV grouping unit)
    rng = np.random.default_rng(SEED)
    counts = np.stack([np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(N_BOOT)])
    W = np.vstack([np.ones((1, len(tasks))), counts])[:, inv].astype(float)  # row 0: the point estimate
    S = lambda x: {"value": round(float(x[0]), 4), "ci95": [round(float(v), 4) for v in np.nanpercentile(x[1:], [2.5, 97.5])]}
    run_y = t.groupby("run").arm.first().eq(POS).astype(int)
    n_run = t.groupby("run").size()
    out = {"checks": checks, "overall": {}, "partitions": {}}
    for rd, col in (("qwen3-32b", "score_q3"), ("qwen3.5", "score_q35")):
        run_mean = t.groupby("run")[col].mean()
        sd = np.sqrt(((run_mean[run_y == 1].var() * ((run_y == 1).sum() - 1)) + run_mean[run_y == 0].var() * ((run_y == 0).sum() - 1))
                     / (len(run_y) - 2))
        out["overall"][rd] = {"run AUROC (common outputs)": round(float(auroc(run_y.to_numpy(), run_mean.to_numpy())[0]), 4),
                              "output AUROC": round(float(auroc(y, t[col].to_numpy())[0]), 4),
                              "run standardized mean difference": round(float((run_mean[run_y == 1].mean() - run_mean[run_y == 0].mean()) / sd), 4),
                              "pooled run SD": float(sd)}
    for pname, c in partitions.items():
        res = {}
        for level in sorted(t[c].unique()):
            m = (t[c] == level).to_numpy()
            r = {"outputs": {"real": int((m & (y == 0)).sum()), "sim": int((m & (y == 1)).sum())}}
            a = {}
            for rd, col in (("qwen3-32b", "score_q3"), ("qwen3.5", "score_q35")):
                s = t[col].to_numpy()
                a[rd] = auroc(y[m], s[m], W[:, m])
                sub = t[m].groupby("run")[col]
                share = sub.sum() / n_run.loc[sub.sum().index]  # this stratum's part of each run's mean
                share = share.reindex(n_run.index, fill_value=0.0)
                contrib = (share[run_y == 1].mean() - share[run_y == 0].mean()) / out["overall"][rd]["pooled run SD"]
                rm = sub.mean()
                r[rd] = {"output AUROC": S(a[rd]), "run AUROC from this stratum alone": round(float(auroc(run_y.loc[rm.index].to_numpy(), rm.to_numpy())[0]), 4),
                         "runs": int(len(rm)), "contribution to the run standardized difference": round(float(contrib), 4),
                         "mean score real": round(float(s[m & (y == 0)].mean()), 4), "mean score sim": round(float(s[m & (y == 1)].mean()), 4)}
            r["reader gap in output AUROC (qwen3-32b minus qwen3.5)"] = S(a["qwen3-32b"] - a["qwen3.5"])
            res[level] = r
        out["partitions"][pname] = res
    # exposure and length go together (later outputs have seen more editor outputs): the two crossed
    cross = {}
    for e in ("0", "1-3", "4-10", "11+"):
        for ln in ("0-8k", "8-16k", "16-32k"):
            m = ((t.exposure == e) & (t.length == ln)).to_numpy()
            n = {"real": int((m & (y == 0)).sum()), "sim": int((m & (y == 1)).sum())}
            if min(n.values()) < 100:
                cross[f"exposure {e}, {ln}"] = {"outputs": n}
                continue
            a = {rd: auroc(y[m], t[col].to_numpy()[m], W[:, m]) for rd, col in (("qwen3-32b", "score_q3"), ("qwen3.5", "score_q35"))}
            cross[f"exposure {e}, {ln}"] = {"outputs": n, **{f"{rd} output AUROC": S(v) for rd, v in a.items()},
                                            "reader gap": S(a["qwen3-32b"] - a["qwen3.5"])}
    out["exposure x length"] = cross
    # the note within real runs, at matched exposure: real outputs after a note minus real outputs with none yet, in each
    # reader's output SD (positive: more sim-like). The simulator writes almost no notes, so this is a real-only contrast.
    note = {}
    for e in ("1-3", "4-10", "11+"):
        base = (t.arm == "real-attack-e1") & (t.exposure == e)
        row = {"real outputs after a note": int((base & t.note_before).sum()), "real outputs with no note yet": int((base & ~t.note_before).sum())}
        for rd, col in (("qwen3-32b", "score_q3"), ("qwen3.5", "score_q35")):
            row[f"{rd}: after a note minus none yet, in output SDs"] = round(float(
                (t.loc[base & t.note_before, col].mean() - t.loc[base & ~t.note_before, col].mean()) / t[col].std()), 4)
        note[f"exposure {e}"] = row
    out["review note within real runs"] = note
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"q1_no_editor_strata_{stamp}.json"
    path.write_text(json.dumps({"readers": {k: {"results": str(v["results"].relative_to(ROOT)), "volume": v["volume"], "id": v["id"]}
                                            for k, v in READERS.items()}, "inputs": INPUTS, "column": COL, "n_boot": N_BOOT, "seed": SEED,
                                **out}, indent=1, default=float))
    print(json.dumps(out["overall"], indent=1))
    for pname, res in out["partitions"].items():
        print(f"== {pname}")
        for level, r in res.items():
            g = r["reader gap in output AUROC (qwen3-32b minus qwen3.5)"]
            print(f"  {level[:22]:22} n {r['outputs']['real']:6}/{r['outputs']['sim']:6}  output AUROC {r['qwen3-32b']['output AUROC']['value']:.3f} / "
                  f"{r['qwen3.5']['output AUROC']['value']:.3f}  gap {g['value']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]  run AUROC alone "
                  f"{r['qwen3-32b']['run AUROC from this stratum alone']:.3f} / {r['qwen3.5']['run AUROC from this stratum alone']:.3f}  contrib "
                  f"{r['qwen3-32b']['contribution to the run standardized difference']:+.3f} / {r['qwen3.5']['contribution to the run standardized difference']:+.3f}")
    print("== exposure x length")
    for k, r in cross.items():
        if "reader gap" in r:
            print(f"  {k:24} n {r['outputs']['real']:5}/{r['outputs']['sim']:5}  {r['qwen3-32b output AUROC']['value']:.3f} / "
                  f"{r['qwen3.5 output AUROC']['value']:.3f}  gap {r['reader gap']['value']:+.3f} [{r['reader gap']['ci95'][0]:+.3f}, {r['reader gap']['ci95'][1]:+.3f}]")
        else:
            print(f"  {k:24} n {r['outputs']} (too few)")
    print("== review note within real runs", json.dumps(note, indent=1))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
