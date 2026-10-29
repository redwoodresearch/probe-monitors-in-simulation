"""Fix-full (PLAN.md here), Qwen3-32B on the simulated honest runs under fix-full: a copy of
../2026-10-12_q3_edit_honest_fa/score_edit_honest.py (main @ 8c204093) reading this folder's extraction and staged arm
(sim-honest-fx-full: all 437 runs, 6 of them unchanged, whose every output must then score as the unedited run), with its
own app and fx_ result names; scoring and checks unchanged. Its docstring follows.

POST HOC (RULE.md), step 5: score the frozen #36 reply readout (output_end) at layers 63 (its committed layer) and 48
on (a) every run of the committed simulated-honest scoring (sim_honest_scores_2026-10-07_124717: 4,254 runs of the v4
all-layers extraction) and (b) the edited copies of the simulated honest runs. Modal CPU; saved activations only; no
model calls; returns scores and checks, no text.

As run_score_sim_honest.py: a run's score per model is the mean over its output_end positions of
((x - mean) / std) . w + b with the frozen probes of run 2026-10-06_181548; outer model f scores fold f, nested model
(f, v) scores folds f and v. Checks returned (the local side stops on any failure):
  (1) layer 63, every run of (a): outer and nested equal the committed scores.npz (simhonest/2026-10-07_124717);
  (2) both layers, the base run's 3,807 runs: outer and nested against the base run's own scores_L<l>.npz;
  (3) every edited run: outputs before its first changed message score as its unedited run (per call, outer model);
  (4) the edited extraction read the staged edited inputs with V4's model, revision, dtype, attention and cut; every
      staged transcript was extracted exactly once and the chunk files tile its metas.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy modal run \
      experiments/2026-10-12_q3_edit_honest_fa/score_edit_honest.py --stage results/stage_<stamp>.json
"""

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-q2-fixfull_qwen3honest_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
V4_EXT, V4_STAGED = "2026-10-06_la_segment-end-v4_qwen3-32b_alllayers_v1", "2026-10-06_la_segment-end-v4_qwen3-32b_v1"
SOURCES = {"sim-honest-fx-full": ("2026-10-13_fixfull_qwen3-32b_l48-63_v2", "2026-10-13_q2_fixfull_inputs_v2", 437),
           "sim-honest-fx-NL": ("2026-10-13_fixfull_qwen3-32b_l48-63_v2", "2026-10-13_q2_fixfull_inputs_v2", 437),
           "sim-honest-fx-fullF9": ("2026-10-13_fixfull_f9_qwen3-32b_l48-63_v1", "2026-10-13_q2_fixfull_inputs_f9r_v1", 45)}  # --arm picks one
ED_ARMS = tuple(SOURCES)
BASE_RUN, COMMITTED = "2026-10-06_181548", "simhonest/2026-10-07_124717"
RUNS = REPO / "experiments/2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717/runs.json"
BASE_RUNS = REPO / "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"
LAYERS, RO = (63, 48), "output_end"

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
work_vol = modal.Volume.from_name(WORK, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3").add_local_python_source("probe_shift")


@app.function(image=image, cpu=8, memory=65536, timeout=3600, retries=0,
              volumes={"/acts": acts_vol.read_only(), "/work": work_vol.read_only()})
def score(runs: list[dict], edited: list[str], n_base: int, ED_ARM: str = "sim-honest-fx-full") -> dict:
    try:
        return _score(runs, edited, n_base, ED_ARM)
    except Exception as e:  # noqa: BLE001 — re-raised with its type and step only, never swallowed
        raise RuntimeError(f"score failed: {type(e).__name__} at {getattr(e, 'where', 'unknown step')}") from None


def _score(runs: list[dict], edited: list[str], n_base: int, ED_ARM: str) -> dict:
    import numpy as np
    ED_EXT, ED_STAGED, _ = SOURCES[ED_ARM]

    from probe_shift.attack_eval import pool

    acts = Path("/acts")
    n, m_ed = len(runs), len(edited)
    want = {(r["arm"], r["key"]): i for i, r in enumerate(runs)}
    meta = [None] * n
    paths = [p for arm in sorted({r["arm"] for r in runs}) for p in (acts / V4_EXT / "meta" / arm).glob("*.json")]
    with ThreadPoolExecutor(32) as ex:
        for m in ex.map(lambda p: json.loads(p.read_text()), paths):
            i = want.get((m["arm"], m["key"]))
            if i is not None:
                meta[i] = m
    if any(m is None for m in meta):
        raise ValueError("runs missing from the v4 extraction")
    ed_meta = {}
    for p in (acts / ED_EXT / "meta" / ED_ARM).glob("*.json"):
        m = json.loads(p.read_text())
        if m["key"] in ed_meta:
            raise ValueError("an edited run extracted twice")
        ed_meta[m["key"]] = m
    staged_ed = {}
    for sh in sorted((acts / ED_STAGED / "inputs").glob("shard_*.jsonl")):
        for line in sh.read_text().splitlines():
            it = json.loads(line)
            if it["arm"] == ED_ARM:
                staged_ed[it["key"]] = it["messages"]
    if set(ed_meta) != set(edited) or set(staged_ed) != set(edited):
        raise ValueError(f"edited: {len(ed_meta)} extracted, {len(staged_ed)} staged, {len(edited)} expected")
    # (4) same reading of the same staged run
    cfg_v4, cfg_ed = (json.loads((acts / r / "config.json").read_text()) for r in (V4_EXT, ED_EXT))
    if cfg_ed["inputs_from"] != ED_STAGED or any(cfg_ed[k] != cfg_v4[k] for k in ("model", "revision", "dtype", "attn_implementation", "max_tokens")):
        raise ValueError("the edited extraction's settings differ from v4's")
    spans = {}  # fix-full: the run holds several arms in shared chunks, so tiling is checked over every arm's metas
    all_metas = [json.loads(p.read_text()) for p in (acts / ED_EXT / "meta").glob("*/*.json")]
    if len({(m["arm"], m["key"]) for m in all_metas}) != len(all_metas):
        raise ValueError("a transcript extracted twice in the run")
    for m in all_metas:
        spans.setdefault(m["chunk"], []).append((m["row_start"], m["row_start"] + m["n_rows"]))
    for l in LAYERS:  # both layers: an orphan chunk written to one layer before a crash must fail
        if {f.stem for f in (acts / ED_EXT / "by_layer" / f"L{l}").glob("*.npy")} != set(spans):
            raise ValueError("edited chunk files without metas, or metas without chunk files")
    for ch, sp in spans.items():
        sp.sort()
        for l in LAYERS:
            rows = np.load(acts / ED_EXT / "by_layer" / f"L{l}" / f"{ch}.npy", mmap_mode="r").shape[0]
            if sp[0][0] != 0 or any(a[1] != b[0] for a, b in zip(sp, sp[1:])) or sp[-1][1] != rows:
                raise ValueError("edited chunk rows do not tile the metas")
    # the first changed message of each edited run, as (step, action) of a tool output (only tool outputs may differ)
    v4_msgs = {}
    for sh in sorted((acts / V4_STAGED / "inputs").glob("shard_*.jsonl")):
        for line in sh.read_text().splitlines():
            it = json.loads(line)
            if it["arm"] == "sim-honest" and it["key"] in staged_ed:
                v4_msgs[it["key"]] = it["messages"]
    first = {}
    for k, msgs in staged_ed.items():
        base, step, action, f = v4_msgs[k], -1, 0, None
        if len(base) != len(msgs):
            raise ValueError("message counts differ")
        for b, m in zip(base, msgs):
            if m["role"] == "assistant":
                step, action = step + 1, 0
            if b != m:
                if m["role"] != "tool" or b["role"] != "tool":
                    raise ValueError("a non-tool message differs")
                f = f or (step, action)
            if m["role"] == "tool":
                action += 1
        first[k] = f if f is not None else (10 ** 9, 0)  # an unchanged run: every output is before the (absent) first change
    sim_idx = {r["key"]: i for i, r in enumerate(runs) if r["arm"] == "sim-honest"}
    fold = np.array([r["fold"] for r in runs] + [runs[sim_idx[k]]["fold"] for k in edited])
    N = n + m_ed
    # rows: (run index, extraction, chunk, row, step, action) of every kept output_end position
    rows = []
    for i, m in enumerate(meta):
        rows += [(i, V4_EXT, m["chunk"], m["row_start"] + p["row"], p["step"], p["action"]) for p in m["positions"] if p["kind"] == RO and p["kept"]]
    for j, k in enumerate(edited):
        m = ed_meta[k]
        rows += [(n + j, ED_EXT, m["chunk"], m["row_start"] + p["row"], p["step"], p["action"]) for p in m["positions"] if p["kind"] == RO and p["kept"]]
    run_of = np.array([r[0] for r in rows])
    out = {"probes_sha256": {}, "feature_sha256": {}, "rows": len(rows), "rows_per_run": np.bincount(run_of, minlength=N).tolist()}
    scores, prefix = {}, {}
    for l in LAYERS:
        X = np.zeros((len(rows), 5120), np.float16)
        by = {}
        for j, (_, ext, ch, row, _, _) in enumerate(rows):
            by.setdefault((ext, ch), []).append((j, row))
        for (ext, ch), its in by.items():
            block = np.load(acts / ext / "by_layer" / f"L{l}" / f"{ch}.npy", mmap_mode="r")
            X[[j for j, _ in its]] = block[[r for _, r in its]]
        if not np.isfinite(X).all():
            raise ValueError(f"L{l}: non-finite features")
        out["feature_sha256"][str(l)] = hashlib.sha256(X.tobytes()).hexdigest()
        X = X.astype(np.float32)
        pp = Path("/work/runs") / BASE_RUN / f"probes_L{l}.npz"
        out["probes_sha256"][str(l)] = hashlib.sha256(pp.read_bytes()).hexdigest()
        z = np.load(pp)

        def call_scores(model, sel):
            key = f"{RO}__{model}"
            return (((X[sel] - z[f"{key}__mean"]) / z[f"{key}__std"]) @ z[f"{key}__w"] + np.float32(z[f"{key}__b"])).astype(np.float64)

        outer, nested = np.full(N, np.nan), np.full((5, 5, N), np.nan)
        call_outer = np.full(len(rows), np.nan)
        for f in range(5):
            sel = np.flatnonzero(fold[run_of] == f)
            s = call_scores(f"task_outer_{f}", sel)
            call_outer[sel] = s
            outer[fold == f] = pool(run_of[sel], s, N, "mean")[fold == f]
            for v in range(5):
                if v != f:
                    sel = np.flatnonzero((fold[run_of] == f) | (fold[run_of] == v))
                    sc = (fold == f) | (fold == v)
                    nested[f, v][sc] = pool(run_of[sel], call_scores(f"task_nested_{f}_{v}", sel), N, "mean")[sc]
        scores[l] = {"outer": outer, "nested": nested}
        # (3) prefix: per edited run, outputs before its first changed message against the unedited run's same outputs
        per_run = {}
        for r, c in zip(rows, call_outer):
            per_run.setdefault(r[0], {})[(r[4], r[5])] = c
        n_cmp, worst, runs_with_prefix, set_mismatch = 0, 0.0, 0, 0
        for j, k in enumerate(edited):
            e = {pa: c for pa, c in per_run.get(n + j, {}).items() if pa < first[k]}
            b = {pa: c for pa, c in per_run.get(sim_idx[k], {}).items() if pa < first[k]}
            if set(e) != set(b):  # every prefix output of the unedited run must be there, and nothing else
                set_mismatch += 1
                continue
            for pa, c in e.items():
                n_cmp, worst = n_cmp + 1, max(worst, abs(c - b[pa]))
            runs_with_prefix += bool(e)
        prefix[str(l)] = {"calls_compared": n_cmp, "max_abs_diff": worst, "edited_runs_with_a_prefix_output": runs_with_prefix,
                          "runs_with_prefix_position_mismatch": set_mismatch}
        ed = slice(n, N)
        base_cols = [sim_idx[k] for k in edited]  # stricter than RULE's dropout clause: a newly missing edited score stops the job
        if not ((np.isfinite(outer[ed]) == np.isfinite(outer[base_cols])).all()
                and (np.isfinite(nested[..., ed]) == np.isfinite(nested[..., base_cols])).all()):
            raise ValueError(f"L{l}: an edited run's outer or nested score is missing where its unedited run has one, or the reverse")
    # (1) committed layer-63 scores of every run of (a)
    zc = np.load(Path("/work") / COMMITTED / "scores.npz")
    co, cn = zc[f"{RO}__outer"], zc[f"{RO}__nested"]
    o63, n63 = scores[63]["outer"][:n], scores[63]["nested"][..., :n]
    check1 = {"nan_pattern_equal": bool(np.array_equal(np.isnan(co), np.isnan(o63)) and np.array_equal(np.isnan(cn), np.isnan(n63))),
              "outer_max_abs_diff": float(np.nanmax(np.abs(co - o63))), "nested_max_abs_diff": float(np.nanmax(np.abs(cn - n63)))}
    # (2) the base run's own scores (its committed layers' features came from the base dump, not v4)
    check2 = {}
    for l in LAYERS:
        zb = np.load(Path("/work/runs") / BASE_RUN / f"scores_L{l}.npz")
        bo, bn = zb[f"{RO}__mean__task_outer"][:n_base], zb[f"{RO}__mean__task_nested"][..., :n_base]
        so, sn = scores[l]["outer"][:n_base], scores[l]["nested"][..., :n_base]
        check2[str(l)] = {"nan_pattern_equal": bool(np.array_equal(np.isnan(bo), np.isnan(so)) and np.array_equal(np.isnan(bn), np.isnan(sn))),
                          "outer_max_abs_diff": float(np.nanmax(np.abs(bo - so))), "nested_max_abs_diff": float(np.nanmax(np.abs(bn - sn)))}
    out |= {"check1_committed_L63": check1, "check2_base_run": check2, "check3_prefix": prefix,
            "check4": {"edited_extracted": len(ed_meta), "chunks": len(spans)},
            "first_changed_output": {k: list(v) for k, v in first.items()}}
    return out | {"scores": {str(l): {"outer": v["outer"], "nested": v["nested"]} for l, v in scores.items()}}


@app.local_entrypoint()
def main(stage: str, arm: str = "sim-honest-fx-full"):
    if arm not in ED_ARMS:
        raise SystemExit(f"--arm must be one of {ED_ARMS}")
    ED_ARM = arm
    import numpy as np

    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO,
                           check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"refusing to run from a dirty tree:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()
    runs = json.loads(RUNS.read_text())
    base = json.loads(BASE_RUNS.read_text())
    if [(r["arm"], r["key"]) for r in runs[:len(base)]] != [(r["arm"], r["key"]) for r in base]:
        raise SystemExit("the scored runs do not start with the base run's runs in order")
    st = json.loads((HERE / stage).read_text())
    edited = sorted(st["runs by arm"][ED_ARM])
    if len(edited) != SOURCES[ED_ARM][2] or st.get("staged") != SOURCES[ED_ARM][1]:
        raise SystemExit(f"{len(edited)} staged honest runs from {st.get('staged')}, not {SOURCES[ED_ARM][2]} from {SOURCES[ED_ARM][1]}")
    t0 = datetime.now()
    res = score.remote(runs, edited, len(base), ED_ARM)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    S = res.pop("scores")
    (HERE / "results").mkdir(exist_ok=True)
    npz = HERE / "results" / f"fx_honest_scores_{ED_ARM}_{stamp}.npz"  # git-ignored (*.npz): archived to S3 with a manifest
    np.savez(npz, **{f"L{l}__{k}": v[k] for l, v in S.items() for k in ("outer", "nested")})
    rec = {"stamp": stamp, "git": head, "app": APP, "stage_record": Path(stage).name, "runs_json_sha256": hashlib.sha256(RUNS.read_bytes()).hexdigest(),
           "honest arm": ED_ARM, "n_runs": len(runs), "n_base": len(base), "edited_keys": edited, "scores_file": npz.name,
           "scores_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(), "seconds": (datetime.now() - t0).total_seconds()} | res
    gates = {"check1": res["check1_committed_L63"]["nan_pattern_equal"] and max(res["check1_committed_L63"]["outer_max_abs_diff"],
                                                                                 res["check1_committed_L63"]["nested_max_abs_diff"]) <= 1e-5,
             "check2": all(c["nan_pattern_equal"] and max(c["outer_max_abs_diff"], c["nested_max_abs_diff"]) <= 1e-3 for c in res["check2_base_run"].values()),
             "check3": all(p["calls_compared"] > 0 and p["max_abs_diff"] <= 1e-4 and p["runs_with_prefix_position_mismatch"] == 0
                           for p in res["check3_prefix"].values())}
    rec["gates"] = gates | {"all": all(gates.values())}
    with (HERE / "results" / f"fx_honest_scoring_{ED_ARM}_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("edited_keys", "first_changed_output", "rows_per_run")}, indent=1))
    if not rec["gates"]["all"]:
        raise SystemExit(f"a scoring check FAILED: {gates}; stop and report")
