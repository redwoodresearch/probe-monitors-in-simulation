"""Score the Needham eval-awareness probes on LinuxArena call_end activations (Modal CPU; a projection needs no GPU).

Sources (Volume probe-shift_aswin-surya_sub-la-activations, mounted read-only):
  alllayers  run 2026-10-06_la_segment-end_qwen3-32b_alllayers_v1: every layer 0-63, per-layer chunk files
             (by_layer/L<l>/<chunk>.npy) with per-transcript metadata (meta/<arm>/<file>.json: positions and rows)
  v1         run 2026-10-06_la_segment-end_qwen3-32b_v1: layers 16-63 saved (8 also), per-transcript npz
Both have the same transcripts, text and positions (probe_shift.reader.segment_end_positions). Only call_end is read
(the call's last content token). A probe ((x - mean) / scale) @ coef is folded to x @ w - b, scored per layer.
Writes data/scores_needham_<source>_<stamp>.parquet (one row per call, one column per readout and layer;
git-ignored, also on the Q3-eval-awareness Volume) and results/scores_needham_<source>_<stamp>.json (counts, checks).

  uv run --with pandas --with pyarrow modal run experiments/2026-10-06_q3_eval_awareness_needham/run_score_la.py \
      --source alllayers [--limit-chunks 2]
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3evalaw_score_kill-2026-11-06"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
VOLUME = "probe-shift_aswin-surya_sub-q3evalaw_data_kill-2026-11-06"
RUNS = {"alllayers": "2026-10-06_la_segment-end_qwen3-32b_alllayers_v1", "v1": "2026-10-06_la_segment-end_qwen3-32b_v1"}
PROBE_RUN = "needham_v1"
READOUTS = ["asst_last", "asst_mean", "all_last", "all_mean"]
META = ("arm", "key", "cell", "main_task", "side_task", "environment")

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS).read_only()
vol = modal.Volume.from_name(VOLUME)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy").add_local_python_source("probe_shift")


@app.function(image=image, volumes={"/acts": acts_vol, "/data": vol}, cpu=2, memory=8192, timeout=1800)
def setup(source: str, layers: list[int]) -> dict:
    """Folded probes {layer: (W [H, R], b [R])} and the work units: chunk names (alllayers) or transcript stems (v1)."""
    import numpy as np

    from probe_shift.offpolicy import load_probe

    folded = {}
    for l in layers:
        ps = [load_probe((Path("/data/probes") / PROBE_RUN / r / f"probe_L{l}.npz").read_bytes()) for r in READOUTS]
        W = np.stack([p["coef"] / p["scaler_scale"] for p in ps], 1).astype(np.float32)
        b = np.array([float(p["scaler_mean"] @ (p["coef"] / p["scaler_scale"])) for p in ps], np.float32)
        folded[l] = (W, b)
    run = Path("/acts") / RUNS[source]
    if source == "alllayers":
        units = sorted(p.stem for p in (run / "chunks").glob("*.json"))
        n_meta = sum(1 for _ in (run / "meta").glob("*/*.json"))
        n_stats = sum(1 for _ in (run / "stats").glob("shard_*.json"))
        return {"folded": folded, "units": units, "transcripts_done": n_meta, "shards_done": n_stats}
    units = sorted(f"{p.parent.name}/{p.name.removesuffix('.json')}" for p in (run / "acts").glob("*/*.json"))
    return {"folded": folded, "units": units, "transcripts_done": len(units)}


@app.function(image=image, volumes={"/acts": acts_vol}, cpu=2, memory=16384, timeout=3600, retries=2)
def score_unit(source: str, units: list[str], folded: dict) -> dict:
    import numpy as np

    run = Path("/acts") / RUNS[source]
    cols = {k: [] for k in META + ("step", "action", "tool", "token", "token_text", "truncated")}
    cols |= {f"{r}_L{l}": [] for l in folded for r in READOUTS}
    for u in units:
        if source == "alllayers":
            ch = json.loads((run / "chunks" / f"{u}.json").read_text())
            rows, metas = [], []
            for t in ch["transcripts"]:
                m = json.loads((run / "meta" / t["arm"] / f"{t['filename']}.json").read_text())
                if m["chunk"] != u or m["row_start"] != t["row_start"] or m["n_rows"] != t["n_rows"]:
                    raise ValueError(f"{u}: metadata of {t['key']} disagrees with the chunk index")
                ce = [p for p in m["positions"] if p["kept"] and p["kind"] == "call_end"]
                rows += [t["row_start"] + p["row"] for p in ce]
                metas += [(m, p) for p in ce]
            rows = np.array(rows, int)
            for l, (W, b) in folded.items():
                x = np.load(run / "by_layer" / f"L{l}" / f"{u}.npy", mmap_mode="r")
                if x.shape[0] != ch["rows"]:
                    raise ValueError(f"{u} L{l}: {x.shape[0]} rows, index says {ch['rows']}")
                s = np.asarray(x[rows], np.float32) @ W - b
                for j, r in enumerate(READOUTS):
                    cols[f"{r}_L{l}"] += s[:, j].tolist()
        else:
            arm, stem = u.split("/", 1)
            m = json.loads((run / "acts" / arm / f"{stem}.json").read_text())
            ce = [p for p in m["positions"] if p["kept"] and p["kind"] == "call_end"]
            z = np.load(run / "acts" / arm / f"{stem}.npz")
            rows = np.array([p["row"] for p in ce], int)
            for l, (W, b) in folded.items():
                s = z[f"L{l}"][rows].astype(np.float32) @ W - b
                for j, r in enumerate(READOUTS):
                    cols[f"{r}_L{l}"] += s[:, j].tolist()
            metas = [(m, p) for p in ce]
        for m, p in metas:
            for k in META:
                cols[k].append(m[k])
            for k in ("step", "action", "tool", "token", "token_text"):
                cols[k].append(p[k])
            cols["truncated"].append(m["truncated"])
    return cols


@app.local_entrypoint()
def main(source: str = "alllayers", limit_chunks: int = 0, batch: int = 4):
    import numpy as np
    import pandas as pd

    from probe_shift.offpolicy import provenance

    layers = list(range(64)) if source == "alllayers" else [16, 24, 32, 40, 48, 56, 63]
    st = setup.remote(source, layers)
    print({k: v for k, v in st.items() if k not in ("folded", "units")}, len(st["units"]), "units", flush=True)
    if source == "alllayers" and not limit_chunks and (st["shards_done"] < 40 or st["transcripts_done"] < 3933):
        raise RuntimeError("the all-layers run has not finished; use --source v1 or wait")
    units = st["units"][:limit_chunks] if limit_chunks else st["units"]
    groups = [units[i:i + batch] for i in range(0, len(units), batch)]
    parts = list(score_unit.map([source] * len(groups), groups, [st["folded"]] * len(groups)))
    df = pd.concat([pd.DataFrame(p) for p in parts], ignore_index=True)
    score_cols = [c for c in df.columns if c.endswith(tuple(f"_L{l}" for l in layers))]
    nonfinite = int((~np.isfinite(df[score_cols].to_numpy())).sum())
    if nonfinite or df.duplicated(["arm", "key", "step", "action"]).any():
        raise ValueError(f"{nonfinite} non-finite scores, or duplicate calls")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S") + ("_test" if limit_chunks else "")
    name = f"scores_needham_{source}_{stamp}"
    (HERE / "data").mkdir(exist_ok=True)
    path = HERE / "data" / f"{name}.parquet"
    df.to_parquet(path, index=False)
    with vol.batch_upload() as up:
        up.put_file(str(path), f"scores/{path.name}")
    summary = {"source": source, "activations": f"{ACTS}/{RUNS[source]}", "probes": f"{VOLUME}/probes/{PROBE_RUN}", "layers": layers,
               "readouts": READOUTS, "calls": len(df), "runs_per_arm": df.groupby("arm").key.nunique().to_dict(),
               "file": str(path.relative_to(HERE)), "volume_file": f"{VOLUME}/scores/{path.name}", "app": APP,
               "provenance": provenance(__file__)}
    (HERE / "results" / f"{name}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("calls", "runs_per_arm")}), "->", path)
