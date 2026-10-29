"""Score an off-policy probe on LinuxArena activations (Modal, CPU containers; a projection needs no GPU).

Two readouts:
  - segment_end: every kept call_end and call_marker position of the segment-end extraction (all 3,933 transcripts).
  - turn_mean: every agent turn's mean activation in the pooled extraction (under-32k transcripts only). A linear
    probe on a mean activation equals the mean of its per-token scores, so this is Apollo's own aggregation (mean
    over the response's tokens), applied to an agent turn: its visible text and tool calls, with template tokens.
Probes are read from their Volume in the local entrypoint and passed to the workers; activations are mounted read
only. Writes data/scores_<probe>_<readout>_<stamp>.parquet (one row per position or turn, one column per layer;
git-ignored, also uploaded to the Q3 Volume) and results/scores_<probe>_<readout>_<stamp>.json (counts, checks).

  PYTHONPATH=src uv run --no-project --python 3.13 --with 'modal>=1.2' --with pyyaml --with numpy --with pandas \
      --with pyarrow modal run experiments/2026-10-06_q3_offpolicy_probes/run_score_probes.py \
      --probe apollo_deception --readout segment_end [--limit 3]
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3_score_kill-2026-11-06"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
META = ("arm", "key", "cell", "main_task", "side_task", "environment")

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy").add_local_python_source("probe_shift")


@app.function(image=image, volumes={"/acts": acts_vol}, cpu=2, memory=4096, timeout=1800)
def list_runs(run: str, suffix: str) -> list[str]:
    """`<arm>/<stem>` for every transcript whose metadata file (written last, so it marks the transcript done) exists."""
    return sorted(f"{p.parent.name}/{p.name.removesuffix(suffix)}" for p in (Path("/acts") / run / "acts").glob(f"*/*{suffix}"))


@app.function(image=image, volumes={"/acts": acts_vol}, cpu=2, memory=8192, timeout=1800, retries=2)
def score_batch(cfg: dict, readout: str, items: list[str], probes: dict) -> dict:
    import numpy as np

    from probe_shift.offpolicy import probe_scores

    cols = {k: [] for k in META + ("kind", "step", "action", "tool", "token", "token_text", "truncated")}
    cols |= {f"L{l}": [] for l in probes}
    for item in items:
        arm, stem = item.split("/", 1)
        if readout == "segment_end":
            d = Path("/acts") / cfg["segment_end_run"] / "acts" / arm
            m = json.loads((d / f"{stem}.json").read_text())
            keep = [p for p in m["positions"] if p["kept"] and p["kind"] in cfg["kinds"]]
            rows = [(p["kind"], p["step"], p["action"], p["tool"], p["token"], p["token_text"]) for p in keep]
            array_of = lambda l: f"L{l}"
            truncated = m["truncated"]
        else:
            d = Path("/acts") / cfg["pooled_run"] / "acts" / arm
            m = json.loads((d / f"{stem}.segments.json").read_text())
            keep = [s for s in m["segments"] if s["kind"] == "agent"]
            rows = [("turn_mean", s["step"], None, s["tool"], None, None) for s in keep]
            array_of = lambda l: f"L{l}_mean"
            truncated = False  # the pooled run read only transcripts under 32k
        if m["arm"] != arm:
            raise ValueError(f"{item}: metadata says arm {m['arm']}")
        z = np.load(d / f"{stem}.npz")
        idx = np.array([p["row"] for p in keep], int)
        for l, probe in probes.items():
            cols[f"L{l}"] += probe_scores(z[array_of(l)][idx], probe).tolist()
        for r in rows:
            for k, v in zip(("kind", "step", "action", "tool", "token", "token_text"), r):
                cols[k].append(v)
            for k in META:
                cols[k].append(m[k])
            cols["truncated"].append(truncated)
    return cols


@app.local_entrypoint()
def main(probe: str = "apollo_deception", readout: str = "segment_end", limit: int = 0):
    """`--limit N` scores N transcripts per arm (a test, written with a _test suffix)."""
    import io

    import numpy as np
    import pandas as pd
    import yaml

    from probe_shift.offpolicy import load_probe, provenance

    if readout not in ("segment_end", "turn_mean"):
        raise ValueError(f"unknown readout {readout}")
    cfg = yaml.safe_load((HERE / "config" / "q3.yaml").read_text())
    pc = cfg["probes"][probe]
    pvol = modal.Volume.from_name(pc["volume"])
    raw = {l: b"".join(pvol.read_file(f"{pc['dir']}/probe_L{l}.npz")) for l in pc["layers"]}
    probes = {l: load_probe(b) for l, b in raw.items()}
    run, suffix = (cfg["segment_end_run"], ".json") if readout == "segment_end" else (cfg["pooled_run"], ".segments.json")
    items = list_runs.remote(run, suffix)
    by_arm = {a: [i for i in items if i.startswith(a + "/")] for a in cfg["cohorts"]}
    print("transcripts per arm:", {a: len(v) for a, v in by_arm.items()})
    if limit:
        items = [i for v in by_arm.values() for i in v[:limit]]
    bs = cfg["batch_size"]
    batches = [items[i:i + bs] for i in range(0, len(items), bs)]
    print(f"scoring {len(items)} transcripts in {len(batches)} containers, layers {pc['layers']}", flush=True)
    parts = list(score_batch.map([cfg] * len(batches), [readout] * len(batches), batches, [probes] * len(batches)))
    df = pd.concat([pd.DataFrame(p) for p in parts], ignore_index=True)
    layer_cols = [f"L{l}" for l in pc["layers"]]
    nonfinite = int((~np.isfinite(df[layer_cols].to_numpy())).sum())
    n_runs = df.groupby("arm")["key"].nunique().to_dict()
    if nonfinite or sum(n_runs.values()) != len(items):
        raise ValueError(f"nonfinite scores {nonfinite}; transcripts with rows {n_runs} of {len(items)} scored")

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S") + ("_test" if limit else "")
    name = f"scores_{probe}_{readout}_{stamp}"
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "data" / f"{name}.parquet"
    with open(path, "xb") as fh:  # append-only: a same-second run fails instead of replacing this table
        df.to_parquet(fh, index=False)
    out_vol = modal.Volume.from_name(cfg["output_volume"], create_if_missing=True)
    with out_vol.batch_upload() as batch:
        batch.put_file(str(path), f"scores/{path.name}")
    summary = {"provenance": provenance(__file__, HERE.parent.parent / "src/probe_shift/offpolicy.py"),
               "probe": probe, "readout": readout, "probe_source": f"{pc['volume']}/{pc['dir']}", "layers": pc["layers"],
               "activations": f"{ACTS}/{run}", "transcripts": len(items), "runs_per_arm": n_runs,
               "rows_per_kind_and_arm": {f"{k}|{a}": int(n) for (k, a), n in df.groupby(["kind", "arm"]).size().items()},
               "score_mean_by_kind_and_arm": {f"{k}|{a}": {c: round(float(v), 4) for c, v in g[layer_cols].mean().items()}
                                              for (k, a), g in df.groupby(["kind", "arm"])},
               "file": str(path.relative_to(HERE)), "volume_file": f"{cfg['output_volume']}/scores/{path.name}",
               "config": cfg | {"modal": {"app": APP, "image": "debian_slim 3.12 + numpy", "batch_size": bs}}}
    with (HERE / "results" / f"{name}.json").open("x") as fh:  # summary (and its provenance) built before this exists
        fh.write(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("runs_per_arm", "rows_per_kind_and_arm", "score_mean_by_kind_and_arm")}, indent=1))
    print(f"-> {path}")
