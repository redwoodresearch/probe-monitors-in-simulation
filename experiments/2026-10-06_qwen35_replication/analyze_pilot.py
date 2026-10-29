"""Pilot check of the Qwen3.5-27B extraction (2026-10-06): does right-padding change the readouts, and what will the
full run cost?

Reads the two `--test` runs of run_extract_qwen35.py (padded to 4,096 tokens, and unpadded) from this track's Volume, on
Modal CPU:
- positions: the same token indices in both runs;
- padding: per transcript, the largest absolute difference between the two runs' rows over all 64 layers, next to the
  difference between two passes of the same padded input (the kernels' own run-to-run noise), and the lowest row cosine;
- timing: per transcript, the first (cold) and second (warm) forward pass, and peak GPU memory.
Then, locally, it fits warm seconds = a*T + b*T^2 (T padded tokens) and projects GPU-hours and cost for every staged
transcript of both texts, from the main session's Qwen3.5 token counts.

  uv run modal run experiments/2026-10-06_qwen35_replication/analyze_pilot.py
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-qwen35_pilot_kill-2026-11-06"
OUTPUT = "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06"
PADDED = "2026-10-06_la_segment-end_qwen35-27b_alllayers_v1_test"
UNPADDED = "2026-10-06_la_segment-end_qwen35-27b_alllayers_nopad_test"
TOKEN_COUNTS = HERE.parent / "2026-10-06_qwen35_reader/results/qwen35_token_counts_2026-10-06_164545.jsonl"
H200_PER_HOUR = 4.54  # Modal's list price, 2026-10
SHARDS = 40

app = modal.App(APP)
vol = modal.Volume.from_name(OUTPUT, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, volumes={"/out": vol}, memory=32768, timeout=1800)
def compare() -> list[dict]:
    import numpy as np

    def load(run):
        d = Path("/out") / run
        metas = {(m["arm"], m["key"]): m for m in (json.loads(p.read_text()) for p in d.glob("meta/*/*.json"))}
        layers = json.loads(next((d / "chunks").glob("*.json")).read_text())["layers"]
        chunks = {c: [np.load(d / "by_layer" / f"L{l}" / f"{c}.npy") for l in layers] for c in {m["chunk"] for m in metas.values()}}
        rows = {k: np.stack([x[m["row_start"]: m["row_start"] + m["n_rows"]] for x in chunks[m["chunk"]]]) for k, m in metas.items()}
        return metas, rows

    pm, prow = load(PADDED)
    um, urow = load(UNPADDED)
    if set(pm) != set(um):
        raise ValueError(f"the runs read different transcripts: {len(set(pm) ^ set(um))} differ")
    out = []
    for k, m in sorted(pm.items(), key=lambda kv: -kv[1]["n_tokens"]):
        if [p["token"] for p in m["positions"]] != [p["token"] for p in um[k]["positions"]]:
            raise ValueError(f"{k}: positions differ between the runs")
        a, b = prow[k].astype(np.float32), urow[k].astype(np.float32)  # [layer, row, hidden]
        cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
        out.append({"arm": m["arm"], "key": k[1], "n_tokens": m["n_tokens"], "n_rows": m["n_rows"],
                    "padded_tokens": m["padded_tokens"], "pad_max_abs_diff": float(np.abs(a - b).max()),
                    "pad_max_abs_value": float(np.abs(a).max()), "pad_min_cosine": float(cos.min()),
                    "repeat_max_abs_diff": m["repeat_max_abs_diff"], "nonfinite": int((~np.isfinite(a)).sum()),
                    **{f"padded_{x}": m[x] for x in ("forward_s", "forward_s_repeat", "peak_gib")},
                    **{f"unpadded_{x}": um[k][x] for x in ("forward_s", "forward_s_repeat", "peak_gib")}})
    return out


def project(rows: list[dict], load_s: float) -> dict:
    """GPU-hours per text: warm seconds from a*T + b*T^2 on padded lengths, plus one cold overhead per new 4,096-token
    bucket in each shard, plus model load per shard."""
    import numpy as np

    T = np.array([r["padded_tokens"] for r in rows], float)
    t = np.array([r["padded_forward_s_repeat"] for r in rows], float)
    a, b = np.linalg.lstsq(np.stack([T, T**2], 1), t, rcond=None)[0]
    overhead = float(np.median([r["padded_forward_s"] - r["padded_forward_s_repeat"] for r in rows[1:]]))  # first: compile
    counts = [json.loads(l) for l in TOKEN_COUNTS.read_text().splitlines()]
    out = {"fit": {"a_s_per_token": a, "b_s_per_token2": b, "cold_overhead_s_per_bucket": overhead, "load_s": load_s}}
    for text in ("original", "scrub5"):
        toks = sorted((c["tokens"] for c in counts if c["text"] == text), reverse=True)
        shard_s = []
        for s in range(SHARDS):
            mine = np.array([-(-x // 4096) * 4096 for x in toks[s::SHARDS]], float)
            shard_s.append(float((a * mine + b * mine**2).sum() + overhead * len(set(mine)) + load_s))
        gpu_h = sum(shard_s) / 3600
        out[text] = {"transcripts": len(toks), "tokens": sum(toks), "gpu_hours": round(gpu_h, 2),
                     "cost_usd": round(gpu_h * H200_PER_HOUR), "longest_shard_hours": round(max(shard_s) / 3600, 2)}
    return out


@app.local_entrypoint()
def main():
    rows = compare.remote()
    stats = {}
    for run in (PADDED, UNPADDED):
        stats[run] = json.loads(b"".join(vol.read_file(f"{run}/stats/shard_0.json")))
    proj = project(rows, stats[PADDED]["load_seconds"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"runs": {"padded": PADDED, "unpadded": UNPADDED}, "transcripts": rows, "shard_stats": stats, "projection": proj,
           "price": {"h200_usd_per_hour": H200_PER_HOUR}}
    path = HERE / "results" / f"pilot_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    for r in rows:
        print(f"{r['n_tokens']:>7} tok {r['n_rows']:>4} rows | pad diff {r['pad_max_abs_diff']:.4g} (repeat {r['repeat_max_abs_diff']:.4g},"
              f" max |x| {r['pad_max_abs_value']:.4g}) min cos {r['pad_min_cosine']:.6f} | padded {r['padded_forward_s']:.1f}/"
              f"{r['padded_forward_s_repeat']:.1f} s, unpadded {r['unpadded_forward_s']:.1f}/{r['unpadded_forward_s_repeat']:.1f} s,"
              f" peak {r['padded_peak_gib']:.0f} GiB | nonfinite {r['nonfinite']}")
    print(json.dumps(proj, indent=1))
    print(f"wrote {path}")
