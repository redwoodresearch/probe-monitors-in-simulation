"""Which positions the Qwen3-32B reader read, for the matched rows of the Qwen3.5-27B analyses (2026-10-06).

Qwen3-32B read each transcript's first 32,768 tokens; Qwen3.5-27B reads it whole. Positions are keyed by (kind, step,
action), which both readers share because they read the same staged messages. For each Qwen3.5 run, this writes
`<run>/qwen3_cut/<qwen3 run>.json` on this track's Volume: {"read": {"<arm>|<key>": [[kind, step, action], ...]}, "truncated":
{"<arm>|<key>": bool}}, the positions the Qwen3-32B run kept (restricted to those the Qwen3.5 run also has a token for)
and whether Qwen3-32B cut the transcript at 32k (for the protocol's exclusions table). It fails loudly if a transcript is
missing from either run, and reports positions Qwen3-32B read that Qwen3.5 has no token for (outputs that are only
whitespace, which Qwen3.5's template trims to nothing).

  uv run modal run experiments/2026-10-06_qwen35_replication/run_qwen3_cut_index.py --texts original,scrub5
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-qwen35_cut-index_kill-2026-11-06"
QWEN3 = "probe-shift_aswin-surya_sub-la-activations"  # the main session's Volume; read only
OUTPUT = "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06"
PAIRS = {  # text -> (Qwen3.5 run on this track's Volume, the Qwen3-32B run of the same text)
    "original": ("2026-10-06_la_segment-end_qwen35-27b_alllayers_v1", "2026-10-06_la_segment-end_qwen3-32b_alllayers_v1"),
    "scrub5": ("2026-10-06_la_segment-end-scrub5_qwen35-27b_alllayers_v1", "2026-10-06_la_segment-end-scrub5_qwen3-32b_alllayers_v1"),
    "scrub7": ("2026-10-06_la_segment-end-scrub7_qwen35-27b_alllayers_v1", "2026-10-06_la_segment-end-scrub7_qwen3-32b_alllayers_v1"),
} | {f"scrub{l}": (f"2026-10-06_la_segment-end-scrub{l}_qwen35-27b_alllayers_v1", f"2026-10-06_la_segment-end-scrub{l}_qwen3-32b_alllayers_v1")
     for l in (1, 2, 3, 4, 6)} | {  # five arms (with simulated honest), every text
    f"v4{t}": (f"2026-10-06_la_segment-end-v4{t}_qwen35-27b_alllayers_v1", f"2026-10-06_la_segment-end-v4{t}_qwen3-32b_alllayers_v1")
    for t in ["", *(f"-scrub{l}" for l in range(1, 8))]}

app = modal.App(APP)
qwen3_vol = modal.Volume.from_name(QWEN3, version=2).read_only()
out_vol = modal.Volume.from_name(OUTPUT, version=2)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, volumes={"/qwen3": qwen3_vol, "/out": out_vol}, cpu=16, memory=32768, timeout=3600)
def index(run: str, qwen3_run: str) -> dict:
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    def metas(d: Path) -> dict:
        paths = sorted(d.glob("meta/*/*.json"))
        with ThreadPoolExecutor(32) as pool:
            ms = list(pool.map(lambda p: json.loads(p.read_text()), paths))
        return {(m["arm"], m["key"]): m for m in ms}

    new, old = metas(Path("/out") / run), metas(Path("/qwen3") / qwen3_run)
    if not new or not old:
        raise ValueError(f"no transcript metadata: {len(new)} in {run}, {len(old)} in {qwen3_run}")
    if set(new) != set(old):
        raise ValueError(f"{run} and {qwen3_run} read different transcripts: {len(set(new) - set(old))} only in the first, "
                         f"{len(set(old) - set(new))} only in the second")
    out, truncated, counts = {}, {}, Counter()
    for k in sorted(new):
        have = {(p["kind"], p["step"], p["action"]) for p in new[k]["positions"] if p["kept"]}
        read = {(p["kind"], p["step"], p["action"]) for p in old[k]["positions"] if p["kept"]}
        every = {(p["kind"], p["step"], p["action"]) for p in new[k]["positions"]}
        if not read <= every:
            raise ValueError(f"{k}: {len(read - every)} Qwen3-32B positions are not positions of the Qwen3.5 run")
        out[f"{k[0]}|{k[1]}"] = sorted([list(x) for x in read & have])
        truncated[f"{k[0]}|{k[1]}"] = bool(old[k]["truncated"])
        counts["qwen35_positions"] += len(have)
        counts["qwen3_read"] += len(read)
        counts["matched"] += len(read & have)
        counts["qwen3_read_but_no_qwen35_token"] += len(read - have)
        counts["transcripts_cut_by_qwen3"] += int(bool(old[k]["truncated"]))
        for kind, *_ in read - have:
            counts[f"no_qwen35_token_{kind}"] += 1
    d = Path("/out") / run / "qwen3_cut"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{qwen3_run}.json").write_text(json.dumps({"read": out, "truncated": truncated}))
    out_vol.commit()
    return {"run": run, "qwen3_run": qwen3_run, "transcripts": len(new), "file": f"{run}/qwen3_cut/{qwen3_run}.json", **counts}


@app.local_entrypoint()
def main(texts: str):
    reports = list(index.starmap([PAIRS[t] for t in texts.split(",")]))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"qwen3_cut_index_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"volume": OUTPUT, "qwen3_volume": QWEN3, "reports": reports}, indent=1))
    print(json.dumps(reports, indent=1))
    print(f"wrote {path}")
