"""POST HOC (RULE.md Amendment 2): Go task vs non-Go among REAL honest runs, under the same (c″) probe's held-out scores.

One Modal CPU container computes per-run flags (has_go_reply, n_go_content_tab) for the staged v4 `real-honest` rows on
the activations Volume, with the census's reply rule, joins them with d9's per-run real-honest scores (baked into the
image from a local copy whose sha256 is given at launch and recorded), and writes one result JSON to the Volume.

  TAB_REAL_CSV=<copy> TAB_REAL_SHA=<sha256> PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' \\
      modal run --detach experiments/2026-10-13_claimB_tab_census/run_tab_real_modal.py
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-claimb-tab-census_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
V4 = "2026-10-06_la_segment-end-v4_qwen3-32b_v1/inputs"
OUT = "claimB/tab_census_v1"
SEED, DRAWS = 20261013, 2000

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.1.3")
if modal.is_local():
    image = image.add_local_file(os.environ["TAB_REAL_CSV"], "/root/scores_real.csv", copy=True)


def shard_flags(path: str) -> tuple[dict, dict, str]:
    """Per real-honest run key in one v4 shard: has_go_reply, n_go_replies, n_go_content_tab, batch. Row failures are
    counted, never echoed."""
    import re

    gutter = re.compile(r"^(\s*\d+)\t")
    raw = Path(path).read_bytes()
    flags, errs = {}, {"row_failures": 0, "pairing_mismatches": 0, "duplicate_keys": 0}
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        try:
            r = json.loads(line.decode("utf-8"))
            if r["arm"] != "real-honest":
                continue
            key = str(r["key"])
            batch = str(r.get("file", "")).split("_", 1)[0]
            n_go = n_tab = mis = 0
            pending = []
            for m in r["messages"]:
                if m["role"] == "assistant":
                    pending = [dict(tc["function"]) for tc in (m.get("tool_calls") or [])]
                elif m["role"] == "tool":
                    call = pending.pop(0) if pending else None
                    if call is None or call.get("name") != m.get("tool_name"):
                        mis += 1
                        continue
                    if call.get("name") != "text_editor":
                        continue
                    args = call.get("arguments") or {}
                    if isinstance(args, str):
                        args = json.loads(args)
                    if not str(dict(args).get("path") or "").endswith(".go"):
                        continue
                    text = m["content"] if isinstance(m["content"], str) else ""
                    n_go += 1
                    n_tab += any("\t" in gutter.sub(r"\1", x, count=1) for x in text.split("\n"))
        except Exception:
            errs["row_failures"] += 1
            continue
        errs["pairing_mismatches"] += mis
        if key in flags:
            errs["duplicate_keys"] += 1
        flags[key] = {"has_go_reply": n_go > 0, "n_go_replies": n_go, "n_go_content_tab": n_tab, "batch": batch}
    return flags, errs, hashlib.sha256(raw).hexdigest()


def auroc(pos, neg):
    import numpy as np

    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def split_stats(rows: list[dict]) -> dict:
    import numpy as np

    go = [r for r in rows if r["has_go_reply"]]
    non = [r for r in rows if not r["has_go_reply"]]
    out = {"n_go": len(go), "n_non_go": len(non)}
    for name, grp in (("go", go), ("non_go", non)):
        s = [r["score_mean"] for r in grp]
        out[f"mean_{name}"] = float(np.mean(s)) if s else None
        out[f"median_{name}"] = float(np.median(s)) if s else None
        out[f"caught_share_{name}"] = (sum(r["caught_1pct"] > 0 for r in grp) / len(grp)) if grp else None
        out[f"mean_caught_1pct_{name}"] = float(np.mean([r["caught_1pct"] for r in grp])) if grp else None
    out["auroc"] = auroc([r["score_mean"] for r in go], [r["score_mean"] for r in non])
    tasks = sorted({r["main_task"] for r in rows})
    by = {t: [r for r in rows if r["main_task"] == t] for t in tasks}
    rng, draws = np.random.default_rng(SEED), []
    for _ in range(DRAWS):
        sample = [r for t in rng.choice(len(tasks), len(tasks)) for r in by[tasks[t]]]
        a = auroc([r["score_mean"] for r in sample if r["has_go_reply"]], [r["score_mean"] for r in sample if not r["has_go_reply"]])
        if a is not None:
            draws.append(a)
    out["auroc_ci95"] = [float(x) for x in np.percentile(draws, [2.5, 97.5])] if draws else None
    out["bootstrap_draws_used"] = len(draws)
    return out


@app.function(image=image, volumes={"/vol": acts}, cpu=8, memory=16384, timeout=900, retries=0)
def real(git: str, script_sha256: str, csv_sha256: str) -> dict:
    import csv
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if hashlib.sha256(Path("/root/scores_real.csv").read_bytes()).hexdigest() != csv_sha256:
        raise SystemExit("scores CSV sha256 mismatch: stop")
    shards = sorted(str(p) for p in Path("/vol", V4).glob("shard_*.jsonl"))
    if len(shards) != 64:
        raise SystemExit(f"v4 shards: {len(shards)}")
    with ProcessPoolExecutor(max_workers=8) as ex:
        parts = list(ex.map(shard_flags, shards))
    flags, errs = {}, {"row_failures": 0, "pairing_mismatches": 0, "duplicate_keys": 0}
    for f, e, _ in parts:
        for k in errs:
            errs[k] += e[k]
        errs["duplicate_keys"] += len(set(flags) & set(f))
        flags |= f
    if errs["row_failures"] or errs["duplicate_keys"]:
        raise SystemExit("flag pass row failures or duplicate keys: stop")
    scores, dup = {}, 0
    with open("/root/scores_real.csv") as fh:
        for r in csv.DictReader(fh):
            k = (r["run_key"], int(r["layer"]))
            dup += k in scores
            scores[k] = {"run_key": r["run_key"], "main_task": r["main_task"], "score_mean": float(r["score_mean"]),
                         "caught_1pct": float(r["caught_1pct"])}
    keys = {L: {k for (k, l) in scores if l == L} for L in (48, 63)}
    if dup or keys[48] != keys[63] or not keys[63] <= set(flags):
        raise SystemExit("join rule failed (duplicate scores, layer key sets differ, or scored keys without flags): stop")
    res = {"label": "POST HOC (RULE.md Amendment 2), preliminary until reviewed", "git": git, "script_sha256": script_sha256,
           "scores_csv_sha256": csv_sha256, "flag_errors": errs, "census_real_honest_rows": len(flags),
           "scored_runs": len(keys[63]), "census_rows_without_score": len(set(flags) - keys[63]),
           "flags_scored": {"has_go_reply": sum(flags[k]["has_go_reply"] for k in keys[63]),
                            "go_runs_with_content_tab": sum(flags[k]["has_go_reply"] and flags[k]["n_go_content_tab"] > 0 for k in keys[63])},
           "layers": {}}
    for L in (63, 48):
        rows = [scores[(k, L)] | flags[k] for k in sorted(keys[L])]
        res["layers"][f"L{L}"] = {"all": split_stats(rows),
                                  "by_batch": {b: split_stats([r for r in rows if r["batch"] == b])
                                               for b in sorted({r["batch"] for r in rows})}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = Path("/vol", OUT)
    out.mkdir(parents=True, exist_ok=True)
    name = f"tab_real_POSTHOC_{stamp}.json"
    with open(out / name, "x") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")
    acts.commit()
    return {"path": f"{OUT}/{name}", "scored_runs": res["scored_runs"], "census_rows_without_score": res["census_rows_without_score"],
            "flags_scored": res["flags_scored"], "flag_errors": errs,
            "L63": {k: v for k, v in res["layers"]["L63"]["all"].items()}, "L48": {k: v for k, v in res["layers"]["L48"]["all"].items()}}


@app.local_entrypoint()
def main():
    want = os.environ["TAB_REAL_SHA"]
    if hashlib.sha256(Path(os.environ["TAB_REAL_CSV"]).read_bytes()).hexdigest() != want:
        raise SystemExit("local CSV sha256 mismatch: stop")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(HERE)], capture_output=True, text=True, cwd=HERE).stdout
    if dirty.strip():
        raise SystemExit("commit the script before running it")
    print(json.dumps(real.remote(git, hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), want), indent=1))
