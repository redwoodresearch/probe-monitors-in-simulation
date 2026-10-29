"""Debug for #203 (Modal CPU, pennies; ran 2026-10-08 about 15:30 PT as ap-QJI7wNr8VCPqTnhrQWP5cQ): why the content null's
real-label catch differs from the headline 2x2's by up to 0.0004.

Returns the null's real-label per-row scores (each fold's layer picked as run_content_null.py picks it, from the
all-layer run's own fits, view '') to null_scores.json; the comparison against the headline table
(.run_cache/impactful_2x2/<reader>__call_end-content__attack_only.jsonl.gz, from the three-way run's attack_only fits)
was done locally with pandas: same rows and the same per-fold layers, scores differing by up to 0.085 (Qwen3.5) and
0.2 (Qwen3-32B) logits in some folds, moving 1-2 positives across the 1% threshold. The returned file is on S3 at
experiments/2026-10-08_impactful_call_2x2/debug/null_scores_2026-10-08.json (sha256 4fabf7c6...).

  uv run modal run experiments/2026-10-08_impactful_call_2x2/debug_null_vs_headline.py
"""
import json
from pathlib import Path

import modal

READERS = {"Qwen3.5-27B": ("probe-shift_aswin-surya_sub-qwen35_q2feats_kill-2026-11-06", "2026-10-06_q2_qwen35_matched_eb014307fd1aab31_as_labeled"),
           "Qwen3-32B": ("probe-shift_aswin-surya_sub-q2_probe-features_kill-2026-11-06", "2026-10-06_q2_features_eb014307fd1aab31_as_labeled")}
ARMS = {"real-attack-e1": "real", "sim-attack": "sim"}
app = modal.App("probe-shift_aswin-surya_sub-q2-nulldiff_kill-2026-11-08")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")
vols = {f"/v{i}": modal.Volume.from_name(v, version=2) for i, (v, _) in enumerate(READERS.values())}


@app.function(image=image, volumes=vols, cpu=2, memory=8192, timeout=900, retries=0)
def scores(i: int) -> dict:
    import numpy as np
    reader, (_, run) = list(READERS.items())[i]
    d = Path(f"/v{i}") / run
    manifest = [m for m in json.loads((d / "manifest.json").read_text()) if m["arm"] in ARMS]
    logs = [d / "alllayers" / "fits_index.jsonl"] + sorted((d / "alllayers" / "fits_index").glob("*.jsonl"))
    fits = [json.loads(l) for log in logs if log.exists() for l in log.read_text().splitlines()]
    by = {(f["train_set"], f["layer"], f["seed"]): f for f in fits if f["position"] == "call_end" and f.get("view", "") == ""}
    layers = sorted({k[1] for k in by})
    fold = np.array([m["fold"] for m in manifest])
    out = {}
    for t in ARMS:  # noqa: PLC0206 (kept as it ran)
        score, picks = np.zeros(len(manifest)), {}
        for f in sorted(set(fold)):
            val = {l: next(fo["val_auroc"] for fo in by[(t, l, None)]["folds"] if fo["fold"] == f) for l in layers}
            pick = min(l for l in layers if val[l] >= max(val.values()) - 0.005)
            a = np.load(d / "alllayers" / "fits" / by[(t, pick, None)]["file"])
            score[fold == f] = a["score"][fold == f]
            picks[int(f)] = pick
            dtype = str(a["score"].dtype)
        out[ARMS[t]] = {"picks": picks, "dtype": dtype,
                        "rows": [[m["arm"], m["key"], int(m["step"]), int(m["action"]), bool(m.get("call_end_content")), float(s)] for m, s in zip(manifest, score)]}
    return {"reader": reader, **out}


@app.local_entrypoint()
def main():
    res = list(scores.map(range(2)))
    Path("null_scores.json").write_text(json.dumps(res))
    print({r["reader"]: {t: (r[t]["picks"], r[t]["dtype"]) for t in ("real", "sim")} for r in res})
