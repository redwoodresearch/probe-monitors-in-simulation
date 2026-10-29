"""E1 stage 1, post hoc (after the pre-registered textual control failed): the textual control read on the appended
sentence's own 15 tokens instead of the whole report. The pre-registered control averages the probe over the report
(median about 630 tokens), so a 15-token sentence moves that mean by only about 15/630 of its own effect. Uses the
per-token scores already saved; in the same frozen SD as analyze_e1.py; 10,000 run resamples (seed 20261011).
Labelled post hoc: it does not change the pre-registered verdict.

  uv run --no-project --with numpy python experiments/2026-10-11_q2_e1_honest/analyze_e1_control_POSTHOC.py <stage-1 folder> <analysis result json>
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
N_SENT = 15


def main(folder: Path, analysis: Path):
    sd = json.loads(analysis.read_text())["frozen SD (honest reports)"]
    sc = {(r["key"], r["seq"]): r for r in (json.loads(p.read_text()) for p in (folder / "scores").glob("*.json"))}
    keys = sorted(k for k, s in sc if s == "honest+conceal")
    c, n = ([sc[(k, s)] for k in keys] for s in ("honest+conceal", "honest+neutral"))
    if any(a["report_tokens"] != b["report_tokens"] for a, b in zip(c, n)):
        raise ValueError("the two appended copies differ in length")
    sent = np.array([(np.mean(a["per_token"][-N_SENT:]) - np.mean(b["per_token"][-N_SENT:])) / sd for a, b in zip(c, n)])
    whole = np.array([(a["score"] - b["score"]) / sd for a, b in zip(c, n)])
    lens = np.array([a["report_tokens"] for a in c])
    boot = sent[np.random.default_rng(20261011).integers(0, len(sent), (10_000, len(sent)))].mean(1)
    out = {"analysis": analysis.name, "frozen SD": sd, "runs": len(keys), "median tokens read (report + sentence)": float(np.median(lens)),
           "whole-report readout (pre-registered)": float(whole.mean()),
           "appended sentence only (post hoc)": {"mean": float(sent.mean()), "95%": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
                                                 "runs positive": int((sent > 0).sum())},
           "dilution check: sentence mean x 15 / tokens read": float((sent * N_SENT / lens).mean())}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"e1_control_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, separators=(",", ":")) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
