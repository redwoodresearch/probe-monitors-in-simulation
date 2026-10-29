"""E1 stage 1, post hoc: the within-run label-swap null for analyze_e1_control_POSTHOC.py's sentence-only contrast
(Codex on #723). The primary analysis carries this null and the post hoc contrast did not. This recomputes the same
per-run differences (conceal minus neutral on the appended sentence's own 15 tokens, in the same frozen SD) and asserts
that they reproduce the committed post hoc mean and interval. It then applies analyze_e1.stats: a 10,000-draw run
bootstrap plus the within-run label swap (one sign flip per run, 10,000 draws, two-sided), with seed 20261011. Labelled
post hoc; it changes no verdict. Local, $0.

  uv run --no-project --with numpy python experiments/2026-10-11_q2_e1_honest/analyze_e1_control_null_POSTHOC.py <stage-1 folder>
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "src"))
from analyze_e1 import DRAWS, stats  # noqa: E402

POSTHOC = HERE / "results/e1_control_POSTHOC_2026-10-11_134120.json"
ANALYSIS = HERE / "results/e1_stage1_analysis_2026-10-11_134021.json"  # the post hoc run's frozen-SD source
N_SENT = 15


def main(folder: Path):
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    me = Path(__file__).resolve()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
    ph = json.loads(POSTHOC.read_text()); sd = json.loads(ANALYSIS.read_text())["frozen SD (honest reports)"]
    if abs(sd - ph["frozen SD"]) > 1e-15:
        raise SystemExit("frozen SD differs from the post hoc run's")
    sc = {(r["key"], r["seq"]): r for r in (json.loads(p.read_text()) for p in (folder / "scores").glob("*.json"))}
    keys = sorted(k for k, s in sc if s == "honest+conceal")
    c, n = ([sc[(k, s)] for k in keys] for s in ("honest+conceal", "honest+neutral"))
    sent = np.array([(np.mean(a["per_token"][-N_SENT:]) - np.mean(b["per_token"][-N_SENT:])) / sd for a, b in zip(c, n)])
    boot = sent[np.random.default_rng(20261011).integers(0, len(sent), (10_000, len(sent)))].mean(1)
    want = ph["appended sentence only (post hoc)"]
    if len(sent) != ph["runs"] or abs(sent.mean() - want["mean"]) > 1e-12 or any(abs(a - b) > 1e-12 for a, b in zip(np.percentile(boot, [2.5, 97.5]), want["95%"])):
        raise SystemExit("the per-run differences don't reproduce the committed post hoc result")
    st = stats(sent, np.random.default_rng(20261011))
    out = {"label": "POST HOC (Codex on #723; no verdict)", "git": head, "post hoc result": POSTHOC.name,
           "post hoc result sha256": hashlib.sha256(POSTHOC.read_bytes()).hexdigest(), "reproduces the post hoc mean and interval": True,
           "runs": len(sent), "runs positive": int((sent > 0).sum()), "analyze_e1.stats (bootstrap + within-run label swap)": st,
           "smallest possible swap p at these draws": 1 / (1 + DRAWS)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"e1_control_null_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
