"""E1 sentence readout: post hoc floor check on the diluted readout check (Bugbot on #764, PLAN.md:92). It was written
after the result was read, so it carries no verdict. The check averaged each honest report's scored tokens together
with the appended 15-token sentence, with no 10-token floor on the honest part. An honest report with few scored tokens
therefore reduces to nearly the appended sentence alone. This script counts those reports and recomputes the check
over honest reports with at least 10 scored tokens. Local CPU, $0.

  uv run --no-project --python 3.12 --with numpy python experiments/2026-10-11_q2_e1_sentences/sensitivity_POSTHOC_floor.py \\
      <stage-1 folder> <archive tar.gz>
"""
import hashlib
import json
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "2026-10-11_q2_e1_honest"))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from analyze_e1 import stats  # noqa: E402

ARCHIVE_SHA = "cbb431246087b3ecbeea373d4f2e24d80b1d3066a36bb99702aeff16c6afd086"
MASKS, RESULT = HERE / "results/sentence_masks_2026-10-11_155002.json", HERE / "results/e1_sentences_2026-10-11_155202.json"
SEED, MIN_TOKENS, APPENDED = 20261011, 10, 15
CHECK = "readout check (diluted: honest scored tokens + the appended 15)"


def main(folder: Path, archive: Path):
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    for f in (Path(__file__).resolve(), HERE.parent / "2026-10-11_q2_e1_honest/analyze_e1.py"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise SystemExit("archive sha256 differs from s3_manifest_v2.json")
    with tarfile.open(archive) as t:
        members = {m.name.split("/", 1)[1]: t.extractfile(m).read() for m in t.getmembers() if m.isfile() and "/scores/" in m.name}
    files = sorted((folder / "scores").glob("*.json"))
    if {f"scores/{p.name}" for p in files} != set(members) or any(p.read_bytes() != members[f"scores/{p.name}"] for p in files):
        raise SystemExit("scores differ from the archive")
    masks, result = json.loads(MASKS.read_text())["masks"], json.loads(RESULT.read_text())
    per_token = {(r["key"], r["seq"]): np.array(r["per_token"], float) for r in (json.loads(p.read_text()) for p in files)}
    sd = result["honest SD"]
    rows = []
    for k in sorted({k.split("|")[0] for k in masks if k.endswith("|honest+conceal")}):
        if f"{k}|honest+neutral" not in masks:
            continue
        c, n = masks[f"{k}|honest+conceal"]["scored"], masks[f"{k}|honest+neutral"]["scored"]
        honest_n = len(c) - APPENDED
        assert honest_n == len(n) - APPENDED == len(masks[f"{k}|honest"]["scored"])
        diff = (per_token[(k, "honest+conceal")][c].mean() - per_token[(k, "honest+neutral")][n].mean()) / sd
        rows.append((honest_n, float(diff)))
    ns = np.array([r[0] for r in rows]); d = np.array([r[1] for r in rows])
    floored = d[ns >= MIN_TOKENS]
    out = {"label": "POST HOC (after the result was read; no verdict)", "git": head, "honest SD (as run)": sd,
           "runs in the check": len(rows), f"honest part under {MIN_TOKENS} scored tokens": int((ns < MIN_TOKENS).sum()),
           "honest part with 0 scored tokens": int((ns == 0).sum()), "honest scored tokens, median": float(np.median(ns)),
           "appended share of the readout tokens, median": float(np.median(APPENDED / (ns + APPENDED))),
           "check as run (result file)": result[CHECK],
           "check recomputed, all runs (fresh generator)": stats(d, np.random.default_rng(SEED)),
           f"check with the floor (honest part >= {MIN_TOKENS}; fresh generator)": stats(floored, np.random.default_rng(SEED)) if len(floored) >= 2 else {"runs": int(len(floored))}}
    f = out[f"check with the floor (honest part >= {MIN_TOKENS}; fresh generator)"]
    out["floored interval above 0"] = bool(f.get("95%", [0])[0] > 0)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"floor_sensitivity_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
