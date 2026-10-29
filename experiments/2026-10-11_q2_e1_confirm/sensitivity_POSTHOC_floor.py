"""E1 confirmation: post hoc floor check on the diluted readout check (the #764 Bugbot point). It was written after the
reading, so it carries no verdict. The check averages each honest report's scored tokens together with the appended
15-token sentence, with no 10-token floor on the honest part. This script counts the honest parts under 10 scored tokens
and recomputes the check over runs whose honest part has at least 10, with analyze_confirm.py's task statistics.
Local CPU, $0.

  uv run --no-project --python 3.12 --with numpy python experiments/2026-10-11_q2_e1_confirm/sensitivity_POSTHOC_floor.py \\
      <confirm stage folder> <confirm index jsonl>
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
sys.path.insert(0, str(HERE))
from analyze_confirm import INDEX_SHA, SD0, SEED, task_stats  # noqa: E402

MASKS = HERE / "results/confirm_masks_2026-10-11_185020.json"
READING = HERE / "results/confirm_reading_2026-10-11_185042.json"
MIN_TOKENS, APPENDED = 10, 15


def main(folder: Path, index: Path):
    if hashlib.sha256(index.read_bytes()).hexdigest() != INDEX_SHA:
        raise SystemExit("confirm index differs")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    for f in (Path(__file__).resolve(), HERE / "analyze_confirm.py", MASKS, READING):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    mk, reading = json.loads(MASKS.read_text()), json.loads(READING.read_text())
    if mk["staged.json sha256"] != hashlib.sha256((folder / "staged.json").read_bytes()).hexdigest():
        raise SystemExit("the stage folder differs from the one the masks were made on")
    masks = mk["masks"]
    task_of = {r["key"]: r["main_task"] for r in map(json.loads, index.open())}
    scores = {(r["key"], r["seq"]): r for r in (json.loads(p.read_text()) for p in (folder / "scores").glob("*.json"))}
    rows = []
    for key in sorted(task_of):
        c, n = masks.get(f"{key}|honest+conceal"), masks.get(f"{key}|honest+neutral")
        sc, sn = scores.get((key, "honest+conceal")), scores.get((key, "honest+neutral"))
        if not (c and n and sc and sn) or sc.get("oom") or sn.get("oom") or len(c["scored"]) < MIN_TOKENS or len(n["scored"]) < MIN_TOKENS:
            continue  # as analyze_confirm.readout: a missing, out-of-memory or short control is skipped
        honest_n = len(c["scored"]) - APPENDED
        assert honest_n == len(n["scored"]) - APPENDED == len(masks[f"{key}|honest"]["scored"])
        diff = (np.asarray(sc["per_token"])[c["scored"]].mean() - np.asarray(sn["per_token"])[n["scored"]].mean()) / SD0
        rows.append((key, task_of[key], honest_n, float(diff)))
    ns = np.array([r[2] for r in rows]); d = np.array([r[3] for r in rows]); task = np.array([r[1] for r in rows])
    keep = ns >= MIN_TOKENS
    out = {"label": "POST HOC (after the reading; no verdict)", "git": head, "runs in the check": len(rows),
           f"honest part under {MIN_TOKENS} scored tokens": int((~keep).sum()), "honest part with 0 scored tokens": int((ns == 0).sum()),
           "honest scored tokens, median": float(np.median(ns)), "appended share of the readout tokens, median": float(np.median(APPENDED / (ns + APPENDED))),
           "check as run (reading file)": reading["readout check (diluted)"],
           "check recomputed, all runs (fresh generator)": task_stats(d, task, np.random.default_rng(SEED)),
           f"check with the floor (honest part >= {MIN_TOKENS}; fresh generator)": task_stats(d[keep], task[keep], np.random.default_rng(SEED)),
           f"check on the short runs only (honest part < {MIN_TOKENS}; fresh generator)": task_stats(d[~keep], task[~keep], np.random.default_rng(SEED))}
    if abs(out["check recomputed, all runs (fresh generator)"]["mean D"] - reading["readout check (diluted)"]["mean D"]) > 1e-12 \
            or len(rows) != reading["readout check (diluted)"]["runs"]:
        raise SystemExit("the recomputed check's mean differs from the reading's")
    f = out[f"check with the floor (honest part >= {MIN_TOKENS}; fresh generator)"]
    out["floored interval above 0"] = bool(f.get("95%", [0])[0] > 0)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"floor_sensitivity_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
