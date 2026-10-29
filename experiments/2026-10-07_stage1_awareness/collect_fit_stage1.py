"""Assemble a Stage 1 fit result from the per-layer files run_fit_stage1.py commits to the Volume, for when the local
client lost its connection before writing results/fit_stage1_<stamp>_<parts>.json (the detached job keeps running and
saves each layer as it finishes). Also rebuilds the Needham drop table (by source, label and reason) from the
extraction's index and failures, as fit_all would have returned it. Reads small JSON only; no activations.

  python collect_fit_stage1.py --parts main_vuln --app <app id> --sets data/sets_<stamp>.jsonl --needham-items <items.jsonl>
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
REPO = HERE.parent.parent
VOL = "probe-shift_aswin-surya_sub-stage1_acts_kill-2026-11-06"
RUN = "2026-10-07_stage1_qwen35-27b_v1"


def vget(remote: str, local: Path):
    subprocess.run(["modal", "volume", "get", "--force", VOL, f"{RUN}/{remote}", str(local)], check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="main_vuln")
    ap.add_argument("--app", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--needham-items", required=True, help="the Needham inputs/items.jsonl (uid, source, label)")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        expected_layers = range(64) if "main" in args.parts else (*range(0, 64, 4), 63)
        layers = []
        for l in expected_layers:  # one file at a time: a directory get can leave an empty folder
            try:
                vget(f"fit_layers_{args.parts}/L{l}.json", t / f"L{l}.json")
                layers.append(json.loads((t / f"L{l}.json").read_text()))
            except subprocess.CalledProcessError:
                pass
        for name in ("index.jsonl", "failures.jsonl"):
            vget(f"needham/{name}", t / name)
        meta = {json.loads(x)["uid"]: json.loads(x) for x in Path(args.needham_items).read_text().splitlines()}
        fails = [json.loads(x) for x in (t / "failures.jsonl").read_text().splitlines()]
        read = {json.loads(x)["item"] for x in (t / "index.jsonl").read_text().splitlines()}
    expected = 64 if "main" in args.parts else 17
    if len(layers) != expected:
        raise SystemExit(f"{len(layers)} layer files on the Volume, expected {expected}: the fit has not finished")
    drops = Counter((meta[f["item"]]["source"], int(meta[f["item"]]["label"]), f["error"].split(":")[0][:40]) for f in fails)
    kept = Counter((meta[u]["source"], int(meta[u]["label"])) for u in read)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"stamp": stamp, "assembled_from_volume": f"{VOL}/{RUN}/fit_layers_{args.parts}", "app": args.app, "git_at_assembly": head,
           "parts": args.parts, "sets": {"file": args.sets, "sha256": hashlib.sha256(Path(args.sets).read_bytes()).hexdigest()},
           "layers": layers,
           "needham_drops": {"dropped": [{"source": a, "label": b, "reason": c, "n": n} for (a, b, c), n in sorted(drops.items())],
                             "read": [{"source": a, "label": b, "n": n} for (a, b), n in sorted(kept.items())]}}
    path = HERE / "results" / f"fit_stage1_{stamp}_{args.parts}.json"
    with path.open("x") as f:
        f.write(json.dumps(rec))
    print(f"{len(layers)} layers -> {path}")


if __name__ == "__main__":
    main()
