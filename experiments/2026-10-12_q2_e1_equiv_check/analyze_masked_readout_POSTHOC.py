"""POST HOC (Codex on #923; after the pre-set reading, no verdict): the masked readout E1's D uses (the frozen masks'
"scored" tokens, at least 10), cached vs one pass, on every rescored sequence that has one, in E1's frozen SD units
(0.5396148830137296). Binds its inputs: the archived stage folder's staged.json and scores/ by sha256, the masks result
and the one-pass file. Local, $0.

  uv run --no-project --with numpy python experiments/2026-10-12_q2_e1_equiv_check/analyze_masked_readout_POSTHOC.py \\
      <one_pass json> <archived stage folder> <confirm_masks_2026-10-11_185020.json>
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
SD0, MIN_TOKENS = 0.5396148830137296, 10
MASKS_SHA, STAGED_SHA = "b3db91f08e361a49d8c1f559edf878c5d26eeb81fdfabbb564865f42e1b6974b", "2fa0706be1e8664f"


def main(one_p: Path, folder: Path, masks_p: Path):
    me = Path(__file__).resolve()
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode \
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode:
        raise SystemExit("the script differs from HEAD: commit it before running")
    if hashlib.sha256(masks_p.read_bytes()).hexdigest() != MASKS_SHA:
        raise SystemExit("not the confirmation's masks result")
    if hashlib.sha256((folder / "staged.json").read_bytes()).hexdigest()[:16] != STAGED_SHA:
        raise SystemExit("the stage folder isn't the one the masks were made on")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    files = sorted((folder / "scores").glob("*.json"))
    scores_sha = hashlib.sha256(b"".join(f.name.encode() + f.read_bytes() for f in files)).hexdigest()
    masks = json.loads(masks_p.read_text())["masks"]
    stored = {(r["key"], r["seq"]): r for r in (json.loads(f.read_text()) for f in files)}
    one = json.loads(one_p.read_text())
    if one["staged.json sha256 prefix"] != STAGED_SHA:
        raise SystemExit("the one-pass job read another stage folder")
    rows = []
    for r in one["rows"]:
        if r.get("oom"):
            continue
        c_ = stored[(r["key"], r["seq"])]
        if r["ids_sha256"] != c_["ids_sha256"] or len(r["one_pass_per_token"]) != len(c_["per_token"]):
            raise SystemExit(f"{r['key']} {r['seq']}: the one-pass sequence isn't the stored one")
        idx = masks.get(f"{r['key']}|{r['seq']}", {}).get("scored", [])
        if len(idx) < MIN_TOKENS:
            continue
        c = float(np.asarray(stored[(r["key"], r["seq"])]["per_token"])[idx].mean()); o = float(np.asarray(r["one_pass_per_token"])[idx].mean())
        rows.append([r["key"], r["seq"], len(idx), round(abs(c - o) / SD0, 6)])
    d = [x[3] for x in rows]
    out = {"label": "POST HOC (Codex on #923; no verdict)", "git": head,
           "inputs": {"one-pass file": hashlib.sha256(one_p.read_bytes()).hexdigest(), "staged.json sha256 prefix": STAGED_SHA,
                      "scores/ (names + bytes, sorted) sha256": scores_sha, "score files": len(files), "masks": MASKS_SHA},
           "sequences with a masked readout (>= 10 scored tokens)": len(rows),
           "max |cached - one pass| masked readout, in E1's SD": max(d) if d else None, "median": float(np.median(d)) if d else None,
           "rows [key, seq, scored tokens, |difference| in SD]": rows}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"masked_readout_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1).replace("\n   ", " ") + "\n")
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("rows")}, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
