"""Design check for the layer-0 control (PLAN.md), $0, counts only: which planted checks can a raw-embedding probe see at
all? At layer E (token embeddings) a tool output's feature is a function of its readout token alone, and a run's score
is the mean of its outputs' linear scores, so it depends only on the run's normalized histogram of readout tokens. A
planted copy whose histogram equals its unplanted twin's (same cell) gets the identical E score under any fit:
"invariant", the negative-control set. A differing histogram is only "potentially distinguishable". Position-aligned
differences (outputs matched by (step, action)) are counted too. Reads only metadata (`meta/<arm>/*.json`) of step A's
level-1 fixed-sim dump; token strings are compared, never printed or saved.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' \\
      python experiments/2026-10-11_claimB_layer0_control/check_plant_readout_tokens.py
"""

import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
DUMP = "2026-10-10_la_fixed-sim-v2-scrub1_qwen3-32b_alllayers_v1"


def main():
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)

    def load(arm):
        paths = sorted(e.path for e in vol.listdir(f"{DUMP}/meta/{arm}"))
        with ThreadPoolExecutor(32) as ex:
            ms = list(ex.map(lambda p: json.loads(b"".join(vol.read_file(p))), paths))
        out = {}
        for m in ms:
            assert m["cell"] not in out, f"two {arm} runs in cell {m['cell']}"
            out[m["cell"]] = {(p["step"], p["action"]): p["token_text"] for p in m["positions"] if p["kind"] == "output_end" and p.get("kept")}
        return out

    base = load("sim-attack-fixed")
    res = {"dump": DUMP, "unplanted_runs": len(base)}
    for arm in ("sim-attack-fixed-plant1", "sim-attack-fixed-plant2"):
        pl = load(arm)
        missing = [c for c in pl if c not in base]
        assert not missing, f"{len(missing)} {arm} cells without an unplanted twin"
        r = {"pairs": len(pl), "pairs_any_readout_token_differs": 0, "pairs_kept_output_set_differs": 0,
             "pairs_histogram_invariant": 0, "pairs_potentially_distinguishable": 0, "outputs_compared": 0,
             "outputs_readout_token_differs": 0}
        for cell, toks in pl.items():
            b = base[cell]
            if not b or not toks:  # no kept readout: the run score is undefined, so it is neither invariant nor distinguishable
                r["pairs_empty_readout"] = r.get("pairs_empty_readout", 0) + 1
                continue
            common = set(b) & set(toks)
            d = sum(b[k] != toks[k] for k in common)
            r["outputs_compared"] += len(common)
            r["outputs_readout_token_differs"] += d
            r["pairs_any_readout_token_differs"] += d > 0
            r["pairs_kept_output_set_differs"] += set(b) != set(toks)
            hb, ht = Counter(b.values()), Counter(toks.values())
            same = all(hb[x] * sum(ht.values()) == ht[x] * sum(hb.values()) for x in set(hb) | set(ht))  # equal proportions
            r["pairs_histogram_invariant"] += same
            r["pairs_potentially_distinguishable"] += not same
        res[arm] = r
    res |= {"label": "design check: planted pairs a token-embedding probe can see (counts only)",
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"plant_readout_tokens_{stamp}.json"
    dst.parent.mkdir(exist_ok=True)
    with open(dst, "x") as fh:
        fh.write(json.dumps(res) + "\n")
    print(json.dumps(res, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
