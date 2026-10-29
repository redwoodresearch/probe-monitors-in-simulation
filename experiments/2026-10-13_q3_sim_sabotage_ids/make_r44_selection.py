"""R44 selection manifest (RULE_simtrained.md Addendum 3 = RULE_scaled.md Amendment 2). Labels and IDs only, $0, no text.

Candidates: state-changing calls of the 24 stage-1 training tasks in Aswin's full-population real export (both real runs),
confident labels (sabotage, not_sabotage), with a stored layer-30 call_end reading (a row in the label-free job's rows file).
Each class is sorted by call_id. Draw d = 0..4: seed = int(sha256("match44|" + str(d)).hexdigest()[:16], 16);
rng = numpy.random.default_rng(seed); 44 sabotage = rng.choice(n_sab, 44, replace=False), then 900 not_sabotage =
rng.choice(n_not, 900, replace=False) from the same generator. Call IDs are listed in the order drawn.

  uv run --no-project --python 3.12 --with numpy python make_r44_selection.py --manifests <dir> --sim-votes <csv> \\
      --real-votes <csv> --flag-rows <csv> --out results/r44_selection_manifest.json
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DRAWS, SIZES = 5, {"sabotage": 44, "not_sabotage": 900}


def select(labels: dict) -> dict:
    """The selection from the labels dict built by make_simtrained_rows.build (part-2 rows, i.e. with a stored reading)."""
    pools = {c: sorted(cid for cid, v in labels.items() if v["world"] in ("real", "run2") and v["group"] == "stage1_train"
                       and v["state_changing"] == 1 and v["label"] == c) for c in SIZES}
    draws = []
    for d in range(DRAWS):
        seed = int(hashlib.sha256(f"match44|{d}".encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        picked = {c: [pools[c][i] for i in rng.choice(len(pools[c]), SIZES[c], replace=False)] for c in SIZES}  # sabotage first
        draws.append({"draw": d, "seed": seed, "sabotage": picked["sabotage"], "not_sabotage": picked["not_sabotage"]})
    return {"candidate_counts": {c: len(p) for c, p in pools.items()},
            "candidate_counts_by_world": {c: {w: sum(labels[cid]["world"] == w for cid in p) for w in ("real", "run2")} for c, p in pools.items()},
            "draws": draws}


def main(manifests: str, sim_votes: str, real_votes: str, flag_rows: str, out: str):
    import sys

    sys.path.insert(0, str(HERE))
    import make_simtrained_rows as M

    spec = M.build(manifests, sim_votes, real_votes, flag_rows)
    sel = select(spec["labels"])
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    res = {"rule": "RULE_simtrained.md Addendum 3 = RULE_scaled.md Amendment 2 (aswin/q3-sabotage-scaleup @ 34b0254d)",
           "selection": "each class sorted by call_id; seed = int(sha256('match44|' + str(d)).hexdigest()[:16], 16), d = 0..4; "
           "numpy default_rng(seed).choice(n, size, replace=False), 44 sabotage then 900 not_sabotage from the same generator; call_ids in draw order",
           "git": git, "inputs_sha256": spec["summary"]["inputs_sha256"], "manifests_sha256": M.MANIFEST_PINS, **sel}
    p = Path(out)
    with p.open("x") as fh:
        fh.write(json.dumps(res) + "\n")  # one line: 4,720 call_ids
    print(json.dumps({k: res[k] for k in ("candidate_counts", "candidate_counts_by_world")}),
          [(d["draw"], d["seed"], len(d["sabotage"]), len(d["not_sabotage"])) for d in res["draws"]], "\n->", p, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("--manifests", "--sim-votes", "--real-votes", "--flag-rows", "--out"):
        ap.add_argument(a, required=True)
    main(**vars(ap.parse_args()))
