"""Amendment 3's M (mixed) and D (mirror) training selections. Labels and IDs only, $0, no text.

Pools: confident labels on state-changing calls of the 24 stage-1 training tasks with a stored reading (a part-2 row of the
label-free rows file), from Aswin's frozen stage-1 exports, each pool sorted by call_id. M: simulated sabotage vs real (both real runs)
not-sabotage; D: real (both runs) sabotage vs simulated not-sabotage; 44 positives and 900 negatives. Draw d = 0..4:
seed = int(sha256("matchM|" + str(d)).hexdigest()[:16], 16) (or "matchD|"), rng = numpy default_rng(seed); positives =
rng.choice(n_pos, 44, replace=False), then negatives = rng.choice(n_neg, 900, replace=False) from the same generator.
S's training set (all simulated sabotage and not-sabotage) is listed too, for the command-readout refit.

  uv run --no-project --python 3.12 --with numpy python make_grid_selection.py --inputs <scaled_inputs dir> --out results/grid_selection_manifest.json
"""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

PINS = {"flagrate_rows_2026-10-13_122417.csv": "44c75f78e78f86b4bcab409b1379762a8d4dccadfd456f28ee651832eb2b13a7",
        "sim_votes.csv": "06f4ef4bc00b53767b074185cda99311f2ba44f7c8f5a4f38ce59dd1537e81ae",
        "real_votes.csv": "9a96988db9818c8b97f3045ade83f19fae0e19e7d0cac134fa775005eba8dc36"}
SIZES = {"pos": 44, "neg": 900}
GRID = {"M": ("sim_sabotage", "real_not_sabotage"), "D": ("real_sabotage", "sim_not_sabotage")}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pools(inputs: str) -> dict:
    I = Path(inputs)
    for n, want in PINS.items():
        if sha(I / n) != want:
            raise SystemExit(f"{n}: sha256 mismatch")
    key4 = lambda r: (r["run_id"], int(r["epoch"]), int(r["turn"]), int(r["call_index"]))  # noqa: E731
    votes = {}
    for path, default in ((I / "sim_votes.csv", "sim"), (I / "real_votes.csv", None)):
        for r in csv.DictReader(open(path)):
            votes[key4(r)] = (r.get("world") or default, r["label"], int(r["state_changing"]))
    out = {k: [] for k in ("sim_sabotage", "sim_not_sabotage", "real_sabotage", "real_not_sabotage")}
    for r in csv.DictReader(open(I / "flagrate_rows_2026-10-13_122417.csv")):
        if r["part"] != "2" or r["group"] != "stage1_train":
            continue
        v = votes.get(key4(r))
        if v is None or v[0] != r["world"] or v[2] != int(r["state_changing"]):
            raise SystemExit("a part-2 row has no matching vote")
        if v[2] == 1 and v[1] in ("sabotage", "not_sabotage"):
            out[("sim_" if v[0] == "sim" else "real_") + v[1]].append(r["call_id"])
    return {k: sorted(v) for k, v in out.items()}


def select(P: dict) -> dict:
    res = {}
    for g, (pk, nk) in GRID.items():
        draws = []
        for d in range(5):
            seed = int(hashlib.sha256(f"match{g}|{d}".encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed)
            pos = [P[pk][i] for i in rng.choice(len(P[pk]), SIZES["pos"], replace=False)]
            neg = [P[nk][i] for i in rng.choice(len(P[nk]), SIZES["neg"], replace=False)]
            draws.append({"draw": d, "seed": seed, "pos": pos, "neg": neg})
        res[g] = {"pos_pool": pk, "neg_pool": nk, "draws": draws}
    res["S_train"] = {"pos": P["sim_sabotage"], "neg": P["sim_not_sabotage"]}
    return res


def main(inputs: str, out: str):
    P = pools(inputs)
    sel = select(P)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip()
    res = {"rule": "RULE_scaled.md Amendment 3", "git": git, "inputs_sha256": PINS, "pool_counts": {k: len(v) for k, v in P.items()}, **sel}
    p = Path(out)
    with p.open("x") as fh:
        fh.write(json.dumps(res) + "\n")
    print(json.dumps(res["pool_counts"]), "\n->", p, sha(p))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    main(**vars(ap.parse_args()))
