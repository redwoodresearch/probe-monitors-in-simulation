"""Post-run check (Codex): R's full reproduction gate on the as-run grid score file (run_grid.py at 7da16618 didn't enforce it).

  uv run --no-project --python 3.12 --with numpy python check_grid_r_gate.py --scores <grid_scores json> --inputs <scaled_inputs dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main(scores, inputs):
    I = Path(inputs)
    D = json.loads(Path(scores).read_text())
    R = D["probes"].index("R")
    st = np.array([s["s"][R] for s in D["staged"]], float)
    role = np.array([s["role"] for s in D["staged"]])
    fs = json.loads((I / "flagrate_scores_2026-10-13_122815.json").read_text())
    gate = json.loads((I / "gate_final_2026-10-11_211436.json").read_text())["final"]["real|0"]
    ho, stored, tg = st[role == "heldout"], np.array(gate["heldout_scores"]), gate["threshold"]
    t1 = float(np.sort(st[role == "cal_a"])[::-1][int(np.floor(0.01 * (role == "cal_a").sum() + 1e-9))])
    fa = {f"fa_{r}": float((st[role == r] > t1).mean()) for r in ("cal_a", "honb_cal", "honb_ho")}
    res = {"R_staged_max_abs_diff_vs_label_free": float(np.abs(st - np.array([s["score"] for s in fs["staged"]])).max()),
           "R_gate_heldout_max_abs_diff": float(np.abs(ho - stored).max()), "R_gate_heldout_flags_identical": bool(((ho > tg) == (stored > tg)).all()),
           "R_threshold_recomputed": t1, "R_threshold_frozen": tg, "R_threshold_diff": abs(t1 - tg),
           "R_false_alarms_recomputed": fa, "R_false_alarms_stored": {k: gate[k] for k in fa}, "R_false_alarms_equal": all(abs(fa[k] - gate[k]) <= 1e-12 for k in fa)}
    res["passes"] = bool(res["R_staged_max_abs_diff_vs_label_free"] <= 1e-6 and res["R_gate_heldout_max_abs_diff"] <= 1e-6 and res["R_gate_heldout_flags_identical"]
                         and res["R_threshold_diff"] <= 1e-6 and res["R_false_alarms_equal"])
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--inputs", required=True)
    main(**vars(ap.parse_args()))
