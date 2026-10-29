"""D's verdicts at 196 tasks (plan_D_redo_196_2026-10-09.md as amended: 500 fixed draws with Monte Carlo SD, Aswin's
planted-check rule, D_control from D's own draws, the D null as a gate). $0, local.

Inputs: the D config's draw file (draws 0..500; cells on_real->{on_real, on_sim, on_run2}, off_real->{off_real, off_sim,
off_e2}), the D null's draw file (draws 1..200; d_null_draw), and analyze_D196_plants.py's output.
Per draw (AUROC on the matched subset):
- D = G_on - G_off, G = AUROC(simulated) - AUROC(real 1) in each phase;
- D_control = G_on(real 2 - real 1) - G_off(real 2 - real 1), real 2 = run 2 on-policy and Sol's epoch 2 off-policy;
- D on standardized pAUC alongside, descriptive.
Each endpoint: 2.5/97.5 percentiles over draws 1..500, failed draws at the floor (-2) for the lower and the ceiling (+2)
for the upper; its Monte Carlo SD as analyze_scaleup196.endpoints; more than 5% failed draws: invalid.
The null: each side's observed real -> real AUROC must lie above its null's 95th percentile (the probe informative on each
side; null cells on the permuted labels), and D_null_k = G_on_null_k - G_off_null_k from the null probes scored on the
true labels ("|true" cells).
Verdicts (the plan's, each also needing the null gate):
- "the shift differs between acting and reading": D's interval excludes 0, D outside D_null's 2.5-97.5% range, D_control
  passes (its interval inside (-0.02, +0.02)), and the 0.030 plant on that side is detected (D up if D > 0, down if < 0);
- "a difference of 0.03 or more is ruled out": D's interval inside (-0.03, +0.03), D_control passes, and both 0.030 plants
  are detected.
A failed control, a missed plant, a failed gate or an invalid endpoint leaves the verdicts it gates inconclusive. Nothing
is read before 500 draws.

The run's config must name the label check (`variant_check`) whose primary variant it is.

  python3 analyze_D196.py <D draws file> <D null draws file> <D plants file> [--draws 500]
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze_scaleup196 import FROZEN_DRAWS, MAX_FAIL, NULL_DRAWS, endpoints, value

HERE = Path(__file__).parent
PLANT_COPIES = 2000
G = {"on": ("on_real->on_sim", "on_real->on_real"), "off": ("off_real->off_sim", "off_real->off_real")}
G2 = {"on": ("on_real->on_run2", "on_real->on_real"), "off": ("off_real->off_e2", "off_real->off_real")}


def dd(draw, pairs, metric="auroc", tag=""):
    v = [value(draw, a + tag, metric) for a, _ in pairs.values()] + [value(draw, b + tag, metric) for _, b in pairs.values()]
    if None in v:
        return None
    (on_a, off_a), (on_b, off_b) = v[:2], v[2:]
    return (on_a - on_b) - (off_a - off_b)


def main(main_path, null_path, plants_path, n):
    if n != FROZEN_DRAWS:  # nothing is computed or printed before the frozen 500 draws (Amendment 3 item 6)
        raise SystemExit(f"D reads only at {FROZEN_DRAWS} draws, not {n}: nothing computed")
    R, N, PL = (json.loads(Path(p).read_text()) for p in (main_path, null_path, plants_path))
    if PL.get("resamples") != PLANT_COPIES:
        raise ValueError(f"the plants file has {PL.get('resamples')} copies, not the frozen {PLANT_COPIES}")
    vc = json.loads((HERE.parents[1] / R["config"]["variant_check"]).read_text())
    if vc["primary"] != R["config"]["variant"] or vc["i"]["draws"] != FROZEN_DRAWS:
        raise ValueError(f"the label check names variant {vc['primary']}; this run is variant {R['config']['variant']}")
    ids = [d["draw"] for d in R["draws"]]
    if len(ids) != len(set(ids)) or sorted(ids) != list(range(n + 1)):
        raise ValueError(f"the D run must hold draws 0..{n} exactly once; it holds {len(ids)} ids, {len(set(ids))} distinct")
    draws = {d["draw"]: d for d in R["draws"]}
    has_control = "on_real->on_run2" in draws[0]["cells"] and "off_real->off_e2" in draws[0]["cells"]
    obs, boots = draws[0], [draws[b] for b in range(1, n + 1)]
    out = {"main_input": Path(main_path).name, "null_input": Path(null_path).name, "plants_input": Path(plants_path).name,
           "git_of_scoring": R.get("git"), "summary": R.get("summary"), "draws": n, "endpoints": {}}
    for name, pairs, m in (("D", G, "auroc"), ("D_control", G2, "auroc"), ("D_pauc (descriptive)", G, "pauc")):
        if name == "D_control" and not has_control:  # the plan: no second real runs -> D_control unavailable
            out["endpoints"][name] = {"unavailable": "no control cells (on_real->on_run2, off_real->off_e2)", "invalid": True}
            continue
        e = endpoints([dd(d, pairs, m) for d in boots], floor=-2.0, ceil=2.0)
        e["observed"] = dd(obs, pairs, m)
        out["endpoints"][name] = e
    nids = [d["draw"] for d in N["draws"]]
    if len(nids) != len(set(nids)) or sorted(nids) != list(range(1, NULL_DRAWS + 1)):
        raise ValueError(f"the D null must hold draws 1..{NULL_DRAWS} exactly once; it holds {len(nids)}")
    gate = {}
    for side, cell in (("on", "on_real->on_real"), ("off", "off_real->off_real")):
        nv = [value(d, cell, "auroc") for d in N["draws"]]
        ok = [x for x in nv if x is not None]
        p95 = float(np.percentile(ok, 95)) if ok else float("nan")
        o = value(obs, cell, "auroc")
        gate[side] = {"observed_real_real_auroc": o, "null_p95": p95, "null_failed": len(nv) - len(ok),
                      "passes": bool(ok and o is not None and o > p95 and len(nv) - len(ok) <= MAX_FAIL * len(nv))}
    dn = [dd(d, G, "auroc", "|true") for d in N["draws"]]
    dn_ok = [x for x in dn if x is not None]
    d_obs = out["endpoints"]["D"]["observed"]
    nlo, nhi = (np.percentile(dn_ok, [2.5, 97.5]) if dn_ok else (np.nan, np.nan))
    gate["D_null_range_95"] = [float(nlo), float(nhi)]
    gate["D_null_failed"] = len(dn) - len(dn_ok)
    gate["D_outside_null_range"] = bool(d_obs is not None and dn_ok and not (nlo <= d_obs <= nhi)
                                        and len(dn) - len(dn_ok) <= MAX_FAIL * len(dn))
    gate["passes"] = gate["on"]["passes"] and gate["off"]["passes"]
    out["null"] = gate
    D, C = out["endpoints"]["D"], out["endpoints"]["D_control"]
    ctrl = has_control and not C["invalid"] and -0.02 < C["ci95_exact"][0] and C["ci95_exact"][1] < 0.02
    up, down = PL["plants"]["D up by 0.03"]["detected"], PL["plants"]["D down by 0.03"]["detected"]
    v = {"D_control_passes": ctrl, "null_gate_passes": gate["passes"], "plant_up_0.030_detected": up, "plant_down_0.030_detected": down}
    if D["invalid"]:
        v["the shift differs"] = v["a difference of 0.03 or more is ruled out"] = "invalid (more than 5% failed draws)"
    else:
        lo, hi = D["ci95_exact"]
        failed = [x for x, ok in (("D_control", ctrl), ("null gate", gate["passes"])) if not ok]
        if lo > 0 or hi < 0:
            side, plant = ("D > 0 (G_on above G_off)", up) if lo > 0 else ("D < 0 (G_on below G_off)", down)
            miss = failed + ([] if gate["D_outside_null_range"] else ["D inside D_null's range"]) + ([] if plant else ["the 0.030 plant on that side"])
            v["the shift differs"] = side if not miss else f"inconclusive: {side} by the interval, but failed: {', '.join(miss)}"
        else:
            v["the shift differs"] = "not established: the interval includes 0" + (f" (also failed: {', '.join(failed)})" if failed else "")
        if -0.03 < lo and hi < 0.03:
            miss = failed + [x for x, ok in (("the up plant", up), ("the down plant", down)) if not ok]
            v["a difference of 0.03 or more is ruled out"] = "yes" if not miss else f"inconclusive: inside +-0.03, but failed: {', '.join(miss)}"
        else:
            v["a difference of 0.03 or more is ruled out"] = "not ruled out: the interval reaches past +-0.03" + (
                f" (also failed: {', '.join(failed)})" if failed else "")
    out["verdicts"] = v
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"D196_verdicts_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("main_path")
    ap.add_argument("null_path")
    ap.add_argument("plants_path")
    ap.add_argument("--draws", type=int, default=500)
    a = ap.parse_args()
    main(a.main_path, a.null_path, a.plants_path, a.draws)
