"""G3's smoke-test projection (0d, 02:30 PT 10-13, condition 3): billed $ per 1M forward tokens, and a projection of the
real arm (and planted, controls) against its $102 stop. $0, local. Inputs: the smoke run's shard stats (stats/shard_0.json,
forward [tokens, seconds] per pass; numbers only), the smoke app's id (its bill is read here with the strict watcher's
bill(), only once the app is stopped, twice 5 minutes apart; the two reads must agree, as a settled bill), and the step-2
render gate record (Seed's per-arm sum(L) and sum(L^2)). A bill that leaves no room for the start and load (inferred
fixed part <= 0) is refused as unsettled or inconsistent.
  - raw: billed $ / smoke forward tokens x 1e6 (dominated by the container start and the 72 GB weight load).
  - projection: forward time scales with Seed's forward FLOPs (reprice_g3.seed_flops: linear 2 * params * L plus full
    causal attention 2 * q_dim * L^2, 64 blocks), at the smoke's measured seconds per FLOP; plus per transcript 1.8 s
    outside the forward pass (Olmo's smoke measure) and, per shard, the smoke's remaining billed time (its start, load and
    idle); all at the H200 effective $/s of price_g3.py. The smoke reads the 4 longest transcripts, so its seconds per
    FLOP is attention-heavy.
Hold (tell 0d) if any arm's projection exceeds its stop (real $102, planted $29, controls $2).

  uv run --no-project --python 3.12 --with huggingface_hub==0.35.3 python experiments/2026-10-13_g3_new_reader/project_smoke.py \\
      --stats <shard_0.json> --app <smoke app id> --gate results/render_g3_gate_<stamp>.json
"""
import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import price_g3 as P  # noqa: E402
import reprice_g3 as RP  # noqa: E402
import watch_billing_strict as W  # noqa: E402

STOPS = {"real": 102.0, "planted": 29.0, "controls": 2.0}
ARMS = {"real": ("real-honest", "real-attack-e1", "real-attack-e2"), "planted": ("real-attack-e2-planted",), "controls": ("real-attack-e2-control",)}
RUNS = {"real": 990 + 939 + 939, "planted": 845, "controls": 20}
NONFWD_S = 1.8


def settled_bill(app: str) -> float:
    if W.state(app)[0] != "stopped":
        raise SystemExit(f"{app} is not stopped yet")
    b1 = W.bill(app); time.sleep(300); b2 = W.bill(app)
    if not (math.isfinite(b1) and b1 > 0 and abs(b1 - b2) < 1e-6):
        raise SystemExit(f"the smoke bill is not settled (reads {b1:.4f}, then {b2:.4f}); read again later")
    return b2


def main(a):
    st, g = json.loads(Path(a.stats).read_text()), json.loads(Path(a.gate).read_text())
    a.bill = settled_bill(a.app)
    if g.get("gate") != "pass" or g.get("scope") != "step 2":
        raise SystemExit("not a passing step-2 gate record")
    passes = st["forward [tokens, seconds]"]
    if len(passes) != 4:
        raise SystemExit(f"{len(passes)} forward passes in the smoke stats, not 4")
    toks, secs = sum(t for t, _ in passes), sum(s for _, s in passes)
    cs = P.cfg(P.SEED["model"], P.SEED["revision"])
    f_smoke = RP.seed_flops(cs, toks, sum(t * t for t, _ in passes))
    eff = P.OLMO["effective $/s (H100, smoke)"]
    eff_h200 = eff * (P.LIST_PER_H["H200"] + eff * 3600 - P.LIST_PER_H["H100"]) / (eff * 3600)
    s_per_flop = secs / f_smoke
    over = a.bill - eff_h200 * (secs + 4 * NONFWD_S)  # the smoke shard's start, load and idle, in $
    if not (toks > 0 and secs > 0 and over > 0):
        raise SystemExit(f"inconsistent smoke inputs: {toks} tokens, {secs:.1f} s, fixed part ${over:.4f} (the bill may be unsettled)")
    proj = {}
    for step, arms in ARMS.items():
        fl = sum(RP.seed_flops(cs, g["arms"][x]["forward tokens"], g["arms"][x]["forward tokens squared, summed over runs"]) for x in arms)
        fwd = eff_h200 * s_per_flop * fl
        tot = fwd + eff_h200 * NONFWD_S * RUNS[step] + RP.P.SEED["shards"][step] * over
        proj[step] = {"forward $": round(fwd, 2), "outside-forward $": round(eff_h200 * NONFWD_S * RUNS[step], 2),
                      "per-shard fixed $": round(RP.P.SEED["shards"][step] * over, 2), "projected $": round(tot, 2),
                      "stop $": STOPS[step], "over the stop": tot > STOPS[step]}
    rec = {"made": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"),
           "inputs": {"stats sha256": hashlib.sha256(Path(a.stats).read_bytes()).hexdigest(), "smoke app": a.app, "smoke bill $ (settled)": a.bill,
                      "gate": Path(a.gate).name, "gate sha256": hashlib.sha256(Path(a.gate).read_bytes()).hexdigest()},
           "smoke forward passes [tokens, seconds]": passes, "smoke forward tokens": toks, "smoke forward seconds": round(secs, 2),
           "billed $ per 1M forward tokens (raw)": round(a.bill / toks * 1e6, 3),
           "seconds per 1e15 forward FLOPs": round(s_per_flop * 1e15, 4), "smoke fixed part $ (start, load, idle)": round(over, 4),
           "H200 effective $/s": round(eff_h200, 6), "projection": proj, "hold": any(v["over the stop"] for v in proj.values()),
           "note": "the 4 smoke transcripts are the longest by Qwen3-32B's count, and 1.8 s per transcript outside the forward pass is Olmo's measure: a rough projection, not a bound"}
    with open(dest := HERE / "results" / f"smoke_projection_{rec['made']}.json", "x") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec, indent=1)); print(dest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True); ap.add_argument("--app", required=True); ap.add_argument("--gate", required=True)
    main(ap.parse_args())
