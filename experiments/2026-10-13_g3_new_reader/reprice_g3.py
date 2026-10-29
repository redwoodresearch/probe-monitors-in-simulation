"""G3's firm re-price (0d, 01:50 PT 10-13): price_g3.py's model with Seed-OSS-36B's own token counts from step 2's render
records, through the gate record (render_g3_gate_<stamp>.json, which must pass). $0, local; runs nothing on Modal.
  - Olmo's rate per forward FLOP and the brackets for its sliding-window layers are price_g3.py's, unchanged (imported).
  - Seed's step-2 FLOPs are exact sums from the render: per block, 2 * linear params * sum(L) + 2 * q_dim * sum(L^2), with
    L each run's forward length to its last kept readout (all 64 blocks run: the forward pass is not truncated).
  - Step 7 (simulated) can't be rendered before the forecast. Its sums are Olmo's simulated-arm sums times Seed's own
    token ratio over Olmo's on the three real arms (for sum(L)) and its square (for sum(L^2)).
  - Billed so far is passed in, from `modal billing report` by app id (strict watcher's bill()).
Decision (0d): launch steps 2-7 if the high bracket's whole job x 1.3 <= $250 - billed so far; otherwise send it to 0d.

  uv run --no-project --python 3.12 --with huggingface_hub==0.35.3 python experiments/2026-10-13_g3_new_reader/reprice_g3.py \\
      --gate results/render_g3_gate_<stamp>.json --stage <q2_reply_stage json> --billed <$ so far>
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import price_g3 as P  # noqa: E402

ALLOCATION = 250.0
REAL = ("real-honest", "real-attack-e1", "real-attack-e2")
STEP2 = (*REAL, "real-attack-e2-planted", "real-attack-e2-control")


def seed_flops(c, s1, s2):
    h, nh, nkv = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"]
    hd = c.get("head_dim") or h // nh
    q = nh * hd
    linear = h * q + 2 * h * nkv * hd + q * h + 3 * h * c["intermediate_size"]
    if c.get("layer_types") or c.get("sliding_window"):
        raise SystemExit("Seed's config now has sliding layers: the full-attention sum no longer applies")
    return c["num_hidden_layers"] * (2 * linear * s1 + 2 * q * s2)


def main(a):
    gate = json.loads(Path(a.gate).read_text())
    if gate["gate"] != "pass" or gate.get("scope") != "step 2":
        raise SystemExit("the step-2 render gate did not pass")
    git = lambda *x: subprocess.run(["git", *x], cwd=P.ROOT, capture_output=True, check=True).stdout  # noqa: E731
    for p_, role in ((P.RUNS_JSON, "runs.json"), (Path(a.stage), "stage")):
        if hashlib.sha256(p_.read_bytes()).hexdigest() != P.PIN[role]:
            raise SystemExit(f"{p_} is not the pinned {role}")
    rows = {r["arm"] + "|" + r["count"]: r["n"] for r in json.loads(git("show", f"{P.OLMO_RENDER[0]}:{P.OLMO_RENDER[1]}"))["rows"]}
    ofwd = {x: rows[f"{x}|forward tokens (to the last kept readout)"] for x in (*REAL, "sim-attack", "sim-attack-fix939")}
    rs = json.loads(P.RUNS_JSON.read_text())
    by_arm = {}
    for r in rs:
        by_arm.setdefault(r["arm"], {})[r["key"]] = r["tokens"]
    scale = {x: ofwd[x] / sum(by_arm[x].values()) for x in (*REAL, "sim-attack")}
    L = {x: [t * scale[x] for t in by_arm[x].values()] for x in scale}
    n_fix, n_sim = rows["sim-attack-fix939|runs"], len(by_arm["sim-attack"])
    L["sim-attack-fix939"] = [v * (ofwd["sim-attack-fix939"] / n_fix) / (ofwd["sim-attack"] / n_sim) for v in L["sim-attack"]]
    W8 = {"sim-attack-fix939": n_fix / n_sim}
    co, cs = P.cfg(P.OLMO["model"], P.OLMO["revision"]), P.cfg(P.SEED["model"], P.SEED["revision"])
    arms = gate["arms"]
    s1 = {x: arms[x]["forward tokens"] for x in STEP2}
    s2 = {x: arms[x]["forward tokens squared, summed over runs"] for x in STEP2}
    ratio = sum(s1[x] for x in REAL) / sum(ofwd[x] for x in REAL)  # Seed tokens per Olmo token, real arms
    sim1 = ratio * sum(L["sim-attack"]); sim2 = ratio ** 2 * sum(v * v for v in L["sim-attack"])
    eff = P.OLMO["effective $/s (H100, smoke)"]
    eff_h200 = eff * (P.LIST_PER_H["H200"] + eff * 3600 - P.LIST_PER_H["H100"]) / (eff * 3600)
    olmo_var = P.OLMO["extraction $"] - P.OLMO["shards"] * P.OLMO["overhead s per shard (smoke)"] * eff
    over = P.OLMO["overhead s per shard (smoke)"] * P.SEED["weights GB"] / P.OLMO["weights GB"] * eff_h200
    sh = P.SEED["shards"]
    out = {}
    for band, full in (("low (Olmo paid full attention)", True), ("high (Olmo paid the window)", False)):
        rate = olmo_var / sum(W8.get(x, 1.0) * P.flops(co, v, full) for x in L for v in L[x]) * eff_h200 / eff
        step2 = {x: rate * seed_flops(cs, s1[x], s2[x]) for x in STEP2}
        e2_ = {"real $": sum(step2[x] for x in REAL) + sh["real"] * over, "planted $": step2["real-attack-e2-planted"] + sh["planted"] * over,
               "controls $": step2["real-attack-e2-control"] + sh["controls"] * over, "smoke $": sh["smoke"] * over + P.SMALL["smoke test forward passes (4 transcripts)"]}
        sim = rate * seed_flops(cs, sim1, sim2) + sh["sim"] * over
        small = sum(v for k, v in P.SMALL.items() if not k.startswith("smoke"))
        raw = sum(e2_.values()) + sim + small
        out[band] = {k: round(v, 2) for k, v in e2_.items()} | {"step 7 simulated $": round(sim, 2), "CPU pieces $": round(small, 2),
                                                                 "whole job raw $": round(raw, 2), "whole job +30% $": round(1.3 * raw, 2)}
    high = out["high (Olmo paid the window)"]["whole job +30% $"]
    rec = {"made": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "git": git("rev-parse", "HEAD").decode().strip(),
           "gate record": {"file": Path(a.gate).name, "sha256": hashlib.sha256(Path(a.gate).read_bytes()).hexdigest()},
           "Seed forward tokens (M)": {x: round(s1[x] / 1e6, 3) for x in STEP2}, "Seed/Olmo token ratio (real arms)": round(ratio, 4),
           "step 7 Seed forward tokens (M, estimated)": round(sim1 / 1e6, 3), "price": out, "billed so far $": a.billed,
           "allocation $": ALLOCATION, "decision": "launch steps 2-7" if high <= ALLOCATION - a.billed else "send to 0d (over the allocation)"}
    with open(dest := HERE / "results" / f"reprice_{rec['made']}.json", "x") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "git"}, indent=1)); print(dest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True); ap.add_argument("--stage", required=True); ap.add_argument("--billed", type=float, required=True)
    main(ap.parse_args())
