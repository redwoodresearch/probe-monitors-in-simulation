"""G3's price, steps 2-7 (RULE.md "Spend"), for the screen's pick, from billed actuals. $0: reads committed records and
two config.json files (anonymous Hugging Face downloads); runs nothing on Modal. A preliminary budget, not a calibrated
estimate: the real token counts come from step 2's render.

Basis: Olmo-3.1-32B's extraction of #36's runs plus fix939's (4,630 transcripts, 106,652,094 forward tokens, 48 shards
on one H100 each) billed $64.35 (SPEND.md row "A third reader, step 2"). Its per-shard fixed part (start and weight
load, 79 s per shard in the smoke test, at the smoke-measured effective $0.001369/s) is taken out. The rest is a rate
per forward FLOP. Per transcript, the forward FLOPs at forward length L are, per block, 2 * linear params * L (no LM
head) plus causal attention: full 2 * q_dim * L^2; sliding window W 4 * q_dim * W * L - 2 * q_dim * W^2 (L > W).
Per-run lengths are runs.json's token counts, scaled per arm to Olmo's forward-token totals (render, C2); planted and
controls take their original e2 runs' lengths. fix939's 823 runs (its per-run file isn't local) take the simulated
arm's length distribution, rescaled to fix939's mean forward length and weighted 823/939.
Olmo has 48 sliding-window layers (W 4,096) and 16 full ones; Seed-OSS-36B is full throughout, with 80 query heads.
How Olmo's SDPA path billed its sliding layers is unknown, so the price brackets it:
  high: Olmo paid the window (fewer FLOPs per dollar, so Seed costs more);
  low: Olmo paid full attention everywhere.
"high, tokens x1.15" scales every length by 1.15, an assumption for Seed's tokenizer, not a measured band.
H200 effective $/h = the H200 list price + Olmo's CPU and memory part (effective H100 rate minus the H100 list price).
  uv run --no-project --python 3.12 --with huggingface_hub==0.35.3 python experiments/2026-10-13_g3_new_reader/price_g3.py --stage <q2_reply_stage json>
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLMO_RENDER = ("c70b5993", "experiments/2026-10-12_q2_olmo_reader/results/olmo_render_r2_2026-10-12_100053.json")
RUNS_JSON = ROOT / "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"
PIN = {"runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96",
       "stage": "f1a68c90a5d26799925bda1c0d574728a547cde6c2ec3b76bfa18c3691c75157"}
OLMO = {"model": "allenai/Olmo-3.1-32B-Instruct", "revision": "ac0587e4a7744a551c059d8cd17ba220bc940dae",
        "extraction $": 64.35, "forward tokens": 106652094, "shards": 48, "overhead s per shard (smoke)": 79,
        "effective $/s (H100, smoke)": 0.001369, "weights GB": 64.4}
SEED = {"model": "ByteDance-Seed/Seed-OSS-36B-Instruct", "revision": "497f1dca95ebdec98e41d517b9f060ee753c902f", "weights GB": 72.3,
        "shards": {"real": 24, "planted": 8, "controls": 1, "sim": 8, "smoke": 1}}
LIST_PER_H = {"H100": 3.95, "H200": 4.54}  # Modal list prices, GPU part
SMALL = {  # Modal CPU pieces (and the smoke test's forward pass), from the Olmo and row 2 actuals, rounded up
    "weights download (CPU)": 0.10, "renders, 4 jobs (CPU, counts only)": 0.12, "smoke test forward passes (4 transcripts)": 0.40,
    "extraction checks, 4 jobs (CPU)": 0.60, "step 3 gates (CPU)": 0.10, "step 4 fit, 7 layers x 20 nested fits (CPU)": 0.50,
    "step 4 score and false-alarm gates (CPU)": 0.05, "step 5 x and step 7 y bootstraps (CPU or local)": 0.10}


def cfg(model, rev):
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(model, "config.json", revision=rev, token=False)).read_text())


def flops(c, L, sliding_as_full):
    h, nh, nkv = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"]
    hd = c.get("head_dim") or h // nh
    q = nh * hd
    linear = h * q + 2 * h * nkv * hd + q * h + 3 * h * c["intermediate_size"]
    W = c.get("sliding_window")
    types = c.get("layer_types") or ["full_attention"] * c["num_hidden_layers"]
    total = 0.0
    for t in types:
        if t == "sliding_attention" and not sliding_as_full and W and L > W:
            att = 4 * q * W * L - 2 * q * W * W
        else:
            att = 2 * q * L * L
        total += 2 * linear * L + att
    return total


def main(stage):
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, check=True).stdout  # noqa: E731
    for p_, role in ((RUNS_JSON, "runs.json"), (Path(stage), "stage")):
        if hashlib.sha256(p_.read_bytes()).hexdigest() != PIN[role]:
            raise SystemExit(f"{p_} is not the pinned {role}")
    rows = {r["arm"] + "|" + r["count"]: r["n"] for r in json.loads(git("show", f"{OLMO_RENDER[0]}:{OLMO_RENDER[1]}"))["rows"]}
    fwd = {a: rows[f"{a}|forward tokens (to the last kept readout)"] for a in ("real-honest", "real-attack-e1", "real-attack-e2", "sim-attack", "sim-attack-fix939")}
    if sum(fwd.values()) != OLMO["forward tokens"]:
        raise SystemExit("Olmo's per-arm forward tokens do not sum to the projection's total")
    rs = json.loads(RUNS_JSON.read_text())
    st = json.loads(Path(stage).read_text())
    by_arm = {}
    for r in rs:
        by_arm.setdefault(r["arm"], {})[r["key"]] = r["tokens"]
    scale = {a: fwd[a] / sum(by_arm[a].values()) for a in ("real-honest", "real-attack-e1", "real-attack-e2", "sim-attack")}
    L = {a: [t * scale[a] for t in by_arm[a].values()] for a in scale}
    n_fix, n_sim = rows["sim-attack-fix939|runs"], len(by_arm["sim-attack"])
    r_fix = (fwd["sim-attack-fix939"] / n_fix) / (fwd["sim-attack"] / n_sim)  # mean-length ratio, fix939 over the sim arm
    L["sim-attack-fix939"] = [x * r_fix for x in L["sim-attack"]]
    W8 = {"sim-attack-fix939": n_fix / n_sim}  # weights: fix939 has 823 runs, drawn here from the 939 sim lengths
    e2 = {k: t * scale["real-attack-e2"] for k, t in by_arm["real-attack-e2"].items()}
    L["planted"] = [e2[x["key"]] for x in st["planted"] if x["arm"] == "real-attack-e2-planted"]
    L["controls"] = [e2[x["key"]] for x in st["controls"]]
    if (len(L["planted"]), len(L["controls"])) != (845, 20):
        raise SystemExit("planted or controls counts differ from the plan's")
    co, cs = cfg(OLMO["model"], OLMO["revision"]), cfg(SEED["model"], SEED["revision"])
    eff = OLMO["effective $/s (H100, smoke)"]
    eff_h200 = eff * (LIST_PER_H["H200"] + eff * 3600 - LIST_PER_H["H100"]) / (eff * 3600)
    olmo_var = OLMO["extraction $"] - OLMO["shards"] * OLMO["overhead s per shard (smoke)"] * eff
    seed_over_s = OLMO["overhead s per shard (smoke)"] * SEED["weights GB"] / OLMO["weights GB"]
    steps = {"step 2": ("real-honest", "real-attack-e1", "real-attack-e2", "planted", "controls"), "step 7": ("sim-attack",)}
    shards = {"step 2": SEED["shards"]["real"] + SEED["shards"]["planted"] + SEED["shards"]["controls"] + SEED["shards"]["smoke"], "step 7": SEED["shards"]["sim"]}
    out = {}
    for band, full, f in (("low (Olmo paid full attention)", True, 1.0), ("high (Olmo paid the window)", False, 1.0),
                          ("high, tokens x1.15", False, 1.15)):
        per_flop = olmo_var / sum(W8.get(a, 1.0) * flops(co, x, full) for a in ("real-honest", "real-attack-e1", "real-attack-e2", "sim-attack", "sim-attack-fix939") for x in L[a])
        rate = per_flop * eff_h200 / eff
        row = {}
        for s_, arms in steps.items():
            var = rate * sum(flops(cs, x * f, False) for a in arms for x in L[a])
            row[f"{s_} extraction $"] = round(var + shards[s_] * seed_over_s * eff_h200, 2)
        raw = sum(row.values()) + sum(SMALL.values())
        out[band] = row | {"small pieces $": round(sum(SMALL.values()), 2), "whole job raw $": round(raw, 2), "whole job +30% $": round(1.3 * raw, 2)}
    rec = {"made": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "git": git("rev-parse", "HEAD").decode().strip(),
           "basis": OLMO, "seed": SEED, "olmo render": f"{OLMO_RENDER[1]}@{OLMO_RENDER[0]}", "pins": PIN,
           "forward tokens (Olmo tokenizer, M)": {a: round(W8.get(a, 1.0) * sum(v) / 1e6, 2) for a, v in L.items()},
           "Olmo layer types": {t: (co.get("layer_types") or []).count(t) for t in set(co.get("layer_types") or [])}, "Olmo sliding window": co.get("sliding_window"),
           "H200 effective $/s": round(eff_h200, 6), "Seed fixed s per shard": round(seed_over_s), "Olmo variable $": round(olmo_var, 2),
           "small pieces": SMALL, "price": out}
    dest = HERE / "results" / f"price_{rec['made']}.json"
    with open(dest, "x") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: rec[k] for k in ("forward tokens (Olmo tokenizer, M)", "Olmo layer types", "H200 effective $/s", "Seed fixed s per shard", "Olmo variable $", "price")}, indent=1))
    print(dest)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--stage":
        raise SystemExit("usage: price_g3.py --stage <q2_reply_stage_2026-10-12_135318.json>")
    main(sys.argv[2])
