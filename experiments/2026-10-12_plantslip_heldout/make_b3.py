"""B3, the historical-outcome forecast, from already-reported gaps only (points, real e2 - simulated, 1%)."""
import json, subprocess, sys
R = sys.argv[1]
def show(ref, path): return json.loads(subprocess.run(["git", "-C", R, "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout)
M = ("c24106ca", "experiments/2026-10-06_q3_attack_detector/results/metrics_2026-10-06_181548__2026-10-06_182739.json")
m = show(*M)
def q(L):  # sim - e2 stored as a fraction; x(-100) gives real - sim in points
    return -100 * float(m["by_layer"]["probe_output_end_mean"][str(L)]["gap"]["fpr_0.01"]["sim_minus_e2"]["value"])
grid = [16, 24, 32, 40, 48, 56, 63]
out = {"source_qwen3": f"{M[1]}@{M[0]}", "key": "by_layer/probe_output_end_mean/<L>/gap/fpr_0.01/sim_minus_e2/value, x(-100)",
       "note": "#36's 10-06 threshold definition, not nested frozen thresholds", "forecasts": {}}
for L in [20, 28, 36, 44, 52, 60]:
    lo = max(g for g in grid if g < L); hi = min(g for g in grid if g > L)
    w_hi = (L - lo) / (hi - lo)
    out["forecasts"][f"qwen3-32b@L{L}"] = {"value": (1 - w_hi) * q(lo) + w_hi * q(hi), "from": [lo, hi], "weights": [1 - w_hi, w_hi]}
for rd, (ref, path) in {"olmo": ("668b902a", "experiments/2026-10-12_q2_olmo_reader/results/reply_break_olmo_2026-10-12_113245.json"),
                        "llama": ("ab5caa5d", "experiments/2026-10-12_q2_llama_reader/results/reply_break_llama_2026-10-12_143812.json")}.items():
    v = -float(show(ref, path)["rates"]["0.01"]["sim - e2 (points)"])
    layers = {"olmo": [16, 24, 48, 63], "llama": [20, 30, 40, 50, 79]}[rd]
    for L in layers:
        out["forecasts"][f"{rd}@L{L}"] = {"value": v, "from": f"{path}@{ref} rates/0.01/'sim - e2 (points)' x(-1), the reader's forest gap"}
out["fallback"] = 2.186
json.dump(out, sys.stdout, indent=1)
