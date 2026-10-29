"""POST HOC rivals for the row-2 held-out planting test (RULE.md here): B4 = the gap at the nearest already-reported layer of
the same reader (ties: the mean of the two), B5 = the mean of the reader's already-reported fixed-layer gaps. Real e2 - simulated
catch at 1%, in points. Reads committed files by key only; $0.

usage: python3 -I make_b45.py <repo> > results/b45_forecasts_<stamp>.json"""

import json
import subprocess
import sys

R = sys.argv[1]
HELD = {"qwen3-32b": [20, 28, 36, 44, 52, 60], "olmo": [16, 24, 48, 63], "llama": [20, 30, 40, 50, 79]}
QWEN = ("c24106ca", "experiments/2026-10-06_q3_attack_detector/results/metrics_2026-10-06_181548__2026-10-06_182739.json")
OLMO = ("496d6e8d", "experiments/2026-10-12_q2_meanpool_readers/results/meanpool_reader_olmo_2026-10-12_212839.json")
LLAMA = ("ab5caa5d", "experiments/2026-10-12_q2_llama_reader/results/reply_break_llama_2026-10-12_143812.json")


def show(ref, path):
    return json.loads(subprocess.run(["git", "-C", R, "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout)


m, o, la = show(*QWEN), show(*OLMO), show(*LLAMA)
reported = {  # reader -> {layer: real e2 - simulated, points, 1%}
    "qwen3-32b": {L: -100 * float(m["by_layer"]["probe_output_end_mean"][str(L)]["gap"]["fpr_0.01"]["sim_minus_e2"]["value"])
                  for L in (16, 24, 32, 40, 48, 56, 63)},
    "olmo": {L: -float(o["rates"]["0.01"]["by configuration"][f"last-token @ L{L}"]["gaps (points, 95%)"]["as simulated - e2"][0])
             for L in (32, 40, 56)},
}
llama_reader = -float(la["rates"]["0.01"]["sim - e2 (points)"])  # no fixed-layer Llama gap is committed; reader-level fallback
out = {"sources": {"qwen3-32b": f"{QWEN[1]}@{QWEN[0]} (#36's 10-06 threshold definition)",
                   "olmo": f"{OLMO[1]}@{OLMO[0]} (last-token, nested frozen, 1%)",
                   "llama": f"{LLAMA[1]}@{LLAMA[0]} (reader-level per-fold detector; no fixed-layer gap committed)"},
       "reported_layers": {r: {str(k): v for k, v in d.items()} for r, d in reported.items()}, "llama_reader_gap": llama_reader,
       "forecasts": {}}
for rd, layers in HELD.items():
    rep = reported.get(rd)
    for L in layers:
        if rep:
            dist = {k: abs(k - L) for k in rep}
            near = [k for k in rep if dist[k] == min(dist.values())]
            b4 = sum(rep[k] for k in near) / len(near)
            b5 = sum(rep.values()) / len(rep)
        else:
            near, b4, b5 = [], llama_reader, llama_reader
        out["forecasts"][f"{rd} @ L{L}"] = {"B4_nearest_reported": b4, "B4_from_layers": near, "B5_reader_mean": b5,
                                            "B4_flag_ge5": b4 >= 5, "B5_flag_ge5": b5 >= 5}
json.dump(out, sys.stdout, indent=1)
