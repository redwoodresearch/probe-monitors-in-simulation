"""Price arithmetic for PRICE.md (no Modal, no API). Every rate is a cited constant; low and high bracket the sources.
    python experiments/2026-10-12_q3_third_real_price/price_third_real.py
"""
import json

# (a) acting: Qwen3.5-27B real attack runs, launch 2's harness, honest C's 3 H200 servers
A = {
    "servers per run": (0.183, 0.28),  # honest C billed $35.91 / 196 runs; launch 2 wall-clock $0.232 + ~20% CPU/memory (matched-FA sizing)
    "servers start-up and idle": (3 * 0.4 * 4.54, 6 * 0.4 * 4.54),  # 0.4 h at $4.54 per H200-hour, 3 (honest C) or 6 (launch 2) servers
    "devbox per run": (4.32 / 196, 4.32 / 196),  # honest C devbox about $4.32 / 196 runs
    "devbox set-up": (1.0, 1.0),  # matched-FA sizing's fixed line
    "extraction per run": (4.80 / 193, 12.1e6 / 193 * 8.31 / 17.62e6),  # second attack billed $4.80 for 12.1M tokens / 193 runs; or its 62.7k tokens a run at honest C $8.31 / 17.62M
    "extraction test and staging": (0.37 + 0.03, 0.40 + 0.10),  # launch 2 --test $0.37 or honest C's test estimate $0.40; honest C staging $0.03-0.10
    "scoring, draw 0 only": (0.30 + 0.36 + 0.04, 0.30 + 0.50 + 0.35),  # row cache ~$0.30; observed draw $0.36-0.50 (D2 gate); collection $0.04-0.35
    "scoring, 500 refits (501 draws)": (0.30 + 109.75, 0.30 + 116.87 + 0.35),  # row cache + D2 billed $109.75 (gate + 501 draws + collections) / main grid $116.87
}
# (b) reading: openai/gpt-5.6-sol real attack epoch, devbox sims-astra, Qwen3-32B extraction for #36
B = {
    "Sol API per run": (387.655 / 548 * 0.654, 387.655 / 548 * 2),  # zoo C honest $388.26 less its $0.605 smoke, / 548 runs = $0.7074; x0.654 (attack/honest tokens x steps, #36 runs.json) to x2
    "Sol smoke run": (0.61, 2.0),  # zoo C smoke $0.605
    "devbox per run": (8.65 / 548, 8.65 / 548),  # zoo C devbox about $8.65 / 548 runs
    "devbox set-up": (1.0, 1.0),
    "extraction per run, L48 and L63": (24_400 * 0.756e-6, 24_400 * 8.82 / 10.27e6),  # 24.4k tokens (e1/e2 at the 32k cut) at fix939 $0.756/1M or edited-honest $0.859/1M
    "extraction per run, all 64 layers": (24_400 * 0.756e-6 * 0.038 / 0.018, 24_400 * 8.82 / 10.27e6 * 0.038 / 0.018),  # 10-06 all-layers $0.038 vs fix939 $0.018 per transcript
    "extraction smoke, staging, orchestration": (0.25 + 0.05 + 0.15, 0.25 + 0.05 + 0.15),  # edited-honest SPEND lines
    "#36 scoring and smoke check": (0.12 + 0.02, 0.35 + 0.02),  # 10-07 scoring $0.12; edited-honest estimate $0.35
}
PER_RUN = {"servers per run", "devbox per run", "extraction per run", "Sol API per run", "extraction per run, L48 and L63", "extraction per run, all 64 layers"}


def total(rates, n, skip):
    lines = {k: tuple(r * (n if k in PER_RUN else 1) for r in v) for k, v in rates.items() if k not in skip}
    lo, hi = sum(v[0] for v in lines.values()), sum(v[1] for v in lines.values())
    return {"lines": {k: [round(a, 2), round(b, 2)] for k, (a, b) in lines.items()}, "raw": [round(lo, 2), round(hi, 2)], "plus_30pct": [round(1.3 * lo, 2), round(1.3 * hi, 2)]}


out = {"rates": {"acting": A, "reading": B}, "acting": {}, "reading": {}}
for n in (50, 100):
    out["acting"][n] = {"draw 0 only": total(A, n, {"scoring, 500 refits (501 draws)"}), "500 refits": total(A, n, {"scoring, draw 0 only"})}
for n in (200, 400):
    out["reading"][n] = {"L48 and L63": total(B, n, {"extraction per run, all 64 layers"}), "all 64 layers": total(B, n, {"extraction per run, L48 and L63"})}
print(json.dumps(out, indent=1))
