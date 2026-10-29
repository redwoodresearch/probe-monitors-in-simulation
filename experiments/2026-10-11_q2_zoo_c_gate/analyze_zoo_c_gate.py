"""Zoo C's positive-count gate (PLAN_zoo_B_C.md as merged in #666, commit c1f9b556, lines 92-122), applied to Aswin's
counts (experiments/2026-10-11_credential_count/results/gate_v3/gate_counts_v3.json in #695; only that file's group
counts are read). No credential code, no monitor; $0.

P(agree) = P(real-vs-sim passes at the Holm level for two, z = 2.24) x P(same-batch real-vs-real passes, z = 1.96), each
2 Phi((0.10 - z SE) / SE) - 1 clamped at 0, with SE^2 = p (1 - p) DE (1/n_a + 1/n_b), p = 0.5 and DE = 1.25. The rule's DE
is max(1.25, 1.2 x Kish); the file has no per-task counts, and a larger DE only lowers P, so DE = 1.25 gives the largest P
the rule allows. Below 0.8 at the primary 5% level, C is "too few to tell". p = 0.5 is the same at both target levels.

  uv run python experiments/2026-10-11_q2_zoo_c_gate/analyze_zoo_c_gate.py e37a2892f4b2a76dce4deacc2b4b4a1bbe9cd35f
"""
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
SOURCE = "experiments/2026-10-11_credential_count/results/gate_v3/gate_counts_v3.json"
P, DE, Z_RS, Z_RR, MARGIN, GATE = 0.5, 1.25, 2.24, 1.96, 0.10, 0.8


def pass_prob(n_a: int, n_b: int, z: float) -> tuple[float, float]:
    se = math.sqrt(P * (1 - P) * DE * (1 / n_a + 1 / n_b))
    return se, max(0.0, math.erf((MARGIN - z * se) / se / math.sqrt(2)))  # 2 Phi(x) - 1 = erf(x / sqrt 2)


def main(commit: str):
    raw = subprocess.run(["git", "show", f"{commit}:{SOURCE}"], capture_output=True, check=True).stdout
    g = json.loads(raw)["groups"]
    n = {"simulated": g["simulated"]["positives"], "matched real (overlap800)": g["matched_real_overlap800"]["positives"],
         "same-batch real run 1": g["same_batch_real_run_1"]["positives"], "same-batch real run 2": g["same_batch_real_run_2"]["positives"]}
    se_rs, p_rs = pass_prob(n["matched real (overlap800)"], n["simulated"], Z_RS)
    se_rr, p_rr = pass_prob(n["same-batch real run 1"], n["same-batch real run 2"], Z_RR)
    joint = p_rs * p_rr
    out = {"source": f"{SOURCE} @ {commit}", "source sha256": hashlib.sha256(raw).hexdigest(), "positives": n,
           "settings": {"p": P, "DE": DE, "z real-vs-sim (Holm, two)": Z_RS, "z real-vs-real": Z_RR, "margin": MARGIN, "gate": GATE},
           "real vs sim": {"SE": se_rs, "pass probability": p_rs}, "same-batch real vs real": {"SE": se_rr, "pass probability": p_rr},
           "predicted P(agree), both target levels": joint,
           "verdict": "too few to tell" if joint < GATE else "gate passed: the monitor may be proposed",
           "what follows": "counts and predicted pass probability only; no agree verdict; no paid monitor launch unless Aswin asks"
           if joint < GATE else "propose the monitor launch to Aswin"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"zoo_c_gate_{stamp}.json"
    with open(dest, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(f"positives {n}")
    print(f"real vs sim: SE {se_rs:.4f}, P(pass) {p_rs:.4f}; same-batch real vs real: SE {se_rr:.4f}, P(pass) {p_rr:.4f}")
    print(f"predicted P(agree) {joint:.4f} vs gate {GATE}: {out['verdict']}")
    print(dest)


if __name__ == "__main__":
    main(sys.argv[1])
