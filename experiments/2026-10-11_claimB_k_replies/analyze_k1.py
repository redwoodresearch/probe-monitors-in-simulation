"""k = 1 stage summary (PLAN.md amendment 1; written before any score is read). Reads claimB/k1_v1/scores_<tag>.json
(numbers only) and applies the rule:
- checks first: self-swap (every site's SELF score within 1e-4 of its stored e1 score at L63) and the planted slip (the
  re-read shift's mean within ±20% of the stored reference mean, 95% interval over 2,000 main-task draws excluding 0);
  if either fails, nothing below is read;
- Δ1 = mean(SIM − SELF), G = mean(stored fixed-sim twin − stored e1) on the same sites, R1 = Δ1 / G with a 95% interval
  from 2,000 main-task draws resampling both together (default_rng(20261011)): "the reply itself" if the lower bound
  ≥ 0.5, "not the reply itself" if the upper bound < 0.5, else too few;
- beside it: Δ1 with its interval, the e2 yardstick gap, R1 against the net gap, step A's layer, per environment and per
  tool (descriptive), identical replies, and lost sites. One figure: Δ1 and G with intervals, and R1 against 0.5.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with matplotlib python \\
      experiments/2026-10-11_claimB_k_replies/analyze_k1.py --tag <plan tag>
"""

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
OUT, DRAWS, SEED = "claimB/k1_v1", 2000, 20261011


def boot(groups, stat, rng):
    """stat over DRAWS resamples of the groups (main tasks), each a list of rows; returns the 95% interval over the
    draws with a value and how many draws had none (for R1: a non-positive G)."""
    keys = sorted(groups)
    vals = []
    for _ in range(DRAWS):
        rows = [r for k in rng.choice(keys, len(keys)) for r in groups[k]]
        v = stat(rows)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None], DRAWS - len(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    raw = b"".join(vol.read_file(f"{OUT}/scores_{a.tag}.json"))
    d = json.loads(raw)
    out = {"label": "k = 1 stage summary (preliminary until reviewed)", "inputs": {f"scores_{a.tag}.json": hashlib.sha256(raw).hexdigest()},
           "lost": d["lost"], "build_counts": d["build_counts"], "gate": d["gate"]["pass"], "fits_converged": d["fits_converged"]}
    sites = d["sites"]
    rng = np.random.default_rng(SEED)
    # check 1: self-swap
    diffs = [abs(r["63"]["self"] - r["63"]["stored_e1"]) for r in sites]
    self_ok = bool(sites) and max(diffs) < 1e-4
    out["check_self_swap"] = {"sites": len(sites), "max_abs_diff": max(diffs) if diffs else None, "pass": self_ok}
    # check 2: the planted slip
    pg = defaultdict(list)
    for x in d["planted"]:
        pg[x["main_task"]].append(x)
    ref = float(np.mean([x["reference"] for x in d["planted"]]))
    shift = float(np.mean([x["read_shift"] for x in d["planted"]]))
    ci, n = boot(pg, lambda rows: float(np.mean([x["read_shift"] for x in rows])), rng)
    planted_ok = bool(abs(shift - ref) <= 0.2 * abs(ref) and (ci[0] > 0 or ci[1] < 0))
    out["check_planted"] = {"sites": len(d["planted"]), "lost_to_cut": d["planted_lost_to_cut"], "reference_mean": ref, "read_shift_mean": shift,
                            "ci95": ci, "pass": planted_ok}
    out["checks_pass"] = self_ok and planted_ok and out["gate"]
    if not out["checks_pass"]:  # nothing below is read: no effect is computed, saved or plotted
        out["reading"] = "checks failed: nothing read"
        out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        with open(HERE / "results" / f"k1_summary_{stamp}.json", "x") as fh:
            fh.write(json.dumps(out, indent=1))
        print(json.dumps({k: out[k] for k in ("reading", "check_self_swap", "check_planted", "gate")}, indent=1))
        return
    g = defaultdict(list)
    for r in sites:
        g[r["main_task"]].append(r)

    def stats(rows, w="63", net=False):
        d1 = np.mean([r[w]["sim"] - r[w]["self"] for r in rows])
        G = np.mean([r[w]["stored_fixed_sim"] - r[w]["stored_e1"] for r in rows])
        if net:
            e2 = [r[w]["stored_e2"] - r[w]["stored_e1"] for r in rows if r[w]["stored_e2"] is not None]
            if not e2:
                return None
            G = G - np.mean(e2)
        return float(d1 / G) if G > 0 else None
    for w in ("63", "A"):
        d1 = float(np.mean([r[w]["sim"] - r[w]["self"] for r in sites]))
        G = float(np.mean([r[w]["stored_fixed_sim"] - r[w]["stored_e1"] for r in sites]))
        e2 = [r[w]["stored_e2"] - r[w]["stored_e1"] for r in sites if r[w]["stored_e2"] is not None]
        ci_r, nonpos = boot(g, lambda rows, w=w: stats(rows, w), rng)
        ci_d, _ = boot(g, lambda rows, w=w: float(np.mean([r[w]["sim"] - r[w]["self"] for r in rows])), rng)
        ci_g, _ = boot(g, lambda rows, w=w: float(np.mean([r[w]["stored_fixed_sim"] - r[w]["stored_e1"] for r in rows])), rng)
        ci_net, _ = boot(g, lambda rows, w=w: stats(rows, w, net=True), rng)
        out[f"layer_{w}"] = {"delta1": d1, "delta1_ci95": ci_d, "G": G, "G_ci95": ci_g, "R1": d1 / G if G > 0 else None, "R1_ci95": ci_r,
                             "draws_with_nonpositive_G": nonpos, "e2_yardstick_gap": float(np.mean(e2)) if e2 else None, "e2_sites": len(e2),
                             "R1_net": stats(sites, w, net=True), "R1_net_ci95": ci_net}
    L = out["layer_63"]
    if L["draws_with_nonpositive_G"] > 0.025 * DRAWS or L["R1"] is None:
        out["reading"] = "too few (the full gap is not reliably positive)"
    elif L["R1_ci95"][0] >= 0.5:
        out["reading"] = "the reply itself"
    elif L["R1_ci95"][1] < 0.5:
        out["reading"] = "not the reply itself"
    else:
        out["reading"] = "too few"
    by_env, by_tool = defaultdict(list), defaultdict(list)
    for r in sites:
        by_env[r["environment"]].append(r)
        by_tool[r["tool"]].append(r)
    out["by_environment"] = {e: {"sites": len(v), "R1": stats(v)} for e, v in sorted(by_env.items())}
    out["by_tool"] = {t: {"sites": len(v), "R1": stats(v)} for t, v in sorted(by_tool.items())}
    out["identical_replies"] = sum(1 for r in sites if r["sim_text_equals_self_text"])
    out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"k1_summary_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8, 3.4), gridspec_kw={"width_ratios": [1.4, 1]})
    for i, (k, lab) in enumerate((("delta1", "one simulated reply (Δ1)"), ("G", "full gap (G)"))):
        v, (lo, hi) = L[k], L[f"{k}_ci95"]
        ax.bar(i, v, 0.55, color=["#dd8452", "#4c72b0"][i])
        ax.vlines(i, lo, hi, color="k")
        ax.text(i, max(v, hi) + 0.02, f"{v:.3f}", ha="center")
    ax.set_xticks([0, 1], ["one simulated reply (Δ1)", "full gap (G)"])
    ax.set_ylabel("probe score shift at the reply (L63)")
    ax.axhline(0, color="k", lw=0.6)
    r, (lo, hi) = L["R1"], L["R1_ci95"]
    bx.vlines(0, lo, hi, color="#dd8452", lw=2)
    bx.plot([0], [r], "o", color="#dd8452")
    bx.axhline(0.5, color="grey", ls=":")
    bx.text(0.06, r, f"{r:.2f}\n[{lo:.2f}, {hi:.2f}]", va="center")
    bx.set_xlim(-0.3, 0.6)
    bx.set_xticks([])
    bx.set_ylabel("R1 = Δ1 / G (0.5 = the bar)")
    bx.set_title(out["reading"], fontsize=10)
    fig.tight_layout()
    fig.savefig(dst.with_suffix(".png"), dpi=150)
    print(json.dumps({k: out[k] for k in ("reading", "check_self_swap", "check_planted", "layer_63", "lost", "identical_replies")}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
