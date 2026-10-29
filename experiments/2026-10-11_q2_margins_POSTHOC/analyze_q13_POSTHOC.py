"""Q1.3 (post hoc, PLAN_POSTHOC.md): Qwen's behaviour in simulated vs real attack runs at 196 tasks, against two margin
sets (set 1: the forecast widths, repurposed; set 2: Aswin's stated resolution), with Holm over the run 1 minus sim
gaps, run 1 minus run 2 (and honest A minus B) beside them, and a plant sweep per measure. Gaps are computed exactly as
analyze_behavior.py does (mean per-task paired difference, 10,000 task-bootstrap draws, seed 20261010) and reproduce
the stored gap file before anything else.

  uv run --no-project --with numpy --with pandas python experiments/2026-10-11_q2_margins_POSTHOC/analyze_q13_POSTHOC.py \
      <per_run_scaleup_counted.csv> <behavior_gaps_2026-10-10_155310.json>
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from probe_shift.margins import LEVEL, difference_p, equivalence_p, holm, inside  # noqa: E402
from probe_shift.planted import detected  # noqa: E402

HERE = Path(__file__).parent
SEED, DRAWS, COPIES = 20261010, 10_000, 200
SET1 = {"sh_edit": 0.02, "reread_rate": 0.05, "sh_failtext": 0.02, "n_steps": 8, "tok_out": None}  # tok_out: 15% of run 1's mean
SET2 = {"sh_failtext": 0.01, "n_steps": 5}
SHARES = {"sh_edit", "reread_rate", "sh_failtext"}
FACTORS = (0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2)


def load(csv):
    df = pd.read_csv(csv)
    df = df[~df.sample_error & (df.n_calls > 0)].copy()
    df["sh_edit"] = (df.n_create + df.n_str_replace + df.n_insert) / df.n_calls
    df["sh_failtext"] = df.n_failtext / df.n_out
    df["reread_rate"] = df.rereads / df.edits.replace(0, np.nan)
    second = df.launch.str.endswith("_second_attack")
    arms = {"run1": df[(df.arm == "attack") & ~second], "run2": df[(df.arm == "attack") & second],
            "sim": df[(df.arm == "sim_attack") & (df.simulator == "september")], "honA": df[df.arm == "honest_a"], "honB": df[df.arm == "honest_b"]}
    for a, d in arms.items():
        assert d.task.is_unique, f"{a}: a task appears twice"
    return {a: d.set_index("task") for a, d in arms.items()}


def gap(a, b, m):
    d = (a[m] - b[m]).dropna(); x = d.to_numpy()
    boot = x[np.random.default_rng(SEED).integers(0, len(x), (DRAWS, len(x)))].mean(1)
    return float(x.mean()), boot, d.index


def pct(x):
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def main(csv: Path, stored_path: Path):
    arms = load(csv); stored = json.loads(stored_path.read_text())["gaps"]
    out = {"inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (csv, stored_path)}, "level for inside after Holm": LEVEL}
    fam = {}
    for m in SET1:
        res = {}
        for a, b in (("run1", "sim"), ("run1", "run2"), ("run2", "sim"), ("honA", "honB")):
            p, boot, idx = gap(arms[a], arms[b], m)
            ref = stored[f"{m}: {a} - {b}"]
            if np.abs(np.round([p, *pct(boot)], 4) - np.array(ref[:3])).max() > 1e-9 or len(idx) != ref[3]:
                raise ValueError(f"{m} {a}-{b}: {p}, {pct(boot)} does not reproduce the stored {ref}")
            res[f"{a} - {b}"] = (p, boot, idx)
        p, boot, idx = res["run1 - sim"]
        m1 = SET1[m] if SET1[m] is not None else 0.15 * float(arms["run1"].loc[idx, m].mean())
        o = {"margin set 1": m1, "margin set 2": SET2.get(m), "paired tasks": len(idx)}
        for k, (pk, bk, ik) in res.items():
            o[k] = {"point": pk, "95%": pct(bk), "tasks": len(ik), "inside set 1": inside(pct(bk), m1)}
            if m in SET2:
                o[k]["inside set 2"] = inside(pct(bk), SET2[m])
        # Plant sweep into the simulated arm, widening run 1 - sim (sign of the unplanted gap)
        sign = 1 if p >= 0 else -1; sim = arms["sim"][m].astype(float); tasks = sim.index.to_numpy(); sweep = {}
        for f in FACTORS:
            size = f * m1; hits = 0
            for c in range(COPIES):
                hit = np.random.default_rng(c).choice(tasks, len(tasks) // 2, replace=False)
                new = sim.copy(); new.loc[hit] = new.loc[hit] - sign * 2 * size
                new = new.clip(0, 1) if m in SHARES else new.clip(lower=0)
                pp, bp, ip = gap(arms["run1"], new.to_frame(m), m)
                exp = -float((new.loc[ip] - sim.loc[ip]).mean())
                hits += exp != 0 and detected(pp, tuple(pct(bp)), p, exp)
            sweep[f"{size:.4g}"] = hits / COPIES
        o["plant sweep (detection rate per size, run 1 - sim)"] = sweep
        o["smallest size detected in at least 80% of copies"] = next((float(z) for z, r in sweep.items() if r >= 0.8), None)
        out[m] = o; fam[m] = boot
    for tag, margins in (("set 1", {m: out[m]["margin set 1"] for m in fam}), ("set 2", SET2)):
        eq = holm({m: equivalence_p(fam[m], margins[m]) for m in margins})
        for m in margins:
            out[m][f"{tag}: equivalence p"] = equivalence_p(fam[m], margins[m])
            out[m][f"{tag}: Holm-adjusted equivalence p"] = eq[m]; out[m][f"{tag}: inside after Holm"] = eq[m] < LEVEL
    dp = holm({m: difference_p(b) for m, b in fam.items()})
    for m in fam:
        out[m]["difference p"] = difference_p(fam[m]); out[m]["Holm-adjusted difference p"] = dp[m]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"q13_margins_POSTHOC_{stamp}.json"; dest.parent.mkdir(exist_ok=True)
    with open(dest, "x") as fh:
        fh.write("{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
    for m in fam:
        o = out[m]; g = lambda d: f"{d['point']:+.4g} [{d['95%'][0]:+.4g}, {d['95%'][1]:+.4g}]"
        s2 = f"; set 2 ±{o['margin set 2']:g}: inside {o['run1 - sim']['inside set 2']}, Holm p {o['set 2: Holm-adjusted equivalence p']:.4f}" if m in SET2 else ""
        print(f"{m}: run1-sim {g(o['run1 - sim'])}; run1-run2 {g(o['run1 - run2'])}; set 1 ±{o['margin set 1']:.4g}: inside {o['run1 - sim']['inside set 1']}, "
              f"Holm p {o['set 1: Holm-adjusted equivalence p']:.4f}{s2}; diff Holm p {o['Holm-adjusted difference p']:.3f}; 80% plant {o['smallest size detected in at least 80% of copies']}")
    print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
