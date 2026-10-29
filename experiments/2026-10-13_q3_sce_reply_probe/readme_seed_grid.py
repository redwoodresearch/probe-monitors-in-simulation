"""README table for Seed's 2x2 grid (RULE.md Amendments 13 and 13b): every number read by key from the metrics file (sha256 checked). Local, $0.

  uv run --no-project --python 3.12 python experiments/2026-10-13_q3_sce_reply_probe/readme_seed_grid.py results/sce_seed_grid_metrics_POSTHOC_2026-10-13_145022.json
"""
import hashlib
import json
import sys
from pathlib import Path

SHA = "8d99fddc6c64b011"  # prefix of the metrics file's sha256
NAMES = {"a": "(a) real attack vs real honest", "b": "(b) simulated attack vs simulated honest", "c": "(c) simulated attack vs real honest",
         "d": "(d) real attack vs simulated honest", "c_fixed": "(c′) fix939 simulated attack vs real honest"}


def pct(v):
    return f"{v * 100:.1f}"


def main(p):
    p = Path(p)
    if not hashlib.sha256(p.read_bytes()).hexdigest().startswith(SHA):
        raise SystemExit("not the committed metrics file")
    d = json.loads(p.read_text())
    out = []
    for k, label in (("fpr_0.01", "1% (headline)"), ("fpr_0.05", "5% (beside it: each 1% threshold is the 2nd or 3rd highest validation honest score)")):
        out += [f"**{label}, G3's layer per test fold (primary view):**", "",
                "| cell | real attack caught (e2) | simulated attack caught | fix939 simulated attack caught | real honest false alarms | simulated honest false alarms | simulated − real gap (points) | e2 − e1 |",
                "|---|---|---|---|---|---|---|---|"]
        for c, name in NAMES.items():
            x = d["results"]["g3_layer_per_fold"][c][k]
            n = x["caught_of_n"]
            cat = lambda a: f"{pct(n[a][0] / n[a][1])}% ({n[a][0]:g} of {n[a][1]})"  # noqa: E731 - from the counts (Codex: no double rounding)
            fa = cat
            g = x["sce_reported_minus_real"]
            gl = "fix939 simulated" if c == "c_fixed" else "simulated"
            out.append(f"| {name} | {cat('real-attack-e2')} | {cat('sim-attack')} | {cat('sim-attack-fixed')} | {fa('real-honest')} | {fa('sim-honest')} "
                       f"| {g['value'] * 100:+.1f} [{g['ci95'][0] * 100:+.1f}, {g['ci95'][1] * 100:+.1f}] ({gl} − e2) | {x['yardstick_e2_minus_e1']['value'] * 100:+.1f} |")
        out.append("")
        r = {c: d["results"]["g3_layer_per_fold"][c][k]["reading"] for c in NAMES}
        fm = d["results"]["g3_layer_per_fold"]["c_fixed"][k]["fixed_minus_c_gap"]
        ro = d["results"]["g3_layer_per_fold"]["c_fixed"][k]["repaired_only"]["gap_repaired_only"]
        out += [f"Readings ({label.split(' ')[0]}): " + "; ".join(f"{c}: " + ", ".join(q for q, v in r[c].items() if v is True) for c in NAMES) + ".",
                f"(c′) − (c) gap {fm['value'] * 100:+.1f} [{fm['ci95'][0] * 100:+.1f}, {fm['ci95'][1] * 100:+.1f}] (repairs rescue: {fm['repairs_rescue']}); "
                f"(c′) on the 823 repaired runs only {ro['value'] * 100:+.1f} [{ro['ci95'][0] * 100:+.1f}, {ro['ci95'][1] * 100:+.1f}].", ""]
    l63 = d["results"]["L63"]
    out.append("Fixed layer 63 beside it, gap at 1%: " + "; ".join(f"{c} {l63[c]['fpr_0.01']['sce_reported_minus_real']['value'] * 100:+.1f}" for c in NAMES) + ".")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
