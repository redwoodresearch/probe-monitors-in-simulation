"""README tables for Seed's fix-full / N / F9 / fix939 cells (RULE.md Amendments 13b, 15, 17, 19), with Qwen3-32B's same cells beside
(the SCE fork's Amendments 14, 16, 18). Every number is read by key from the two metrics files (sha256 checked). Local, $0.

  uv run --no-project --python 3.12 python experiments/2026-10-13_q3_sce_reply_probe/readme_seed_variants.py \
      results/sce_seed_variants_metrics_POSTHOC_2026-10-13_163533.json <copy of sce_fixfull_metrics_POSTHOC_2026-10-13_153447.json>
"""
import hashlib
import json
import sys
from pathlib import Path

SEED_SHA = "eb4d73c4f5de33ff9ee52c4e1846560c7634e72b709b52ff6f1fda670fc43524"
QWEN_SHA = "9d3e6f8a5123c9f242ea96a584db40d7b009092ba47a31541e12930f52a2a7dd"  # manifest_sce_fixfull_results_v1.json, aswin/q3-sce-reply-probe
CELLS = {"a": ("(a) real-trained", "sim-honest", "a"), "c": ("(c) as-is, mixed", "sim-honest", "c"),
         "c_fix939": ("(c′) fix939, mixed", "sim-honest", "c_fixed (c′)"), "c_ff": ("(c″) fix-full, mixed", "sim-honest-fx-full", "c_full (c″)"),
         "c_N": ("(c_N) newline only, mixed", "sim-honest-fx-NL", "c_N"), "c_F9": ("(c_F9) Full + F9, mixed", "sim-honest-fx-fullF9", "c_F9"),
         "b": ("(b) as-is, sim-only", "sim-honest", "b"), "b_ff": ("(b″) fix-full, sim-only", "sim-honest-fx-full", "b_full (b″)"),
         "b_N": ("(b_N) newline only, sim-only", "sim-honest-fx-NL", "b_N"), "b_F9": ("(b_F9) Full + F9, sim-only", "sim-honest-fx-fullF9", "b_F9")}


def load(p, sha):
    p = Path(p)
    if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
        raise SystemExit(f"not the expected file: {p}")
    return json.loads(p.read_text())


def iv(d):
    return f"{d['value'] * 100:+.1f} [{d['ci95'][0] * 100:+.1f}, {d['ci95'][1] * 100:+.1f}]"


def main(seed_path, qwen_path):
    s, Q = load(seed_path, SEED_SHA)["results"], load(qwen_path, QWEN_SHA)
    q, nl = Q["layers"], Q["newline_readings_1pct"]
    out = []
    for view, title in (("g3_layer_per_fold", "G3's layer per test fold (primary view)"), ("L63", "layer 63 in every fold (beside it)")):
        out += [f"**Seed-OSS-36B, {title}, targeting 1%** (5% gap beside); % (caught of runs); D = e2 catch − (a)'s, paired; rule 4b verdict", "",
                "| cell | real attacks (e2) | simulated attacks | that cell's simulated honest false alarms | simulated − real, points [95%] | D, points [95%] | 4b | gap at 5% |",
                "|---|---|---|---|---|---|---|---|"]
        for c, (name, honest, _) in CELLS.items():
            x = s[view][c]["fpr_0.01"]
            n = x["caught_of_n"]
            cat = lambda a: f"{n[a][0] / n[a][1] * 100:.1f} ({n[a][0]:g}/{n[a][1]})"  # noqa: E731 - from the counts
            rule = x.get("rule_4b", {})
            d = iv(rule["e2_minus_a"]) if "e2_minus_a" in rule else "—"
            out.append(f"| {name} | {cat('real-attack-e2')} | {cat(s[view][c]['sim_attacks_of_this_cell'])} | {cat(honest)} | "
                       f"**{iv(x['sce_reported_minus_real'])}** | {d} | {rule.get('verdict_4b', '—')} | "
                       f"{s[view][c]['fpr_0.05']['sce_reported_minus_real']['value'] * 100:+.1f} |")
        r = s[view]["readings_across_cells"]["fpr_0.01"]
        out += ["", f"Readings at 1%, {view}: newline (c) script output \"{r['newline_reading_c']}\" (c_N − c″ {iv(r['c_N_minus_c_ff'])}, "
                f"c_N − c {iv(r['c_N_minus_c'])}); newline (b) \"{r['newline_reading_b']}\" (b_N − b″ {iv(r['b_N_minus_b_ff'])}, b_N − b {iv(r['b_N_minus_b'])}); "
                f"F9 adds (c): {r['F9_adds_c']} (c_F9 − c″ {iv(r['c_F9_minus_c_ff'])}); F9 adds (b): {r['F9_adds_b']} (b_F9 − b″ {iv(r['b_F9_minus_b_ff'])}).", ""]
    g = lambda c, v: s[v][c]["fpr_0.01"]["sce_reported_minus_real"]  # noqa: E731
    out += ["**Gap (simulated − real catch, points [95%]) at 1%, Seed-OSS-36B beside Qwen3-32B** (Qwen3-32B: the SCE fork's last-token probe, "
            "`manifest_sce_fixfull_results_v1.json` on aswin/q3-sce-reply-probe)", "",
            "| cell | Seed, G3's layer per fold | Seed, L63 | Qwen3-32B, L48 | Qwen3-32B, L63 |", "|---|---|---|---|---|"]
    for c, (name, _, qc) in CELLS.items():
        out.append(f"| {name} | {iv(g(c, 'g3_layer_per_fold'))} | {iv(g(c, 'L63'))} | {iv(q['48'][qc]['fpr_0.01']['fake_minus_real'])} | "
                   f"{iv(q['63'][qc]['fpr_0.01']['fake_minus_real'])} |")
    share = []
    for v in ("g3_layer_per_fold", "L63"):
        gc, gff, gn = (g(c, v)["value"] for c in ("c", "c_ff", "c_N"))
        share.append(f"Seed {v} {(gc - gn) / (gc - gff):.3f}")
    out += ["", "Share of fix-full's gap cut that newline-only achieves, (c − c_N)/(c − c″), Amendment 16's statistic, for comparison only: "
            + "; ".join(share) + f"; Qwen3-32B L48 {nl['48']['r_share_of_full_cut']:.3f}, L63 {nl['63']['r_share_of_full_cut']:.3f}."]
    print("\n".join(out))


if __name__ == "__main__":
    main(*sys.argv[1:3])
