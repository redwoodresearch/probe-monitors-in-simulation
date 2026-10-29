"""Print run_q1b.py --mode fulla's result JSONs (RULE.md Amendment 1). Reads the JSONs only.

  python print_q1b_fulla.py <folder holding q1b_<setting>_fulla.json files> [settings, default avg,lt]
"""
import json
import sys
from pathlib import Path

f = lambda d: f"{100 * d['value']:+.1f} [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}]"  # noqa: E731
NAMES = {"avg": "averaged, full + F9, L48", "lt": "last token, fix-full, L48", "savg": "Seed-OSS-36B, averaged, fix-full, L48",
         "slt": "Seed-OSS-36B, last token, fix-full, L48"}
for st in (sys.argv[2] if len(sys.argv) > 2 else "avg,lt").split(","):
    name = NAMES[st]
    r = json.loads((Path(sys.argv[1]) / f"q1b_{st}_fulla.json").read_text())
    print(f"== {name}: gate passes {r['gate_passes']} | {json.dumps(r['gate']['metrics'])} | expected {json.dumps(r['gate_expected'])}")
    for a, x in r["results"].items():
        ff = (f", fake/fake real {100 * x['fakefake_e2']:.1f} sim {100 * x['fakefake_sim']:.1f} D vs full (a) {100 * x['fakefake_D_full']:+.1f}"
              if "fakefake_D_full" in x else "")
        print(f"{a}: full-count (a) real catch {100 * x['a_full_e2']:.1f}, a_m real catch (draw mean) {100 * x['a_m_e2_draw_mean']:.1f}{ff}  (points)")
        for k in ("k0", "k10"):
            v = x[k]
            line = (f"  {k:4s} real {100 * v['real_catch_e2']['value']:.1f}, sim {100 * v['sim_catch']['value']:.1f} | gap {f(v['gap'])} | "
                    f"D vs full (a) {f(v['D_vs_full_a'])} | D vs a_m {f(v['D_vs_a_m'])}")
            if "Dfull_minus_plain" in v:
                line += f" | D vs full (a), k10 − plain {f(v['Dfull_minus_plain'])}"
            print(line)
