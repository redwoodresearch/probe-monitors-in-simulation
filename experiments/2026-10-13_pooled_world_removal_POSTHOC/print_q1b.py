"""Print run_q1b.py's result JSONs as plain tables. Reads the JSONs only.

  python print_q1b.py <folder holding q1b_avg.json and q1b_lt.json>
"""
import json
import sys
from pathlib import Path

NAMES = {"avg": "averaged, full + F9, L48 (primary)", "lt": "last token, fix-full, L48 (secondary)"}
pct = lambda s: f"{100 * s['value']:+.1f} [{100 * s['ci95'][0]:+.1f}, {100 * s['ci95'][1]:+.1f}]"  # noqa: E731
for st in ("avg", "lt"):
    p = Path(sys.argv[1]) / f"q1b_{st}.json"
    if not p.exists():
        print(f"{NAMES[st]}: no result")
        continue
    r = json.loads(p.read_text())
    print(f"== {NAMES[st]}: gate passes {r['gate_passes']} | gate {json.dumps(r['gate'])} | expected {json.dumps(r.get('gate_expected'))}")
    if not r["gate_passes"]:
        continue
    print(f"primary k {r['primary_k']} | room {r['room']} | reading: {r['reading']}")
    print("held-out honest realism after removal (mean of 20 models) vs its label-shuffle null:")
    for k, v in r["realism_heldout"].items():
        print(f"  {k:4s} {v['observed']:.4f} | null mean {v['null_mean']:.4f}, 95th {v['null_p95']:.4f} | p above {v['p_above']:.4f}")
    for a, x in r["results"].items():
        extra = f", fake/fake real {100 * x['fakefake_e2']:.1f} sim {100 * x['fakefake_sim']:.1f}" if "fakefake_e2" in x else ""
        print(f"{a}: a_m real catch (draw mean) {100 * x['a_m_e2_draw_mean']:.1f}{extra}  (points)")
        for k, v in x.items():
            if not isinstance(v, dict):
                continue
            line = (f"  {k:4s} real {100 * v['real_catch_e2']['value']:.1f}, sim {100 * v['sim_catch']['value']:.1f} | gap {pct(v['gap'])} | "
                    f"D vs a_m {pct(v['D_vs_a_m'])}")
            if "gap_minus_plain" in v:
                line += f" | gap − plain {pct(v['gap_minus_plain'])} | D − plain {pct(v['D_minus_plain'])}"
            print(line)
