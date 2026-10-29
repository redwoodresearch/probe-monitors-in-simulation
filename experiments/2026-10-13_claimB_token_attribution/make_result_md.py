"""Token attribution: the numbers block of RESULT.md, printed from the result JSON by key (no number typed by hand).

  python3 experiments/2026-10-13_claimB_token_attribution/make_result_md.py results/token_attr_<stamp>.json
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}" if isinstance(x, float) else str(x)


def tok_table(rows, role, k=10):
    out = [f"| token | n fake / e1 / e2 | T_{role} | T random max | m fake / e1 / e2 | sites | tasks | candidate |", "|---|---|---|---|---|---|---|---|"]
    for r in rows[:k]:
        n, m = r["n"], r["m"]
        out.append(f"| `{json.dumps(r['token'])[1:-1]}` | {n.get('fake', 0)} / {n.get('e1', 0)} / {n.get('e2', 0)} | {f(r['T_' + role], 2)} | "
                   f"{f(r['T_random_max'], 2)} | {f(m.get('fake'))} / {f(m.get('e1'))} / {f(m.get('e2'))} | {r['sites']} | {r['tasks']} | {r['candidate']} |")
    return "\n".join(out)


def main(path):
    d = json.loads((HERE / path).read_text())
    g, rp = d["gate"], d["refit"]
    L = [f"Numbers printed by make_result_md.py from `{path}` (git {d['git'][:8]}); preliminary until reviewed.", "",
         "**Refit reproduction (pre-condition, 1e-6):** " + ", ".join(f"{k} max abs diff {v['max_abs_diff']:.2e}" for k, v in rp["directions"].items())
         + f"; pass = {rp['pass']}.", "",
         "**Gate (gutter plants, plant-2 direction of the other half; PASS iff G <= 0.01 and every random G > 0.01):**", "",
         "| direction | G (best planted) | G gutter lines only | G final newline only | newline best in |", "|---|---|---|---|---|"]
    for k in ["plant2"] + [f"plant2_random{j}" for j in range(1, 6)] + ["main"] + [f"main_random{j}" for j in range(1, 6)]:
        v = g[k]
        L.append(f"| {k} | {f(v['G'], 4)} | {f(v['G_gutter_lines_only'], 4)} | {f(v['G_final_newline_only'], 4)} | {v['newline_is_best_planted']} of {len(v['per_reply_r'])} |")
    L += ["", f"Gate pass = **{g['pass']}**.", ""]
    for k, v in d["note_descriptive"].items():
        L.append(f"Review note (descriptive), {k}: planted minus twin at output_end mean {f(v['output_end_planted_minus_twin']['mean'])}, "
                 f"at the window's last token mean {f(v['window_last_planted_minus_twin']['mean'])}; template tail "
                 + ", ".join(f"{t} {f(x['mean'])} ({'/'.join(json.dumps(y)[1:-1] for y in x['tokens_planted'])})" for t, x in v["template_tail"].items()) + ".")
    if "main" in d:
        m = d["main"]
        re_ = m["reply_end"]
        L += ["", f"**Main direction, reply end (output_end):** mean score {', '.join(f'{a} {f(x)}' for a, x in re_['mean_score'].items())}; "
              f"AUROC fake vs first real {f(re_['auroc_fake_vs_first_real'])}, second vs first real {f(re_['auroc_second_vs_first_real'])}.", ""]
        for part in ("all_sites", "new_sites", "natural_sites"):
            p = m[part]
            L.append(f"- {part} ({p['sites']} sites): candidates in the fake top 20 {p['candidates_fake']} (known slips excluded "
                     f"{p['candidates_fake_excluded']}); second-real control {p['candidates_control']}.")
        L += ["", "Top 10 token strings by total push in fake site replies (all tokens):", "", tok_table(m["all_sites"]["top_fake_all_tokens"], "fake"),
              "", "Same, known-slip tokens excluded (the \"unnamed tells\" list):", "", tok_table(m["all_sites"]["top_fake_known_slips_excluded"], "fake"),
              "", "Second-real control, top 10 by total push in second-real replies:", "", tok_table(m["all_sites"]["control_top_second_real"], "e2"),
              "", f"Whole runs ({m['whole_runs']['runs_per_arm']} per arm), top 10 in simulated runs, known-slip tokens excluded:", "",
              tok_table(m["whole_runs"]["top_simulated_known_slips_excluded"], "fake")]
    for extra in extras:
        x = json.loads((HERE / extra).read_text())
        if "verdict" in x:
            L += ["", f"**POST HOC amendment 1** (`{extra}`, git {x['git'][:8]}): plant-2 planted minus twin at output_end on {x['pairs']} gutter pairs: "
                  f"planted > twin {x['planted_gt_twin']}, < {x['planted_lt_twin']}, ties {x['ties']}; one-sided sign test p = {x['sign_test_p_one_sided']:.2g}; "
                  f"mean {f(x['mean_diff'], 2)}, median {f(x['median_diff'], 2)}; verdict **{x['verdict']}**: \"{x['closing']}\"."]
        else:
            L += ["", f"**Throughput** (`{extra}`): tokens per second "
                  + ", ".join(f"{k} {v}" for k, v in x["tokens_per_second"].items())
                  + f"; rows assembled at {x['rows_assembled_s']} s, reader loaded at {x['reader_loaded_s']} s, done at {x['done_s']} s of "
                  f"{x['app_seconds']} s app time (GPU idle until the reader loaded: {x['gpu_idle_share_of_app_time']:.1%})."]
    print("\n".join(L))


if __name__ == "__main__":
    extras = sys.argv[2:]
    main(sys.argv[1])
