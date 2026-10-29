"""Writes README_F10_F11.md from the result JSONs, every number read by key (no hand-copied numbers).

  python3 experiments/2026-10-13_q2_fixfull/make_readme_f10_f11.py experiments/2026-10-13_q2_fixfull
"""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
R = D / "results"
q = json.loads((R / "f10f11_qwen3_POSTHOC_2026-10-13_223404.json").read_text())
s = json.loads((R / "f10f11_seed_POSTHOC_2026-10-13_223418.json").read_text())
pg = json.loads((R / "prefix_gate_f10_2026-10-13_220241.json").read_text())
ex = json.loads((R / "f10_real_tabs_2026-10-13_185536.json").read_text())["counts"]
st10 = json.loads((R / "stage_f10_stage_2026-10-13_175148.json").read_text())["counts"]
st11 = json.loads((R / "stage_f11_stage_2026-10-13_183525.json").read_text())["counts"]
g11 = json.loads((R / "g3_scores_f11_2026-10-13_222636.json").read_text())["arms"]


def p(x):
    return f"{x:+.2f}"


def iv(v):
    return f"{p(v[0])} [{p(v[1][0])}, {p(v[1][1])}]"


def gap(v):
    return iv(v["gap = second real - variant (points)"])


rows = []
for name, ro, al in (("Qwen3-32B L63, 1%", "output_end (layer 63)", "0.01"), ("Qwen3-32B L63, 5%", "output_end (layer 63)", "0.05"),
                     ("Qwen3-32B L48, 1%", "output_end@48 (layer 48)", "0.01"), ("Qwen3-32B L48, 5%", "output_end@48 (layer 48)", "0.05")):
    r = q["readouts"][ro][al]; v = r["variants"]; pc = r["paired change vs full + F9 (points, 95%)"]
    rows.append(f"| {name} | {iv(r['as-is gap = second real - as simulated (points)'])} | {gap(v['fullF9'])} | {gap(v['fullF9F10'])} | {gap(v['fullF9F11'])} | {iv(pc['fullF9F10'])} | {iv(pc['fullF9F11'])} |")
for al in ("0.01", "0.05"):
    r = s["rates"][al]; v = r["variants"]; pc = r["paired change vs full + F9 (points, 95%)"]
    rows.append(f"| Seed-OSS-36B, {float(al):.0%} | {iv(r['as-is gap = second real - as simulated (points)'])} | {gap(v['fullF9'])} | {gap(v['fullF9F10'])} | {gap(v['fullF9F11'])} | {iv(pc['fullF9F10'])} | {iv(pc['fullF9F11'])} |")

qh = q["false alarms, simulated honest"]; qp = q["false alarms, simulated honest: paired change vs full + F9 (points, 95%)"]
hrows = []
for k, lab in (("output_end (layer 63) @ 0.01", "Qwen3-32B L63, 1%"), ("output_end (layer 63) @ 0.05", "Qwen3-32B L63, 5%")):
    f9 = qh["fullF9"][k]["false alarms, fullF9 (points, 95%)"]
    a10 = qh["fullF9F10"][k]["false alarms, fullF9F10 (points, 95%)"]; a11 = qh["fullF9F11"][k]["false alarms, fullF9F11 (points, 95%)"]
    hrows.append(f"| {lab} | {f9[0]:.2f} | {a10[0]:.2f} | {a11[0]:.2f} | {iv(qp['fullF9F10'][k])} | {iv(qp['fullF9F11'][k])} |")
for al in ("0.01", "0.05"):
    h = s["rates"][al]["false alarms on simulated honest: full + F9 and own arms"]
    hrows.append(f"| Seed-OSS-36B, {float(al):.0%} | {h['fullF9']['share caught (points)']:.2f} | {h['fullF9F10']['share caught (points)']:.2f} | {h['fullF9F11']['share caught (points)']:.2f} | "
                 f"{iv(h['fullF9F10']['paired change vs full + F9 (points, 95%)'])} | {iv(h['fullF9F11']['paired change vs full + F9 (points, 95%)'])} |")

q1 = q["readouts"]["output_end (layer 63)"]["0.01"]["paired change vs full + F9 (points, 95%)"]
s1 = s["rates"]["0.01"]["paired change vs full + F9 (points, 95%)"]
qh10 = qp["fullF9F10"]["output_end (layer 63) @ 0.01"]; qh11 = qp["fullF9F11"]["output_end (layer 63) @ 0.01"]
sh = s["rates"]["0.01"]["false alarms on simulated honest: full + F9 and own arms"]
qg, sg = pg["qwen3-32b"], pg["seed-oss-36b"]
honest_changed = {k.split("|")[-1].removeprefix("environment "): v for k, v in ex.items() if k.startswith("real-honest|changed|environment")}
n_honest = ex["real-honest|changed|view"] + ex.get("real-honest|changed|undo_edit", 0)
left11 = len(g11["sim-honest-fx-fullF9F11"]["left out (main task not in the fit)"])

text = f"""# F10 and F11: frozen-detector results (POST HOC, PRELIMINARY until Aswin reviews)

Rules: `PLAN_F10.md` (Amendments 8, 8a–8c) and `PLAN_F11.md` (Amendment 1). Aswin, via 0d: F10 "only as its own arm"; matched
false alarms skipped for F10 and F11 ("descriptive only; they don't change the story"). Every number below is read by key
from the result files named here (this README is generated from them).

- **F10:** literal tabs in simulated text_editor replies expanded as the real tool does. It changed {st10['sim-attack|runs changed']} simulated attack
  and {st10['sim-honest|runs changed']} simulated honest runs.
- **F11:** simulated insert replies re-rendered as the real `cat -n` snippet. It changed {st11['sim-attack|runs changed']} attack and {st11['sim-honest|runs changed']} honest runs.
- **Each is its own arm:** full + F9 + F10, and full + F9 + F11, each full + F9 with that fix's changed runs swapped in by
  key. They are reported beside fix-full (full + F9) and never folded into the fix-full headline.

## Tables (frozen detectors, no refit)

Files: `results/f10f11_qwen3_POSTHOC_2026-10-13_223404.json` (Qwen3-32B, #36's reply detector) and
`results/f10f11_seed_POSTHOC_2026-10-13_223418.json` (Seed-OSS-36B, G3's frozen probes; 5% thresholds by #36's rule).

How to read the tables:
- **Gap** = catch(second real) − catch(arm), in points, over the 939 cells.
- **Paired change** = catch(arm) − catch(full + F9) on simulated runs.
- **Intervals** are 95% over 2,000 main-task draws (seed 20261012), with thresholds fixed.
- **Reproduction gate:** the full and full + F9 rows equal the committed fix-full tables exactly, attack and honest.

| Reader, rate | as-is gap | full + F9 | + F10 | + F11 | paired change, F10 | paired change, F11 |
|---|---|---|---|---|---|---|
""" + "\n".join(rows) + f"""

**False alarms on the simulated honest runs** (share caught, points; paired change vs full + F9):

| Reader, rate | full + F9 | + F10 | + F11 | paired change, F10 | paired change, F11 |
|---|---|---|---|---|---|
""" + "\n".join(hrows) + f"""

Seed-OSS-36B's F11 honest arm leaves out {left11} runs whose main task is not in the frozen fit.

## Predictions (written before any count)

**F10** (PLAN_F10.md):
- **Seed-OSS-36B, 1%:** closes 0.5 ± 1.0 points of full + F9's gap. Result: {iv(s1['fullF9F10'])} (paired change). **Met.** The paired
  interval also lies inside ±2, so the change is "small" under Amendment 8a.
- **Qwen3-32B L63, 1%:** 0 ± 1. Result: {iv(q1['fullF9F10'])}. **Met.**
- **Honest false alarms:** b7's tail check came back positive, so the committed branch applies. Qwen3-32B L63 at 1% falls by
  1 ± 1; result {iv(qh10)}. **Met, at the interval's edge.** Seed-OSS-36B: change under 1; result
  {iv(sh['fullF9F10']['paired change vs full + F9 (points, 95%)'])}. **Met.**
- **Changed runs:** honest 35–50 → {st10['sim-honest|runs changed']}, met; attack 70–95 → {st10['sim-attack|runs changed']}, met.
- **Reading:** expanding tabs changes nothing measurable at 1% for either reader.

**F11** (PLAN_F11.md):
- **Seed-OSS-36B, 1%:** closes 1.5 ± 2.0 points. Result: {iv(s1['fullF9F11'])}. **Met.**
- **Qwen3-32B L63, 1%:** the simulated − real gap moves by −1 ± 2 points. Result: {iv(q1['fullF9F11'])} (paired change in
  simulated catch). **Met.**
- **Honest false alarms at 1%:** Qwen3-32B falls by 1 ± 1.5, result {iv(qh11)}, **met**. Seed-OSS-36B falls by 0.5 ± 1.5, result
  {iv(sh['fullF9F11']['paired change vs full + F9 (points, 95%)'])}, **met**.
- **Changed runs:** attack 340–420 → {st11['sim-attack|runs changed']}, **missed**; honest 140–260 → {st11['sim-honest|runs changed']}, met.
- **Rendering:** "exact at least 70%" **missed**: most renders are approx_window (Amendment 1).
- **Caveat (PLAN_F11 Amendment 1), kept beside every F11 number:** most re-rendered inserts are approx_window, 1–2 lines
  narrower than the real snippet, and some diff-format inserts remain unrendered. F11 is therefore the real insert template
  with a slightly narrower window, not a byte-faithful real rendering.

## Gates

- **F10 prefix gate** (`results/prefix_gate_f10_2026-10-13_220241.json`): Qwen3-32B **{qg['gate']}**, Seed-OSS-36B **{sg['gate']}**.
  - Prefix positions matched: {qg['totals']['prefix positions']} (Qwen3-32B) and {sg['totals']['prefix positions']} (Seed), with 0 different tokens.
  - Same-GPU rows are bit-exact. Rows across H100 and B200 sit at cosine ≥ 0.9999997.
  - The 1e-3 tolerance is score-level (Amendment 8c). There, the scorers' prefix checks read 0.0.
- **Scoring checks:** Qwen3-32B passes V1 vs committed and the prefix check; its honest gates are all true. Seed-OSS-36B's
  reproduction gate passes. Records: `fx_frozen_POSTHOC_2026-10-13_221522.json` (F10), `…_222301.json` (F11),
  `fx_honest_scoring_*_2026-10-13_2224*/2225*.json`, `g3_scores_f10_2026-10-13_222539.json`, `g3_scores_f11_2026-10-13_222636.json`.

## Real tabbed outputs (F10's invariance control)

F10 would change {ex.get('real-attack-e1|changed|view', 0)} real e1, {ex.get('real-attack-e2|changed|view', 0)} real e2 and {n_honest} real honest
text_editor messages (`results/f10_real_tabs_2026-10-13_185536.json`, counts only). None starts with the real tool's
`cat -n` preamble, so these are not the tool's standard view output. Recorded as **source unidentified, mostly
file_indexing** ({honest_changed.get('file_indexing', 0)} of {n_honest} honest; honest by environment: {honest_changed}).

## How it ran

Everything ran on Modal CPU. `run_in_modal.py` runs the as-run scorers and table scripts byte-identical in one container,
with a git shim that replays the laptop's verified clean, pushed HEAD and its `git show` blobs. The table scripts are copies
with the own-arm additions: `analyze_f10f11_qwen3.py`, `analyze_f10f11_seed.py`, `score_qwen3_honest_f10f11.py`,
`g3_score_f10.py` and `g3_score_f11.py`. The post-launch Codex review is in `results/codex_postlaunch_f10_2026-10-13.md`.
"""
(D / "README_F10_F11.md").write_text(text)
print(len(text.splitlines()), "lines")
