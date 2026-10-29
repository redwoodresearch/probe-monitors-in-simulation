"""Write RESULT.md for the Qwen3.5 halves re-run from its result JSON (numbers read by key)."""
import json, sys
from pathlib import Path
D = Path("experiments/2026-10-11_claimB_halves_qwen35")
RES = "results/halves_qwen35_2026-10-12_005827.json"
d = json.load(open(D / RES))
r = d["results"]
dt, td, po, cmp = r["dev_to_test"], r["test_to_dev"], r["pooled"], r["qwen35_vs_qwen3_32b"]
pts = lambda x: f"{100 * x:+.2f}"
ci = lambda c: f"[{pts(c[0])}, {pts(c[1])}]"
def plant(x):
    p = x["planted"]
    return f"AUROC {p['auroc']:.3f} against the {p['floor']:.3f} floor (refit-null p {p['p']:.3f}): **{'pass' if p['pass'] else 'FAIL'}**"
def gap(x):
    c = x["catch_1pct"]
    return f"{pts(c['gap'])} pts {ci(c['gap_ci95'])}, sim null p {c['sim_null_p']:.4f}"
md = f"""# Halves on Qwen3.5-27B: result (EXPLORATORY; preliminary until reviewed)

Plan: `PLAN.md` here. Result: `{RES}` (analysis `analyze_halves_qwen35.py` at {d['git'][:8]}). Every number below is
read from that file by key by `make_result_md.py` here. Codex's fact-check record is
`results/codex_factcheck_halves_qwen35_2026-10-12_005827.txt`.

**The dev → test planted check failed.** Plant 2 on the held-out half scored {plant(dt)}. Test → dev's passed:
{plant(td)}. The plan reads a direction whose planted check fails as descriptive only, and the pooled reading needs both
to pass, so **nothing here is a confirmation**.

**Registered reading (the plan's C2 line): "{d['c2_verdict']}".** Catch at 1%, simulated runs flagged minus a second
real run (e2) flagged, at the threshold set on held-out real e1 runs:
- **Pooled:** {gap(po)}. Descriptive only (a planted check failed).
- **Dev → test (planted check failed):** {gap(dt)}: "{dt['catch_reading']}".
- **Test → dev (planted check passed):** {gap(td)}: "{td['catch_reading']}".

**Beside it, no verdict:**
- **Direct-e2 yardstick** (e2 scored by the sim-trained probe at the same threshold): pooled {pts(po['catch_1pct']['direct_gap'])} pts
  {ci(po['catch_1pct']['direct_gap_ci95'])}; dev → test {pts(dt['catch_1pct']['direct_gap'])} {ci(dt['catch_1pct']['direct_gap_ci95'])}; test → dev
  {pts(td['catch_1pct']['direct_gap'])} {ci(td['catch_1pct']['direct_gap_ci95'])}. Every interval includes 0.
- **Qwen3.5 against Qwen3-32B:** pooled gaps {pts(cmp['qwen35_pooled_gap'])} and {pts(cmp['qwen3_32b_pooled_gap'])} pts; difference
  {pts(cmp['difference'])} {ci(cmp['difference_ci95'])} ({cmp['note']}).
- **AUROC, fixed sim vs e1:** dev → test {dt['auroc']['fixed_vs_e1']:.3f} (e2 vs e1 {dt['auroc']['e2_vs_e1']:.3f}); test → dev
  {td['auroc']['fixed_vs_e1']:.3f} (e2 vs e1 {td['auroc']['e2_vs_e1']:.3f}).

**Reading.** The pooled gap's lower bound is just above 0 and its sim null p is just under 0.05. Test → dev shows a gap
beyond real-run noise; dev → test, where the planted check failed, does not. The direct yardstick includes 0 in every
direction. Under the plan's table, a failed planted check makes this descriptive only: not a confirmation.
"""
(D / "RESULT.md").write_text(md)
print(md)
