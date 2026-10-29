"""Write RESULT.md for the hidden-errors check from its result JSON (numbers read by key; the JSON is on S3 behind
results_manifest_v1.json and its sha256 is checked first).
  uv run --no-project --python 3.12 python experiments/2026-10-12_claimB_halves_errors/make_result_md.py <result json>
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
NAME = "hidden_errors_POSTHOC_2026-10-12_102409.json"
raw = Path(sys.argv[1]).read_bytes()
if hashlib.sha256(raw).hexdigest() != json.loads((HERE / "results_manifest_v1.json").read_text())["files"][f"results/{NAME}"]["sha256"]:
    raise SystemExit("the result JSON does not match results_manifest_v1.json")
d = json.loads(raw)
p = d["primary"]
pt = lambda x: f"{100 * x:+.2f}"
iv = lambda v: f"{pt(v['value'])} pts [{pt(v['ci95'][0])}, {pt(v['ci95'][1])}]"
def row(g, k):
    x = p[g][k]
    return f"sim {x['flagged_sim']}/{x['n_sim']}, e2 {x['flagged_e2']}/{x['n_e2']}, gap sim − e2 {iv(x['G'])}"
pl, dl, cf = p["placebo"], p["delta_G_E_minus_G_N"], p["confound"]
sg, l0 = d["secondary_grouping"], d["level0"]
f0, e0 = l0["fixed_vs_e1"], l0["e2_vs_e1"]
assert f0["gate"]["got"] == f0["gate"]["want"] and e0["gate"]["got"] == e0["gate"]["want"]
pc = lambda v: f"{100 * v['value']:.2f}% [{100 * v['ci95'][0]:.2f}, {100 * v['ci95'][1]:.2f}]"
md = f"""# Hidden tool errors and the halves gap (POST HOC; preliminary until reviewed)

Question (Aswin, about 10:08 PT 10-12, via 0d): does the halves gap at the frozen 1-in-100 alarm (fixed sim 51/939 =
5.4% flagged vs second-real e2 15/939 = 1.6%) come from tasks where the real run shows a tool error? Rule: `RULE.md`
(b5194e30) and script (78916efd), both committed at 10:22:35 PT, before this saved result (10:24:09 PT). The rule
was not blind to the group sizes: `experiments/2026-10-07_sim_failure_hiding/README.md` (e76432bb, lines 51 and 116)
already reported that 47.9% of e1 runs have a structured tool error, and 379 of 920 cells a non-parsing one. Result:
`results/{NAME}` on S3 (`results_manifest_v1.json`); this file is written from it by `make_result_md.py`.

**Grouping.** {p['grouping']}. Group E: the cell's real e1 run has a recorded structured tool error; group N: it has
**no recorded structured tool error**. Sim and e2 runs take their cell's group.

**Reading under the committed rule: "{p['reading']}"** (RULE.md's label; it means the gap persists in cells with no
recorded structured tool error). It rests on the **pooled** interval, as the rule specifies:
- N, pooled: {row('N', 'pooled')}; {p['N']['pooled']['cells']} cells, {p['N']['pooled']['main_tasks']} tasks.
- N by direction: dev → test {iv(p['N']['dev_to_test']['G'])}; test → dev {iv(p['N']['test_to_dev']['G'])}. Neither
  direction alone has a lower bound strictly above 0.
- E, pooled: {row('E', 'pooled')}; {p['E']['pooled']['cells']} cells, {p['E']['pooled']['main_tasks']} tasks.
- E − N, pooled: {iv(dl['pooled'])}. It includes 0, and the observed difference lies inside the range of
  the central 95% range of {pl['splits']:,} seeded placebo splits ({pt(pl['delta_range_95'][0])} to {pt(pl['delta_range_95'][1])} pts), so the
  gap is not shown to concentrate in cells with a recorded error.
- Plant 2 passes in both groups (pooled AUROC {p['planted_plant2']['E']['pooled_auroc']:.3f} in E,
  {p['planted_plant2']['N']['pooled_auroc']:.3f} in N). Yardstick e2 − e1: {iv(p['E']['pooled']['Y'])} in E,
  {iv(p['N']['pooled']['Y'])} in N.
- Power: {p['power']['E']['expected_sim_flags_at_5.4pct']:.1f} and {p['power']['N']['expected_sim_flags_at_5.4pct']:.1f}
  expected sim flags at 5.4% in E and N.

**Secondary grouping** (RULE.md line 33; {sg['grouping']}): N {iv(sg['N']['pooled']['G'])}
({sg['N']['pooled']['cells']} cells), E {iv(sg['E']['pooled']['G'])} ({sg['E']['pooled']['cells']} cells); both lower bounds above 0.

**Secondary: step A level 0** (RULE.md lines 74–81; labelled, no verdict, since level 0 has no direct e2 scores):
{l0['label']}. Both gates reproduced: fixed sim vs e1 caught {f0['gate']['got'][0]}/{f0['gate']['got'][2]} with {f0['gate']['got'][1]} false alarms;
e2 vs e1 {e0['gate']['got'][0]}/{e0['gate']['got'][2]} with {e0['gate']['got'][1]} (as required). Fixed-sim vs e1 detector, flagged per group: sim {pc(f0['rates']['pos_E'])} in E vs
{pc(f0['rates']['pos_N'])} in N; e1 {pc(f0['rates']['neg_E'])} vs {pc(f0['rates']['neg_N'])}. Separately trained e2-vs-e1 detector: e2
{pc(e0['rates']['pos_E'])} vs {pc(e0['rates']['pos_N'])}; e1 {pc(e0['rates']['neg_E'])} vs {pc(e0['rates']['neg_N'])}. ({f0['n']['pos_E']} and {f0['n']['pos_N']} cells; the catch
rows' cross-fit draws.)

**Caveats.**
- Group N means no recorded structured tool error, not no failure the reader saw. Per
  `experiments/2026-10-07_sim_failure_hiding/README.md` (e76432bb, lines 7–9 and 50–57), the simulator writes tool
  errors as plain text without setting Inspect's `error` field; counted in the text, 85.6% of sim runs and 86.3% of
  real e1 runs show a failed call. Those text counts are that README's, not this result's.
- The grouping counts errors on any tool output of the e1 run, including outputs past the reader's 32,768-token cut,
  which neither the detector nor any scored readout saw. A grouping restricted to the read prefix was not run.
- Groups are selected, not random: cells in E are longer (median e1 tool outputs {cf['E']['median_e1_tool_outputs']:.0f}
  vs {cf['N']['median_e1_tool_outputs']:.0f}).

**Inputs and rerun.** The score inputs (`runs.json`, the per-direction `fixed_vs_e1.npz` and `plant2.npz`, and the
direct-e2 `POSTHOC_e2/*/fixed_vs_e1.npz`, from the Volume folder `claimB/halves_v1/5b07509a98fc`) are on S3 behind
`results_manifest_v2.json`, each equal to the sha256 the result records. The `--pack-runs` file (step A's
`packed_39f842231a46/runs.L63.json`) was not recorded by the script; every copy on the machine that ran it has sha256
6957bc60…, the object on S3 behind `experiments/2026-10-11_claimB_h2_probe_best_found/results_manifest_v5.json`. The
script stays as run (78916efd): it writes to `results/` without creating it, and imports halves' code from a
`git archive` of 53da7f48, which this PR's provenance merge makes reachable.
"""
(HERE / "RESULT.md").write_text(md)
print(md)
