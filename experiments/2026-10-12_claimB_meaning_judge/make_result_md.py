"""Write RESULT.md for the meaning judge from its final result JSON and the cost-basis v2 result (numbers read by key).
The result JSON is on S3 behind results_manifest_v4.json, and its sha256 is checked first; so is the cost-basis file's.
  uv run --no-project --python 3.12 python experiments/2026-10-12_claimB_meaning_judge/make_result_md.py \\
      <meaning_result_2026-10-12_164829.json> [<cost_basis_2026-10-12_165335.json>]
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
NAME = "meaning_result_2026-10-12_164829.json"
COST = "cost_basis_2026-10-12_165335.json"  # cost-basis v2, committed at db71927d
COST_SHA = "09a4b2caf85cf1d746c3b67b388229810ea5a2a1d9be9f30e1a32263df01e712"


def load(path, want):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != want:
        raise SystemExit(f"{path} does not match its recorded sha256")
    return json.loads(raw)


man = json.loads((HERE / "results_manifest_v4.json").read_text())["files"]
d = load(sys.argv[1], man[f"experiments/{HERE.name}/results/{NAME}"]["sha256"])
cb = load(sys.argv[2] if len(sys.argv) > 2 else HERE / "results" / COST, COST_SHA)
sol, gem = d["judges"]["sol"], d["judges"]["gemini"]
f3 = lambda x: f"{x:.3f}"  # noqa: E731
iv = lambda v: f"[{v[0]:+.3f}, {v[1]:+.3f}]"  # noqa: E731


def row(label, r, verdict=True):
    cells = [label, str(r["sites"]), str(r["retained"]), f3(r["agree_sim_e1"]), f3(r["agree_e2_e1"]), f"{r['D']:+.3f}",
             iv(r["D_ci90"]), iv(r["D_ci95"]), f3(r["p_D_le_minus_margin"])]
    return "| " + " | ".join(cells + ([f'"{r["verdict"]}"'] if verdict else [])) + " |"


g = sol["planted_gate"]
gg = gem["planted_gate"]
strata = [("first command (k = 0)", "first_command"), ("later commands (k ≥ 1)", "later"),
          ("e1 and e2 from the same batch", "same_batch"), ("e1 and e2 from different batches", "mixed_batch"),
          ("the simulated run's reference holds the command", "reference_has_command"),
          ("the simulated run's reference lacks the command", "reference_lacks_command")]
head = "| Set | Sites | Retained | Agreement (sim, e1) | Agreement (e2, e1) | D | 90% interval | 95% interval | p(D ≤ −0.10) |"
md = f"""# Meaning judge result (unrepaired simulator, honest runs; preliminary until reviewed)

{d['label']}. Plan: `PLAN.md` (@ 2d35f421, amendment 1 @ eb0ff6d5). Final result: `results/{NAME}` (committed at
40dc3ffb, analysis v3 @ 6b4a669a, run git {d['git'][:8]}), on S3 behind `results_manifest_v4.json`; this file is written
from it by `make_result_md.py`. **Superseded interim files**, kept because results are append-only:
`meaning_result_2026-10-12_154506.json` (analysis v1, @ 1d12b342) and `meaning_result_2026-10-12_163423.json` (analysis
v2, @ e95abe8b), both on S3 behind the same manifest.

**What the numbers are.** A site is a point where the first real run (e1) and the second real run (e2) issue the same
command after the same earlier commands and outputs (PLAN.md). At the primary sites the n437 simulated run also matched
there, and its own reply is used; at the secondary sites the reply is a new call through the Oct-6 pipeline
(`run_simulate_pairs.py` @ cba62df4, same simulator and n437 inputs) on e1's history. The judge sees the command and two replies and
answers whether they report the same facts, in both orders. A pair's site score is the share of its two orders judged
the same; agreement is the mean over a set's sites. D = agreement(sim, e1) − agreement(e2, e1). Intervals are
percentiles of 2,000 main-task bootstrap draws. Verdict rule (`analyze_meaning.py` docstring): "as often, within 0.10"
if the 90% lower end is above −0.10; "less often, by at least 0.10" if the 95% upper end is below −0.10; else "too few".

## GPT-5.6 Sol (judge `sol`): {sol['statuses'].get('ok', 0)} of {sum(sol['statuses'].values())} jobs ok

**Planted check** (read first): {g['breaking_n']} meaning-breaking plants judged different {f3(g['judged_different'])}
(lower 95% bound {f3(g['lower95'])}); {g['controls_n']} controls judged the same {f3(g['controls_judged_same'])}; complete
{g['complete']}, pass **{g['pass']}**. The {g['excluded_9_descriptive']['n']} plants Codex excluded (descriptive only):
judged different {f3(g['excluded_9_descriptive']['judged_different'])}.

**Registered sets:**

{head} Verdict |
|---|---|---|---|---|---|---|---|---|---|
{row("primary: sites with the n437 run's own reply", sol["primary_33_natural"])}
{row("secondary: sites with a new simulator reply", sol["secondary_58_new"])}

Primary verdict after Holm over the judges whose primary verdict is read (Sol only): "{sol['primary_33_natural']['verdict_holm']}"
(adjusted p(D ≤ −0.10) {d['holm_primary']['p_le']['sol']}, p(D ≥ −0.10) {d['holm_primary']['p_ge']['sol']}).

**Descriptive (no verdict):**

{head}
|---|---|---|---|---|---|---|---|---|
{row("all strict sites pooled", sol["pooled_91"], False)}
""" + "\n".join(row(lab, sol[k], False) for lab, k in strata) + f"""

Oct-6 typical positions (no second real run): agreement (sim, e1) {f3(sol['oct6_descriptive']['agree_sim_e1'])} over
{sol['oct6_descriptive']['sites']} sites. Repeat check: {sol['retest']['pairs']} pairs, agreement {f3(sol['retest']['agreement'])}.

## Gemini 3.1 Pro (judge `gemini`): closed as (A) after 429s; not read

Standard-pin records only (google-vertex/global): statuses {json.dumps(gem['statuses'], sort_keys=True)}. Planted check
incomplete: {gg['breaking_n']} meaning-breaking plants answered, {gg['controls_n']} controls; complete {gg['complete']}.
Primary verdict: "{gem['primary_33_natural']['verdict']}". Its per-set fields in the JSON are not read. Sol–Gemini
agreement on the {d['sol_gemini_agreement']['jobs']} jobs both answered: {f3(d['sol_gemini_agreement']['agree'])}.

## Cost, billed (unique requests)

From `results/{COST}` (cost-basis v2, @ db71927d), where each judge's unique-request total equals its ledger:
Sol ${cb['judges']['sol']['unique_request_usd']:.2f} ({cb['judges']['sol']['unique_requests']} unique requests), Gemini
${cb['judges']['gemini']['unique_request_usd']:.2f} ({cb['judges']['gemini']['unique_requests']} unique requests, flex and standard).
The result JSON's `cost_usd` fields are record sums that count cache-served duplicates, not billed amounts. The
simulator's spend is in its `SPEND.md` row.

Also: the POST HOC identity counts and score cross-tables (descriptive) are in `results/identity_POSTHOC_2026-10-12_155010.json`.
"""
(HERE / "RESULT.md").write_text(md)
print(md)
