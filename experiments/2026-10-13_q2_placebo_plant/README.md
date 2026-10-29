# 2026-10-13_q2_placebo_plant

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Placebo control for planting: does an edit at the same output sites and line positions, but changing a different number of characters, that the simulator does not make move the in-domain tool-output attack detector as much as planting the simulator's own editor formats into real runs did (-11.8 points)?

## What we did

Planted a placebo formatting change matched on output sites and changed lines, but not character count, into GPT-5.6-Sol's second real attack runs (845 runs, 7,337 outputs), re-extracted Qwen3-32B layers 48 and 63 and rescored with the frozen detector, compared against the earlier plant of the simulator's editor formats ([2026-10-12_plantslip_transfer](../2026-10-12_plantslip_transfer/)) and the whitespace plant ([2026-10-13_q2_row5_whitespace](../2026-10-13_q2_row5_whitespace/)). Modal GPU, \$20.51.

## What we found

placebo_delta_2026-10-13_104222.json: at layer 63, 5%: placebo moves catch -5.78 [-7.65, -3.99] against the simulator-format plant's -11.79, difference +6.02, reading 'in between'; at 1% -3.01 vs -9.13. At layer 48 the placebo moves catch up (+4.47 at 5%), opposite to the whitespace plant's -2.64. This placebo's layer-63 drop is about half the simulator-format plant's, descriptively; it does not establish a generic effect of any edit. The interventions also differ in information removal and can retain different outputs under the 32k token limit.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-12_plantslip_transfer](../2026-10-12_plantslip_transfer/) (planted editor slips); [2026-10-13_q2_row5_whitespace](../2026-10-13_q2_row5_whitespace/); the attack detector ([2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/))

## Main scripts

- [`placebo.py`](placebo.py)
- [`run_stage_placebo.py`](run_stage_placebo.py)
- [`run_score_placebo.py`](run_score_placebo.py)
- [`analyze_placebo.py`](analyze_placebo.py)
- [`run_sensitivity_intersection.py`](run_sensitivity_intersection.py)
- [`guard/guard_tasks_v2.py`](guard/guard_tasks_v2.py)

Result files the index names:

- `placebo_delta_2026-10-13_104222.json` (not copied: result left out to keep the release under 120 MB)
- [`placebo_scores_placebo_2026-10-13_102123.json`](results/placebo_scores_placebo_2026-10-13_102123.json)
- [`placebo_scores_controls_2026-10-13_024126.json`](results/placebo_scores_controls_2026-10-13_024126.json)
- `SENSITIVITY_intersection_2026-10-13_104529.json` (not copied: result left out to keep the release under 120 MB)
- `stage_placebo_2026-10-13_015900.json` (not copied: result left out to keep the release under 120 MB)

## Files here

8 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-13_q2_placebo_plant/` at commit `4e8190456d2689e5051a7ae887afba87f6a5194c` of our internal repository. 15 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
