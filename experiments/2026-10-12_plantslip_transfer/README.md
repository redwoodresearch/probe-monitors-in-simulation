# 2026-10-12_plantslip_transfer

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Does a monitor's sensitivity to the planted output fixes (catch lost when the simulator's slips are planted into real runs) line up with its real minus simulated gap in the forest figure?

## What we did

Planted both editor slips into real runs for eight monitors (command and reply readers on Qwen3-32B, Qwen3.5-27B and Olmo-3.1-32B, on- and off-policy), re-extracted activations on Modal, scored with frozen probes and thresholds, and plotted planted catch loss against the forest gap.

## What we found

Only the reply readers on Qwen3-32B and Olmo-3.1-32B lose catch (9.13 and 4.10 points) and they have the largest gaps; the others lose about 0; rank correlation 0.586; descriptive. The planted trailing newline reaches Qwen3-32B and Olmo token ids but never Qwen3.5-27B.

## In the post?

Not in the post.

## Status

Exploratory (descriptive)

## Builds on

[2026-10-12_monitor_forest](../2026-10-12_monitor_forest/); [2026-10-12_q2_reverse_plant](../2026-10-12_q2_reverse_plant/); [2026-10-12_q2_template_slips](../2026-10-12_q2_template_slips/); [2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/)

## Main scripts

- [`plant_both.py`](plant_both.py)
- [`run_plant_splice.py`](run_plant_splice.py)
- [`run_plantslip_onpolicy.py`](run_plantslip_onpolicy.py)
- [`q2_reply_stage.py`](q2_reply_stage.py)
- [`q2_reply_score.py`](q2_reply_score.py)
- [`q2_reply_delta.py`](q2_reply_delta.py)
- [`q2_cmd/run_score_cmd.py`](q2_cmd/run_score_cmd.py)
- [`check_reach_full.py`](check_reach_full.py)

Result files the index names:

- [`summary_onpolicy_2026-10-12_145842.json`](results/summary_onpolicy_2026-10-12_145842.json)
- [`refit_2026-10-12_142646.json`](results/refit_2026-10-12_142646.json)

## Files here

28 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-12_plantslip_transfer/` at commit `60ebad77aa1ac138c3f40498427c8d5a65813aea` of our internal repository. 39 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
