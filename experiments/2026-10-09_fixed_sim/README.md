# 2026-10-09_fixed_sim

2026-10-09 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

The fixed-simulator test: if the simulator's systematic formatting errors are fixed by general rules, can a probe or text classifier still tell simulated runs from real ones, and what does it read?

## What we did

Rewrote simulated GPT-5.6-Sol runs with frozen general fix rules (real runs untouched; 9,440 of 26,397 simulated outputs changed), audited by hand, then ran a TF-IDF gate on held-out environments, a Qwen3-32B end-of-output probe (the main realism probe), every-token and max-pooled readouts, planted checks, LLM describe judges, a study of the residual after all seven cumulative text cleanups (times and dates, numbers and IDs, whitespace, harness format, diff lines, empty diff lines, editor output format) and catch-at-1% metrics.

## What we found

The fixed simulator fails the text bar (TF-IDF 0.661 [0.612, 0.744] on held-out environments, bar 0.55). The probe still separates (0.840 raw, 0.822 with times masked, vs 0.504 real-versus-real) but the first planted check fails. Catch at 1% only 2.7% [1.1, 20.8]. Thread closed.

## In the post?

Not in the post.

## Status

Closed; several sub-rows exploratory

## Builds on

[2026-10-06_q1_segment_end](../2026-10-06_q1_segment_end/); [2026-10-06_q1_text_baselines](../2026-10-06_q1_text_baselines/); [2026-10-06_all_layers_extraction](../2026-10-06_all_layers_extraction/)

## Main scripts

- [`fixes.py`](fixes.py)
- [`fixes_round2.py`](fixes_round2.py)
- [`run_fixed_sim_modal.py`](run_fixed_sim_modal.py)
- [`run_gate_a_modal.py`](run_gate_a_modal.py)
- [`run_claimB_output_scores_modal.py`](run_claimB_output_scores_modal.py)
- [`run_plant_oracle_modal.py`](run_plant_oracle_modal.py)
- [`run_describe_judge.py`](run_describe_judge.py)
- [`analyze_tpr_at_fpr.py`](analyze_tpr_at_fpr.py)
- [`level7_residual/run_level7_residual_modal.py`](level7_residual/run_level7_residual_modal.py)

Result files the index names:

- `tpr_at_fpr_2026-10-10_235259.json` (not in this folder at the source commit)
- `results_manifest_v20.json` (not copied: JSON outside results/ and config/)
- `threshold_stability_2026-10-11_002451.png` (not copied: result file type not copied (.png))
- `audit.md` (not copied: notes or README (internal working labels))
- `RULES.md` (not copied: notes or README (internal working labels))

## Files here

60 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-09_fixed_sim/` at commit `53181ef24fe5cb4174a49ea6f421be74e45c9df0` of our internal repository. 40 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
