# 2026-10-14_priority1_spec_09

2026-10-14 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

The analysis plan for the supervisor's top-priority question: when Qwen3.5-27B attacks, is a probe's catch on its simulated runs within some margin X of its catch on its real runs, for probes trained in many ways? It became the plan governing every analysis through to the post's final numbers.

## What we did

The frozen analysis plan (a 4,980-line RULE.md with numbered appendices covering the quantity estimated, the groups of runs, each setting's own thresholds, the bootstrap, which probes are eligible, the scale-up, scoring in the agent's served context, probes trained on simulated runs and the on-policy probe trained on all tasks), side rules for a false-alarm check and a bundle of sanity checks, answers to questions from the 507-task alarm-level sweep, the gate checklists for the final pooled analysis on all 1,313 tasks (GATES_routec_2026-10-17.md), and that analysis's release records (versions 1 to 2.8) with the jq and zsh scripts that build and self-check them.

## What we found

final_release/routec_FINAL_release_v2.8_2026-10-26.json is the newest release record in the folder; the spec's numbered additions govern [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) (the final pooled analysis), whose records our table of the post's number sources lists for the post's main numbers.

## In the post?

In the post: Figure 1, Tables 5 and 8, Appendices C, D and H (the analysis rules and final release records of the pooled analysis behind them)

## Status

Current

## Builds on

All the 10-14 analysis folders it governs; [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/)

## Main scripts

- RULE.md
- GATES_routec_2026-10-17.md
- [`fa_check.py`](fa_check.py)
- final_release build and selfcheck scripts ([`build_v2_4.zsh`](final_release/v2_4_tooling/build/build_v2_4.zsh), build_v2_5.zsh, tooling_gate.zsh)

Result files the index names:

- `routec_FINAL_release_v1_2026-10-20.json` (not copied: JSON outside results/ and config/)
- `routec_FINAL_release_v2.8_2026-10-26.json` (not copied: JSON outside results/ and config/)
- `fpr507_pool_export_09_2026-10-15.json` (not copied: JSON outside results/ and config/)

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. Other result files, from earlier or side runs, give 1 script-and-commit pair that match and 3 where the script was not in the folder; we did not copy those versions.

## Files here

100 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_priority1_spec_09/` at commit `0f1924b6597b72b6e3408c54f296c2893039ce48` of our internal repository. 268 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
