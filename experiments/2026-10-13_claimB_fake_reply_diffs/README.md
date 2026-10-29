# 2026-10-13_claimB_fake_reply_diffs

2026-10-13 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Beyond the two known editor-format slips, what differs between simulated and real tool replies to the same command?

## What we did

On 91 matched honest command sites in 72 main tasks (same commands and byte-identical earlier outputs; first real, second real and simulated reply each), counted surface features before and after the frozen fixed-simulation repairs, ran a reply-text classifier with a label-shuffle null, and post hoc checks of clock patterns, pipeline levels, a marker control, masked plants and a newline render check. \$0 local plus a \$0.0014 Modal render check.

## What we found

fake_reply_diffs_POSTHOC_2026-10-13_131300.json: a text classifier separates simulated from first-real replies with AUROC 0.81 (null 95th percentile 0.57); of 33 surface features x 2 versions, 2 'survive' and 31 show no difference; byte-identical pairs: simulated = first real at 21 sites, second real = first real at 38.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-12_claimB_meaning_judge](../2026-10-12_claimB_meaning_judge/) (matched sites); [2026-10-09_fixed_sim](../2026-10-09_fixed_sim/) (repair rules)

## Main scripts

- [`analyze_POSTHOC_fake_reply_diffs.py`](analyze_POSTHOC_fake_reply_diffs.py)
- [`analyze_POSTHOC_controls_v2.py`](analyze_POSTHOC_controls_v2.py)
- [`analyze_POSTHOC_marker_control.py`](analyze_POSTHOC_marker_control.py)
- [`analyze_POSTHOC_clock_patterns.py`](analyze_POSTHOC_clock_patterns.py)
- [`analyze_POSTHOC_pipeline_levels.py`](analyze_POSTHOC_pipeline_levels.py)
- [`analyze_POSTHOC_masked_planted.py`](analyze_POSTHOC_masked_planted.py)
- [`run_render_newlines_modal.py`](run_render_newlines_modal.py)

Result files the index names:

- `fake_reply_diffs_POSTHOC_2026-10-13_131300.json` (not copied: result left out to keep the release under 120 MB)
- `controls_v2_POSTHOC_2026-10-13_140400.json` (not copied: result left out to keep the release under 120 MB)
- `marker_control_POSTHOC_2026-10-13_141158.json` (not copied: result left out to keep the release under 120 MB)
- [`clock_patterns_POSTHOC_2026-10-13_135124.json`](results/clock_patterns_POSTHOC_2026-10-13_135124.json)
- `pipeline_levels_POSTHOC_2026-10-13_135038.json` (not copied: strings over 300 characters)
- [`masked_planted_POSTHOC_2026-10-13_134618.json`](results/masked_planted_POSTHOC_2026-10-13_134618.json)
- [`render_newlines_2026-10-13_135117.json`](results/render_newlines_2026-10-13_135117.json)

## Files here

7 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-13_claimB_fake_reply_diffs/` at commit `0056175e5992031cfea917d7bba3b7b719d8478d` of our internal repository. 12 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
