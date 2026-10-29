# 2026-10-11_q2_b_feasibility

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

On the target-touch labels, are there enough positives to compare a monitor of target-touching commands between real and simulated Qwen3.5-27B runs?

## What we did

\$0 count feasibility from metadata and numeric labels: joined labels to the activation readouts at the end of each tool call and applied the credential-monitor comparison's positive-count power rule, first on version-2 labels and then on version-3 labels with calls that touch only targets shared with the main task dropped.

## What we found

README: on v2 labels feasible (predicted chance of agreement 0.995); commit dfa3012d3: on v3 labels with shared-only rows dropped, too few to tell (chance 0 at the committed rule).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-11_target_touch_labels](../2026-10-11_target_touch_labels/); [2026-10-11_q2_zoo_plans](../2026-10-11_q2_zoo_plans/)

## Main scripts

- [`analyze_b_feasibility.py`](analyze_b_feasibility.py)
- [`analyze_b_gate_v3.py`](analyze_b_gate_v3.py)

Result files the index names:

- `b_feasibility_2026-10-11_124535.json` (not copied: result left out to keep the release under 120 MB)
- `b_gate_v3_2026-10-11_135544.json` (not copied: result left out to keep the release under 120 MB)
- `inputs_provenance.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_q2_b_feasibility/` at commit `dfa3012d38bd79a67d212a353f606041357418c9` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
