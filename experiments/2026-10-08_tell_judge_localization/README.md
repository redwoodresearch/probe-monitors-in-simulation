# 2026-10-08_tell_judge_localization

2026-10-08 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Does the impactful-command probe's unexplained before-call shift on simulated runs concentrate right after outputs an LLM judge flags as fabricated?

## What we did

Built flags from the LinuxArena pairwise judge's content flags on the 100 rendering-swap runs and regressed the frozen Qwen3-32B before-call probe's misranking on them, on provisional labels; the pre-registered labelling model never ran.

## What we found

README: no localization on provisional labels, simulated-minus-real difference -0.113 [-0.231, +0.020]; real runs' calls after a real content flag are misranked more.

## In the post?

Not in the post.

## Status

Exploratory (incomplete: pre-registered classifier not run)

## Builds on

[2026-10-07_q2_rendering_swap](../2026-10-07_q2_rendering_swap/); [2026-10-06_q2_impactful_call_transfer](../2026-10-06_q2_impactful_call_transfer/)

## Main scripts

- [`build_flags.py`](build_flags.py)
- [`build_blind_list.py`](build_blind_list.py)
- [`run_classify_flags.py`](run_classify_flags.py)
- [`analyze_localization.py`](analyze_localization.py)
- [`analyze_label_agreement.py`](analyze_label_agreement.py)

Result files the index names:

- `localization_provisional_2026-10-08_010636.md` (not copied: lines over 300 characters)
- `preregistration.md` (not copied: notes or README (internal working labels))

## Files here

8 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-08_tell_judge_localization/` at commit `44513ad281d88737224eadf5e511a4ff3f642c17` of our internal repository. 13 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
