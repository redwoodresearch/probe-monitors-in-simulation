# 2026-10-12_probe_impl_audit

2026-10-12 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Do five side probes reproduce from their stored artifacts under seven fixed checks (identity, readout position, frozen layer, convergence, thresholds, nulls and plants, headline)?

## What we did

Read-only agents checked the evaluation-awareness probe, is-this-a-test part 2, the fixed-simulator test's max-pooled planted-line probe, the stage-1 sabotage gate and the frozen deception probes from git and S3; small Modal CPU reads filled blocked checks; Codex reviewed the verdicts.

## What we found

RESULTS.md: evaluation-awareness, max-pooled planted-line probe and sabotage gate reproduced; is-this-a-test not (last-digit difference); frozen deception probes not (plant sizing on a grid instead of bisection; no value changes); the sabotage gate reads past the command's end in 967 of 4,452 rows.

## In the post?

Not in the post.

## Status

Current (on a side branch)

## Builds on

[2026-10-10_eval_awareness_probe](../2026-10-10_eval_awareness_probe/); [2026-10-11_is_this_a_test](../2026-10-11_is_this_a_test/); [2026-10-11_sabotage_stage1_labels](../2026-10-11_sabotage_stage1_labels/)

## Main scripts

- [`run_ea_spans_modal.py`](run_ea_spans_modal.py)
- [`run_p7a_spans_modal.py`](run_p7a_spans_modal.py)
- [`run_p7a_scope_modal.py`](run_p7a_scope_modal.py)
- [`run_sg_calscores_modal.py`](run_sg_calscores_modal.py)

Result files the index names:

- `RESULTS.md` (not copied: notes or README (internal working labels))
- `codex_review.md` (not copied: notes or README (internal working labels))
- `ea_it.md` (not copied: notes or README (internal working labels))
- `fd.md` (not copied: notes or README (internal working labels))
- `p7a_sg.md` (not copied: notes or README (internal working labels))

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_probe_impl_audit/` at commit `ce82d39e554e5cba445ce2adfd29e4d1cb10e8d3` of our internal repository. 9 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
