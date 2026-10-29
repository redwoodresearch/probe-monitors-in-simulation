# 2026-10-11_tool_output_cue_probe

2026-10-11 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

Evaluation awareness, new design: does a probe trained on simulation cues inserted into real tool outputs, read at the end of each tool output, generalize to held-out cue types while ignoring a neutral line (and only then, do simulated runs look like cued runs)?

## What we did

Built cue and neutral pairs in tool outputs, extracted Qwen3.5-27B activations on Modal, fit content and marker probes, and ran stage-1 validation gates (held-out implied cues, explicit phrasings, neutral control, attack panel, planted line); stage 2 gated on all passing.

## What we found

README: probe A does not validate, so stage 2 was not run; held-out implied cues 0.529 (null 95th percentile 0.546) fail; explicit phrasings 0.95 pass; neutral control fails; a planted line lowers run scores (-1.17).

## In the post?

Not in the post.

## Status

Exploratory (stage 1 failed)

## Builds on

[2026-10-11_la_format_sim_probe](../2026-10-11_la_format_sim_probe/); [2026-10-10_eval_awareness_probe](../2026-10-10_eval_awareness_probe/)

## Main scripts

- [`toc_templates.py`](toc_templates.py)
- [`run_build_toc.py`](run_build_toc.py)
- [`run_extract_toc.py`](run_extract_toc.py)
- [`spawn_fit_toc.py`](spawn_fit_toc.py)
- [`analyze_stage1.py`](analyze_stage1.py)
- [`analyze_power.py`](analyze_power.py)
- [`run_attachment_census.py`](run_attachment_census.py)

Result files the index names:

- `stage1_A_2026-10-11_180323.json` (not copied: result left out to keep the release under 120 MB)
- [`extract_2026-10-11_161002.json`](results/extract_2026-10-11_161002.json)
- [`extract_2026-10-11_173852.json`](results/extract_2026-10-11_173852.json)

## Files here

7 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-11_tool_output_cue_probe/` at commit `4df7390d8185159af673982fcee9bd5dc043068d` of our internal repository. 11 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
