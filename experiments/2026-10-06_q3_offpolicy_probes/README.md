# 2026-10-06_q3_offpolicy_probes

2026-10-06 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Do off-the-shelf probes trained on other data (deception, evaluation awareness, code vulnerability) score real and simulated LinuxArena runs the same?

## What we did

Phase 1: Qwen3-32B probes from other datasets (Apollo's deception probe, a templated evaluation-awareness probe from 60 hand-written contrast tasks, vulnerability probes trained on Emergent Misalignment and SafeCoder data) read at call ends and edits of real vs simulated GPT-5.6-Sol runs, with positive controls.

## What we found

README status: none is a usable LinuxArena monitor (attack vs honest AUROC at most 0.61), so the real-versus-simulated results are uninformative; deception shift +0.003 [-0.025, +0.032]; evaluation-awareness -0.299 [-0.461, -0.148].

## In the post?

Not in the post.

## Status

Current (kept as a record)

## Builds on

[2026-10-06_apollo_qwen_gap](../2026-10-06_apollo_qwen_gap/) (Apollo probe on Qwen3-32B)

## Main scripts

- [`run_score_probes.py`](run_score_probes.py)
- [`run_eval_awareness_probe.py`](run_eval_awareness_probe.py)
- [`run_eval_awareness_control.py`](run_eval_awareness_control.py)
- [`run_vuln_probe.py`](run_vuln_probe.py)
- [`run_vuln_safecoder_probe.py`](run_vuln_safecoder_probe.py)
- [`run_edit_mean_readout.py`](run_edit_mean_readout.py)
- [`analyze_shift.py`](analyze_shift.py)
- [`plot_shift.py`](plot_shift.py)

Result files the index names:

- `shift_apollo_deception_2026-10-07_231050.json` (not copied: result left out to keep the release under 120 MB)
- `shift_eval_awareness_2026-10-07_124023.json` (not copied: result left out to keep the release under 120 MB)
- [`control_2026-10-06_152934.json`](results/control_2026-10-06_152934.json)
- `shift_vulnerability_2026-10-07_124146.json` (not copied: result left out to keep the release under 120 MB)

## Files here

13 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-06_q3_offpolicy_probes/` at commit `76a6ab5c64d7af8854435a5e3f6f4196801ecc2a` of our internal repository. 64 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
