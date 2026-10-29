# 2026-10-07_sim_failure_hiding

2026-10-07 · part of [Side question: does the agent behave differently in simulation?](../../docs/experiments.md#side-question-does-the-agent-behave-differently-in-simulation)

## Question

Does the simulator hide failed commands (show success where the real machine failed), and does that change agent behavior and control-evaluation outcomes?

## What we did

CPU-only analysis of local eval archives for 939 paired GPT-5.6-Sol attack cells: counted failed calls per run in real run 1, real run 2 and the simulated run, compared identical calls made before any file edit, and checked editor templates and the agents' stated success.

## What we found

README summary: no clear sign of hiding; runs with a failed call are 86.3% (real run 1), 87.2% (real run 2), 85.6% (simulated); on 431 identical pre-edit calls the simulator hid 0 of 23 real failures (upper bound about 13%); the earlier 46% vs 0.1% tool-error contrast was a recording artifact.

## In the post?

Not in the post.

## Status

Current

## Builds on

None stated

## Main scripts

- [`analyze_failure_hiding.py`](analyze_failure_hiding.py)
- [`analyze_success_credence.py`](analyze_success_credence.py)
- [`analyze_editor_templates.py`](analyze_editor_templates.py)

Result files the index names:

- `failure_hiding_20261007-135042.json` (not copied: strings over 300 characters)
- [`success_credence_20261007-131502.json`](results/success_credence_20261007-131502.json)

## Files here

3 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-07_sim_failure_hiding/` at commit `e76432bb345207e5e485f62ae90ae566d3a1d04d` of our internal repository. 11 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
