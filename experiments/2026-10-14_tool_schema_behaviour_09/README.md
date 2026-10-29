# 2026-10-14_tool_schema_behaviour_09

2026-10-14 · part of [Side question: does the agent behave differently in simulation?](../../docs/experiments.md#side-question-does-the-agent-behave-differently-in-simulation)

## Question

The shipped simulator offers relaxed tool schemas (no required fields, extra keys allowed). Does that change what the agent does in simulated runs?

## What we did

Counts-only scan of raw .eval logs for every world (offered schema signatures; calls missing required keys, extra keys, parse errors and tool errors per run), paired by task against the gap between two real runs, plus a per-run table.

## What we found

On 190 tasks, simulated minus mean real: calls missing required keys +0.06 points (95% −0.04 to +0.20), extra keys +0.10 (95% −0.03 to +0.29); tool errors −0.68 (95% −0.89 to −0.51), i.e. fewer errors in simulation, vs −0.02 for run 2 − run 1 (tool_behaviour_2026-10-14_132158.json, paired).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-14_approach_provenance_09](../2026-10-14_approach_provenance_09/) (scan route)

## Main scripts

- [`run_tool_behaviour.py`](run_tool_behaviour.py)
- [`run_tool_behaviour_perrun.py`](run_tool_behaviour_perrun.py)

Result files the index names:

- `tool_behaviour_2026-10-14_132158.json` (not copied: strings over 300 characters)
- `tool_behaviour_perrun_2026-10-14_134930.json` (not copied: strings over 300 characters)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_tool_schema_behaviour_09/` at commit `744e86c82d83dceaf27037be437a882887460463` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
