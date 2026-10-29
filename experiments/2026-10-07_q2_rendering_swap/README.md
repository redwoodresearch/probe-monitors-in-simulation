# 2026-10-07_q2_rendering_swap

2026-10-07 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Does the September simulator's text-editor insert template (a diff instead of the real tool's snippet plus review note) drive the impactful-command probe's real-to-simulated gap when read just before a call?

## What we did

Re-rendered every insert output of 100 simulated GPT-5.6-Sol runs in the real tool's style (five arms: snippet, review note, both), extracted fresh Qwen3-32B activations and scored them with frozen probes trained on uncleaned text; a follow-up swapped the find-and-replace tells (line-number gutter, trailing newline).

## What we found

Re-rendering closes 35% [22%, 51%] of the after-insert ranking gap. The review note accounts for +40% (snippet format) and +34% (diff format). Fixing the gutter and the newline together closes 101% of the recall gap after find-and-replace edits.

## In the post?

Not in the post.

## Status

Current (Phase 1 record)

## Builds on

[2026-10-06_q2_impactful_call_transfer](../2026-10-06_q2_impactful_call_transfer/) (frozen probes, predictions)

## Main scripts

- [`run_swap.py`](run_swap.py)
- [`rerender.py`](rerender.py)
- [`select_runs.py`](select_runs.py)
- [`select_extension.py`](select_extension.py)
- [`str_replace_tells.py`](str_replace_tells.py)
- [`analyze_swap.py`](analyze_swap.py)
- [`analyze_note.py`](analyze_note.py)
- [`analyze_sr.py`](analyze_sr.py)
- [`analyze_checks.py`](analyze_checks.py)

Result files the index names:

- [`swap_2026-10-08_003813.json`](results/swap_2026-10-08_003813.json)
- [`checks_2026-10-07_234747.json`](results/checks_2026-10-07_234747.json)
- `note_2026-10-08_003833.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `note_presence_2026-10-08_003312.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- [`sr_2026-10-08_014938.json`](results/sr_2026-10-08_014938.json)

## Files here

19 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-07_q2_rendering_swap/` at commit `f5b45f59c13f6254055bcba2797b6925bda65fe2` of our internal repository. 21 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
