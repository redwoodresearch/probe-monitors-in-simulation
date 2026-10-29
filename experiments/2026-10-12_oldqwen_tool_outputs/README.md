# 2026-10-12_oldqwen_tool_outputs

2026-10-12 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Did tool outputs actually reach Qwen3-32B's input when it read GPT-5.6-Sol's transcripts (the project manager's setup worry)?

## What we did

One Modal CPU container read the v4 Qwen3-32B extraction's staged inputs and per-run metadata for all 1,000 real honest runs, counted tool outputs inside the 32,768-token cut, and re-rendered one seeded example with the extraction's own code.

## What we found

CHECK.md: yes: 23,174 of 34,237 tool outputs (67.7%) lie inside the input, 28.0% beyond the cut (517 long runs), 4.3% empty; all 1,000 runs have outputs inside; the re-render matched the metadata exactly.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-06_segment_end_extraction](../2026-10-06_segment_end_extraction/)

## Main scripts

- [`run_check_modal.py`](run_check_modal.py)
- [`make_check_md.py`](make_check_md.py)

Result files the index names:

- `CHECK.md` (not copied: notes or README (internal working labels))
- `oldqwen_tool_outputs_2026-10-12_164139Z.json` (not in this folder at the source commit)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_oldqwen_tool_outputs/` at commit `26c1534ed1cbcafec3ffd4032a3282a35c1b67b0` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
