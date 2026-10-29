# 2026-10-13_reader_render_fixture

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

What does each activation reader actually see at the tool-output readout: does a reply's trailing newline reach it, and which token is read?

## What we did

Rendered synthetic fixtures (a bash output ending in a newline, real-form and simulator-form str_replace outputs, an insert ending in the review note) through the chat templates and readout rules of Qwen3-32B, Qwen3.5-27B, Olmo-3.1-32B, Llama-3.3-70B and Seed-OSS-36B on Modal CPU (tokenizers only).

## What we found

render_readers_2026-10-13_115027.printout.txt: for Qwen3.5-27B the trailing newline never reaches the reader (its template trims content and writes its own newline), so the newline slip is invisible to it; for Qwen3-32B and Olmo-3.1-32B it does reach the reader.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-13_g3_new_reader](../2026-10-13_g3_new_reader/) (fixture tool_newline.json)

## Main scripts

- [`run_render_readers.py`](run_render_readers.py)

Result files the index names:

- [`render_readers_2026-10-13_115027.json`](results/render_readers_2026-10-13_115027.json)
- [`render_readers_2026-10-13_115027.printout.txt`](results/render_readers_2026-10-13_115027.printout.txt)

## Files here

1 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-13_reader_render_fixture/` at commit `8d46b355cb58e2783748b5889c4b60a7696c7801` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
