# 2026-10-11_q2_placeholder_census

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Which of the impactful-tool-call probe pipelines rebuild run histories with unresolved Inspect "attachment://" placeholders instead of text, and which results do they feed?

## What we did

\$0 counts over honest logs (and counts-only scans of attack arms): placeholders in sample messages, tool-call arguments, model-event inputs and call records, for the GPT-5.6-Sol and Qwen3.5-27B runs, with S3 and manifest checks.

## What we found

CENSUS.md: sample-message contents hold 0 placeholders, but call records hold many (28,007 references in 35,057 messages for Qwen's first real honest arm), so pipelines rebuilding histories from call records get placeholders; model-event inputs carry them too.

## In the post?

Not in the post.

## Status

Current (preliminary)

## Builds on

The pipelines of the second research question's folders (2026-10-06 to 2026-10-10)

## Main scripts

- [`count_placeholders.py`](count_placeholders.py)
- [`count_placeholders_more.py`](count_placeholders_more.py)
- [`count_placeholders_part3.py`](count_placeholders_part3.py)

Result files the index names:

- [`placeholders_q2_2026-10-11_165548.json`](results/placeholders_q2_2026-10-11_165548.json)
- [`placeholders_q2_more_2026-10-11_170117.json`](results/placeholders_q2_more_2026-10-11_170117.json)
- [`placeholders_q2_part3_2026-10-11_174112.json`](results/placeholders_q2_part3_2026-10-11_174112.json)

## Files here

3 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-11_q2_placeholder_census/` at commit `45a92d649ee26974aecb93d7440b3a3274015b6f` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
