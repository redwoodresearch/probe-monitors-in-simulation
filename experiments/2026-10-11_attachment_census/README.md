# 2026-10-11_attachment_census

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Do the pipelines that rebuild run histories for the reader models pass unresolved Inspect "attachment://" placeholders (stand-ins Inspect writes for long strings) to the model instead of the real text, and which results do they feed?

## What we did

Counted placeholders (integer counts only) in every staged reader input of the Qwen3.5-27B on-policy replays, in 44 off-policy Qwen3-32B stagings, and in the source logs of five GPT-5.6-Sol and GPT-6-Astra cohorts and 20 Phase 2 folders, using three small Modal CPU jobs and a planted positive control.

## What we found

census.md: 0 unresolved placeholders in every staged reader input scanned (305.0M decoded on-policy tokens; 88,308 off-policy staged items) and in the source messages; placeholders appear only in the model events' raw provider requests; the planted control was counted correctly.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/); [2026-10-06_segment_end_extraction](../2026-10-06_segment_end_extraction/); [2026-10-07_q3_codex_followups](../2026-10-07_q3_codex_followups/)

## Main scripts

- [`count_eval_attachments.py`](count_eval_attachments.py)
- [`count_staged_attachments_modal.py`](count_staged_attachments_modal.py)

Result files the index names:

- `eval_counts_2026-10-11_170123.json` (not copied: result left out to keep the release under 120 MB)
- [`plant_control_2026-10-11_170420.json`](results/plant_control_2026-10-11_170420.json)
- `staged_offpolicy_2026-10-11_170233.json` (not copied: result left out to keep the release under 120 MB)
- `staged_onpolicy_all_2026-10-11_170233.json` (not copied: result left out to keep the release under 120 MB)
- [`staged_onpolicy_honest_2026-10-11_170128.json`](results/staged_onpolicy_honest_2026-10-11_170128.json)

## Files here

2 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-11_attachment_census/` at commit `75b27fa6a64e861d71739f950de12ac7c4225b62` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
