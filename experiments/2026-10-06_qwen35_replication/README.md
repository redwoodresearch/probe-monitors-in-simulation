# 2026-10-06_qwen35_replication

2026-10-06 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Do the Phase 1 results (real vs simulated, impactful-call transfer) hold when Qwen3.5-27B, the Phase 2 agent model, reads the GPT-5.6-Sol transcripts instead of Qwen3-32B?

## What we did

Extracted Qwen3.5-27B activations at all 64 layers at segment ends on the Phase 1 cohorts (original text; seven cumulative cleanup stages covering times/dates, numbers/IDs, whitespace, harness formatting, diff lines, empty diff lines and editor-output formatting; and a v4 dump that adds the real and simulated honest arms), reran the real-vs-simulated and impactful-call transfer analyses, and built a side-by-side table against Qwen3-32B; follow-ups on the no-editor separation and the insert review-note swap.

## What we found

No probe advantage over text, probe 0.572 vs text classifier 0.555 on held-out environments. The end-of-call impactful probe transfers (recall 0.870 real vs 0.873 simulated). Adding the insert review note removes 57% of the simulated false-alarm excess.

## In the post?

Not in the post.

## Status

Current (Phase 1 record); its v4 Qwen3.5-27B dump is read by [2026-10-14_q35_sol_avg_POSTHOC](../2026-10-14_q35_sol_avg_POSTHOC/) only in a code-equivalence test, while the post's off-policy probe fit uses a 2026-10-14 extraction

## Builds on

[2026-10-06_qwen35_reader](../2026-10-06_qwen35_reader/); [2026-10-06_q1_segment_end](../2026-10-06_q1_segment_end/); [2026-10-06_q2_impactful_call_transfer](../2026-10-06_q2_impactful_call_transfer/)

## Main scripts

- [`run_extract_qwen35.py`](run_extract_qwen35.py)
- [`run_q2_probes_qwen35.py`](run_q2_probes_qwen35.py)
- [`run_q2_alllayers_qwen35.py`](run_q2_alllayers_qwen35.py)
- [`run_q2_review_note_qwen35.py`](run_q2_review_note_qwen35.py)
- [`analyze_side_by_side.py`](analyze_side_by_side.py)
- [`analyze_text_baselines_qwen35.py`](analyze_text_baselines_qwen35.py)

Result files the index names:

- `side_by_side_2026-10-07_002447.md` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- [`q1_protocol_2026-10-06_222630.json`](results/q1/q1_protocol_2026-10-06_222630.json)
- `q2_review_note_qwen35_2026-10-08_020417.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- [`extract_2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1_2026-10-06_213622.json`](results/extract_2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1_2026-10-06_213622.json)

## Files here

183 code and configuration files and 10 result files, copied byte for byte from `experiments/2026-10-06_qwen35_replication/` at commit `129c8642d6f3cf443e5f0034604d90eca5e616b4` of our internal repository. 190 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
