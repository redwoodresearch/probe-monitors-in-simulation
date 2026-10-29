# 2026-10-12_claimB_convergence_check

2026-10-12 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Did the environment-halves probe selections (trained on one half of the environments, tested on the other) pick logistic-regression fits that never converged, and does dropping unconverged fits change any result?

## What we did

Post hoc \$0 code check on stored score arrays: applied the main realism probe's convergence rule (final gradient below tolerance) to the halves selections for Qwen3-32B, the Qwen3.5-27B rerun, the planted-check refits, the TF-IDF halves and the halves rerun with numbers, IDs and whitespace masked in both arms, and recomputed affected numbers with unconverged fits excluded from selection.

## What we found

Codex recomputation: PASS; every real-label selection had converged, so no headline changes; masking moved only null p-values (two for Qwen3.5-27B, one for mask23: 0.267 to 0.287).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-11_claimB_halves](../2026-10-11_claimB_halves/); [2026-10-11_claimB_halves_qwen35](../2026-10-11_claimB_halves_qwen35/); [2026-10-12_claimB_planted_oppoint](../2026-10-12_claimB_planted_oppoint/); [2026-10-12_claimB_tfidf_halves](../2026-10-12_claimB_tfidf_halves/); [2026-10-12_claimB_halves_mask23](../2026-10-12_claimB_halves_mask23/); [2026-10-12_probe_code_reuse](../2026-10-12_probe_code_reuse/) (raised the issue)

## Main scripts

- [`analyze_POSTHOC_convergence.py`](analyze_POSTHOC_convergence.py)
- [`analyze_POSTHOC_mask23_toplevel.py`](analyze_POSTHOC_mask23_toplevel.py)
- [`rerun_POSTHOC_halves_masked.py`](rerun_POSTHOC_halves_masked.py)
- [`run_POSTHOC_mask23_checks.py`](run_POSTHOC_mask23_checks.py)

Result files the index names:

- `mask23_toplevel_POSTHOC_2026-10-12_122055.json` (not copied: result left out to keep the release under 120 MB)
- `codex_factcheck_convergence_POSTHOC_2026-10-12.txt` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `results_manifest_v1.json` (not copied: JSON outside results/ and config/)

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_claimB_convergence_check/` at commit `762dc08a3cf7aa26969ece5900d9f4a03704de6a` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
