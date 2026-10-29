# 2026-10-11_claimB_halves_qwen35

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Does the environment-halves result hold when Qwen3.5-27B, not Qwen3-32B, reads the same evened fixed-simulator runs?

## What we did

Staged the evened texts for a Qwen3.5-27B reader, extracted activations on Modal, and reran the halves analysis (train on one half of the environments, test on the other) with planted checks and the direct real-run-2 yardstick.

## What we found

RESULT.md: the dev-to-test planted check failed (0.899 vs the 0.910 floor), so the result is descriptive only; pooled gap +1.60 points [+0.11, +3.99]; direct real-run-2 yardstick +0.43 [-0.53, +2.65]; AUROC 0.532 and 0.620; Qwen3.5 minus Qwen3-32B pooled gap -3.30 [-6.86, +0.21].

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-11_claimB_halves](../2026-10-11_claimB_halves/); [2026-10-11_claimB_h3_fix_to_plateau](../2026-10-11_claimB_h3_fix_to_plateau/)

## Main scripts

- [`run_stage_evened_modal.py`](run_stage_evened_modal.py)
- [`run_halves_qwen35_modal.py`](run_halves_qwen35_modal.py)
- [`analyze_halves_qwen35.py`](analyze_halves_qwen35.py)
- [`analyze_gate_standin.py`](analyze_gate_standin.py)
- [`make_result_md.py`](make_result_md.py)

Result files the index names:

- `halves_qwen35_2026-10-12_005827.json` (not copied: result left out to keep the release under 120 MB)
- `gate_standin_2026-10-11_235636.json` (not copied: result left out to keep the release under 120 MB)
- `codex_factcheck_halves_qwen35_2026-10-12_005827.txt` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

6 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_claimB_halves_qwen35/` at commit `6820984aca51c0d0c1bd56d30005d54ea4eb2e9b` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
