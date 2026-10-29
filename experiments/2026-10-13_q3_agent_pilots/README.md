# 2026-10-13_q3_agent_pilots

2026-10-13 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Could gpt-oss-120b or Qwen3-Coder-Next serve as the acting agent instead of Qwen3.5-27B: do they call tools cleanly and solve enough LinuxArena main tasks?

## What we did

Served each model with vLLM on Modal H200s and ran 20 tasks x 2 honest runs through the September harness, after smoke gates on templates, tool calling and byte-exact tool-output delivery; scored with the real scorer against a pre-set bar (pass 15 of 38, clearly below 9). Modal billed \$8.90 (Qwen3-Coder-Next) and \$5.28 (gpt-oss-120b).

## What we found

README (combined_40 files, Codex-recomputed): Qwen3-Coder-Next solved 13 of 38 scored and gpt-oss-120b 10 of 40, both 'marginal', neither selected; Qwen3.5-27B solved 23 of 38 on the same 20 tasks. gpt-oss-120b had 8.3% parsing rejections; one launch was discarded after a false canary alarm.

## In the post?

Not in the post.

## Status

Exploratory (neither agent selected)

## Builds on

[2026-10-13_q3_seed_agent_pilot](../2026-10-13_q3_seed_agent_pilot/) (same pipeline and 20 tasks); [2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/) (harness and server configs)

## Main scripts

- [`analyze_smoke_pilots.py`](analyze_smoke_pilots.py)
- [`combine_40_pilots.py`](combine_40_pilots.py)
- [`check_tool_outputs_pilots.py`](check_tool_outputs_pilots.py)
- [`slip_counts_pilots.py`](slip_counts_pilots.py)
- [`pricing_stats_pilots.py`](pricing_stats_pilots.py)
- [`check_empty_new_str_gptoss.py`](check_empty_new_str_gptoss.py)
- [`devbox/qcoder_full.sh`](devbox/qcoder_full.sh)
- [`devbox/gptoss_full.sh`](devbox/gptoss_full.sh)

Result files the index names:

- [`report_2026-10-13_155403_qwen3_coder_next_smoke.json`](results/report_2026-10-13_155403_qwen3_coder_next_smoke.json)
- [`report_2026-10-13_155631_gpt_oss_120b_smoke.json`](results/report_2026-10-13_155631_gpt_oss_120b_smoke.json)
- [`tool_outputs_qcoder_2026-10-13_160155.json`](results/tool_outputs_qcoder_2026-10-13_160155.json)
- [`tool_outputs_gptoss_2026-10-13_160849.json`](results/tool_outputs_gptoss_2026-10-13_160849.json)
- `manifest_pilot_results_s3_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

15 code and configuration files and 4 result files, copied byte for byte from `experiments/2026-10-13_q3_agent_pilots/` at commit `ee536b955cd8e52c5bd47d3133a09e743af3a77c` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
