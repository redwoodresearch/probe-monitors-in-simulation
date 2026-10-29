# 2026-10-13_q3_seed_agent_pilot

2026-10-13 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Could Seed-OSS-36B serve as the acting agent: does it call tools cleanly and solve enough LinuxArena main tasks?

## What we did

Served Seed-OSS-36B with vLLM on Modal H200s and ran 20 tasks x 2 honest runs with smoke gates (template, tool calling, byte-exact delivery), scored by the real scorer against a pre-registered bar and prediction; priced a full on-policy run.

## What we found

README (combined_40_2026-10-13_144715.json): 3 of 40 succeeded (predicted 14 [5, 24]), reading 'clearly below'; 17 runs made 0 non-submit calls; tool-error share 17.3%; Qwen3.5-27B solved 23 of 38 on the same tasks. Not selected.

## In the post?

Not in the post.

## Status

Exploratory (not selected)

## Builds on

[2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/) (harness)

## Main scripts

- [`analyze_smoke_seed.py`](analyze_smoke_seed.py)
- [`check_tool_outputs_seed.py`](check_tool_outputs_seed.py)
- [`combine_40.py`](combine_40.py)
- [`pricing_stats.py`](pricing_stats.py)
- [`price_full_onpolicy.py`](price_full_onpolicy.py)
- [`devbox/seed_full.sh`](devbox/seed_full.sh)

Result files the index names:

- [`report_2026-10-13_140142_seed_oss_36b_smoke.json`](results/report_2026-10-13_140142_seed_oss_36b_smoke.json)
- [`tool_outputs_seed_2026-10-13_141045.json`](results/tool_outputs_seed_2026-10-13_141045.json)
- `price_full_onpolicy_2026-10-13_145318.json` (not copied: result left out to keep the release under 120 MB)
- `manifest_pilot_results_s3_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

8 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-13_q3_seed_agent_pilot/` at commit `3995ff9eaa4daa83087665baa411f4790906c6fd` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
