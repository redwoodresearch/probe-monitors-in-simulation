# 2026-10-12_q2_llama_reader

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Does the reply-readout gap (the in-domain tool-output attack detector, a probe trained on real attack vs real honest runs and read at the end of each tool output, catches simulated attack runs less often than real ones) replicate with Llama-3.3-70B as the reader model instead of Qwen3-32B?

## What we did

After a check that Llama-3.3-70B's chat template keeps every simulator formatting difference intact, extracted its activations for 3,807 GPT-5.6-Sol transcripts (24 shards of 2 H100s), refit the same detector recipe and compared catch on simulated runs with catch on a second, held-out real attack run of the same tasks; part 2 rescored the 823 simulated runs changed by the editor-output formatting fixes (simulated text-editor outputs rendered the way the real tool renders them). Billed \$223.37 and \$45.57.

## What we found

README: "replicates": simulated minus second real at 5% -12.41 points [-15.53, -9.11] (-13.55 at 1%), swap null p 0.0001; part 2 label "partial"; label-shuffle null passes.

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/); [2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/)

## Main scripts

- [`run_extract_llama.py`](run_extract_llama.py)
- [`run_score_fix_llama.py`](run_score_fix_llama.py)
- [`analyze_reply_break_llama.py`](analyze_reply_break_llama.py)
- [`analyze_edit_effect_llama.py`](analyze_edit_effect_llama.py)
- [`check_label_shuffle_null.py`](check_label_shuffle_null.py)

Result files the index names:

- [`reply_break_llama_2026-10-12_143812.json`](results/reply_break_llama_2026-10-12_143812.json)
- `render_r2j_gate_2026-10-12_123520.json` (not in this folder at the source commit)

## Files here

10 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-12_q2_llama_reader/` at commit `c20ca2bda58538b00428e9dc526e312dc7119cd3` of our internal repository. 12 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
