# 2026-10-06_q1_segment_end

2026-10-06 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Can a linear probe on Qwen3-32B activations tell real LinuxArena attack runs (GPT-5.6-Sol's) from simulated ones, and does it see anything a text classifier does not?

## What we did

Phase 1: logistic-regression probes on Qwen3-32B activations at the end of each tool output and tool call, trained out-of-fold by main task on 939 paired real and simulated attack cells, compared with a term-frequency text classifier (TF-IDF) across text cleanup levels from none up to all seven cumulative cleanups (times and dates, numbers and IDs, whitespace, harness format, diff lines, empty diff lines, editor output format), with real-versus-real and label-shuffle nulls; amendments added a smoothed (exponential moving average) readout, the honest arm and a length check.

## What we found

results.md: under the frozen rule with all seven cleanups, no probe advantage. The probe scores 0.541 on all its outputs; on outputs shared with the text classifier, it scores 0.536 versus 0.555, gap −0.019 [−0.046, +0.004]. On original text the probe reached 0.952 vs 0.515 real-versus-real, but the text classifier scored 0.977.

## In the post?

Not in the post.

## Status

Current (Phase 1 record); its first 8-layer analysis is superseded within the folder by the 64-layer frozen-rule analyses

## Builds on

[2026-10-06_segment_end_extraction](../2026-10-06_segment_end_extraction/); [2026-10-06_all_layers_extraction](../2026-10-06_all_layers_extraction/); [2026-10-06_q1_text_baselines](../2026-10-06_q1_text_baselines/)

## Main scripts

- [`analyze_q1_gpu.py`](analyze_q1_gpu.py)
- [`analyze_q1_segment_end.py`](analyze_q1_segment_end.py)
- [`analyze_level7_length.py`](analyze_level7_length.py)
- [`report_q1.py`](report_q1.py)
- [`report_final.py`](report_final.py)
- [`plot_layer_curves.py`](plot_layer_curves.py)

Result files the index names:

- `q1_gpu_2026-10-06_190522.json` (not copied: result file 1 MB or larger)
- `q1_gpu_2026-10-06_181139.json` (not copied: result file 1 MB or larger)
- [`q1_protocol_2026-10-06_200311.json`](results/q1_protocol_2026-10-06_200311.json)
- [`layer_curves_full_all_outputs_2026-10-06_2305.md`](results/layer_curves_full_all_outputs_2026-10-06_2305.md)
- `results.md` (not copied: notes or README (internal working labels))

## Files here

201 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-06_q1_segment_end/` at commit `eb08a9d51291024ff1f48c6c0dbfeafb63aa1df9` of our internal repository. 178 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
