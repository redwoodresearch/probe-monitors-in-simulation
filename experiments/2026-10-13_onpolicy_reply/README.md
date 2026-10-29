# 2026-10-13_onpolicy_reply

2026-10-13 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

The main comparison on-policy: does a tool-output probe trained on Qwen3.5-27B's own runs catch simulated attacks as often as real ones?

## What we did

On 183 main tasks of Qwen3.5-27B's own runs (two real attack runs, one simulated, two real honest sets), scored a last-token tool-output readout (layers 39, 51, 63) for probes trained on real run 1, probes trained on the simulated run, and frozen probes trained on GPT-5.6-Sol's runs, with planted 10-point shifts, a label-shuffle null, a token-averaged readout, first-k-replies checks and two tests of why on-policy reading protected the probe (attack-signal strength with a noise-reduction sweep, and how well real and simulated runs can be told apart). Modal GPU, about \$25 cap.

## What we found

stage1_cell_b_POSTHOC_2026-10-13_120555.printout.txt: the on-policy probe trained on Qwen3.5-27B's real attack versus real honest runs, with honest system-prompt and submit definitions substituted and times/dates masked, 1% alarm: catch real 1 96.17%, real 2 97.27%, simulated 93.44%; simulated minus real 2 -3.83 [-8.70, 0.00]; without those substitutions, using uncleaned text: 97.81 / 97.81 / 96.72; null AUROC 0.991 vs 95th percentile 0.617; both planted 10-point shifts detected; reading 'a 10-point gap is ruled out'.

## In the post?

Not in the post (an early 183-task version; the post's on-policy tool-output numbers in Appendix F come from the 1,313-task records).

## Status

Superseded by the 1,313-task on-policy probe on tool outputs ([2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/), Appendix F)

## Builds on

The clean attack-detector test's cache ([2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/), 196-task frozen folds); [2026-10-12_q2_qwen35_reply_readout](../2026-10-12_q2_qwen35_reply_readout/)

## Main scripts

- [`run_cell_a.py`](run_cell_a.py)
- [`run_cell_c.py`](run_cell_c.py)
- [`run_planted.py`](run_planted.py)
- [`run_extract_pooled.py`](run_extract_pooled.py)
- [`run_avg_cells.py`](run_avg_cells.py)
- [`run_t1t2.py`](run_t1t2.py)
- [`run_t1_sweep.py`](run_t1_sweep.py)
- [`analyze_onpolicy_cells.py`](analyze_onpolicy_cells.py)
- [`analyze_cell_a_null.py`](analyze_cell_a_null.py)

Result files the index names:

- [`stage1_cell_b_POSTHOC_2026-10-13_120555.printout.txt`](results/stage1_cell_b_POSTHOC_2026-10-13_120555.printout.txt)
- [`stage2_cell_a_POSTHOC_2026-10-13_121124.printout.txt`](results/stage2_cell_a_POSTHOC_2026-10-13_121124.printout.txt)
- [`stage4_cell_c_POSTHOC_2026-10-13_121906.printout.txt`](results/stage4_cell_c_POSTHOC_2026-10-13_121906.printout.txt)
- [`stage5_planted_POSTHOC_2026-10-13_123424.printout.txt`](results/stage5_planted_POSTHOC_2026-10-13_123424.printout.txt)
- [`stage7_avg_cells_POSTHOC_2026-10-13_133239.printout.txt`](results/stage7_avg_cells_POSTHOC_2026-10-13_133239.printout.txt)
- [`t1t2_POSTHOC_2026-10-13_135644.printout.txt`](results/t1t2_POSTHOC_2026-10-13_135644.printout.txt)
- `t1sweep_POSTHOC_v2_2026-10-13_142945.printout.txt` (not copied: lines over 300 characters)
- [`cell_a_null_POSTHOC_2026-10-13_145628.printout.txt`](results/cell_a_null_POSTHOC_2026-10-13_145628.printout.txt)

## Files here

22 code and configuration files and 8 result files, copied byte for byte from `experiments/2026-10-13_onpolicy_reply/` at commit `3b86857d8e30a20545e75dc136c57bd31b890c42` of our internal repository. 58 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
