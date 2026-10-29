# 2026-10-13_cell_a_transfer_POSTHOC

2026-10-13 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Why does the tool-output (reply) probe trained on GPT-5.6-Sol's transcripts and frozen transfer poorly to Qwen3.5-27B's own runs (area under the ROC curve, AUROC, 0.66 to 0.77), while a probe trained on Qwen3.5's own runs catches 93 to 98%?

## What we did

Qwen3.5-27B read its own runs two ways (reader-style rendering vs replay of the acting context), with token-level render parity checks, truncation-matched and system-prompt-swapped replays, re-centering on a third honest set, and the Sol-trained probes at every stored layer (16 to 63). Small Modal jobs, \$11.29.

## What we found

README (from printouts): second real run vs honest AUROC is 0.747 / 0.752 reader-style against 0.669 / 0.660 replay (Sol in-domain 0.963); truncation-matched replay 0.643 / 0.628; the swapped replay matches reader-style (0.7518 vs 0.7521); re-centering leaves AUROC 0.67; layer 48 beats 63 in AUROC for every variant. So the context the agent acted in (tools block, past reasoning, truncation) explains part of the weak transfer.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_onpolicy_reply](../2026-10-13_onpolicy_reply/) (imports analyze_onpolicy_cells.py and analyze_cell_a_null.py; cell (a) and cell (b) come from there)

## Main scripts

- [`run_xfer_cpu.py`](run_xfer_cpu.py)
- [`run_xfer_layers.py`](run_xfer_layers.py)
- [`run_replay_matched.py`](run_replay_matched.py)
- [`run_replay_matched_swapped.py`](run_replay_matched_swapped.py)
- [`stage_honest_reader.py`](stage_honest_reader.py)
- [`run_score_honest_reader.py`](run_score_honest_reader.py)
- [`analyze_xfer.py`](analyze_xfer.py)
- [`analyze_honest_reader.py`](analyze_honest_reader.py)

Result files the index names:

- `honest_reader_POSTHOC_2026-10-13_160153.printout.txt` (not copied: lines over 300 characters)
- `replay_matched_analysis_POSTHOC_2026-10-13_160210.printout.txt` (not copied: lines over 300 characters)
- `replay_matched_swapped_analysis_POSTHOC_2026-10-13_160437.printout.txt` (not copied: lines over 300 characters)
- [`xfer_layers_analysis_POSTHOC_2026-10-13_155125.printout.txt`](results/xfer_layers_analysis_POSTHOC_2026-10-13_155125.printout.txt)
- `xfer_POSTHOC_2026-10-13_153458.printout.txt` (not copied: lines over 300 characters)

## Files here

18 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_cell_a_transfer_POSTHOC/` at commit `16a61162feade1549ccd3e287ac25dcc68bb94fe` of our internal repository. 25 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
