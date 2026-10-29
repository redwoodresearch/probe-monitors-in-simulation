# 2026-10-12_q2_repair_sensitivity

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

The editor repair rules were chosen on 217 of the 939 cells; do the repair and planting results hold on the other 722 cells?

## What we did

\$0: copied each original analysis with a cell filter, required it to reproduce the committed 939-cell result exactly, then recomputed on the 722 outside cells and on the 217 alone.

## What we found

Qwen3-32B second real minus as simulated at 5% 12.36 [8.94, 16.18] on the 722 (14.27 on all); second real minus both repairs 0.66 [-2.30, 4.37]; planting deltas -8.93 (Qwen3-32B) and -4.47 (Olmo) on the 722, close to their all-cell values.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/); [2026-10-12_q2_per_repair](../2026-10-12_q2_per_repair/); [2026-10-12_plantslip_transfer](../2026-10-12_plantslip_transfer/); [2026-10-12_q2_llama_plant](../2026-10-12_q2_llama_plant/); [2026-10-12_plantslip_heldout](../2026-10-12_plantslip_heldout/)

## Main scripts

- [`run_subsets.sh`](run_subsets.sh)
- [`subset_cells.py`](subset_cells.py)
- [`build_subset_copies.py`](build_subset_copies.py)
- [`analyze_subset_reply_delta.py`](analyze_subset_reply_delta.py)
- [`analyze_subset_per_repair.py`](analyze_subset_per_repair.py)
- [`analyze_subset_plant_llama.py`](analyze_subset_plant_llama.py)
- [`analyze_subset_y.py`](analyze_subset_y.py)
- [`summarize_subsets.py`](summarize_subsets.py)

Result files the index names:

- `s3_manifest_v1.json` (not copied: result left out to keep the release under 120 MB)
- `s3_manifest_v2.json` (not copied: result left out to keep the release under 120 MB)

## Files here

8 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q2_repair_sensitivity/` at commit `e36151d8e9bfd65f45ce46b263e7e3e76c8bbfdd` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
