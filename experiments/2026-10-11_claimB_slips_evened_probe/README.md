# 2026-10-11_claimB_slips_evened_probe

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Do the four formatting slips left after the simulator fixes carry the main realism probe's separation of formatting-fixed simulated runs from real ones? (Tested by evening them out in both arms, on the development environments only.)

## What we did

Applied the TF-IDF check's evening rules for the four slips (trailing newlines on file-create outputs, an empty last line in file views, error-line order, and the simulator's invented "Invalid view_range" errors) to formatting-fixed simulated, first real and second real inputs on the 5 development environments, re-read them with Qwen3-32B on Modal, and re-scored the frozen probe before and after, with a planted check and catch at 1% and 5% false alarms.

## What we found

summary_2026-10-11.json: reading 'partly'; evening removes 0.478 [0.420, 0.537] of the gap; AUROC 0.850 before vs 0.711 after (null 95th percentile 0.534); catch at 1% 94 of 466 before vs 26 of 466 after; our experiment log calls this 'about half the gap'.

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/) (its second probe step's inputs with times masked)

## Main scripts

- [`run_slips_evened_modal.py`](run_slips_evened_modal.py)
- [`analyze_catch_slips_evened.py`](analyze_catch_slips_evened.py)
- [`plot_slips_evened.py`](plot_slips_evened.py)

Result files the index names:

- `summary_2026-10-11.json` (not copied: result left out to keep the release under 120 MB)
- `slips_evened_2026-10-11_013350.png` (not copied: result file type not copied (.png))

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_claimB_slips_evened_probe/` at commit `66d338627c0970ed4f45efb9250ead22d330b39f` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
