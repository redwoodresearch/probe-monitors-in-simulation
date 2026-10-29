# 2026-10-16_honest_sim_detector_all

2026-10-16 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Can a detector tell simulated honest runs from real honest runs on all tasks, and how much do two real honest draws differ?

## What we did

Same detector recipe on honest runs only: simulated honest vs the second real honest arm, and the first real honest arm vs the second as a control; rerun on the final simulated-honest pool (171 + 20 pilot runs); boring-explanation slices.

## What we found

The post's Table 7: simulated vs real honest 0.75 (tool outputs) and 0.60 (commands); two real honest sets 0.52.

## In the post?

In the post: Table 7 rows 4-5; Appendix E

## Status

Current

## Builds on

[2026-10-16_cmd_sim_detector_uncut_all](../2026-10-16_cmd_sim_detector_uncut_all/); [2026-10-15_cut_sensitivity](../2026-10-15_cut_sensitivity/) inventory_b_fix_v2

## Main scripts

- [`run_honest_all.py`](run_honest_all.py)
- [`honest_inputs.py`](honest_inputs.py)
- [`routec_keys.py`](routec_keys.py)
- [`analyze_honest_slices.py`](analyze_honest_slices.py)

Result files the index names:

- `honest_all_routec_2026-10-18_121211.json` (not copied: result file 1 MB or larger)
- `honest_all_2026-10-16_120353.json` (not copied: result file 1 MB or larger)
- `honest_slices_2026-10-16_130259.json` (not copied: result left out to keep the release under 120 MB)

## As-run versions

The copies above are the folder's latest versions. These versions, which differ from them, produced results the index names or the post uses (see each note):

- [`as_run/56c36262b/run_honest_all.py`](as_run/56c36262b/run_honest_all.py) from commit `56c36262b`: results/honest_all_2026-10-16_120353.json and honest_slices_2026-10-16_130259.json record commits 56c36262b and 3fc73d5c8, where this script had this version. The later honest_all_routec_2026-10-18_121211.json records commit 4c29bb573, where the script matches the copy in this folder.
- [`as_run/56c36262b/honest_inputs.py`](as_run/56c36262b/honest_inputs.py) from commit `56c36262b`: results/honest_all_2026-10-16_120353.json and honest_slices_2026-10-16_130259.json record commits 56c36262b and 3fc73d5c8, where this script had this version. The later honest_all_routec_2026-10-18_121211.json records commit 4c29bb573, where the script matches the copy in this folder.

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. For the 3 result files the index names, they match in 5 script-and-commit pairs, differ in 4, and in 3 the script was not in the folder at that commit. The differing versions behind results the index names or the post uses are under `as_run/` (above).

## Files here

19 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-16_honest_sim_detector_all/` at commit `92452a656b621c7752fff368cbea137d48c8c0c4` of our internal repository. 23 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). The 2 files under `as_run/` come from the commits named above.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
