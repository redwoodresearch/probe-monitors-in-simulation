# 2026-10-16_attack_refit3

2026-10-16 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Certify the all-tasks simulation detector's three fold fits that missed the convergence bar, without moving the other 37.

## What we did

Repaired 3 of 40 detector fits (layer 63 commands, layer 48 tool outputs) by an exact Newton step from the stored solution on Modal H100, with replay and identity checks against the frozen inputs.

## What we found

All 40 fits now meet the 1e-6 gradient bar; the simulated-versus-real command AUROC moved by about 6e-7 (at most about 3.5e-6 across all comparisons) (NOTE_attack_all_refit3); the post: AUROC 0.88 on tool outputs and 0.72 on commands for simulated vs real attack runs.

## In the post?

In the post: Figure 2; Table 7; Appendix E (and the detector used in Appendix C's off-policy check)

## Status

Current

## Builds on

[2026-10-16_cmd_sim_detector_uncut_all](../2026-10-16_cmd_sim_detector_uncut_all/)

## Main scripts

- [`run_refit3.py`](run_refit3.py)
- [`check_frozen_identity.sh`](check_frozen_identity.sh)

Result files the index names:

- `attack_all_refit3_2026-10-16_141813.json` (not copied: result file 1 MB or larger)
- `nonbitwise_2026-10-16_143825.json` (not copied: result left out to keep the release under 120 MB)

## As-run versions

The copies above are the folder's latest versions. These versions, which differ from them, produced results the index names or the post uses (see each note):

- [`as_run/4a9322473/run_refit3.py`](as_run/4a9322473/run_refit3.py) from commit `4a9322473`: Refitted the three fits of the simulation detector behind Figure 2 and Table 7 that had missed the convergence bar. The copy in this folder's snapshot is a newer version of the script.

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. For the 2 result files the index names, they match in 2 script-and-commit pairs, and in 2 the script was not in the folder at that commit. The differing versions behind results the index names or the post uses are under `as_run/` (above).

## Files here

14 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-16_attack_refit3/` at commit `cb1da6112352117578ac05eb7b81861d68773f84` of our internal repository. 40 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). The 1 file under `as_run/` come from the commits named above.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
