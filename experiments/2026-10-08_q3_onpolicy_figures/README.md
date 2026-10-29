# 2026-10-08_q3_onpolicy_figures

2026-10-08 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Does a probe still work the same when Qwen3.5-27B does the tasks itself, in simulation as on real machines (the on-policy comparison, at 20 and then 196 tasks)?

## What we did

On-policy pipeline for Qwen3.5-27B's own runs: replay extraction of its activations (all 64 layers; command, call-end and output-end readouts) for the 20-task strict rerun, the 196-task September scale-up, its top-up and the second real attack run, then an on-policy 2x2 for the impactful-command probe with refitting bootstraps and label-shuffle nulls; also registered and launched a third real honest batch (C) on the 196 tasks, plus many post hoc attack-detector, threshold and false-alarm analyses.

## What we found

196-task real minus simulated catch at 1% false alarms +0.46 points [-2.32, +2.50], a 10-point gap ruled out. README (Figure 3, 20 tasks): AUROC gap +0.0003 [-0.0111, +0.0132] with the September simulator. The two real honest arms and the third real honest batch alarm in their first 20 calls in 8.4%, 7.4% and 10.0% of runs, respectively.

## In the post?

In the post: Appendix A runs (indirectly: the third real honest batch and the extracted 196-task runs are part of the oldest stratum); its own numbers are not in the post

## Status

Current as a data source (extractions, the third real honest batch); its 20- and 196-task scoring results are superseded by the final pooled analysis on all 1,313 tasks in [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/); the 10-09 Sol-trained attack detector rows are superseded

## Builds on

[2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/) (runs); [2026-10-09_q2_september_sizing](../2026-10-09_q2_september_sizing/) (scale-up plan); [2026-10-07_phase2_probe_proposal](../2026-10-07_phase2_probe_proposal/)

## Main scripts

- [`run_replay_readouts.py`](run_replay_readouts.py)
- [`run_extract_replay.py`](run_extract_replay.py)
- [`run_score_onpolicy.py`](run_score_onpolicy.py)
- [`analyze_onpolicy_2x2.py`](analyze_onpolicy_2x2.py)
- [`analyze_scaleup196.py`](analyze_scaleup196.py)
- [`analyze_retrain.py`](analyze_retrain.py)
- [`run_POSTHOC_score_honest_c.py`](run_POSTHOC_score_honest_c.py)
- [`plot_figure3.py`](plot_figure3.py)

Result files the index names:

- `RESULTS_scaleup196.md` (not copied: notes or README (internal working labels))
- [`scaleup196_verdicts_2026-10-10_164622.json`](results/scaleup196_verdicts_2026-10-10_164622.json)
- `scaleup_plants_2026-10-10_164614.json` (not copied: result left out to keep the release under 120 MB)
- [`a9_real_null_2026-10-09_112159.json`](results/a9_real_null_2026-10-09_112159.json)
- `figure3_manifest_2026-10-09_v2.json` (not copied: result left out to keep the release under 120 MB)
- `A2_batches_POSTHOC_2026-10-11_150751.json` (not copied: result left out to keep the release under 120 MB)

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. For the 1 result file the index names, they match in 5 script-and-commit pairs, differ in 2, and in 1 the script was not in the folder at that commit. Other result files, from earlier or side runs, give 166 script-and-commit pairs that match, 56 that differ and 26 where the script was not in the folder; we did not copy those versions. This folder's own numbers are not in the post, so we did not copy the differing versions either.

## Files here

197 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-08_q3_onpolicy_figures/` at commit `0a9497d1fd8cb67748a9762b50a83fabef335fb9` of our internal repository. 262 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
