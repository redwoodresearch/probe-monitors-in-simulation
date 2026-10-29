# 2026-10-14_p2_family_expansion

2026-10-14 · part of [Training and scoring the probes](../../docs/experiments.md#training-and-scoring-the-probes)

## Question

Started as adding probes with mid-range real catch to the probe family; grew into the main scoring and fitting job for the scale-up, including the on-policy probe trained on all tasks.

## What we did

Pass 1 fitted a fixed list of probe variants (training fractions 10, 25 and 100%, several layers) on saved Qwen3.5-27B activations; pass 2 scored the frozen probes on every scale-up set (the 311 added tasks, new main tasks, interleaved runs, simulated honest, second-epoch honest). Later it fitted the on-policy probe on all tasks at layer 48 (a post hoc recipe registered after earlier results were read, before its training) and versions of it trained on full-length transcripts (with 10% and 25% training subsets), five folds by main task, plus coverage, leakage and held-out checks.

## What we found

The input config of the final pooled analysis on all 1,313 tasks ([2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/)/config/INPUTS_FULL_2026-10-17_233212.json, fits) uses own_all_rs_uncut_L48 and its 10% and 25% versions from own_all_rs_uncut_fit_2026-10-16_030614.json in this folder; the fit record labels it a new post hoc probe trained on full transcripts, 30 models.

## In the post?

In the post: Table 1 and Appendix A (on-policy probe); Appendix C (on-policy probe trained on a tenth of its tasks)

## Status

Current

## Builds on

[2026-10-14_q3_overnight_chain](../2026-10-14_q3_overnight_chain/); [2026-10-14_q3_scaleup_311](../2026-10-14_q3_scaleup_311/); [2026-10-14_q3_new_mains_launch](../2026-10-14_q3_new_mains_launch/); [2026-10-14_priority1_spec_09](../2026-10-14_priority1_spec_09/)

## Main scripts

- [`run_famexp.py`](run_famexp.py)
- [`run_famexp_pass2.py`](run_famexp_pass2.py)
- [`run_famexp_pack.py`](run_famexp_pack.py)
- [`run_own_all.py`](run_own_all.py)
- [`cutfull.py`](cutfull.py)
- [`run_leakage_check.py`](run_leakage_check.py)
- [`check_heldout_overlap.py`](check_heldout_overlap.py)

Result files the index names:

- `own_all_fit_2026-10-15_173309.json` (not copied: result left out to keep the release under 120 MB)

## As-run versions

The copies above are the folder's latest versions. These versions, which differ from them, produced results the index names or the post uses (see each note):

- [`as_run/63096838c/run_own_all.py`](as_run/63096838c/run_own_all.py) from commit `63096838c`: Fitted and scored the post's on-policy probe (Table 1, Appendix A). The copy in this folder's snapshot is a newer version of the script.

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. Other result files, from earlier or side runs, give 270 script-and-commit pairs that match, 102 that differ and 279 where the script was not in the folder; we did not copy those versions. The differing versions behind results the index names or the post uses are under `as_run/` (above).

## Files here

27 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_p2_family_expansion/` at commit `1f5ca22318beecebff4c0c1fa76370f3109a3fb2` of our internal repository. 464 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). The 1 file under `as_run/` come from the commits named above.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
