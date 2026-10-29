# 2026-10-15_fpr_sweep_507_prep

2026-10-15 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

Main pipeline: on all 1,313 tasks, how often do the six probes catch simulated vs real attack runs, at each alarm level, with intervals from resampling main tasks?

## What we did

Started as the unit-level false-alarm sweep for the 507-task scale-up; became the final pooled pipeline (the final pooled analysis): input freezes (INPUTS_FINAL/FULL), exclusions, epoch 1+2 pools, run_peraction311.py producing the main records (pooled, 1% and 5%), the sweep-of-levels record and the strict-levels record, plus group catch/excess, per-task, kappa, detector-vs-drop, appendix rates and figure ports; Modal CPU.

## What we found

The post (reading the records via slots): six-probe simulated minus real catch -1.0 points (95% -3.0 to +1.0); real run 2 minus run 1 -0.2 (Table 5).

## In the post?

In the post: Figure 1; Tables 1, 5, 8; Figures 4-7; Appendices A, C, D (gap comparison), F, H, I; Appendix D's extra-probe comparison; the footnotes on task counts, the four probes' intervals and the tool-output intervals

## Status

Current

## Builds on

[2026-10-15_cut_sensitivity](../2026-10-15_cut_sensitivity/); [2026-10-15_epoch2_poolkeys](../2026-10-15_epoch2_poolkeys/); [2026-10-16_own_all_rs](../2026-10-16_own_all_rs/); 2026-10-15_q3_\* launches

## Main scripts

- [`run_peraction311.py`](run_peraction311.py)
- [`run_group_catch.py`](run_group_catch.py)
- [`run_group_excess.py`](run_group_excess.py)
- [`run_pw_shares_routec.py`](run_pw_shares_routec.py)
- [`run_pertask_anchor.py`](run_pertask_anchor.py)
- [`run_kappa_extra.py`](run_kappa_extra.py)
- [`run_detector_vs_drop.py`](run_detector_vs_drop.py)
- [`make_exclusions.py`](make_exclusions.py)

Result files the index names:

- `routec_A_..._2026-10-19_180327.json` (not in this folder at the source commit)

## As-run versions

The copies above are the folder's latest versions. These versions, which differ from them, produced results the index names or the post uses (see each note):

- [`as_run/7797f2cc2/run_peraction311.py`](as_run/7797f2cc2/run_peraction311.py) from commit `7797f2cc2`: Made the agreement (kappa) and per-task records (Figures 6 and 7, Appendix H). The main record (Figure 1, Tables 5 and 8) came from commit f2d47f19f, whose version is byte-identical to the copy in this folder's snapshot.
- [`as_run/35dff7132/make_appendix_a_rates.py`](as_run/35dff7132/make_appendix_a_rates.py) from commit `35dff7132`: Computed the failure rates in Appendix A. It is on another branch only (its one commit), not in this folder's snapshot.
- [`as_run/7be398471/run_peraction311.py`](as_run/7be398471/run_peraction311.py) from commit `7be398471`: Made the record behind the post's Table 1. The copy in this folder's snapshot comes from a different line of the history.
- [`as_run/d8b087792/run_detector_vs_drop.py`](as_run/d8b087792/run_detector_vs_drop.py) from commit `d8b087792`: Made the detector-versus-drop input to the statistics tables read in [2026-10-17_routec_slot_readout](../2026-10-17_routec_slot_readout/). The copy in this folder's snapshot is a newer version of the script.

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. Other result files, from earlier or side runs, give 20 script-and-commit pairs that match, 47 that differ and 389 where the script was not in the folder; we did not copy those versions. The differing versions behind results the index names or the post uses are under `as_run/` (above).

## Files here

201 code and configuration files and 10 result files, copied byte for byte from `experiments/2026-10-15_fpr_sweep_507_prep/` at commit `904bb61c5b94a41557bcb60bb273f3fd68a29c1c` of our internal repository. 604 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). We also copied 6 main scripts that the index names but that commit lacks, each from its own latest commit on another branch; the manifest gives each one's source commit. The 4 files under `as_run/` come from the commits named above.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
