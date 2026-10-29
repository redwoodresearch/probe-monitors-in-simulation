# 2026-10-16_examples_v81

2026-10-16 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

What do the probes' alarms look like: random false alarms and one random attack per outcome group?

## What we did

Seeded draws of training examples, six false alarms (pooled over probes and settings) and four attacks (one per caught/missed group, on-policy probe, commands), plus threshold and outcome tables, rebuilt on the final main pooled results record at 1% and 5% alarm levels; Modal CPU.

## What we found

The post's Table 4 context: 1,304 tasks caught in both real run 1 and simulation, 6 only real, 2 only simulated, 1 neither.

## In the post?

In the post: Appendix A Table 2 (on-policy training examples); Appendix B Tables 3 and 4

## Status

Current

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) main record

## Main scripts

- [`run_examples_v81.py`](run_examples_v81.py)
- [`run_tables_v81.py`](run_tables_v81.py)
- [`run_outcome_meta_v81.py`](run_outcome_meta_v81.py)

Result files the index names:

- `examples_v81_tables_pooled_2026-10-20_003359.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `oaru_training_keys_2026-10-16_132653.json` (not in this folder at the source commit)

## As-run versions

The copies above are the folder's latest versions. These versions, which differ from them, produced results the index names or the post uses (see each note):

- [`as_run/4cc8da1ba/run_examples_v81.py`](as_run/4cc8da1ba/run_examples_v81.py) from commit `4cc8da1ba`: Counted the on-policy probe's training runs (record oaru_keys). The copy in this folder's snapshot comes from a different line of the history.

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. For the 1 result file the index names, they match in 3 script-and-commit pairs. The differing versions behind results the index names or the post uses are under `as_run/` (above).

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-16_examples_v81/` at commit `34c9c5998ae209c6485a5f0ad556a9d75b4f2c7a` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). The 1 file under `as_run/` come from the commits named above.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
