# 2026-10-12_q2_e1_equiv_check

2026-10-12 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Does the hide-or-mention confirmation (deception probe on concealment) give the same probe scores with its cached-prefix scoring as with one uncached forward pass on its long contexts?

## What we did

On one H200 (\$0.75), rescored 33 sequences from 8 runs (the 3 longest, up to 210,770 tokens, and 5 seeded) in one uncached pass through layer 31 and compared report-mean scores with the stored cached scores; post hoc checks on the effect size and the masked readout.

## What we found

README: "cached scoring verified": largest report-mean difference 0.0030 against a 0.0084 tolerance; single tokens differ up to 0.198; post hoc the effect size moves by 0.008 and 0.006 in 2 runs against the hide-or-mention test's +0.58.

## In the post?

Not in the post.

## Status

Current (preliminary)

## Builds on

[2026-10-11_q2_e1_confirm](../2026-10-11_q2_e1_confirm/)

## Main scripts

- [`run_equiv_check.py`](run_equiv_check.py)
- [`analyze_equiv_check.py`](analyze_equiv_check.py)
- [`analyze_d_change_POSTHOC.py`](analyze_d_change_POSTHOC.py)
- [`analyze_masked_readout_POSTHOC.py`](analyze_masked_readout_POSTHOC.py)

Result files the index names:

- `equiv_check_2026-10-12_071806.json` (not in this folder at the source commit)
- `d_change_POSTHOC_2026-10-12_071928.json` (not in this folder at the source commit)
- `masked_readout_POSTHOC_2026-10-12_074711.json` (not in this folder at the source commit)

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q2_e1_equiv_check/` at commit `400ca5e7e5769b94870bb8b406fc2368bbef54c1` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
